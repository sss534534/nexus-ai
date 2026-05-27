# Nexus-LLM Agent 开发指南

> 本文档详细介绍如何使用 Nexus-LLM 的 Agent 框架开发智能体应用，包括自定义工具开发、ReAct Agent 和 Function Calling Agent 的使用、多 Agent 协调以及与外部 API 的集成。

---

## 目录

- [Agent 框架概览](#agent-框架概览)
- [创建自定义工具](#创建自定义工具)
  - [工具基类](#工具基类)
  - [工具开发示例](#工具开发示例)
  - [工具注册与管理](#工具注册与管理)
- [ReAct Agent](#react-agent)
  - [工作原理](#工作原理)
  - [基本使用](#基本使用)
  - [高级配置](#高级配置)
- [Function Calling Agent](#function-calling-agent)
  - [工作原理](#工作原理)
  - [基本使用](#基本使用)
  - [动态函数注册](#动态函数注册)
- [多 Agent 协调](#多-agent-协调)
  - [Agent 编排器](#agent-编排器)
  - [Agent 通信](#agent-通信)
  - [实际示例](#实际示例)
- [内置工具参考](#内置工具参考)
  - [SearchTool](#searchtool)
  - [CalculatorTool](#calculatortool)
  - [CodeExecutorTool](#codeexecutortool)
  - [FileReaderTool](-filereadertool)
  - [WebBrowserTool](-webbrowsertool)
  - [DatabaseTool](-databasetool)
- [外部 API 集成](#外部-api-集成)
  - [REST API 集成](#rest-api-集成)
  - [GraphQL API 集成](#graphql-api-集成)
  - [WebSocket 集成](#websocket-集成)

---

## Agent 框架概览

Nexus-LLM Agent 框架提供了一套完整的智能体开发工具包，支持两种主流 Agent 范式：

```
+=====================================================================+
|                      Agent 框架架构                                  |
+=====================================================================+
                                                                     |
  +------------------------------------------------------------------+ |
  |                        Agent 层                                   | |
  |                                                                  | |
  |  +------------------+          +----------------------------+   | |
  |  |   ReAct Agent    |          |  Function Calling Agent    |   | |
  |  |  (推理+行动循环)  |          |  (原生函数调用)             |   | |
  |  +--------+---------+          +-------------+--------------+   | |
  |           |                                |                   | |
  |           +----------------+---------------+                   | |
  |                            |                                   | |
  |                   +--------v--------+                          | |
  |                   |   ToolRegistry   |                          | |
  |                   |  (工具注册中心)   |                          | |
  |                   +--------+--------+                          | |
  |                            |                                   | |
  +----------------------------+-----------------------------------+ |
                               |                                      |
  +----------------------------+-----------------------------------+ |
  |                        工具层                                    | |
  |                                                                  | |
  |  +----------+ +----------+ +----------+ +----------+ +--------+ | |
  |  | Search   | |Calculator| | Code     | | File     | | Web    | | |
  |  | Tool     | | Tool     | | Executor | | Reader   | |Browser | | |
  |  +----------+ +----------+ +----------+ +----------+ +--------+ | |
  |                                                                  | |
  |  +----------+ +----------+ +----------+ +----------+           | |
  |  | Database | | Email    | | Calendar | | Custom   |           | |
  |  | Tool     | | Tool     | | Tool     | | Tools    |           | |
  |  +----------+ +----------+ +----------+ +----------+           | |
  +------------------------------------------------------------------+ |
                                                                     |
+=====================================================================+
```

### 两种 Agent 范式对比

| 特性 | ReAct Agent | Function Calling Agent |
|------|------------|----------------------|
| 工作方式 | 思考-行动-观察循环 | 模型原生函数调用 |
| 推理过程 | 可解释（输出思考链） | 高效（直接调用） |
| 适用场景 | 复杂推理、多步任务 | 简单工具调用、API 集成 |
| 模型要求 | 通用模型即可 | 需要支持函数调用 |
| 延迟 | 较高（多轮交互） | 较低（单次调用） |
| 灵活性 | 高（可动态调整策略） | 中（依赖模型能力） |

---

## 创建自定义工具

### 工具基类

所有自定义工具必须继承 `BaseTool` 并实现 `execute` 方法：

```python
from nexus_llm.agent import BaseTool
from typing import Any, Optional, Dict, List
from pydantic import BaseModel, Field

class MyCustomTool(BaseTool):
    """自定义工具示例。

    每个工具需要定义:
    - name: 工具名称（唯一标识）
    - description: 工具描述（供模型理解工具用途）
    - parameters_schema: 参数的 JSON Schema 定义
    - execute(): 工具执行逻辑
    """

    name: str = "my_custom_tool"
    description: str = "这是一个自定义工具，用于执行特定任务"

    # 使用 Pydantic 定义参数（推荐）
    class InputSchema(BaseModel):
        query: str = Field(..., description="查询关键词")
        limit: int = Field(default=10, description="返回结果数量上限")
        category: Optional[str] = Field(default=None, description="结果分类过滤")

    parameters_schema = InputSchema.schema()

    def execute(self, query: str, limit: int = 10, category: Optional[str] = None) -> str:
        """执行工具逻辑。

        参数:
            query: 查询关键词
            limit: 返回结果数量上限
            category: 可选的分类过滤

        返回:
            str: 工具执行结果（文本格式）
        """
        # 在这里实现你的工具逻辑
        results = self._do_something(query, limit, category)
        return str(results)

    def _do_something(self, query, limit, category):
        """内部实现方法。"""
        # ... 实际业务逻辑
        return [{"result": f"查询 '{query}' 的结果"}]
```

### 工具开发示例

#### 示例 1：天气查询工具

```python
"""weather_tool.py - 天气查询工具"""
import requests
from nexus_llm.agent import BaseTool
from typing import Optional
from pydantic import BaseModel, Field


class WeatherTool(BaseTool):
    """天气查询工具，获取指定城市的天气信息。"""

    name: str = "get_weather"
    description: str = (
        "获取指定城市的当前天气信息，包括温度、湿度、风速等。"
        "当用户询问天气相关问题时使用此工具。"
    )

    class InputSchema(BaseModel):
        city: str = Field(..., description="城市名称，如'北京'、'上海'")
        unit: str = Field(default="celsius", description="温度单位：celsius 或 fahrenheit")

    parameters_schema = InputSchema.schema()

    def __init__(self, api_key: str = ""):
        super().__init__()
        self.api_key = api_key

    def execute(self, city: str, unit: str = "celsius") -> str:
        """查询天气信息。"""
        try:
            # 调用天气 API
            response = requests.get(
                "https://api.weather.example.com/v1/current",
                params={"city": city, "unit": unit, "key": self.api_key},
                timeout=10,
            )
            data = response.json()

            if response.status_code == 200:
                temp = data["temperature"]
                humidity = data["humidity"]
                wind = data["wind_speed"]
                desc = data["description"]

                return (
                    f"{city}当前天气：\n"
                    f"  温度：{temp}°{'C' if unit == 'celsius' else 'F'}\n"
                    f"  湿度：{humidity}%\n"
                    f"  风速：{wind} km/h\n"
                    f"  天气状况：{desc}"
                )
            else:
                return f"查询失败：{data.get('message', '未知错误')}"

        except requests.Timeout:
            return "查询超时，请稍后重试"
        except Exception as e:
            return f"查询出错：{str(e)}"
```

#### 示例 2：数据库查询工具

```python
"""database_tool.py - 数据库查询工具"""
import sqlite3
from nexus_llm.agent import BaseTool
from typing import Optional
from pydantic import BaseModel, Field


class DatabaseQueryTool(BaseTool):
    """数据库查询工具，支持执行 SQL 查询。"""

    name: str = "database_query"
    description: str = (
        "查询数据库并返回结果。支持 SELECT 查询语句。"
        "可用于查询用户信息、订单数据、统计数据等。"
        "注意：仅支持只读查询，不支持修改操作。"
    )

    class InputSchema(BaseModel):
        query: str = Field(..., description="SQL 查询语句（仅支持 SELECT）")
        database: str = Field(default="main", description="数据库名称")

    parameters_schema = InputSchema.schema()

    # 允许的表名白名单（安全措施）
    ALLOWED_TABLES = {"users", "orders", "products", "categories"}

    def __init__(self, db_path: str = "./data/app.db"):
        super().__init__()
        self.db_path = db_path

    def validate_args(self, **kwargs) -> bool:
        """验证参数安全性。"""
        query = kwargs.get("query", "").upper().strip()

        # 仅允许 SELECT 语句
        if not query.startswith("SELECT"):
            return False

        # 检查是否包含危险操作
        dangerous_keywords = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"]
        for keyword in dangerous_keywords:
            if keyword in query:
                return False

        return True

    def execute(self, query: str, database: str = "main") -> str:
        """执行 SQL 查询。"""
        if not self.validate_args(query=query):
            return "错误：仅支持 SELECT 查询语句。"

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)

            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]

            if not rows:
                return "查询结果为空。"

            # 格式化输出
            result = [columns]
            for row in rows:
                result.append(list(row))

            # 生成表格格式的文本
            col_widths = [max(len(str(item)) for item in col) for col in zip(*result)]
            output_lines = []
            for i, row in enumerate(result):
                line = " | ".join(str(item).ljust(width) for item, width in zip(row, col_widths))
                output_lines.append(line)
                if i == 0:
                    output_lines.append("-+-".join("-" * width for width in col_widths))

            return f"查询结果（共 {len(rows)} 行）：\n```\n" + "\n".join(output_lines) + "\n```"

        except sqlite3.Error as e:
            return f"数据库错误：{str(e)}"
        finally:
            conn.close()
```

#### 示例 3：知识库检索工具

```python
"""knowledge_tool.py - 知识库检索工具"""
from nexus_llm.agent import BaseTool
from typing import List, Optional
from pydantic import BaseModel, Field


class KnowledgeRetrievalTool(BaseTool):
    """知识库检索工具，基于向量相似度搜索。"""

    name: str = "knowledge_search"
    description: str = (
        "从知识库中检索相关文档。适用于查找公司内部文档、"
        "产品手册、FAQ 等结构化知识。"
    )

    class InputSchema(BaseModel):
        query: str = Field(..., description="搜索查询文本")
        top_k: int = Field(default=5, description="返回最相关的文档数量")
        category: Optional[str] = Field(default=None, description="文档分类过滤")

    parameters_schema = InputSchema.schema()

    def __init__(self, index_path: str = "./data/knowledge_index"):
        super().__init__()
        self.index_path = index_path
        self._load_index()

    def _load_index(self):
        """加载向量索引。"""
        # 实际实现中可使用 FAISS、Milvus 等向量数据库
        pass

    def execute(self, query: str, top_k: int = 5, category: Optional[str] = None) -> str:
        """检索相关文档。"""
        # 1. 将查询文本转换为向量
        query_vector = self._embed(query)

        # 2. 在索引中搜索
        results = self._search(query_vector, top_k=top_k, category=category)

        # 3. 格式化输出
        if not results:
            return "未找到相关文档。"

        output = f"找到 {len(results)} 条相关文档：\n\n"
        for i, doc in enumerate(results, 1):
            output += f"[{i}] {doc['title']}\n"
            output += f"    相关度: {doc['score']:.2f}\n"
            output += f"    摘要: {doc['snippet']}\n\n"

        return output.strip()

    def _embed(self, text: str) -> List[float]:
        """文本向量化。"""
        # 实际实现中使用嵌入模型
        pass

    def _search(self, vector, top_k, category) -> List[dict]:
        """向量搜索。"""
        # 实际实现中使用向量数据库
        pass
```

### 工具注册与管理

```python
from nexus_llm.agent import ToolRegistry, BaseTool

# 创建工具注册中心
registry = ToolRegistry()

# 注册工具
registry.register(WeatherTool(api_key="your-key"))
registry.register(DatabaseQueryTool(db_path="./data/app.db"))
registry.register(KnowledgeRetrievalTool())

# 列出所有工具
for tool_info in registry.list_tools():
    print(f"  {tool_info['name']}: {tool_info['description']}")

# 搜索工具
tools = registry.search("数据库")
print(f"匹配'数据库'的工具: {[t.name for t in tools]}")

# 获取工具
weather_tool = registry.get("get_weather")
result = weather_tool.execute(city="北京")
print(result)

# 注销工具
registry.unregister("get_weather")

# 创建包含所有内置工具的注册中心
registry = ToolRegistry.from_default()
```

---

## ReAct Agent

### 工作原理

ReAct（Reasoning + Acting）Agent 通过交替执行"思考-行动-观察"循环来解决复杂任务：

```
用户: "2024年中国GDP是多少？换算成美元是多少？"

Agent 内部推理过程:

Thought 1: 用户询问2024年中国GDP，我需要先搜索这个数据。
Action 1: search("2024年中国GDP")
Observation 1: 2024年中国GDP为126.06万亿元人民币。

Thought 2: 我已经获得了中国GDP数据（126.06万亿元人民币），
          现在需要查询人民币对美元的汇率来进行换算。
Action 2: search("人民币对美元汇率 2024")
Observation 2: 2024年人民币对美元平均汇率约为7.10。

Thought 3: 我有了GDP数据和汇率，现在可以计算美元金额。
Action 3: calculator(126.06 / 7.10)
Observation 3: 17.76

Thought 4: 我已经获得了所有需要的信息，可以给出最终答案。
Answer: 2024年中国GDP约为126.06万亿元人民币，按平均汇率7.10计算，
        约合17.76万亿美元。
```

### 基本使用

```python
from nexus_llm.agent import ReActAgent, AgentConfig
from nexus_llm.agent.tools import SearchTool, CalculatorTool

# 创建 ReAct Agent
agent = ReActAgent(
    config=AgentConfig(
        model_path="nexus-ai/nexus-7b",
        max_iterations=10,           # 最大推理轮数
        max_tokens_per_step=512,     # 每步最大 token 数
        temperature=0.7,             # 生成温度
        verbose=True,                # 打印推理过程
        return_intermediate_steps=True,  # 返回中间步骤
    ),
    tools=[SearchTool(), CalculatorTool()],
    system_prompt=(
        "你是一个智能助手，能够使用工具来回答问题。"
        "对于需要计算或查找信息的问题，请使用可用工具。"
        "在给出最终答案前，确保你已经收集了足够的信息。"
    ),
)

# 执行任务
result = agent.run("2024年中国GDP是多少？换算成美元是多少？")

# 查看结果
print(f"\n最终答案: {result.answer}")
print(f"\n中间步骤:")
for step in result.intermediate_steps:
    print(f"  Thought: {step.thought}")
    print(f"  Action: {step.action}")
    print(f"  Observation: {step.observation}")
    print()
```

### 高级配置

```python
from nexus_llm.agent import ReActAgent, AgentConfig
from nexus_llm.agent.tools import SearchTool, CalculatorTool, CodeExecutorTool

# 高级配置
agent = ReActAgent(
    config=AgentConfig(
        model_path="nexus-ai/nexus-7b",
        max_iterations=15,
        max_tokens_per_step=1024,
        temperature=0.3,                # 降低温度以获得更确定性的推理
        verbose=True,
        return_intermediate_steps=True,
    ),
    tools=[
        SearchTool(api_key="your-search-api-key"),
        CalculatorTool(),
        CodeExecutorTool(timeout=30),    # 代码执行工具，30秒超时
    ],
    system_prompt=(
        "你是一个强大的研究助手。在回答问题时：\n"
        "1. 首先分析问题，确定需要哪些信息\n"
        "2. 使用搜索工具查找相关信息\n"
        "3. 使用计算器进行数值计算\n"
        "4. 如需复杂计算，使用代码执行工具\n"
        "5. 综合所有信息给出准确答案\n"
        "注意：始终引用信息来源，确保答案准确可靠。"
    ),
)

# 对话模式
history = []
while True:
    user_input = input("用户: ")
    if user_input.lower() in ("quit", "exit"):
        break

    response = agent.chat(user_input, history=history)
    print(f"助手: {response}")

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
```

---

## Function Calling Agent

### 工作原理

Function Calling Agent 利用模型原生的函数调用能力，通过结构化的 JSON 格式直接指定要调用的函数和参数：

```
用户: "北京和上海今天的天气怎么样？"

模型输出（函数调用）:
{
  "function_calls": [
    {"name": "get_weather", "arguments": {"city": "北京"}},
    {"name": "get_weather", "arguments": {"city": "上海"}}
  ]
}

工具执行结果:
[
  "北京：25°C，晴天，湿度 45%",
  "上海：28°C，多云，湿度 65%"
]

模型输出（最终回答）:
"北京今天25°C，晴天，湿度45%；上海今天28°C，多云，湿度65%。"
```

### 基本使用

```python
from nexus_llm.agent import FunctionCallingAgent, AgentConfig
from nexus_llm.agent.tools import SearchTool, CalculatorTool

# 创建 Function Calling Agent
agent = FunctionCallingAgent(
    config=AgentConfig(
        model_path="nexus-ai/nexus-7b",
        max_iterations=5,
        temperature=0.7,
    ),
    system_prompt="你是一个天气和数学助手。",
)

# 注册函数
agent.register_function(
    name="get_weather",
    description="获取指定城市的当前天气信息",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
        },
        "required": ["city"],
    },
    handler=lambda city, unit="celsius": f"{city}: 25{unit[0]}, 晴天",
)

agent.register_function(
    name="calculate",
    description="执行数学计算",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "数学表达式"},
        },
        "required": ["expression"],
    },
    handler=lambda expression: str(eval(expression)),
)

# 执行任务
result = agent.run("北京天气怎么样？如果温度是华氏度，换算成摄氏度是多少？")
print(result.answer)
```

### 动态函数注册

```python
# 动态注册和注销函数
agent.register_function(
    name="send_email",
    description="发送电子邮件",
    parameters={
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "收件人邮箱"},
            "subject": {"type": "string", "description": "邮件主题"},
            "body": {"type": "string", "description": "邮件正文"},
        },
        "required": ["to", "subject", "body"],
    },
    handler=send_email_handler,
)

# 根据用户权限动态注册函数
def setup_agent_for_user(user_role: str):
    agent = FunctionCallingAgent(config=AgentConfig(model_path="nexus-ai/nexus-7b"))

    # 所有用户可用的函数
    agent.register_function(name="search", ...)

    # 管理员专属函数
    if user_role == "admin":
        agent.register_function(name="delete_user", ...)
        agent.register_function(name="view_logs", ...)

    # 普通用户函数
    else:
        agent.register_function(name="view_profile", ...)

    return agent
```

---

## 多 Agent 协调

### Agent 编排器

```python
from nexus_llm.agent import AgentOrchestrator, AgentConfig, ReActAgent
from nexus_llm.agent.tools import SearchTool, CodeExecutorTool, DatabaseQueryTool

# 创建专业化的 Agent
researcher = ReActAgent(
    config=AgentConfig(model_path="nexus-ai/nexus-7b", max_iterations=5),
    tools=[SearchTool()],
    system_prompt="你是一个信息研究员，负责搜索和收集信息。给出简洁、准确的信息摘要。",
)

coder = ReActAgent(
    config=AgentConfig(model_path="nexus-ai/nexus-7b", max_iterations=5),
    tools=[CodeExecutorTool()],
    system_prompt="你是一个编程专家，负责编写和执行代码。给出完整的代码和运行结果。",
)

analyst = ReActAgent(
    config=AgentConfig(model_path="nexus-ai/nexus-7b", max_iterations=5),
    tools=[DatabaseQueryTool(), CalculatorTool()],
    system_prompt="你是一个数据分析师，负责分析数据并给出结论。",
)

# 创建编排器
orchestrator = AgentOrchestrator(
    agents={
        "researcher": researcher,
        "coder": coder,
        "analyst": analyst,
    },
    coordinator_model_path="nexus-ai/nexus-7b",
    max_rounds=10,
)

# 执行多 Agent 任务
result = orchestrator.run(
    task="分析最近一周的股票市场趋势，并用Python绘制趋势图",
    strategy="sequential",  # sequential / parallel / hierarchical
)

print(f"最终答案: {result.answer}")
print(f"执行路径: {result.execution_trace}")
```

### Agent 通信

```python
from nexus_llm.agent import MessageBus, AgentMessage

# 创建消息总线
bus = MessageBus()

# Agent 之间通过消息通信
class ResearchAgent(ReActAgent):
    def on_message(self, message: AgentMessage):
        """收到消息时的处理。"""
        if message.type == "data_request":
            # 执行搜索
            search_result = self.run(message.content)
            # 发送结果
            bus.send(
                AgentMessage(
                    sender=self.name,
                    receiver=message.sender,
                    type="data_response",
                    content=search_result.answer,
                )
            )

class AnalysisAgent(ReActAgent):
    def on_message(self, message: AgentMessage):
        """收到消息时的处理。"""
        if message.type == "data_response":
            # 分析数据
            analysis = self.run(f"分析以下数据：{message.content}")
            bus.send(
                AgentMessage(
                    sender=self.name,
                    receiver="coordinator",
                    type="analysis_result",
                    content=analysis.answer,
                )
            )
```

### 实际示例

```python
"""multi_agent_example.py - 多 Agent 协作示例"""
from nexus_llm.agent import AgentOrchestrator, ReActAgent, AgentConfig
from nexus_llm.agent.tools import SearchTool, CodeExecutorTool

# 场景：自动化研究报告生成

# 1. 信息收集 Agent
info_agent = ReActAgent(
    config=AgentConfig(model_path="nexus-ai/nexus-7b"),
    tools=[SearchTool()],
    system_prompt="你负责收集信息。根据给定的主题，搜索相关资料并整理成结构化的信息摘要。",
)

# 2. 数据分析 Agent
data_agent = ReActAgent(
    config=AgentConfig(model_path="nexus-ai/nexus-7b"),
    tools=[CodeExecutorTool()],
    system_prompt="你负责数据分析。根据提供的数据，进行统计分析并生成可视化图表。",
)

# 3. 写作 Agent
writer_agent = ReActAgent(
    config=AgentConfig(model_path="nexus-ai/nexus-7b"),
    system_prompt="你负责撰写报告。根据收集的信息和分析结果，撰写专业的研究报告。",
)

# 创建编排器
orchestrator = AgentOrchestrator(
    agents={
        "info_collector": info_agent,
        "data_analyst": data_agent,
        "writer": writer_agent,
    },
    coordinator_model_path="nexus-ai/nexus-7b",
)

# 定义工作流
workflow = {
    "steps": [
        {"agent": "info_collector", "task": "收集关于人工智能在教育领域应用的最新信息"},
        {"agent": "data_analyst", "task": "分析收集到的数据，生成趋势图表"},
        {"agent": "writer", "task": "根据信息和分析结果，撰写一份完整的研究报告"},
    ],
    "strategy": "sequential",
}

# 执行工作流
result = orchestrator.execute_workflow(workflow)
print(result.answer)
```

---

## 内置工具参考

### SearchTool

网络搜索工具。

```python
from nexus_llm.agent.tools import SearchTool

search_tool = SearchTool(
    api_key="your-api-key",           # 搜索 API 密钥
    engine="google",                   # 搜索引擎：google / bing / duckduckgo
    max_results=5,                     # 最大返回结果数
    language="zh-CN",                  # 搜索语言
)

# 直接使用
results = search_tool.execute(query="量子计算最新进展", max_results=3)
print(results)
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `api_key` | str | 必填 | 搜索 API 密钥 |
| `engine` | str | `google` | 搜索引擎 |
| `max_results` | int | 5 | 最大返回结果数 |
| `language` | str | `zh-CN` | 搜索语言 |

### CalculatorTool

数学计算工具。

```python
from nexus_llm.agent.tools import CalculatorTool

calc_tool = CalculatorTool(
    precision=10,                      # 计算精度
    safe_mode=True,                    # 安全模式（禁用危险函数）
)

# 直接使用
result = calc_tool.execute(expression="sqrt(144) * pi")
print(result)  # 37.6991118431
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `precision` | int | 10 | 浮点数精度 |
| `safe_mode` | bool | True | 启用安全模式 |

### CodeExecutorTool

代码执行工具，支持 Python 代码沙箱执行。

```python
from nexus_llm.agent.tools import CodeExecutorTool

code_tool = CodeExecutorTool(
    timeout=30,                        # 执行超时（秒）
    max_memory="512MB",                # 内存限制
    allowed_modules=["math", "json", "re", "datetime"],  # 允许导入的模块
)

# 直接使用
result = code_tool.execute(
    code="""
import math

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 计算前 10 个斐波那契数
results = [fibonacci(i) for i in range(10)]
print(results)
"""
)
print(result)
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `timeout` | int | 30 | 执行超时（秒） |
| `max_memory` | str | `512MB` | 内存限制 |
| `allowed_modules` | list | `[]` | 允许导入的模块白名单 |

### FileReaderTool

文件读取工具。

```python
from nexus_llm.agent.tools import FileReaderTool

file_tool = FileReaderTool(
    base_dir="/data/documents",        # 基础目录（安全限制）
    allowed_extensions=[".txt", ".md", ".csv", ".json"],  # 允许的文件类型
    max_file_size=10485760,            # 最大文件大小（10MB）
)

# 直接使用
content = file_tool.execute(file_path="report.txt")
print(content)
```

### WebBrowserTool

网页浏览工具，支持访问网页和提取内容。

```python
from nexus_llm.agent.tools import WebBrowserTool

browser_tool = WebBrowserTool(
    headless=True,                     # 无头模式
    timeout=30,                        # 页面加载超时
    max_pages=10,                      # 最大浏览页面数
)

# 直接使用
content = browser_tool.execute(url="https://example.com", action="read")
print(content)
```

### DatabaseTool

数据库操作工具。

```python
from nexus_llm.agent.tools import DatabaseTool

db_tool = DatabaseTool(
    connection_string="postgresql://user:pass@localhost:5432/mydb",
    read_only=True,                    # 只读模式
    max_rows=100,                      # 最大返回行数
    timeout=30,                        # 查询超时
)

# 直接使用
result = db_tool.execute(query="SELECT * FROM users WHERE age > 18 LIMIT 10")
print(result)
```

---

## 外部 API 集成

### REST API 集成

```python
"""rest_api_tool.py - REST API 集成工具"""
import requests
from nexus_llm.agent import BaseTool
from typing import Optional
from pydantic import BaseModel, Field


class RESTAPITool(BaseTool):
    """通用 REST API 调用工具。"""

    name: str = "rest_api_call"
    description: str = "调用 REST API 接口获取数据"

    class InputSchema(BaseModel):
        url: str = Field(..., description="API 端点 URL")
        method: str = Field(default="GET", description="HTTP 方法：GET, POST, PUT, DELETE")
        headers: Optional[dict] = Field(default=None, description="请求头")
        body: Optional[dict] = Field(default=None, description="请求体（JSON）")
        params: Optional[dict] = Field(default=None, description="查询参数")

    parameters_schema = InputSchema.schema()

    def __init__(self, base_url: str = "", default_headers: dict = None, timeout: int = 30):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.default_headers = default_headers or {}
        self.timeout = timeout

    def execute(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[dict] = None,
        body: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> str:
        """调用 REST API。"""
        full_url = f"{self.base_url}/{url.lstrip('/')}" if self.base_url else url
        merged_headers = {**self.default_headers, **(headers or {})}

        try:
            response = requests.request(
                method=method.upper(),
                url=full_url,
                json=body,
                params=params,
                headers=merged_headers,
                timeout=self.timeout,
            )

            return f"状态码: {response.status_code}\n响应: {response.text[:2000]}"

        except requests.Timeout:
            return f"请求超时（{self.timeout}秒）"
        except Exception as e:
            return f"请求失败：{str(e)}"


# 使用示例：集成 GitHub API
github_tool = RESTAPITool(
    base_url="https://api.github.com",
    default_headers={"Accept": "application/vnd.github.v3+json"},
)

# 注册到 Agent
agent.register_function(
    name="github_api",
    description="调用 GitHub REST API",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "API 路径，如 /repos/nexus-ai/nexus-llm"},
            "method": {"type": "string", "enum": ["GET", "POST"], "default": "GET"},
        },
        "required": ["url"],
    },
    handler=lambda url, method="GET": github_tool.execute(url=url, method=method),
)
```

### GraphQL API 集成

```python
"""graphql_tool.py - GraphQL API 集成工具"""
import requests
from nexus_llm.agent import BaseTool
from pydantic import BaseModel, Field


class GraphQLTool(BaseTool):
    """GraphQL API 调用工具。"""

    name: str = "graphql_query"
    description: str = "执行 GraphQL 查询"

    class InputSchema(BaseModel):
        query: str = Field(..., description="GraphQL 查询语句")
        variables: dict = Field(default={}, description="查询变量")
        operation_name: str = Field(default=None, description="操作名称")

    parameters_schema = InputSchema.schema()

    def __init__(self, endpoint: str, headers: dict = None):
        super().__init__()
        self.endpoint = endpoint
        self.headers = headers or {}

    def execute(self, query: str, variables: dict = None, operation_name: str = None) -> str:
        """执行 GraphQL 查询。"""
        payload = {
            "query": query,
            "variables": variables or {},
        }
        if operation_name:
            payload["operationName"] = operation_name

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=self.headers,
                timeout=30,
            )
            data = response.json()

            if "errors" in data:
                errors = data["errors"]
                return f"GraphQL 错误：{errors}"

            return f"查询结果：{data.get('data', {})}"

        except Exception as e:
            return f"查询失败：{str(e)}"
```

### WebSocket 集成

```python
"""websocket_tool.py - WebSocket 集成工具"""
import asyncio
import websockets
from nexus_llm.agent import BaseTool
from pydantic import BaseModel, Field


class WebSocketTool(BaseTool):
    """WebSocket 通信工具。"""

    name: str = "websocket_send"
    description: str = "通过 WebSocket 发送消息并接收响应"

    class InputSchema(BaseModel):
        message: str = Field(..., description="要发送的消息")
        endpoint: str = Field(..., description="WebSocket 端点 URL")

    parameters_schema = InputSchema.schema()

    def __init__(self, default_endpoint: str = None, timeout: int = 10):
        super().__init__()
        self.default_endpoint = default_endpoint
        self.timeout = timeout

    def execute(self, message: str, endpoint: str = None) -> str:
        """发送 WebSocket 消息。"""
        uri = endpoint or self.default_endpoint

        async def _send():
            async with websockets.connect(uri) as ws:
                await ws.send(message)
                response = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                return response

        try:
            return asyncio.run(_send())
        except asyncio.TimeoutError:
            return "WebSocket 响应超时"
        except Exception as e:
            return f"WebSocket 错误：{str(e)}"
```

---

> 更多信息请参阅 [API 参考](./api_reference.md) 和 [快速入门](./getting_started.md)。
