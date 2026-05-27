"""
Agent Tool System - 工具调用和ReAct推理框架
支持Function Calling、Tool Use和多Agent协作
"""

import json
import re
import inspect
import asyncio
from typing import (
    Dict, List, Any, Optional, Callable, Union, 
    Type, Tuple, get_type_hints, get_origin, get_args
)
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from functools import wraps
import logging

from . import NexusForCausalLM, NexusTokenizer


logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema格式
    func: Callable = field(default=None, repr=False)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """转换为OpenAI格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


@dataclass
class ToolCall:
    """工具调用"""
    id: str
    name: str
    arguments: Dict[str, Any]
    result: Any = None
    error: str = None


@dataclass
class Message:
    """对话消息"""
    role: MessageRole
    content: str
    tool_calls: List[ToolCall] = None
    tool_call_id: str = None
    name: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    }
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.name:
            result["name"] = self.name
        return result


class ToolResult:
    """工具执行结果"""
    def __init__(self, tool_call_id: str, content: str, error: str = None):
        self.tool_call_id = tool_call_id
        self.content = content
        self.error = error
    
    def to_message(self) -> Message:
        """转换为消息"""
        return Message(
            role=MessageRole.TOOL,
            content=self.content if not self.error else f"Error: {self.error}",
            tool_call_id=self.tool_call_id,
        )


class BaseTool(ABC):
    """工具基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """参数定义 (JSON Schema)"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具"""
        pass
    
    def to_tool(self) -> Tool:
        """转换为Tool对象"""
        return Tool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            func=self.execute,
        )


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._instances: Dict[str, BaseTool] = {}
    
    def register(self, tool: Union[Tool, BaseTool]) -> None:
        """注册工具"""
        if isinstance(tool, BaseTool):
            tool_obj = tool.to_tool()
            self._instances[tool.name] = tool
        else:
            tool_obj = tool
        
        self._tools[tool_obj.name] = tool_obj
        logger.info(f"Registered tool: {tool_obj.name}")
    
    def unregister(self, name: str) -> None:
        """取消注册工具"""
        if name in self._tools:
            del self._tools[name]
        if name in self._instances:
            del self._instances[name]
    
    def get(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tools.get(name)
    
    def list_tools(self) -> List[Tool]:
        """列出所有工具"""
        return list(self._tools.values())
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """获取所有工具的schema"""
        return [tool.to_openai_format() for tool in self._tools.values()]
    
    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """执行工具"""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult("", f"Tool '{name}' not found", error="Tool not found")
        
        try:
            if name in self._instances:
                result = self._instances[name].execute(**arguments)
            elif tool.func:
                result = tool.func(**arguments)
            else:
                result = "Tool has no executable function"
            
            return ToolResult("", str(result))
        except Exception as e:
            return ToolResult("", "", error=str(e))


class ToolRegistryDecorator:
    """工具注册装饰器"""
    
    _registry: Optional[ToolRegistry] = None
    
    @classmethod
    def set_registry(cls, registry: ToolRegistry):
        """设置全局注册表"""
        cls._registry = registry
    
    @classmethod
    def tool(
        cls,
        name: str = None,
        description: str = None,
    ):
        """装饰器：自动注册函数为工具"""
        def decorator(func: Callable) -> Callable:
            nonlocal name, description
            
            func_name = name or func.__name__
            func_desc = description or func.__doc__ or ""
            
            # 获取类型提示生成参数schema
            hints = get_type_hints(func)
            sig = inspect.signature(func)
            
            properties = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                if param_name in ('self', 'cls'):
                    continue
                
                param_type = hints.get(param_name, str)
                json_type = cls._python_type_to_json(param_type)
                
                properties[param_name] = {
                    "type": json_type,
                    "description": f"Parameter {param_name}",
                }
                
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)
            
            parameters = {
                "type": "object",
                "properties": properties,
                "required": required,
            }
            
            tool = Tool(
                name=func_name,
                description=func_desc.strip(),
                parameters=parameters,
                func=func,
            )
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            # 注册到全局注册表
            if cls._registry:
                cls._registry.register(tool)
            
            return wrapper
        
        return decorator
    
    @staticmethod
    def _python_type_to_json(py_type: Type) -> str:
        """Python类型转JSON Schema类型"""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        
        origin = get_origin(py_type)
        if origin is Union:
            args = get_args(py_type)
            # 取第一个非None类型
            for arg in args:
                if arg is not type(None):
                    return ToolRegistryDecorator._python_type_to_json(arg)
        
        return type_map.get(py_type, "string")


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent
    实现思考-行动-观察循环
    """
    
    def __init__(
        self,
        model: NexusForCausalLM,
        tokenizer: NexusTokenizer,
        tools: Union[ToolRegistry, List[Tool]],
        max_iterations: int = 10,
        system_prompt: str = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_iterations = max_iterations
        
        # 设置工具
        if isinstance(tools, ToolRegistry):
            self.tool_registry = tools
        else:
            self.tool_registry = ToolRegistry()
            for tool in tools:
                self.tool_registry.register(tool)
        
        # 默认系统提示
        if system_prompt is None:
            system_prompt = """你是一个智能助手，可以使用工具来完成任务。

你具有以下工具可用：
{tool_schemas}

对于每个用户请求，你应该：
1. 思考需要做什么
2. 如果需要使用工具，决定使用哪个工具及参数
3. 执行工具并观察结果
4. 根据结果决定下一步行动
5. 最终给出回答

格式要求：
- 当需要使用工具时，输出：Action: tool_name\nAction Input: {{"param": "value"}}
- 当完成时，输出：Final Answer: 你的回答
"""
        
        self.system_prompt = system_prompt
    
    def _format_messages(self, messages: List[Message]) -> str:
        """格式化消息为文本"""
        formatted = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                formatted.append(f"System: {msg.content}")
            elif msg.role == MessageRole.USER:
                formatted.append(f"User: {msg.content}")
            elif msg.role == MessageRole.ASSISTANT:
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        formatted.append(f"Action: {tc.name}\nAction Input: {json.dumps(tc.arguments)}")
                else:
                    formatted.append(f"Assistant: {msg.content}")
            elif msg.role == MessageRole.TOOL:
                formatted.append(f"Observation: {msg.content}")
        
        return "\n\n".join(formatted)
    
    def _parse_action(self, text: str) -> Tuple[Optional[str], Optional[Dict], Optional[str]]:
        """解析模型输出中的Action和Action Input"""
        # 匹配 Action: tool_name
        action_match = re.search(r'Action:\s*(\w+)', text)
        if not action_match:
            return None, None, None
        
        tool_name = action_match.group(1)
        
        # 匹配 Action Input: {...}
        input_match = re.search(r'Action Input:\s*(\{[^}]+\}|\[[^\]]+\]|"[^"]*")', text, re.DOTALL)
        
        arguments = {}
        if input_match:
            input_str = input_match.group(1)
            try:
                arguments = json.loads(input_str)
            except json.JSONDecodeError:
                return tool_name, {}, f"Failed to parse arguments: {input_str}"
        
        # 检查Final Answer
        final_match = re.search(r'Final Answer:\s*(.*)', text, re.DOTALL)
        final_answer = final_match.group(1) if final_match else None
        
        return tool_name, arguments, final_answer
    
    async def think(
        self,
        prompt: str,
        messages: List[Message] = None,
    ) -> Tuple[str, List[ToolCall]]:
        """
        单步思考：给定当前状态，决定下一步行动
        """
        messages = messages or []
        
        # 构建完整上下文
        context = self._format_messages(messages)
        
        # 构建输入
        tool_schemas = json.dumps(self.tool_registry.get_tool_schemas(), ensure_ascii=False, indent=2)
        system = self.system_prompt.format(tool_schemas=tool_schemas)
        
        full_input = f"{system}\n\n{context}\n\nUser: {prompt}\n\nAssistant:"
        
        # 生成
        input_ids = self.tokenizer.encode(full_input, add_bos=True, add_eos=False)
        input_ids = input_ids[:self.tokenizer.config.max_seq_length]
        
        with torch.no_grad():
            output_ids = self.model.generate(
                torch.tensor([input_ids], device=self.model.device),
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
            )
        
        output_text = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        
        # 解析Action
        tool_name, arguments, final_answer = self._parse_action(output_text)
        
        tool_calls = []
        if tool_name:
            tc = ToolCall(
                id=f"call_{len(messages)}_{tool_name}",
                name=tool_name,
                arguments=arguments,
            )
            tool_calls.append(tc)
        
        return final_answer, tool_calls
    
    async def run(self, prompt: str) -> str:
        """
        运行Agent处理请求
        """
        messages: List[Message] = []
        
        for iteration in range(self.max_iterations):
            # 思考
            final_answer, tool_calls = await self.think(prompt, messages)
            
            if final_answer:
                return final_answer
            
            if not tool_calls:
                # 没有工具调用但也没有最终答案，可能是模型输出格式问题
                messages.append(Message(
                    role=MessageRole.ASSISTANT,
                    content="I need to think more carefully about this.",
                ))
                continue
            
            # 执行工具
            for tc in tool_calls:
                result = self.tool_registry.execute(tc.name, tc.arguments)
                
                if result.error:
                    messages.append(Message(
                        role=MessageRole.ASSISTANT,
                        content=f"Error executing {tc.name}: {result.error}",
                    ))
                else:
                    messages.append(Message(
                        role=MessageRole.TOOL,
                        content=result.content,
                        tool_call_id=tc.id,
                    ))
        
        # 达到最大迭代次数
        return "我需要更多信息来回答您的问题。"
    
    def run_sync(self, prompt: str) -> str:
        """同步运行Agent"""
        return asyncio.get_event_loop().run_until_complete(self.run(prompt))


class FunctionCallingAgent(ReActAgent):
    """
    Function Calling Agent
    专门处理函数调用场景
    """
    
    def __init__(
        self,
        model: NexusForCausalLM,
        tokenizer: NexusTokenizer,
        tools: List[Tool],
        strict: bool = True,
    ):
        super().__init__(model, tokenizer, tools)
        self.strict = strict
    
    async def chat(
        self,
        messages: List[Message],
        stream: bool = False,
    ) -> Tuple[str, List[ToolCall]]:
        """
        聊天接口，返回文本回复和工具调用
        """
        # 格式化消息
        formatted = self._format_messages(messages)
        
        # 构建输入
        tool_schemas = json.dumps(self.tool_registry.get_tool_schemas(), ensure_ascii=False)
        system = f"""你是一个智能助手。工具定义：{tool_schemas}

当用户请求需要执行操作时，使用工具。
当只需要回答问题时，直接回复。"""
        
        full_input = f"{system}\n\n{formatted}"
        
        # 生成
        input_ids = self.tokenizer.encode(full_input, add_bos=True)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                torch.tensor([input_ids], device=self.model.device),
                max_new_tokens=512,
                temperature=0.7,
            )
        
        output_text = self.tokenizer.decode(output_ids[0].tolist())
        
        # 解析
        tool_calls = self._extract_function_calls(output_text)
        
        if tool_calls:
            return None, tool_calls
        else:
            return output_text, []
    
    def _extract_function_calls(self, text: str) -> List[ToolCall]:
        """从文本中提取函数调用"""
        # 尝试匹配函数调用格式
        pattern = r'```(?:json)?\s*(\{[^}]*"name"[^}]*\})\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        tool_calls = []
        for i, match in enumerate(matches):
            try:
                data = json.loads(match)
                tool_calls.append(ToolCall(
                    id=f"call_{i}",
                    name=data.get("name", ""),
                    arguments=data.get("arguments", {}),
                ))
            except json.JSONDecodeError:
                continue
        
        return tool_calls


class MultiAgentCoordinator:
    """
    多Agent协作协调器
    支持多个专业Agent协作完成任务
    """
    
    def __init__(self):
        self.agents: Dict[str, ReActAgent] = {}
        self.coordinator_prompt = """你是一个任务协调员。

可用Agent：
{agent_descriptions}

根据任务需求，选择合适的Agent来处理。
"""
    
    def register_agent(self, name: str, agent: ReActAgent, description: str):
        """注册Agent"""
        self.agents[name] = agent
        self.agent_descriptions = getattr(self, 'agent_descriptions', {})
        self.agent_descriptions[name] = description
    
    async def delegate_task(
        self,
        task: str,
        available_agents: List[str] = None,
    ) -> str:
        """
        将任务委托给合适的Agent
        """
        agents_to_use = available_agents or list(self.agents.keys())
        
        results = {}
        
        # 并行执行（如果允许多个Agent）
        tasks = []
        for agent_name in agents_to_use:
            if agent_name in self.agents:
                tasks.append(self.agents[agent_name].run(task))
        
        if tasks:
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            for agent_name, result in zip(agents_to_use, results_list):
                if isinstance(result, Exception):
                    results[agent_name] = f"Error: {result}"
                else:
                    results[agent_name] = result
        
        # 整合结果
        final_response = "\n\n".join([
            f"[{name}]: {result}"
            for name, result in results.items()
        ])
        
        return final_response


# ============================================================================
# 内置工具示例
# ============================================================================

class Calculator(BaseTool):
    """计算器工具"""
    
    @property
    def name(self) -> str:
        return "calculate"
    
    @property
    def description(self) -> str:
        return "执行数学计算，支持加减乘除和常见数学函数"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '2 + 3 * 4' 或 'sin(pi/2)'",
                }
            },
            "required": ["expression"],
        }
    
    def execute(self, expression: str) -> str:
        try:
            # 安全评估（仅支持基本运算）
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return f"Error: Invalid characters in expression"
            
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"


class WebSearch(BaseTool):
    """网络搜索工具"""
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "搜索互联网获取最新信息"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
                "num_results": {
                    "type": "integer",
                    "description": "返回结果数量",
                    "default": 5,
                }
            },
            "required": ["query"],
        }
    
    def execute(self, query: str, num_results: int = 5) -> str:
        # 这里应该调用实际的搜索API
        return f"Searching for '{query}'... (Web search API not configured)"


class KnowledgeBase(BaseTool):
    """知识库查询工具"""
    
    def __init__(self):
        self.knowledge: Dict[str, str] = {}
    
    @property
    def name(self) -> str:
        return "knowledge_lookup"
    
    @property
    def description(self) -> str:
        return "查询企业内部知识库"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "查询内容",
                },
                "category": {
                    "type": "string",
                    "description": "知识类别（可选）",
                    "enum": ["product", "policy", "faq", "general"],
                }
            },
            "required": ["query"],
        }
    
    def execute(self, query: str, category: str = None) -> str:
        # 这里应该查询实际的知识库
        return f"Knowledge base query for '{query}' in category '{category or 'general'}'..."
    
    def add_knowledge(self, key: str, value: str):
        """添加知识"""
        self.knowledge[key] = value
    
    def remove_knowledge(self, key: str):
        """删除知识"""
        if key in self.knowledge:
            del self.knowledge[key]


# ============================================================================
# 使用示例
# ============================================================================

def agent_example():
    """Agent使用示例"""
    print("=" * 60)
    print("Agent System Example")
    print("=" * 60)
    
    print("""
    # 1. 创建工具注册表
    registry = ToolRegistry()
    
    # 2. 注册内置工具
    registry.register(Calculator())
    registry.register(WebSearch())
    
    # 3. 注册自定义工具
    @ToolRegistryDecorator.tool(name="custom_tool", description="自定义工具")
    def custom_function(param1: str, param2: int) -> str:
        return f"Result: {param1}, {param2}"
    
    # 4. 创建Agent
    agent = ReActAgent(
        model=model,
        tokenizer=tokenizer,
        tools=registry,
    )
    
    # 5. 运行Agent
    result = agent.run_sync("请计算 (15 + 25) * 3 的结果")
    print(result)
    
    # 6. Function Calling模式
    fc_agent = FunctionCallingAgent(
        model=model,
        tokenizer=tokenizer,
        tools=registry.list_tools(),
    )
    
    messages = [
        Message(role=MessageRole.USER, content="请帮我计算 100 / 5"),
    ]
    
    response, tool_calls = await fc_agent.chat(messages)
    
    if tool_calls:
        for tc in tool_calls:
            result = registry.execute(tc.name, tc.arguments)
            print(f"Tool result: {result.content}")
    """)
    
    print("\n" + "=" * 60)
    print("Multi-Agent Example")
    print("=" * 60)
    
    print("""
    # 1. 创建多个专业Agent
    finance_agent = ReActAgent(model, tokenizer, finance_tools)
    legal_agent = ReActAgent(model, tokenizer, legal_tools)
    general_agent = ReActAgent(model, tokenizer, general_tools)
    
    # 2. 创建协调器
    coordinator = MultiAgentCoordinator()
    coordinator.register_agent("finance", finance_agent, "处理金融相关问题")
    coordinator.register_agent("legal", legal_agent, "处理法律咨询")
    coordinator.register_agent("general", general_agent, "处理通用问题")
    
    # 3. 委托任务
    result = await coordinator.delegate_task(
        task="分析这笔投资的税务影响",
        available_agents=["finance", "legal"],
    )
    """)
