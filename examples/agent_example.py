"""
Agent Framework Example
Demonstrates how to use the Agent module with tool calling
"""
import asyncio
from typing import List
from nexus_llm.agent import (
    BaseTool,
    ToolResult,
    ToolRegistry,
    ReActAgent,
    FunctionCallingAgent,
    MultiAgentCoordinator,
    Message,
    ToolCall,
    Calculator,
    WebSearch,
    KnowledgeBase
)


# Custom Tool Examples
class WeatherTool(BaseTool):
    """Weather查询工具"""
    name = "get_weather"
    description = "获取指定城市的天气信息"
    parameters = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "温度单位"
            }
        },
        "required": ["city"]
    }
    
    def execute(self, city: str, unit: str = "celsius") -> ToolResult:
        # 模拟天气数据
        weather_data = {
            "北京": {"temp": 22, "condition": "晴"},
            "上海": {"temp": 25, "condition": "多云"},
            "深圳": {"temp": 28, "condition": "雷阵雨"},
        }
        
        if city in weather_data:
            data = weather_data[city]
            temp = data["temp"]
            if unit == "fahrenheit":
                temp = temp * 9/5 + 32
            
            return ToolResult(
                success=True,
                output=f"{city}天气：{data['condition']}，温度{temp}°{'F' if unit == 'fahrenheit' else 'C'}"
            )
        
        return ToolResult(success=False, error=f"未找到城市 {city} 的天气信息")


class DateTimeTool(BaseTool):
    """日期时间工具"""
    name = "get_datetime"
    description = "获取当前日期和时间"
    parameters = {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "description": "日期时间格式"
            }
        }
    }
    
    def execute(self, format: str = "%Y-%m-%d %H:%M:%S") -> ToolResult:
        from datetime import datetime
        now = datetime.now()
        return ToolResult(success=True, output=now.strftime(format))


class TextAnalysisTool(BaseTool):
    """文本分析工具"""
    name = "analyze_text"
    description = "分析文本的统计信息"
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "要分析的文本"
            },
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要计算的指标"
            }
        },
        "required": ["text"]
    }
    
    def execute(self, text: str, metrics: List[str] = None) -> ToolResult:
        if metrics is None:
            metrics = ["length", "words", "chars"]
        
        result = {}
        
        if "length" in metrics:
            result["字符数"] = len(text)
        if "words" in metrics:
            result["单词数"] = len(text.split())
        if "chars" in metrics:
            result["字符（不含空格）"] = len(text.replace(" ", ""))
        
        return ToolResult(success=True, output=str(result))


def example_tool_registry():
    """Example: Tool Registry Usage"""
    print("=" * 50)
    print("Example: Tool Registry")
    print("=" * 50)
    
    # Create registry
    registry = ToolRegistry()
    
    # Register built-in tools
    calc = Calculator()
    registry.register(calc)
    
    # Register custom tools
    weather = WeatherTool()
    registry.register(weather)
    
    datetime_tool = DateTimeTool()
    registry.register(datetime_tool)
    
    print(f"Registered tools: {registry.list_tools()}")
    
    # Use calculator
    result = registry.execute_tool("calculator", expression="(100 + 200) * 2")
    print(f"Calculator (100+200)*2 = {result.output}")
    
    # Use weather tool
    result = registry.execute_tool("get_weather", city="北京", unit="celsius")
    print(f"Weather: {result.output}")
    
    # Use datetime tool
    result = registry.execute_tool("get_datetime")
    print(f"Current time: {result.output}")
    
    return registry


async def example_react_agent(registry: ToolRegistry):
    """Example: ReAct Agent"""
    print("\n" + "=" * 50)
    print("Example: ReAct Agent")
    print("=" * 50)
    
    # Create agent with tools
    agent = ReActAgent(
        model_name="nexus-7b-chat",
        max_steps=5,
        tool_registry=registry
    )
    
    print(f"Agent: {agent.model_name}")
    print(f"Max steps: {agent.max_steps}")
    print(f"Available tools: {agent.tool_registry.list_tools()}")
    
    # Mock the think function for demonstration
    async def mock_think(prompt: str, messages=None):
        prompt_lower = prompt.lower()
        
        if "天气" in prompt or "weather" in prompt_lower:
            return "需要查询天气", [ToolCall(tool_name="get_weather", arguments={"city": "北京"})]
        elif "计算" in prompt or "calculate" in prompt_lower or "+" in prompt:
            return "需要计算", [ToolCall(tool_name="calculator", arguments={"expression": "15 * 25 + 100"})]
        elif "时间" in prompt or "time" in prompt_lower:
            return "需要获取时间", [ToolCall(tool_name="get_datetime", arguments={})]
        else:
            return "直接回答", []
    
    agent._think = mock_think
    
    # Run agent
    queries = [
        "北京今天的天气怎么样？",
        "帮我计算 (15 * 25 + 100) 的结果",
        "现在几点了？"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = await agent.run(query)
        print(f"Response: {response}")
    
    return agent


async def example_function_calling_agent():
    """Example: Function Calling Agent"""
    print("\n" + "=" * 50)
    print("Example: Function Calling Agent")
    print("=" * 50)
    
    # Create agent
    agent = FunctionCallingAgent(
        model_name="nexus-7b-function",
        temperature=0.0  # Lower temperature for function calling
    )
    
    # Register tools
    calc = Calculator()
    agent.tool_registry.register(calc)
    
    text_tool = TextAnalysisTool()
    agent.tool_registry.register(text_tool)
    
    print(f"Agent: {agent.model_name}")
    print(f"Temperature: {agent.temperature}")
    print(f"Tools: {agent.tool_registry.list_tools()}")
    
    # Mock function call generation
    async def mock_generate(prompt: str, messages=None):
        prompt_lower = prompt.lower()
        
        if "分析" in prompt and "文本" in prompt:
            return [ToolCall(
                tool_name="analyze_text",
                arguments={"text": "这是一个测试文本", "metrics": ["length", "words"]}
            )]
        elif "统计" in prompt:
            return [ToolCall(
                tool_name="analyze_text",
                arguments={"text": prompt, "metrics": ["length", "chars"]}
            )]
        return []
    
    agent._generate_function_calls = mock_generate
    
    # Run queries
    queries = [
        "分析文本'Hello World, this is a test!'的统计信息",
        "统计这段文字的长度"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = await agent.run(query)
        print(f"Response: {response}")
    
    return agent


async def example_multi_agent():
    """Example: Multi-Agent Coordination"""
    print("\n" + "=" * 50)
    print("Example: Multi-Agent Coordination")
    print("=" * 50)
    
    # Create coordinator
    coordinator = MultiAgentCoordinator()
    
    # Create specialized agents
    researcher = ReActAgent(
        model_name="researcher-agent",
        max_steps=3
    )
    researcher.tool_registry.register(WebSearch())
    
    analyst = FunctionCallingAgent(model_name="analyst-agent")
    analyst.tool_registry.register(Calculator())
    
    coder = ReActAgent(model_name="coder-agent", max_steps=5)
    
    # Register agents
    coordinator.register_agent("researcher", researcher)
    coordinator.register_agent("analyst", analyst)
    coordinator.register_agent("coder", coder)
    
    print(f"Coordinator agents: {coordinator.list_agents()}")
    
    # Mock agent responses
    async def mock_run(prompt):
        return f"Processed by agent: {prompt[:30]}..."
    
    for name in coordinator.list_agents():
        coordinator.get_agent(name).run = mock_run
    
    # Coordinate task across agents
    task = "分析AI发展趋势并计算市场规模"
    
    print(f"\nTask: {task}")
    print("Executing with [researcher, analyst]...")
    
    results = await coordinator.coordinate(task, ["researcher", "analyst"])
    
    for agent_name, result in results.items():
        print(f"  {agent_name}: {result}")
    
    return coordinator


def example_knowledge_base():
    """Example: Knowledge Base Tool"""
    print("\n" + "=" * 50)
    print("Example: Knowledge Base")
    print("=" * 50)
    
    # Create knowledge base
    kb = KnowledgeBase()
    
    # Add documents
    documents = [
        {
            "id": "doc1",
            "content": "Nexus-7B是基于Transformer架构的大语言模型，使用RoPE位置编码和SwiGLU激活函数。",
            "metadata": {"topic": "模型架构", "version": "1.0"}
        },
        {
            "id": "doc2",
            "content": "RLHF（从人类反馈中学习强化学习）是一种训练方法，可以使模型更符合人类偏好。",
            "metadata": {"topic": "训练方法", "version": "1.0"}
        },
        {
            "id": "doc3",
            "content": "模型压缩技术包括知识蒸馏、剪枝和量化，可以减小模型体积并提高推理效率。",
            "metadata": {"topic": "模型优化", "version": "1.0"}
        }
    ]
    
    for doc in documents:
        kb.add_document(doc["id"], doc["content"], doc["metadata"])
    
    print(f"Added {len(documents)} documents")
    
    # Search
    queries = ["Transformer", "RLHF", "模型压缩"]
    
    for query in queries:
        results = kb.search(query, top_k=2)
        print(f"\nQuery: '{query}'")
        print(f"Found {len(results)} results:")
        for r in results:
            print(f"  - {r['content'][:50]}... (score: {r['score']:.3f})")
    
    return kb


async def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Agent Framework Examples")
    print("=" * 60)
    
    # Basic examples
    registry = example_tool_registry()
    
    # Async examples
    await example_react_agent(registry)
    await example_function_calling_agent()
    await example_multi_agent()
    
    # Knowledge base
    example_knowledge_base()
    
    print("\n" + "=" * 60)
    print("All Agent examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
