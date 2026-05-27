"""
Agent Module Tests
"""
import pytest
import asyncio
from typing import List, Dict, Any
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


class TestBaseTool:
    """Test Base Tool Interface"""
    
    def test_tool_metadata(self):
        class MyTool(BaseTool):
            name = "my_tool"
            description = "A test tool"
            parameters = {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
            
            def execute(self, query: str) -> ToolResult:
                return ToolResult(success=True, output=f"Processed: {query}")
        
        tool = MyTool()
        assert tool.name == "my_tool"
        assert tool.description == "A test tool"
        assert "query" in tool.parameters["required"]
    
    def test_tool_execution(self):
        class EchoTool(BaseTool):
            name = "echo"
            description = "Echo back input"
            parameters = {"type": "object", "properties": {}}
            
            def execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=True, output="echo")
        
        tool = EchoTool()
        result = tool.execute()
        
        assert result.success is True
        assert result.output == "echo"
    
    def test_tool_error_handling(self):
        class FailingTool(BaseTool):
            name = "fail"
            description = "Always fails"
            parameters = {"type": "object", "properties": {}}
            
            def execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=False, error="Intentional failure")
        
        tool = FailingTool()
        result = tool.execute()
        
        assert result.success is False
        assert "failure" in result.error.lower()


class TestToolRegistry:
    """Test Tool Registry"""
    
    @pytest.fixture
    def registry(self):
        reg = ToolRegistry()
        
        class TestTool1(BaseTool):
            name = "tool1"
            description = "First tool"
            parameters = {"type": "object", "properties": {}}
            def execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=True, output="tool1")
        
        class TestTool2(BaseTool):
            name = "tool2"
            description = "Second tool"
            parameters = {"type": "object", "properties": {}}
            def execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=True, output="tool2")
        
        reg.register(TestTool1())
        reg.register(TestTool2())
        return reg
    
    def test_register_tools(self, registry):
        tools = registry.list_tools()
        assert len(tools) == 2
        assert "tool1" in tools
        assert "tool2" in tools
    
    def test_get_tool(self, registry):
        tool = registry.get_tool("tool1")
        assert tool is not None
        assert tool.name == "tool1"
    
    def test_get_nonexistent_tool(self, registry):
        tool = registry.get_tool("nonexistent")
        assert tool is None
    
    def test_execute_tool(self, registry):
        result = registry.execute_tool("tool1")
        assert result.success is True
        assert result.output == "tool1"
    
    def test_execute_with_args(self, registry):
        class ArgTool(BaseTool):
            name = "arg_tool"
            description = "Tool with args"
            parameters = {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"]
            }
            def execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=True, output=str(kwargs.get("value", 0)))
        
        registry.register(ArgTool())
        result = registry.execute_tool("arg_tool", value=42)
        assert result.success is True
        assert result.output == "42"
    
    def test_execute_nonexistent_tool(self, registry):
        result = registry.execute_tool("nonexistent")
        assert result.success is False
        assert "not found" in result.error.lower()


class TestCalculator:
    """Test Calculator Tool"""
    
    def test_basic_arithmetic(self):
        calc = Calculator()
        
        result = calc.execute(expression="2 + 3")
        assert result.success is True
        assert result.output == 5
        
        result = calc.execute(expression="10 - 4")
        assert result.success is True
        assert result.output == 6
        
        result = calc.execute(expression="3 * 7")
        assert result.success is True
        assert result.output == 21
        
        result = calc.execute(expression="20 / 4")
        assert result.success is True
        assert result.output == 5
    
    def test_complex_expressions(self):
        calc = Calculator()
        
        # Test expression with parentheses
        result = calc.execute(expression="(2 + 3) * 4")
        assert result.success is True
        assert result.output == 20
        
        # Test with variables
        result = calc.execute(expression="x = 10")
        assert result.success is True
        assert result.output == 10
        
        result = calc.execute(expression="x * 2")
        assert result.success is True
        assert result.output == 20
    
    def test_invalid_expression(self):
        calc = Calculator()
        result = calc.execute(expression="invalid ++ syntax")
        assert result.success is False
        assert result.error is not None


class TestWebSearch:
    """Test Web Search Tool"""
    
    def test_search_tool_initialization(self):
        search = WebSearch()
        assert search.name == "web_search"
        assert "search" in search.description.lower()
    
    def test_search_parameters(self):
        search = WebSearch()
        assert "query" in search.parameters.get("required", [])
    
    @pytest.mark.asyncio
    async def test_async_search(self):
        search = WebSearch()
        # Mock response for testing
        result = await search._async_search("test query")
        # Web search returns a result object
        assert result is not None


class TestKnowledgeBase:
    """Test Knowledge Base Tool"""
    
    @pytest.fixture
    def kb(self):
        kb = KnowledgeBase()
        kb.add_document("test_doc_1", "This is a test document about AI.", {"topic": "AI"})
        kb.add_document("test_doc_2", "Machine learning is a subset of AI.", {"topic": "ML"})
        return kb
    
    def test_add_document(self, kb):
        docs = kb.search("test")
        assert len(docs) >= 2
    
    def test_search_documents(self, kb):
        results = kb.search("machine learning")
        assert len(results) > 0
        # Should find the ML document
        assert any("machine learning" in r["content"].lower() for r in results)
    
    def test_search_with_filter(self, kb):
        results = kb.search("test", filter_dict={"topic": "AI"})
        assert all(r["metadata"].get("topic") == "AI" for r in results)


class TestReActAgent:
    """Test ReAct Agent"""
    
    @pytest.fixture
    def agent(self):
        agent = ReActAgent(
            model_name="test-model",
            max_steps=5
        )
        
        # Add calculator tool
        calc = Calculator()
        agent.tool_registry.register(calc)
        
        return agent
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        assert agent.max_steps == 5
        assert agent.model_name == "test-model"
        assert "calculator" in agent.tool_registry.list_tools()
    
    @pytest.mark.asyncio
    async def test_think_generates_tool_calls(self, agent):
        # Mock the model call
        async def mock_think(prompt, messages=None):
            return "Let me calculate 2 + 2", []
        
        agent._think = mock_think
        
        thought, tool_calls = await agent.think("What is 2 + 2?")
        
        assert isinstance(thought, str)
        assert isinstance(tool_calls, list)
    
    @pytest.mark.asyncio
    async def test_run_with_calculator(self, agent):
        # Mock the model to return tool calls
        tool_call = ToolCall(
            tool_name="calculator",
            arguments={"expression": "2 + 2"}
        )
        
        async def mock_think(prompt, messages=None):
            return "I need to calculate", [tool_call]
        
        agent._think = mock_think
        
        result = await agent.run("What is 2 + 2?")
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_max_steps_limit(self, agent):
        call_count = 0
        
        async def mock_think(prompt, messages=None):
            nonlocal call_count
            call_count += 1
            # Keep calling tools
            if call_count < 10:
                return "thinking", [ToolCall(tool_name="calculator", arguments={"expression": "1"})]
            return "done", []
        
        agent._think = mock_think
        agent.max_steps = 3
        
        await agent.run("Calculate something")
        # Should stop at max_steps
        assert call_count <= 4  # initial + up to max_steps


class TestFunctionCallingAgent:
    """Test Function Calling Agent"""
    
    @pytest.fixture
    def func_agent(self):
        agent = FunctionCallingAgent(model_name="test-model")
        
        calc = Calculator()
        agent.tool_registry.register(calc)
        
        return agent
    
    def test_agent_initialization(self, func_agent):
        assert func_agent.model_name == "test-model"
        assert "calculator" in func_agent.tool_registry.list_tools()
    
    @pytest.mark.asyncio
    async def test_function_call_generation(self, func_agent):
        # Mock function call generation
        func_calls = await func_agent._generate_function_calls(
            "Calculate 5 + 3",
            []
        )
        
        assert isinstance(func_calls, list)
    
    @pytest.mark.asyncio
    async def test_execute_function_calls(self, func_agent):
        func_call = ToolCall(
            tool_name="calculator",
            arguments={"expression": "10 * 10"}
        )
        
        results = await func_agent._execute_function_calls([func_call])
        
        assert len(results) == 1
        assert results[0].success is True


class TestMultiAgentCoordinator:
    """Test Multi Agent Coordinator"""
    
    @pytest.fixture
    def coordinator(self):
        coord = MultiAgentCoordinator()
        
        # Create sub-agents
        agent1 = ReActAgent(model_name="agent1")
        agent2 = FunctionCallingAgent(model_name="agent2")
        
        coord.register_agent("researcher", agent1)
        coord.register_agent("executor", agent2)
        
        return coord
    
    def test_register_agents(self, coordinator):
        agents = coordinator.list_agents()
        assert len(agents) == 2
        assert "researcher" in agents
        assert "executor" in agents
    
    def test_get_agent(self, coordinator):
        agent = coordinator.get_agent("researcher")
        assert agent is not None
        assert agent.model_name == "agent1"
    
    def test_get_nonexistent_agent(self, coordinator):
        agent = coordinator.get_agent("nonexistent")
        assert agent is None
    
    @pytest.mark.asyncio
    async def test_coordinate_task(self, coordinator):
        # Mock sub-agent responses
        async def mock_run(prompt):
            return f"Processed: {prompt}"
        
        coordinator.get_agent("researcher").run = mock_run
        coordinator.get_agent("executor").run = mock_run
        
        result = await coordinator.coordinate(
            "Test task",
            ["researcher", "executor"]
        )
        
        assert isinstance(result, dict)
        assert "researcher" in result
        assert "executor" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
