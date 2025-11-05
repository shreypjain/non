# Agents Module

The agents module provides a framework for building LLM-powered agents with tool execution capabilities. Agents orchestrate reasoning (via NoN networks) with tool execution (via ToolRegistry) to accomplish complex tasks autonomously.

## Architecture

The agent system consists of three core components:

1. **Agent**: Orchestrates the reasoning-execution loop
2. **ToolRegistry**: Manages tool registration and execution with validation
3. **NoN Network**: Provides LLM reasoning for routing and decision making

## Core Components

### Agent (`agent.py`)

The Agent class orchestrates LLM reasoning with tool execution through an async generator pattern.

Key responsibilities:
- Execute LLM forward passes to get routing decisions
- Execute tools based on LLM decisions
- Manage state across execution steps
- Handle errors and retry logic
- Stream results incrementally

Constructor parameters:
- `network`: NoN network for LLM reasoning
- `registry`: ToolRegistry with registered tools
- `max_steps`: Maximum execution steps before termination
- `max_llm_retry`: Maximum retries for LLM failures
- `state_adapter`: Function to adapt initial input to state dict
- `merge_fn`: Function to merge tool results into state

Key methods:
- `step()`: Execute a single reasoning-execution cycle
- `run()`: Main orchestration loop that yields results incrementally

### ToolRegistry (`registry.py`)

The ToolRegistry manages tool registration with Pydantic validation and standardized error handling.

Key features:
- Decorator-based tool registration
- Automatic parameter validation via Pydantic models
- Standardized result format
- Stop tool support for agent termination

Tool registration pattern:
```python
registry = ToolRegistry()

@registry.tool(
    name="tool_name",
    description="What this tool does",
    param_model=ParamsModel,  # Optional Pydantic model
    is_stop_tool=False  # Set True to mark termination
)
async def tool_function(params, **context):
    return {"result": "value"}
```

Result format:
```python
{
    "success": True/False,
    "tool_name": "name",
    "output": {...},
    "is_stop_tool": True/False,
    "error": "error message"  # Only on failure
}
```

## Usage Patterns

### Basic Agent Pattern

```python
from nons.core.agents.agent import Agent
from nons.core.agents.registry import ToolRegistry
from nons.core.network import create_network
from nons.core.node import Node

# 1. Create registry and register tools
registry = ToolRegistry()

@registry.tool(name="example", description="Example tool")
async def example_tool(params, **context):
    return {"result": "success"}

# 2. Create reasoning network
network = create_network(
    layers=[Node("route")],
    provider="openai",
    model="gpt-4o-mini",
    system_prompt="You are an agent..."
)

# 3. Create and run agent
agent = Agent(network=network, registry=registry)

async for result in agent.run("Task description"):
    print(result)
```

### Multi-Agent Pattern

For complex tasks, create specialized sub-agents that can be called by a planner agent:

```python
# Create specialized agents
code_agent = Agent(network=code_network, registry=code_registry)
file_agent = Agent(network=file_network, registry=file_registry)

# Store in dict for delegation
sub_agents = {"code": code_agent, "file": file_agent}

# Planner agent delegates to sub-agents via tools
@planner_registry.tool(name="delegate_to_code", ...)
async def delegate_to_code(params, **context):
    results = []
    async for result in sub_agents["code"].run(params.task):
        results.append(result)
    return {"results": results[-1]}
```

### State Management

The agent maintains state across steps using a merge function:

```python
def custom_merge(state, tool_output):
    # Merge strategy depends on your use case
    if isinstance(tool_output.get("output"), dict):
        return {**state, **tool_output["output"]}
    return {**state, "last_result": tool_output["output"]}

agent = Agent(network=network, registry=registry, merge_fn=custom_merge)
```

### Dynamic Tool Registration

Tools can be added at runtime to adapt agent capabilities:

```python
# Agent already created
agent = create_agent()

# Add new tool dynamically
@agent.registry.tool(name="new_tool", description="...")
async def new_tool(params, **context):
    return {"result": "value"}
```

## Design Patterns

### Separation of Concerns

- **Agent**: Orchestration logic only
- **Network**: LLM reasoning only
- **Registry**: Tool management only

This separation allows each component to be tested and modified independently.

### Streaming Results

Agents use async generators to stream results incrementally:

```python
async for step_result in agent.run(task):
    # Process each step as it completes
    # Enables real-time monitoring and UI updates
    if step_result.get("is_stop"):
        break
```

### Standardized Tool Interface

All tools follow the same async signature:

```python
async def tool(params: BaseModel, **context) -> Dict[str, Any]:
    """
    params: Validated Pydantic model
    context: Additional context passed through from agent.run()
    returns: Dict with tool-specific results
    """
```

This standardization enables:
- Consistent error handling
- Easy tool composition
- Predictable behavior

### Stop Tool Pattern

Mark completion tools with `is_stop_tool=True`:

```python
@registry.tool(name="finish", is_stop_tool=True, ...)
async def finish(params, **context):
    return {"final_result": params.result}
```

When a stop tool is executed, the agent terminates its execution loop.

## Examples

See the examples directory for complete working examples:

- `single_agent_calculator.py`: Basic agent with calculator tools
- `multi_agent_planner.py`: Hierarchical agents with delegation
- `python_interpreter_agent.py`: Agent with code execution capabilities

## Error Handling

### Tool Execution Errors

Tool execution is wrapped in try-catch with standardized error responses:

```python
{
    "success": False,
    "error": "Execution failed: ...",
    "tool_name": "tool_name"
}
```

### LLM Retry Logic

The agent automatically retries LLM calls on validation failures:

```python
agent = Agent(network=network, registry=registry, max_llm_retry=3)
```

After max retries, raises ValueError with details.

### Step Limits

Prevent infinite loops with max_steps:

```python
agent = Agent(network=network, registry=registry, max_steps=100)
```

When max_steps is reached, the agent yields a warning and terminates.

## Testing Considerations

When testing agents:

1. Mock the NoN network to control LLM responses
2. Test tools independently via registry.execute()
3. Use small max_steps for faster test execution
4. Verify state management with custom merge functions
5. Test error paths (invalid tools, validation failures)

## Performance Considerations

- Tools should be async to prevent blocking
- Use context parameter to pass shared resources
- Consider caching for expensive operations
- Stream results for long-running tasks
- Set appropriate max_steps to prevent runaway execution

## Future Enhancements

Potential areas for extension:

- Tool composition and chaining
- Parallel tool execution
- Conversation history management
- Memory systems for long-term context
- Tool approval workflows for sensitive operations
- Observability integration for production monitoring
