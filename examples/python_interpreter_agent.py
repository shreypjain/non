#!/usr/bin/env python3
"""
Python Interpreter Agent Example

This example demonstrates an agent with Python code execution capabilities.
The agent can write, execute, and debug Python code dynamically, similar to
a Jupyter notebook or REPL environment.

Key Patterns Demonstrated:
1. Safe code execution with sandboxing considerations
2. State preservation across code executions
3. Error handling and debugging workflows
4. Variable inspection and manipulation
5. Multi-step code development (write, test, fix)

Architecture:
- Python Interpreter Agent: Writes and executes Python code
- Execution Environment: Maintains state across executions
- Safety Tools: Variable inspection and environment reset

Safety Note: This example uses exec() for demonstration. In production,
consider using restricted execution environments or containers.
"""

import asyncio
import sys
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import traceback
from io import StringIO
import contextlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nons.core.network import NoN, create_network
from nons.core.node import Node
from nons.core.agents.agent import Agent
from nons.core.agents.registry import ToolRegistry


# Parameter Models
class PythonCodeParams(BaseModel):
    """Parameters for Python code execution"""

    code: str = Field(description="Python code to execute")
    capture_output: bool = Field(
        default=True, description="Whether to capture stdout/stderr"
    )


class VariableInspectParams(BaseModel):
    """Parameters for inspecting variables"""

    variable_name: Optional[str] = Field(
        default=None, description="Specific variable to inspect, or None for all"
    )


class StopParams(BaseModel):
    """Parameters for stop tool"""

    final_output: str = Field(description="Final result to return")


# Pattern: Global execution environment to maintain state across tool calls
# In production, consider using isolated environments per session
execution_environment: Dict[str, Any] = {
    "__builtins__": __builtins__,
}

# Create registry
python_registry = ToolRegistry()


# ============================================================================
# PYTHON EXECUTION TOOLS
# ============================================================================


@python_registry.tool(
    name="execute_python",
    description="Execute Python code and return the result. Code runs in a persistent environment where variables are preserved between executions.",
    param_model=PythonCodeParams,
)
async def execute_python(params: PythonCodeParams, **context) -> Dict[str, Any]:
    """
    Execute Python code in a persistent environment.

    Pattern: Maintain state between executions to support multi-step development.
    Capture stdout/stderr to provide feedback to the agent.
    """
    global execution_environment

    try:
        # Pattern: Capture stdout/stderr for agent feedback
        output_capture = StringIO()
        error_capture = StringIO()

        with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(
            error_capture
        ):
            # Execute code in persistent environment
            exec(params.code, execution_environment)

        stdout = output_capture.getvalue()
        stderr = error_capture.getvalue()

        return {
            "success": True,
            "stdout": stdout if stdout else "No output",
            "stderr": stderr if stderr else None,
            "code": params.code,
            "environment_size": len(
                [k for k in execution_environment.keys() if not k.startswith("__")]
            ),
        }

    except Exception as e:
        # Pattern: Provide detailed error information for debugging
        error_details = {
            "success": False,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "code": params.code,
        }
        return error_details


@python_registry.tool(
    name="inspect_variables",
    description="Inspect variables in the execution environment. Useful for debugging and understanding current state.",
    param_model=VariableInspectParams,
)
async def inspect_variables(
    params: VariableInspectParams, **context
) -> Dict[str, Any]:
    """
    Inspect variables in the execution environment.

    Pattern: Allow agent to understand current state before making decisions.
    """
    global execution_environment

    # Filter out builtins and private variables
    user_vars = {
        k: v
        for k, v in execution_environment.items()
        if not k.startswith("__") and k != "__builtins__"
    }

    if params.variable_name:
        # Inspect specific variable
        if params.variable_name in user_vars:
            value = user_vars[params.variable_name]
            return {
                "variable": params.variable_name,
                "type": type(value).__name__,
                "value": str(value),
                "repr": repr(value),
            }
        else:
            return {
                "error": f"Variable '{params.variable_name}' not found",
                "available_variables": list(user_vars.keys()),
            }
    else:
        # List all variables
        variables_info = {
            name: {"type": type(value).__name__, "value": str(value)[:100]}
            for name, value in user_vars.items()
        }
        return {"variables": variables_info, "count": len(variables_info)}


@python_registry.tool(
    name="reset_environment",
    description="Reset the Python execution environment, clearing all variables. Use this to start fresh.",
)
async def reset_environment(params: None, **context) -> Dict[str, Any]:
    """
    Reset the execution environment.

    Pattern: Allow agent to start fresh when needed.
    """
    global execution_environment
    old_count = len(
        [k for k in execution_environment.keys() if not k.startswith("__")]
    )

    execution_environment = {
        "__builtins__": __builtins__,
    }

    return {
        "success": True,
        "message": "Environment reset",
        "variables_cleared": old_count,
    }


@python_registry.tool(
    name="finish",
    description="Complete the Python programming task and return the final result",
    param_model=StopParams,
    is_stop_tool=True,
)
async def finish_task(params: StopParams, **context) -> Dict[str, Any]:
    """Complete the programming task"""
    return {"final_output": params.final_output}


# ============================================================================
# AGENT CREATION
# ============================================================================


def create_python_interpreter_agent() -> Agent:
    """
    Create Python interpreter agent.

    Pattern: System prompt guides the agent through iterative development:
    write code -> execute -> check results -> fix errors -> repeat
    """
    reasoning_node = Node("generate")
    network = create_network(
        layers=[reasoning_node],
        provider="openai",
        model="gpt-4o-mini",
        system_prompt="""You are a Python programming agent with code execution capabilities.

Your workflow:
1. Write Python code to solve the task
2. Execute code using execute_python
3. Check the output and any errors
4. If there are errors, fix them and re-execute
5. Use inspect_variables to check the state when needed
6. Use reset_environment if you need to start fresh
7. Use finish when the task is complete

Available tools:
- execute_python: Run Python code
- inspect_variables: Check current variables
- reset_environment: Clear all variables
- finish: Complete the task

The execution environment persists between calls, so variables remain available.

Return decisions in this format:
{
    "selected_path": "tool_name",
    "routing_confidence": 0.95,
    "reasoning": "why this action",
    "params": {...}
}""",
    )

    return Agent(network=network, registry=python_registry, max_steps=30)


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================


async def demo_simple_execution():
    """Demonstrate simple code execution"""
    print("=== Simple Python Execution Demo ===\n")

    agent = create_python_interpreter_agent()

    task = "Calculate the factorial of 5 and print the result"
    print(f"Task: {task}\n")

    async for result in agent.run(task):
        if "step" in result:
            print(f"Step {result['step']}:")
            print(f"  Tool: {result.get('tool', 'N/A')}")
            tool_result = result.get("result", {})
            if "code" in tool_result:
                print(f"  Code: {tool_result['code'][:100]}...")
            if "stdout" in tool_result:
                print(f"  Output: {tool_result['stdout']}")
            if tool_result.get("error_type"):
                print(f"  Error: {tool_result['error_type']}: {tool_result['error_message']}")
            print()

    print("\n" + "=" * 50 + "\n")


async def demo_iterative_development():
    """Demonstrate iterative code development with error fixing"""
    print("=== Iterative Development Demo ===\n")

    agent = create_python_interpreter_agent()

    task = "Create a function that calculates Fibonacci numbers, then use it to get the 10th Fibonacci number"
    print(f"Task: {task}\n")

    async for result in agent.run(task):
        if "step" in result:
            print(f"Step {result['step']}:")
            print(f"  Tool: {result.get('tool', 'N/A')}")
            print(f"  Reasoning: {result.get('reasoning', 'N/A')[:80]}...")

            tool_result = result.get("result", {})
            if "stdout" in tool_result and tool_result["stdout"] != "No output":
                print(f"  Output: {tool_result['stdout'].strip()}")
            if tool_result.get("error_type"):
                print(
                    f"  Error: {tool_result['error_type']} - Agent will attempt to fix"
                )
            print()

    print("\n" + "=" * 50 + "\n")


async def demo_state_persistence():
    """Demonstrate state persistence across executions"""
    print("=== State Persistence Demo ===\n")

    agent = create_python_interpreter_agent()

    task = """First, create a list with numbers 1 to 5.
Then, create a function that squares each number.
Finally, apply the function to the list and print the result."""

    print(f"Task: {task}\n")

    async for result in agent.run(task):
        if "step" in result:
            print(f"Step {result['step']}:")
            print(f"  Tool: {result.get('tool', 'N/A')}")

            tool_result = result.get("result", {})

            # Show environment size to demonstrate persistence
            if "environment_size" in tool_result:
                print(f"  Variables in environment: {tool_result['environment_size']}")

            if "variables" in tool_result:
                print(f"  Current variables: {list(tool_result['variables'].keys())}")

            if "stdout" in tool_result and tool_result["stdout"] != "No output":
                print(f"  Output: {tool_result['stdout'].strip()}")

            print()

    print("\n" + "=" * 50 + "\n")


async def demo_variable_inspection():
    """Demonstrate variable inspection capabilities"""
    print("=== Variable Inspection Demo ===\n")

    agent = create_python_interpreter_agent()

    task = """Create a dictionary with some data about a person (name, age, city).
Then inspect the variables to see what was created."""

    print(f"Task: {task}\n")

    async for result in agent.run(task):
        if "step" in result:
            print(f"Step {result['step']}:")
            print(f"  Tool: {result.get('tool', 'N/A')}")

            tool_result = result.get("result", {})

            if tool_result.get("tool") == "inspect_variables":
                print(f"  Inspection result: {tool_result}")

            if "stdout" in tool_result and tool_result["stdout"] != "No output":
                print(f"  Output: {tool_result['stdout'].strip()}")

            print()

    print("\n" + "=" * 50 + "\n")


async def main():
    """Run all demonstrations"""
    print("\n" + "=" * 50)
    print("Python Interpreter Agent Example")
    print("=" * 50 + "\n")

    print("Pattern Notes:")
    print("1. Code executes in persistent environment")
    print("2. Variables remain available across executions")
    print("3. Agent can inspect state and debug errors")
    print("4. Iterative development: write -> test -> fix")
    print("5. Safety through error handling and feedback")
    print()

    await demo_simple_execution()
    await demo_iterative_development()
    await demo_state_persistence()
    await demo_variable_inspection()

    # Clean up
    await reset_environment(None)


if __name__ == "__main__":
    asyncio.run(main())
