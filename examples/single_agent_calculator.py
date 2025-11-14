#!/usr/bin/env python3
"""
Single Agent Calculator Example

This example demonstrates how to create a simple agent with calculator tools.
The agent uses a NoN network for reasoning and a ToolRegistry for tool execution.

Key Patterns Demonstrated:
1. Tool registration using decorator pattern
2. Agent initialization with network and registry
3. Streaming agent execution with step-by-step results
4. State management and context passing
5. Termination tools for agent completion

Architecture:
- NoN Network: Handles LLM reasoning and routing decisions
- ToolRegistry: Manages tool registration and execution
- Agent: Orchestrates LLM reasoning with tool execution
"""

import asyncio
import sys
import os
from typing import Dict, Any
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nons.core.network import NoN, create_network
from nons.core.node import Node
from nons.core.agents.agent import Agent
from nons.core.agents.registry import ToolRegistry
import nons.operators.base  # This registers all the base operators


# Tool Parameter Models - Define schemas for validation
class CalculatorParams(BaseModel):
    """Parameters for calculator operations"""

    a: float = Field(description="First number")
    b: float = Field(description="Second number")


class StopParams(BaseModel):
    """Parameters for stop tool"""

    final_answer: str = Field(description="The final answer to return to the user")


# Pattern: Create registry first, then use decorators to register tools
registry = ToolRegistry()


# Pattern: Use @registry.tool decorator for clean registration
# The decorator automatically registers the function with validation
@registry.tool(
    name="add",
    description="Add two numbers together. Returns the sum.",
    param_model=CalculatorParams,
)
async def add_numbers(params: CalculatorParams, **context) -> Dict[str, Any]:
    """Add two numbers together"""
    result = params.a + params.b
    return {"result": result, "operation": "addition"}


@registry.tool(
    name="subtract",
    description="Subtract the second number from the first. Returns the difference.",
    param_model=CalculatorParams,
)
async def subtract_numbers(params: CalculatorParams, **context) -> Dict[str, Any]:
    """Subtract second number from first number"""
    result = params.a - params.b
    return {"result": result, "operation": "subtraction"}


@registry.tool(
    name="multiply",
    description="Multiply two numbers together. Returns the product.",
    param_model=CalculatorParams,
)
async def multiply_numbers(params: CalculatorParams, **context) -> Dict[str, Any]:
    """Multiply two numbers together"""
    result = params.a * params.b
    return {"result": result, "operation": "multiplication"}


@registry.tool(
    name="divide",
    description="Divide the first number by the second. Returns the quotient.",
    param_model=CalculatorParams,
)
async def divide_numbers(params: CalculatorParams, **context) -> Dict[str, Any]:
    """Divide first number by second number"""
    if params.b == 0:
        raise ValueError("Cannot divide by zero")
    result = params.a / params.b
    return {"result": result, "operation": "division"}


# Pattern: Mark termination tools with is_stop_tool=True
# This signals the agent to stop when this tool is called
@registry.tool(
    name="stop",
    description="Call this when you have the final answer to return to the user.",
    param_model=StopParams,
    is_stop_tool=True,
)
async def stop_agent(params: StopParams, **context) -> Dict[str, Any]:
    """Stop the agent and return final answer"""
    return {"final_answer": params.final_answer}


def create_reasoning_network() -> NoN:
    """
    Create the LLM reasoning network.

    Pattern: Simple single-node network for basic reasoning.
    For more complex agents, you can add multiple layers and nodes.
    """
    # Create a single reasoning node
    # The 'generate' operator handles LLM calls for decision making and tool selection
    reasoning_node = Node("generate")

    # Create network with single layer
    network = create_network(
        layers=[reasoning_node],  # Single node = single layer
        provider="openai",
        model="gpt-4o-mini",
        system_prompt="""You are a calculator agent. You have access to basic math operations.

When given a math problem:
1. Break it down into steps if needed
2. Use the appropriate tool for each operation
3. When you have the final answer, use the stop tool to return it

Available tools: add, subtract, multiply, divide, stop

Return your decisions in this format:
{
    "selected_path": "tool_name",
    "routing_confidence": 0.95,
    "reasoning": "why this tool was selected",
    "params": {"a": 10, "b": 5}
}""",
    )

    return network


async def demo_simple_calculation():
    """Demonstrate simple calculation"""
    print("=== Simple Calculation Demo ===\n")

    # Setup
    network = create_reasoning_network()
    agent = Agent(
        network=network,
        registry=registry,  # Use the globally decorated registry
        max_steps=10,  # Limit steps to prevent infinite loops
    )

    # Run agent
    print("Question: What is 15 + 7?\n")
    async for result in agent.run("What is 15 + 7?"):
        if "step" in result:
            print(f"Step {result['step']}:")
            print(f"  Tool: {result.get('tool', 'N/A')}")
            print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
            print(f"  Result: {result.get('result', 'N/A')}")
            print()

    print("\n" + "=" * 50 + "\n")


async def demo_multi_step_calculation():
    """Demonstrate multi-step calculation"""
    print("=== Multi-Step Calculation Demo ===\n")

    # Setup
    network = create_reasoning_network()
    agent = Agent(network=network, registry=registry, max_steps=20)

    # Run agent with complex query
    print("Question: Calculate (10 + 5) * 3 - 8\n")
    async for result in agent.run("Calculate (10 + 5) * 3 - 8"):
        if "step" in result:
            print(f"Step {result['step']}:")
            print(f"  Tool: {result.get('tool', 'N/A')}")
            print(f"  Result: {result.get('result', {}).get('result', 'N/A')}")
            if result.get("is_stop"):
                print(f"  Final Answer: {result.get('result', {}).get('final_answer')}")
            print()

    print("\n" + "=" * 50 + "\n")


async def demo_error_handling():
    """Demonstrate error handling in tool execution"""
    print("=== Error Handling Demo ===\n")

    # Setup
    network = create_reasoning_network()
    agent = Agent(network=network, registry=registry, max_steps=10)

    # Run agent with division by zero
    print("Question: What is 10 divided by 0?\n")
    async for result in agent.run("What is 10 divided by 0?"):
        if "step" in result:
            print(f"Step {result['step']}:")
            print(f"  Tool: {result.get('tool', 'N/A')}")
            if not result.get("success", True):
                print(f"  Error: {result.get('result', {}).get('error')}")
            else:
                print(f"  Result: {result.get('result', 'N/A')}")
            print()

    print("\n" + "=" * 50 + "\n")


async def main():
    """Run all demonstrations"""
    print("\n" + "=" * 50)
    print("Single Agent Calculator Example")
    print("=" * 50 + "\n")

    print("Pattern Notes:")
    print("1. Tools registered using @registry.tool decorator")
    print("2. Pydantic models validate tool parameters")
    print("3. Agent streams results step-by-step")
    print("4. Stop tool marks agent completion")
    print()

    await demo_simple_calculation()
    await demo_multi_step_calculation()
    await demo_error_handling()


if __name__ == "__main__":
    asyncio.run(main())
