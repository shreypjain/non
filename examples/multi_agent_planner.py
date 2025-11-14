#!/usr/bin/env python3
"""
Multi-Agent Planner Example

This example demonstrates a hierarchical multi-agent system similar to Claude Code.
A planner agent coordinates multiple specialized sub-agents that can dynamically
adjust their tools based on the task.

Key Patterns Demonstrated:
1. Hierarchical agent architecture (planner + sub-agents)
2. Dynamic tool registration and modification
3. Agent-to-agent delegation
4. Context passing between agents
5. Specialized agents with different capabilities

Architecture:
- Planner Agent: Breaks down tasks and delegates to sub-agents
- Code Writer Agent: Handles code generation tasks
- File Manager Agent: Handles file operations
- Research Agent: Handles information gathering

Pattern Note: Each agent has its own network and registry, allowing for
specialized behavior and tool sets.
"""

import asyncio
import sys
import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nons.core.network import NoN, create_network
from nons.core.node import Node
from nons.core.agents.agent import Agent
from nons.core.agents.registry import ToolRegistry
import nons.operators.base  # This registers all the base operators


# Shared Parameter Models
class DelegateParams(BaseModel):
    """Parameters for delegating to a sub-agent"""

    agent_name: str = Field(description="Name of the agent to delegate to")
    task: str = Field(description="Task description for the sub-agent")


class PlanStepParams(BaseModel):
    """Parameters for planning a task"""

    steps: List[str] = Field(description="List of steps to complete the task")


class CodeWriteParams(BaseModel):
    """Parameters for code writing"""

    language: str = Field(description="Programming language")
    description: str = Field(description="What the code should do")
    context: str = Field(default="", description="Additional context")


class FileOperationParams(BaseModel):
    """Parameters for file operations"""

    operation: str = Field(description="Operation type: read, write, delete")
    filepath: str = Field(description="Path to the file")
    content: str = Field(default="", description="Content for write operations")


class ResearchParams(BaseModel):
    """Parameters for research tasks"""

    query: str = Field(description="Research query")
    depth: str = Field(default="basic", description="Research depth: basic or deep")


class StopParams(BaseModel):
    """Parameters for stop tool"""

    final_result: str = Field(description="Final result to return")


# Pattern: Create specialized registries for different agent types
planner_registry = ToolRegistry()
code_writer_registry = ToolRegistry()
file_manager_registry = ToolRegistry()
research_registry = ToolRegistry()


# Sub-Agent Storage - Pattern: Store sub-agents for delegation
sub_agents: Dict[str, Agent] = {}


# ============================================================================
# PLANNER AGENT TOOLS
# ============================================================================
# Pattern: Planner has delegation tools to coordinate sub-agents


@planner_registry.tool(
    name="create_plan",
    description="Break down a complex task into steps",
    param_model=PlanStepParams,
)
async def create_plan(params: PlanStepParams, **context) -> Dict[str, Any]:
    """Create a plan by breaking down the task"""
    return {
        "plan": params.steps,
        "total_steps": len(params.steps),
        "message": "Plan created successfully",
    }


@planner_registry.tool(
    name="delegate_to_code_writer",
    description="Delegate a code writing task to the code writer agent",
    param_model=DelegateParams,
)
async def delegate_to_code_writer(params: DelegateParams, **context) -> Dict[str, Any]:
    """Delegate to code writer sub-agent"""
    if "code_writer" not in sub_agents:
        return {"error": "Code writer agent not available"}

    # Pattern: Collect results from sub-agent execution
    results = []
    async for result in sub_agents["code_writer"].run(params.task):
        results.append(result)

    return {"sub_agent": "code_writer", "results": results[-1] if results else {}}


@planner_registry.tool(
    name="delegate_to_file_manager",
    description="Delegate a file operation task to the file manager agent",
    param_model=DelegateParams,
)
async def delegate_to_file_manager(
    params: DelegateParams, **context
) -> Dict[str, Any]:
    """Delegate to file manager sub-agent"""
    if "file_manager" not in sub_agents:
        return {"error": "File manager agent not available"}

    results = []
    async for result in sub_agents["file_manager"].run(params.task):
        results.append(result)

    return {"sub_agent": "file_manager", "results": results[-1] if results else {}}


@planner_registry.tool(
    name="delegate_to_research",
    description="Delegate a research task to the research agent",
    param_model=DelegateParams,
)
async def delegate_to_research(params: DelegateParams, **context) -> Dict[str, Any]:
    """Delegate to research sub-agent"""
    if "research" not in sub_agents:
        return {"error": "Research agent not available"}

    results = []
    async for result in sub_agents["research"].run(params.task):
        results.append(result)

    return {"sub_agent": "research", "results": results[-1] if results else {}}


@planner_registry.tool(
    name="complete_task",
    description="Mark the overall task as complete and return final result",
    param_model=StopParams,
    is_stop_tool=True,
)
async def complete_task(params: StopParams, **context) -> Dict[str, Any]:
    """Complete the planning task"""
    return {"final_result": params.final_result}


# ============================================================================
# CODE WRITER AGENT TOOLS
# ============================================================================
# Pattern: Specialized tools for code generation


@code_writer_registry.tool(
    name="write_code",
    description="Generate code in the specified language",
    param_model=CodeWriteParams,
)
async def write_code(params: CodeWriteParams, **context) -> Dict[str, Any]:
    """Generate code based on description"""
    # Simulated code generation
    code = f"""# {params.description}
# Language: {params.language}
# Context: {params.context}

def generated_function():
    # Implementation would go here
    pass
"""
    return {"code": code, "language": params.language}


@code_writer_registry.tool(
    name="finish_code_task",
    description="Mark code writing task as complete",
    param_model=StopParams,
    is_stop_tool=True,
)
async def finish_code_task(params: StopParams, **context) -> Dict[str, Any]:
    """Complete code writing"""
    return {"final_result": params.final_result}


# ============================================================================
# FILE MANAGER AGENT TOOLS
# ============================================================================
# Pattern: File operations with validation


@file_manager_registry.tool(
    name="file_operation",
    description="Perform file operations: read, write, or delete",
    param_model=FileOperationParams,
)
async def file_operation(params: FileOperationParams, **context) -> Dict[str, Any]:
    """Simulate file operations"""
    if params.operation == "read":
        return {
            "operation": "read",
            "filepath": params.filepath,
            "content": "Simulated file content",
        }
    elif params.operation == "write":
        return {
            "operation": "write",
            "filepath": params.filepath,
            "status": "success",
            "bytes_written": len(params.content),
        }
    elif params.operation == "delete":
        return {
            "operation": "delete",
            "filepath": params.filepath,
            "status": "success",
        }
    else:
        raise ValueError(f"Unknown operation: {params.operation}")


@file_manager_registry.tool(
    name="finish_file_task",
    description="Mark file management task as complete",
    param_model=StopParams,
    is_stop_tool=True,
)
async def finish_file_task(params: StopParams, **context) -> Dict[str, Any]:
    """Complete file management"""
    return {"final_result": params.final_result}


# ============================================================================
# RESEARCH AGENT TOOLS
# ============================================================================
# Pattern: Information gathering tools


@research_registry.tool(
    name="search",
    description="Search for information on a topic",
    param_model=ResearchParams,
)
async def search(params: ResearchParams, **context) -> Dict[str, Any]:
    """Simulate research/search"""
    depth_info = "comprehensive" if params.depth == "deep" else "basic"
    return {
        "query": params.query,
        "results": f"Simulated {depth_info} research results for: {params.query}",
        "sources": ["source1.com", "source2.com"],
    }


@research_registry.tool(
    name="finish_research_task",
    description="Mark research task as complete",
    param_model=StopParams,
    is_stop_tool=True,
)
async def finish_research_task(params: StopParams, **context) -> Dict[str, Any]:
    """Complete research"""
    return {"final_result": params.final_result}


# ============================================================================
# AGENT CREATION FUNCTIONS
# ============================================================================


def create_planner_agent() -> Agent:
    """
    Create the main planner agent.

    Pattern: Planner uses a more sophisticated system prompt to coordinate
    multiple sub-agents and break down complex tasks.
    """
    reasoning_node = Node("generate")
    network = create_network(
        layers=[reasoning_node],
        provider="openai",
        model="gpt-4o-mini",
        system_prompt="""You are a planner agent that coordinates specialized sub-agents.

Your job is to:
1. Break down complex tasks into steps using create_plan
2. Delegate steps to appropriate sub-agents:
   - code_writer: For code generation tasks
   - file_manager: For file operations
   - research: For information gathering
3. Use complete_task when all steps are done

Return decisions in this JSON format:
{
    "selected_path": "tool_name",
    "routing_confidence": 0.95,
    "reasoning": "why this action",
    "params": {...}
}""",
    )

    return Agent(network=network, registry=planner_registry, max_steps=20)


def create_code_writer_agent() -> Agent:
    """Create specialized code writer agent"""
    reasoning_node = Node("generate")
    network = create_network(
        layers=[reasoning_node],
        provider="openai",
        model="gpt-4o-mini",
        system_prompt="""You are a code writer agent specialized in generating code.

Use write_code to generate code based on requirements.
Use finish_code_task when complete.

Return decisions in this JSON format:
{
    "selected_path": "tool_name",
    "routing_confidence": 0.95,
    "reasoning": "why this action",
    "params": {...}
}""",
    )

    return Agent(network=network, registry=code_writer_registry, max_steps=10)


def create_file_manager_agent() -> Agent:
    """Create specialized file manager agent"""
    reasoning_node = Node("generate")
    network = create_network(
        layers=[reasoning_node],
        provider="openai",
        model="gpt-4o-mini",
        system_prompt="""You are a file manager agent specialized in file operations.

Use file_operation for read, write, or delete operations.
Use finish_file_task when complete.

Return decisions in this JSON format:
{
    "selected_path": "tool_name",
    "routing_confidence": 0.95,
    "reasoning": "why this action",
    "params": {...}
}""",
    )

    return Agent(network=network, registry=file_manager_registry, max_steps=10)


def create_research_agent() -> Agent:
    """Create specialized research agent"""
    reasoning_node = Node("generate")
    network = create_network(
        layers=[reasoning_node],
        provider="openai",
        model="gpt-4o-mini",
        system_prompt="""You are a research agent specialized in gathering information.

Use search to find information on topics.
Use finish_research_task when complete.

Return decisions in this JSON format:
{
    "selected_path": "tool_name",
    "routing_confidence": 0.95,
    "reasoning": "why this action",
    "params": {...}
}""",
    )

    return Agent(network=network, registry=research_registry, max_steps=10)


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================


async def demo_multi_agent_system():
    """Demonstrate the multi-agent system with task delegation"""
    print("=== Multi-Agent Planner System ===\n")

    # Pattern: Initialize all sub-agents before starting planner
    print("Initializing sub-agents...")
    sub_agents["code_writer"] = create_code_writer_agent()
    sub_agents["file_manager"] = create_file_manager_agent()
    sub_agents["research"] = create_research_agent()
    print("Sub-agents initialized: code_writer, file_manager, research\n")

    # Create planner
    planner = create_planner_agent()

    # Run complex task
    task = "Create a Python script that reads data from a file and generates a report"
    print(f"Task: {task}\n")
    print("Planner execution:\n")

    async for result in planner.run(task):
        if "step" in result:
            print(f"Step {result['step']}:")
            print(f"  Tool: {result.get('tool', 'N/A')}")
            print(f"  Reasoning: {result.get('reasoning', 'N/A')[:100]}...")
            if "sub_agent" in result.get("result", {}):
                print(f"  Delegated to: {result['result']['sub_agent']}")
            print()

    print("\n" + "=" * 50 + "\n")


async def demo_dynamic_tool_modification():
    """
    Demonstrate dynamic tool addition to an agent.

    Pattern: Tools can be added at runtime to adapt agent capabilities.
    This is useful for context-specific tools or temporary capabilities.
    """
    print("=== Dynamic Tool Modification Demo ===\n")

    # Create a basic agent
    code_writer = create_code_writer_agent()

    # Pattern: Add a new tool dynamically
    @code_writer.registry.tool(
        name="validate_syntax",
        description="Validate code syntax",
        param_model=CodeWriteParams,
    )
    async def validate_syntax(params: CodeWriteParams, **context) -> Dict[str, Any]:
        """Dynamically added syntax validation"""
        return {
            "valid": True,
            "language": params.language,
            "message": "Syntax validation passed",
        }

    print("Added dynamic tool: validate_syntax")
    print(f"Available tools: {list(code_writer.registry.get_tool_descriptions().keys())}\n")
    print("This pattern allows agents to adapt their capabilities on-the-fly\n")
    print("=" * 50 + "\n")


async def main():
    """Run all demonstrations"""
    print("\n" + "=" * 50)
    print("Multi-Agent Planner Example")
    print("=" * 50 + "\n")

    print("Pattern Notes:")
    print("1. Hierarchical agent architecture with planner + sub-agents")
    print("2. Each agent has specialized tools and system prompts")
    print("3. Planner delegates tasks to appropriate sub-agents")
    print("4. Tools can be added dynamically at runtime")
    print("5. Sub-agents return results back to planner")
    print()

    await demo_multi_agent_system()
    await demo_dynamic_tool_modification()


if __name__ == "__main__":
    asyncio.run(main())
