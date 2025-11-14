#!/usr/bin/env python3
"""
Quick test of multi-agent system using mock provider for fast testing.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "examples"))

from multi_agent_planner import (
    create_planner_agent,
    create_code_writer_agent,
    create_file_manager_agent,
    create_research_agent,
    sub_agents,
)


# Patch to use mock provider
from nons.core.types import ModelProvider
import nons.core.node as node_module

# Override create_provider to always use mock
original_create_provider = None

def mock_create_provider(config):
    from nons.utils.providers import MockProvider
    return MockProvider(config)


async def test_multi_agent_mock():
    """Test multi-agent with mock provider"""
    # Patch the provider creation
    global original_create_provider
    original_create_provider = node_module.create_provider
    node_module.create_provider = mock_create_provider

    print("=== Multi-Agent Test with Mock Provider ===\n")

    # Initialize sub-agents
    print("Initializing sub-agents...")
    sub_agents["code_writer"] = create_code_writer_agent()
    sub_agents["file_manager"] = create_file_manager_agent()
    sub_agents["research"] = create_research_agent()
    print("Sub-agents initialized\n")

    # Create planner
    print("Creating planner agent...")
    planner = create_planner_agent()
    print("Planner created\n")

    # Run task
    task = "Create a Python script that reads data from a file"
    print(f"Task: {task}\n")
    print("Executing (will use mock responses):\n")

    step_count = 0
    try:
        async for result in planner.run(task):
            step_count += 1
            if "step" in result:
                print(f"Step {result['step']}:")
                print(f"  Tool: {result.get('tool', 'N/A')}")
                print(f"  Success: {result.get('success', False)}")
                if result.get("is_stop"):
                    print("  -> Task complete!")
                    break
            if step_count >= 3:  # Limit for testing
                print("\n(Stopping after 3 steps for testing)")
                break
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nCompleted {step_count} steps")
    print("✓ Multi-agent system is working with mock provider")

    # Restore original
    node_module.create_provider = original_create_provider


if __name__ == "__main__":
    asyncio.run(test_multi_agent_mock())
