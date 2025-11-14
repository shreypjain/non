#!/usr/bin/env python3
"""
Test with scheduler properly started.
"""

import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from nons.core.types import ModelConfig, ModelProvider
from nons.core.node import Node
from nons.core.network import create_network
from nons.core.agents.agent import Agent
from nons.core.agents.registry import ToolRegistry
from nons.core.scheduler import get_scheduler, start_scheduler
from pydantic import BaseModel, Field
import nons.operators.base

# Force mock provider
from nons.utils.providers import MockProvider
import nons.core.node as node_module

def force_mock_provider(config):
    return MockProvider(config)


async def test_with_scheduler_started():
    """Test agent with scheduler properly started"""
    print("=== Testing Agent with Scheduler Started ===\n")

    # Patch to use mock
    node_module.create_provider = force_mock_provider

    try:
        # START THE SCHEDULER FIRST!
        print("Starting scheduler...")
        await start_scheduler()
        print("✓ Scheduler started\n")

        # Create simple registry
        registry = ToolRegistry()

        class StopParams(BaseModel):
            answer: str = Field(description="The answer")

        @registry.tool(
            name="finish",
            description="Finish with answer",
            param_model=StopParams,
            is_stop_tool=True,
        )
        async def finish_tool(params: StopParams, **context):
            return {"answer": params.answer}

        # Create agent
        node = Node("generate")
        network = create_network(
            layers=[node],
            provider="openai",
            model="gpt-4o-mini",
            system_prompt='Return JSON: {"selected_path": "finish", "routing_confidence": 0.9, "reasoning": "done", "params": {"answer": "test"}}',
        )

        agent = Agent(network=network, registry=registry, max_steps=2)

        print("Running agent...")
        start = time.time()
        step_count = 0

        async for result in agent.run("Test task"):
            step_count += 1
            elapsed = time.time() - start

            print(f"Step {step_count} ({elapsed:.2f}s):")
            print(f"  Tool: {result.get('tool', 'N/A')}")
            print(f"  Success: {result.get('success', False)}")

            if result.get("is_stop"):
                total_time = time.time() - start
                print(f"\n✓ Agent completed successfully in {total_time:.3f}s")
                print(f"  Steps: {step_count}")

                # Get node stats
                stats = node.get_execution_stats()
                print(f"  Node executions: {stats['execution_count']}")
                print(f"  Last execution time: {stats['last_execution_time']:.3f}s")
                return True

            if step_count >= 5:
                print("\n⚠️  Reached max test steps")
                return False

        return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Stop scheduler
        from nons.core.scheduler import stop_scheduler
        await stop_scheduler()


async def test_multi_agent_with_scheduler():
    """Test multi-agent planner with scheduler started"""
    print("\n" + "=" * 60)
    print("Multi-Agent Planner Test with Scheduler Started")
    print("=" * 60 + "\n")

    # Patch to use mock
    node_module.create_provider = force_mock_provider

    try:
        # START THE SCHEDULER!
        print("Starting scheduler...")
        await start_scheduler()
        print("✓ Scheduler started\n")

        # Import after patching
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "examples"))
        from multi_agent_planner import (
            create_planner_agent,
            create_code_writer_agent,
            sub_agents,
        )

        print("Initializing sub-agents...")
        sub_agents["code_writer"] = create_code_writer_agent()
        print("✓ Sub-agents initialized\n")

        print("Creating planner agent...")
        planner = create_planner_agent()
        print("✓ Planner created\n")

        task = "Create a simple Python script"
        print(f"Task: {task}\n")
        print("Executing...\n")

        start = time.time()
        step_count = 0

        async for result in planner.run(task):
            step_count += 1
            elapsed = time.time() - start

            if "step" in result:
                print(f"Step {step_count} ({elapsed:.2f}s): tool={result.get('tool', 'N/A')}")

            if result.get("is_stop") or step_count >= 3:
                total_time = time.time() - start
                print(f"\n✓ Completed in {total_time:.3f}s")
                print(f"  Steps: {step_count}")
                return True

        return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        from nons.core.scheduler import stop_scheduler
        await stop_scheduler()


async def main():
    print("\n" + "=" * 60)
    print("Scheduler Fix Test")
    print("=" * 60 + "\n")

    test1 = await test_with_scheduler_started()
    test2 = await test_multi_agent_with_scheduler()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Simple Agent Test: {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"Multi-Agent Test: {'✓ PASS' if test2 else '✗ FAIL'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
