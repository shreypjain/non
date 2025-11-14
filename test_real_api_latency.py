#!/usr/bin/env python3
"""
Test to verify real API calls are being made and measure actual latency.
"""

import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from nons.core.types import ModelConfig, ModelProvider
from nons.utils.providers import create_provider, OPENAI_AVAILABLE
from nons.core.config import get_api_key


async def test_provider_selection():
    """Test which provider is actually being created"""
    print("=== Provider Selection Test ===\n")

    # Check API key availability
    openai_key = get_api_key(ModelProvider.OPENAI)
    print(f"OpenAI API key available: {bool(openai_key)}")
    print(f"OpenAI package installed: {OPENAI_AVAILABLE}\n")

    # Create config
    config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens=100,
    )

    # Create provider
    provider = create_provider(config)
    print(f"Provider created: {type(provider).__name__}")
    print(f"Provider class: {provider.__class__.__module__}.{provider.__class__.__name__}\n")

    # Test a simple completion
    print("Testing actual API call...\n")
    start = time.time()

    try:
        result, metrics = await provider.generate_completion("Say 'Hello from real API' if you are a real LLM")
        elapsed = time.time() - start

        print(f"Response: {result}")
        print(f"\nLatency: {elapsed:.3f}s ({metrics.response_time_ms:.1f}ms)")
        print(f"Tokens used: {metrics.token_usage.total_tokens}")
        print(f"Cost: ${metrics.cost_info.total_cost_usd:.6f}")
        print(f"Provider: {metrics.provider}")
        print(f"Model: {metrics.model_name}")

        # Check if this was a real API call
        if "Mock" in type(provider).__name__:
            print("\n⚠️  WARNING: Using MockProvider - not making real API calls!")
            return False
        elif metrics.token_usage.total_tokens > 0:
            print("\n✓ Confirmed: Real API call with actual token usage")
            return True
        else:
            print("\n⚠️  WARNING: No token usage recorded - might be mock")
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_with_real_api():
    """Test agent execution with real API and measure latency"""
    print("\n" + "=" * 60)
    print("Agent Execution Test with Real API")
    print("=" * 60 + "\n")

    from nons.core.node import Node
    from nons.core.network import create_network
    from nons.core.agents.agent import Agent
    from nons.core.agents.registry import ToolRegistry
    from pydantic import BaseModel, Field
    import nons.operators.base

    # Create a simple registry with one tool
    registry = ToolRegistry()

    class StopParams(BaseModel):
        final_answer: str = Field(description="The final answer")

    @registry.tool(
        name="finish",
        description="Return the final answer",
        param_model=StopParams,
        is_stop_tool=True,
    )
    async def finish_tool(params: StopParams, **context):
        return {"final_answer": params.final_answer}

    # Create agent
    node = Node("generate")
    network = create_network(
        layers=[node],
        provider="openai",
        model="gpt-4o-mini",
        system_prompt="""You are a test agent.

Respond with the finish tool and provide a simple greeting.

Return JSON format:
{
    "selected_path": "finish",
    "routing_confidence": 0.95,
    "reasoning": "completing task",
    "params": {"final_answer": "Hello from real API!"}
}""",
    )

    agent = Agent(network=network, registry=registry, max_steps=1)

    print("Running agent with real API calls...")
    print("Task: Say hello\n")

    start = time.time()
    step_count = 0

    try:
        async for result in agent.run("Say hello"):
            step_count += 1
            elapsed = time.time() - start

            print(f"Step {step_count} (elapsed: {elapsed:.3f}s):")
            print(f"  Tool: {result.get('tool', 'N/A')}")
            print(f"  Success: {result.get('success', False)}")

            if result.get("is_stop"):
                print(f"  Final result: {result.get('result', {})}")
                print("\n✓ Agent completed successfully")
                break

        total_time = time.time() - start
        print(f"\nTotal execution time: {total_time:.3f}s")
        print(f"Steps completed: {step_count}")

        # Get node stats to see actual API metrics
        node_stats = node.get_execution_stats()
        print(f"\nNode execution stats:")
        print(f"  Executions: {node_stats['execution_count']}")
        print(f"  Total tokens: {node_stats['total_tokens']}")
        print(f"  Total cost: ${node_stats['total_cost']:.6f}")

        if node_stats['total_tokens'] > 0:
            print("\n✓ Confirmed: Real API calls with actual token usage and costs")
            return True
        else:
            print("\n⚠️  WARNING: No tokens used - likely using mock provider")
            return False

    except Exception as e:
        print(f"\n❌ Error during agent execution: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("\n" + "=" * 60)
    print("Real API Verification and Latency Test")
    print("=" * 60 + "\n")

    # Test 1: Provider selection
    provider_real = await test_provider_selection()

    # Test 2: Agent execution
    agent_real = await test_agent_with_real_api()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Provider uses real API: {'✓ YES' if provider_real else '✗ NO (mock)'}")
    print(f"Agent uses real API: {'✓ YES' if agent_real else '✗ NO (mock)'}")

    if provider_real and agent_real:
        print("\n✓ All tests passed - using real API with measured latency")
    else:
        print("\n⚠️  WARNING: Some tests using mock provider - not real API calls")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
