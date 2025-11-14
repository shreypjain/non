#!/usr/bin/env python3
"""
Simplified agent test to identify where the hang occurs.
Uses forced mock provider to rule out API issues.
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
from pydantic import BaseModel, Field
import nons.operators.base


# Force MockProvider
from nons.utils.providers import MockProvider
import nons.core.node as node_module

original_create_provider = node_module.create_provider


def force_mock_provider(config):
    print(f"[TRACE] Creating MockProvider for {config.model_name}")
    return MockProvider(config)


async def test_mock_provider_directly():
    """Test MockProvider directly to ensure it works"""
    print("=== Testing MockProvider Directly ===\n")

    config = ModelConfig(provider=ModelProvider.MOCK, model_name="mock-model")
    provider = MockProvider(config)

    print("Making test call to MockProvider...")
    start = time.time()

    try:
        result, metrics = await provider.generate_completion("Test prompt")
        elapsed = time.time() - start

        print(f"Response: {result[:100]}...")
        print(f"Latency: {elapsed:.3f}s")
        print("✓ MockProvider works\n")
        return True
    except Exception as e:
        print(f"❌ MockProvider failed: {e}\n")
        return False


async def test_node_execution():
    """Test Node execution with mock provider"""
    print("=== Testing Node Execution ===\n")

    # Patch to use mock
    node_module.create_provider = force_mock_provider

    try:
        config = ModelConfig(provider=ModelProvider.OPENAI, model_name="gpt-4o-mini")
        node = Node("generate", model_config=config)

        print("Executing node...")
        start = time.time()

        # Set a timeout
        try:
            result = await asyncio.wait_for(
                node.execute("Test input"), timeout=10.0
            )
            elapsed = time.time() - start

            print(f"Result: {result[:100] if result else 'None'}...")
            print(f"Latency: {elapsed:.3f}s")
            print("✓ Node execution works\n")
            return True
        except asyncio.TimeoutError:
            print("❌ Node execution HUNG (timeout after 10s)\n")
            return False

    except Exception as e:
        print(f"❌ Node execution failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    finally:
        node_module.create_provider = original_create_provider


async def test_network_execution():
    """Test Network execution with mock provider"""
    print("=== Testing Network Execution ===\n")

    # Patch to use mock
    node_module.create_provider = force_mock_provider

    try:
        node = Node("generate")
        network = create_network(
            layers=[node],
            provider="openai",
            model="gpt-4o-mini",
            system_prompt="You are a test assistant",
        )

        print("Executing network...")
        start = time.time()

        # Set a timeout
        try:
            result = await asyncio.wait_for(
                network.forward({"content": "Test"}), timeout=10.0
            )
            elapsed = time.time() - start

            print(f"Result: {result}")
            print(f"Latency: {elapsed:.3f}s")
            print("✓ Network execution works\n")
            return True
        except asyncio.TimeoutError:
            print("❌ Network execution HUNG (timeout after 10s)\n")
            return False

    except Exception as e:
        print(f"❌ Network execution failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    finally:
        node_module.create_provider = original_create_provider


async def test_agent_execution():
    """Test Agent execution with mock provider"""
    print("=== Testing Agent Execution ===\n")

    # Patch to use mock
    node_module.create_provider = force_mock_provider

    try:
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
            system_prompt="Return JSON: {\"selected_path\": \"finish\", \"routing_confidence\": 0.9, \"reasoning\": \"done\", \"params\": {\"answer\": \"test\"}}",
        )

        agent = Agent(network=network, registry=registry, max_steps=2)

        print("Running agent (with 10s timeout per step)...")
        start = time.time()
        step_count = 0

        try:
            async for result in agent.run("Test task"):
                step_count += 1
                elapsed = time.time() - start

                print(f"Step {step_count} ({elapsed:.2f}s): tool={result.get('tool', 'N/A')}, success={result.get('success', False)}")

                if result.get("is_stop"):
                    print(f"✓ Agent completed in {elapsed:.3f}s\n")
                    return True

                if elapsed > 15:  # Safety timeout
                    print("❌ Agent HUNG (exceeded 15s)\n")
                    return False

        except asyncio.TimeoutError:
            print("❌ Agent execution HUNG (timeout)\n")
            return False

    except Exception as e:
        print(f"❌ Agent execution failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    finally:
        node_module.create_provider = original_create_provider


async def main():
    print("\n" + "=" * 60)
    print("Progressive Hang Diagnosis Test")
    print("=" * 60 + "\n")

    tests = [
        ("MockProvider Direct", test_mock_provider_directly),
        ("Node Execution", test_node_execution),
        ("Network Execution", test_network_execution),
        ("Agent Execution", test_agent_execution),
    ]

    results = {}

    for name, test_func in tests:
        print(f"Running: {name}")
        print("-" * 60)
        results[name] = await test_func()

        if not results[name]:
            print(f"⚠️  Test '{name}' failed or hung - stopping here\n")
            break

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL/HUNG"
        print(f"{name:30s}: {status}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
