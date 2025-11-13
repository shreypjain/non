#!/usr/bin/env python3
"""
Fallback Models Demo

This demo showcases the new fallback functionality:
- Latency-based fallback: Switch models if response time exceeds threshold
- Rate-limit-based fallback: Automatically use backup when rate limited
- Cascading fallback: Try multiple backup models in order

Run this demo to see intelligent model fallback in action!
"""

import asyncio
import sys
import os

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nons.core.node import Node
from nons.core.network import NoN
from nons.core.types import ModelConfig, ModelProvider
from nons.core.scheduler import start_scheduler, stop_scheduler
import nons.operators.base


async def demo_latency_fallback():
    """Demonstrate latency-based fallback."""
    print("=" * 70)
    print("LATENCY-BASED FALLBACK DEMO")
    print("=" * 70)
    print()

    # Create primary model with very low latency threshold
    # This will likely trigger fallback to faster model
    primary_config = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=50,
        max_latency_ms=10,  # Very low threshold - will likely trigger fallback
        fallback_on_rate_limit=True,
    )

    # Create fallback model (faster/cheaper alternative)
    fallback_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini",
        max_tokens=50,
        max_latency_ms=None,  # No latency limit for fallback
    )

    # Set fallback models
    primary_config.fallback_models = [fallback_config]

    # Create node with fallback configuration
    node = Node("generate", model_config=primary_config)

    print(f"Primary model: {primary_config.model_name} (max latency: {primary_config.max_latency_ms}ms)")
    print(f"Fallback model: {fallback_config.model_name}")
    print()

    # Start scheduler
    await start_scheduler()

    try:
        # Execute the node
        print("Executing request...")
        result = await node.execute("Write a very short haiku about AI")
        print(f"\nResult: {result}")
        print()

        # Check metrics to see if fallback was used
        if node._last_metrics:
            metrics = node._last_metrics
            print("Execution Metrics:")
            print(f"  Model used: {metrics.model_name}")
            print(f"  Response time: {metrics.response_time_ms:.2f}ms")
            print(f"  Fallback used: {metrics.fallback_used}")
            if metrics.fallback_reason:
                print(f"  Fallback reason: {metrics.fallback_reason}")
    finally:
        await stop_scheduler()

    print()


async def demo_rate_limit_awareness():
    """Demonstrate rate limit awareness."""
    print("=" * 70)
    print("RATE LIMIT AWARENESS DEMO")
    print("=" * 70)
    print()

    # Create node that monitors rate limits
    config = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=30,
        fallback_on_rate_limit=True,
    )

    node = Node("generate", model_config=config)

    print(f"Model: {config.model_name}")
    print(f"Monitoring rate limits: {config.fallback_on_rate_limit}")
    print()

    # Start scheduler
    await start_scheduler()

    try:
        # Execute multiple requests
        print("Executing 3 requests to monitor rate limits...")
        for i in range(3):
            result = await node.execute(f"Say hello number {i+1}")

            if node._last_metrics and node._last_metrics.rate_limit_info:
                rate_info = node._last_metrics.rate_limit_info
                print(f"\nRequest {i+1}:")
                print(f"  Result: {result[:50]}...")
                if rate_info.requests_remaining is not None:
                    print(f"  Requests remaining: {rate_info.requests_remaining}")
                if rate_info.tokens_remaining is not None:
                    print(f"  Tokens remaining: {rate_info.tokens_remaining}")
                if rate_info.should_consider_fallback():
                    print(f"  WARNING: Rate limits low, consider fallback!")
    finally:
        await stop_scheduler()

    print()


async def demo_cascading_fallback():
    """Demonstrate cascading through multiple fallback models."""
    print("=" * 70)
    print("CASCADING FALLBACK DEMO")
    print("=" * 70)
    print()

    # Create a chain of fallback models
    # Primary -> Fallback 1 -> Fallback 2
    primary_config = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=40,
    )

    fallback1_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini",
        max_tokens=40,
    )

    fallback2_config = ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_name="gemini-2.0-flash",
        max_tokens=40,
    )

    # Set up cascading fallbacks
    primary_config.fallback_models = [fallback1_config, fallback2_config]

    node = Node("generate", model_config=primary_config)

    print(f"Fallback chain:")
    print(f"  1. {primary_config.model_name} (Anthropic)")
    print(f"  2. {fallback1_config.model_name} (OpenAI)")
    print(f"  3. {fallback2_config.model_name} (Google)")
    print()

    # Start scheduler
    await start_scheduler()

    try:
        print("Executing with fallback chain...")
        result = await node.execute("Say hello briefly")
        print(f"\nResult: {result}")

        if node._last_metrics:
            print(f"Model actually used: {node._last_metrics.model_name}")
            print(f"Fallback was used: {node._last_metrics.fallback_used}")
    finally:
        await stop_scheduler()

    print()


async def demo_network_with_fallbacks():
    """Demonstrate fallback in a multi-node network."""
    print("=" * 70)
    print("NETWORK WITH FALLBACKS DEMO")
    print("=" * 70)
    print()

    # Create configs with fallbacks
    fast_config = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=80,
        max_latency_ms=5000,
    )

    backup_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini",
        max_tokens=80,
    )

    fast_config.fallback_models = [backup_config]

    # Create network with fallback-enabled nodes
    network = NoN.from_operators(
        ["generate", "condense"],
        model_config=fast_config,
    )

    print(f"Network created with {len(network.layers)} layers")
    print(f"Each node has fallback: {fast_config.model_name} -> {backup_config.model_name}")
    print()

    # Start scheduler
    await start_scheduler()

    try:
        print("Executing network...")
        result = await network.forward("Explain AI in simple terms")
        print(f"\nFinal result: {result.final_output[:100]}...")
        print(f"Total execution time: {result.execution_time:.3f}s")
        print(f"Layers executed: {result.total_layers}")
    finally:
        await stop_scheduler()

    print()


async def main():
    """Run all fallback demonstrations."""
    print("\n")
    print("=" * 70)
    print("NoN FALLBACK MODELS DEMONSTRATION")
    print("=" * 70)
    print()
    print("This demo showcases intelligent model fallback strategies:")
    print("  - Latency-based switching")
    print("  - Rate limit awareness")
    print("  - Cascading fallback chains")
    print("  - Network-wide fallback support")
    print()

    # Run demonstrations
    try:
        await demo_latency_fallback()
        await demo_rate_limit_awareness()
        await demo_cascading_fallback()
        await demo_network_with_fallbacks()

        print("=" * 70)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print()
        print("Key features demonstrated:")
        print("  ✓ Automatic fallback when latency exceeds threshold")
        print("  ✓ Rate limit monitoring and proactive fallback")
        print("  ✓ Cascading through multiple fallback models")
        print("  ✓ Fallback tracking in execution metrics")
        print("  ✓ Network-wide fallback configuration")
        print()

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
