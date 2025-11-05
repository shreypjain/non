#!/usr/bin/env python3
"""
Model Discovery and Capability Testing

This script tests available models and their capabilities with the current API keys,
helping determine what models are accessible and their rate limits.
"""

import asyncio
import sys
import os
import time

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nons.core.types import ModelConfig, ModelProvider
from nons.utils.providers import create_provider, test_provider


async def test_openai_models():
    """Test various OpenAI models to see what's accessible."""
    print("🔵 TESTING OPENAI MODELS")
    print("=" * 50)

    # List of OpenAI models to test
    models_to_test = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
    ]

    accessible_models = []

    for model in models_to_test:
        print(f"Testing {model}...")
        config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name=model,
            temperature=0.7,
            max_tokens=50,
        )

        result = await test_provider(config)

        if result["status"] == "success":
            print(f"  ✅ {model}: ACCESSIBLE")
            print(f"     Tokens: {result['tokens']}")
            print(f"     Cost: {result['cost']}")
            print(f"     Response time: {result['response_time_ms']:.1f}ms")
            accessible_models.append(model)
        else:
            print(f"  ❌ {model}: {result['error'][:80]}...")

        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)

    print(
        f"\n📊 OpenAI Summary: {len(accessible_models)}/{len(models_to_test)} models accessible"
    )
    if accessible_models:
        print(f"✅ Accessible: {', '.join(accessible_models)}")
    print()


async def test_anthropic_models():
    """Test various Anthropic models to see what's accessible."""
    print("🟡 TESTING ANTHROPIC MODELS")
    print("=" * 50)

    # List of Anthropic models to test
    models_to_test = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-instant-1.2",
    ]

    accessible_models = []

    for model in models_to_test:
        print(f"Testing {model}...")
        config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name=model,
            temperature=0.7,
            max_tokens=50,
        )

        result = await test_provider(config)

        if result["status"] == "success":
            print(f"  ✅ {model}: ACCESSIBLE")
            print(f"     Tokens: {result['tokens']}")
            print(f"     Cost: {result['cost']}")
            print(f"     Response time: {result['response_time_ms']:.1f}ms")
            accessible_models.append(model)
        else:
            print(f"  ❌ {model}: {result['error'][:80]}...")

        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)

    print(
        f"\n📊 Anthropic Summary: {len(accessible_models)}/{len(models_to_test)} models accessible"
    )
    if accessible_models:
        print(f"✅ Accessible: {', '.join(accessible_models)}")
    print()


async def test_rate_limits():
    """Test rate limits by making rapid successive calls."""
    print("⚡ TESTING RATE LIMITS")
    print("=" * 50)

    # Use the fastest/cheapest model for rate limit testing
    config = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        temperature=0.7,
        max_tokens=10,
    )

    print("Making 5 rapid successive calls to test rate limits...")

    times = []
    for i in range(5):
        start = time.time()
        result = await test_provider(config)
        end = time.time()

        times.append(end - start)

        if result["status"] == "success":
            print(f"  Call {i+1}: ✅ {(end-start)*1000:.0f}ms")
        else:
            print(f"  Call {i+1}: ❌ {result['error'][:50]}...")
            if "rate" in result["error"].lower():
                print("    ^ Rate limit detected!")
                break

    if len(times) >= 2:
        avg_time = sum(times) / len(times)
        print(f"\n📊 Average response time: {avg_time*1000:.1f}ms")
        print(f"📊 Requests completed: {len(times)}/5")
    print()


async def test_token_limits():
    """Test maximum token limits for accessible models."""
    print("📏 TESTING TOKEN LIMITS")
    print("=" * 50)

    # Test with different token limits
    token_limits = [100, 1000, 4000, 8000]

    for max_tokens in token_limits:
        print(f"Testing max_tokens={max_tokens}...")

        config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-haiku-20240307",
            temperature=0.7,
            max_tokens=max_tokens,
        )

        # Use a prompt that would generate a long response
        try:
            provider = create_provider(config)
            result, metrics = await provider.generate_completion(
                "Write a detailed explanation of machine learning algorithms"
            )

            print(
                f"  ✅ {max_tokens} tokens: Generated {metrics.token_usage.completion_tokens} tokens"
            )
            print(f"     Cost: ${metrics.cost_info.total_cost_usd:.6f}")

        except Exception as e:
            print(f"  ❌ {max_tokens} tokens: {str(e)[:60]}...")

        await asyncio.sleep(0.5)
    print()


async def main():
    """Run comprehensive model and capability testing."""
    print("🔍 MODEL DISCOVERY & CAPABILITY TESTING")
    print("=" * 60)
    print("Testing available models and their capabilities...")
    print("=" * 60)
    print()

    await test_openai_models()
    await test_anthropic_models()
    await test_rate_limits()
    await test_token_limits()

    print("🎉 MODEL DISCOVERY COMPLETED!")
    print("=" * 60)
    print("✨ Use the accessible models shown above in your configurations!")
    print("✨ Be mindful of rate limits and token costs!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
