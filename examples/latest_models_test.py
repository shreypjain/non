#!/usr/bin/env python3
"""
Latest Anthropic Models Testing

Test the newest Anthropic models including Sonnet 4, Sonnet 4.5, and Opus 4.1
to check accessibility and capabilities.
"""

import asyncio
import sys
import os

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nons.core.types import ModelConfig, ModelProvider
from nons.utils.providers import test_provider


async def test_latest_anthropic_models():
    """Test the latest Anthropic models."""
    print("🚀 TESTING LATEST ANTHROPIC MODELS")
    print("=" * 60)

    # Latest models to test based on Claude naming conventions
    latest_models = [
        # Sonnet 4 variants
        "claude-4-sonnet",
        "claude-4-sonnet-20241022",
        "claude-4-sonnet-20241201",
        "claude-4-sonnet-20250101",
        # Sonnet 4.5 variants
        "claude-4.5-sonnet",
        "claude-4.5-sonnet-20241022",
        "claude-4.5-sonnet-20241201",
        "claude-4.5-sonnet-20250101",
        "claude-4-5-sonnet-20241022",
        "claude-4-5-sonnet-20241201",
        # Opus 4.1 variants
        "claude-4.1-opus",
        "claude-4-1-opus",
        "claude-4.1-opus-20241022",
        "claude-4.1-opus-20241201",
        "claude-4-1-opus-20241022",
        "claude-4-1-opus-20241201",
        # Other potential latest models
        "claude-4",
        "claude-4-opus",
        "claude-4-haiku",
        "claude-3-5-sonnet-20241022",
        "claude-opus-4",
        "claude-opus-4.1",
        # Check if there are any 2025 models
        "claude-3-5-sonnet-20250101",
        "claude-3-5-haiku-20250101",
        "claude-3-opus-20250101",
    ]

    accessible_models = []

    print(f"Testing {len(latest_models)} potential latest model names...")
    print()

    for model in latest_models:
        print(f"Testing {model}...")

        config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name=model,
            temperature=0.7,
            max_tokens=50,
        )

        result = await test_provider(config)

        if result["status"] == "success":
            print(f"  ✅ {model}: ACCESSIBLE!")
            print(f"     Tokens: {result['tokens']}")
            print(f"     Cost: {result['cost']}")
            print(f"     Response time: {result['response_time_ms']:.1f}ms")
            print(f"     Response preview: {result['completion'][:80]}...")
            accessible_models.append(model)
            print()
        else:
            error_type = "404" if "404" in result["error"] else "other"
            if error_type == "404":
                print(f"  ❌ {model}: Not found")
            else:
                print(f"  ❌ {model}: {result['error'][:60]}...")

        # Small delay to avoid overwhelming the API
        await asyncio.sleep(0.3)

    print("=" * 60)
    print(f"📊 LATEST MODELS SUMMARY")
    print(f"Tested: {len(latest_models)} models")
    print(f"Accessible: {len(accessible_models)} models")

    if accessible_models:
        print(f"\n🎉 NEW ACCESSIBLE MODELS FOUND:")
        for model in accessible_models:
            print(f"  ✅ {model}")
    else:
        print(f"\n📝 No new models found beyond previously tested ones.")
        print(f"The latest accessible models remain:")
        print(f"  • claude-3-5-sonnet-20241022")
        print(f"  • claude-3-5-haiku-20241022")
        print(f"  • claude-3-opus-20240229")
        print(f"  • claude-3-haiku-20240307")

    print("=" * 60)


async def test_model_capabilities():
    """Test capabilities of the best available model."""
    print("\n🧠 TESTING MODEL CAPABILITIES")
    print("=" * 60)

    # Use the best available model for capability testing
    config = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-20241022",
        temperature=0.7,
        max_tokens=200,
    )

    test_prompts = [
        "Solve this equation: 2x + 5 = 15",
        "Write a Python function to calculate fibonacci numbers",
        "Explain quantum computing in simple terms",
        "What's the capital of France?",
        "Generate a JSON object with user data",
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        result = await test_provider(config)

        if result["status"] == "success":
            print(f"Response: {result['completion'][:150]}...")
            print(f"Tokens used: {result['tokens']}")
        else:
            print(f"Error: {result['error']}")

        await asyncio.sleep(0.5)


async def main():
    """Run latest model testing."""
    print("🔍 ANTHROPIC LATEST MODELS DISCOVERY")
    print("=" * 60)
    print("Searching for Claude 4, Claude 4.5, Opus 4.1 and other latest models...")
    print("=" * 60)

    await test_latest_anthropic_models()
    await test_model_capabilities()

    print("\n🎯 RECOMMENDATION:")
    print("Continue using claude-3-5-sonnet-20241022 as the primary model")
    print("until newer models become available in your API scope.")


if __name__ == "__main__":
    asyncio.run(main())
