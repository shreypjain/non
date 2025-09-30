#!/usr/bin/env python3
"""
Google Gemini Integration Demo

This demo showcases the Google Gemini integration with the NoN system,
testing the latest Gemini models and cost tracking.
"""

import asyncio
import sys
import os

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nons.core.node import Node
from nons.core.network import NoN
from nons.core.types import ModelConfig, ModelProvider
from nons.utils.providers import test_provider
import nons.operators.base


async def test_google_models():
    """Test different Google Gemini models."""
    print("🚀 TESTING GOOGLE GEMINI MODELS")
    print("=" * 60)

    models_to_test = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]

    for model in models_to_test:
        print(f"Testing {model}...")
        config = ModelConfig(
            provider=ModelProvider.GOOGLE,
            model_name=model,
            temperature=0.7,
            max_tokens=50
        )

        result = await test_provider(config)

        if result["status"] == "success":
            print(f"  ✅ {model}: ACCESSIBLE")
            print(f"     Tokens: {result['tokens']}")
            print(f"     Cost: {result['cost']}")
            print(f"     Response time: {result['response_time_ms']:.1f}ms")
            print(f"     Response: {result['completion'][:60]}...")
        else:
            print(f"  ❌ {model}: {result['error'][:80]}...")

        await asyncio.sleep(0.5)
    print()


async def demo_google_node():
    """Demonstrate Google Gemini with Node execution."""
    print("🤖 GOOGLE GEMINI NODE DEMO")
    print("=" * 60)

    node = Node(
        'generate',
        model_config=ModelConfig(
            provider=ModelProvider.GOOGLE,
            model_name="gemini-2.5-flash",
            temperature=0.7,
            max_tokens=100
        )
    )

    print("Before execution:")
    print(node)
    print()

    # Test prompts
    prompts = [
        "What is 2+2?",
        "Write a haiku about technology",
        "Explain machine learning in one sentence"
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"Execution {i}: {prompt}")
        result = await node.execute(prompt)
        print(f"  Result: {result[:80]}...")

        metrics = node.get_last_metrics()
        if metrics:
            print(f"  Tokens: {metrics.token_usage.total_tokens}")
            print(f"  Cost: ${metrics.cost_info.total_cost_usd:.6f}")
            print(f"  Response time: {metrics.response_time_ms:.1f}ms")
        print()

    print("After all executions:")
    print(node)
    print()


async def demo_mixed_provider_network():
    """Demonstrate network with Google, Anthropic, and OpenAI providers."""
    print("🌐 MIXED PROVIDER NETWORK WITH GOOGLE")
    print("=" * 60)

    # Create network with mixed providers
    network = NoN.from_operators([
        'generate',                    # Layer 0: Single generate
        ['generate', 'generate']       # Layer 1: Parallel generates
    ])

    # Configure different providers
    # Layer 0: Google Gemini
    network.layers[0].nodes[0].configure_model(
        provider=ModelProvider.GOOGLE,
        model_name="gemini-2.5-flash",
        max_tokens=50
    )

    # Layer 1: Mixed Google and Anthropic
    network.layers[1].nodes[0].configure_model(
        provider=ModelProvider.GOOGLE,
        model_name="gemini-2.0-flash",
        max_tokens=50
    )
    network.layers[1].nodes[1].configure_model(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        max_tokens=50
    )

    print("Network architecture:")
    print(network)
    print()

    # Execute the network
    print("Executing mixed provider network...")
    result = await network.forward("What are the benefits of renewable energy?")
    print()

    print("Network after execution:")
    print(network)
    print()

    # Cost breakdown by provider
    google_cost = 0.0
    anthropic_cost = 0.0
    google_tokens = 0
    anthropic_tokens = 0

    print("📊 Cost Breakdown by Provider:")
    for i, layer in enumerate(network.layers):
        print(f"  Layer {i}:")
        for j, node in enumerate(layer.nodes):
            metrics = node.get_last_metrics()
            if metrics:
                cost = node.get_total_cost()
                tokens = node.get_total_tokens()
                print(f"    Node {j} ({metrics.provider}): {tokens} tokens, ${cost:.6f}")

                if metrics.provider == "google":
                    google_cost += cost
                    google_tokens += tokens
                else:
                    anthropic_cost += cost
                    anthropic_tokens += tokens

    print()
    print("🏆 PROVIDER TOTALS:")
    print(f"  Google: {google_tokens} tokens, ${google_cost:.6f}")
    print(f"  Anthropic: {anthropic_tokens} tokens, ${anthropic_cost:.6f}")
    print(f"  Total: {google_tokens + anthropic_tokens} tokens, ${google_cost + anthropic_cost:.6f}")

    print("\n" + "="*60 + "\n")


async def main():
    """Run Google Gemini integration demo."""
    print("🎯 GOOGLE GEMINI + NoN INTEGRATION")
    print("="*60)
    print("Testing the latest Gemini models with NoN system!")
    print("="*60)
    print()

    await test_google_models()
    await demo_google_node()
    await demo_mixed_provider_network()

    print("🎉 GOOGLE GEMINI INTEGRATION COMPLETED!")
    print("="*60)
    print("✨ Gemini 2.5 models integrated with cost tracking!")
    print("✨ Mixed provider networks with Google + Anthropic!")
    print("✨ Latest AI models available in NoN system!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())