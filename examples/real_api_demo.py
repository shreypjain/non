#!/usr/bin/env python3
"""
Real API Cost and Token Tracking Demo

This demo showcases the cost and token tracking with real Anthropic API
and mock OpenAI (due to invalid key), demonstrating mixed provider usage.
"""

import asyncio
import sys
import os

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nons.core.node import Node
from nons.core.network import NoN
from nons.core.types import ModelConfig, ModelProvider
import nons.operators.base


async def demo_mixed_providers():
    """Demonstrate mixed real and mock provider usage."""
    print("🔵 MIXED PROVIDER DEMO")
    print("=" * 50)

    # Create nodes with different providers
    anthropic_node = Node(
        'generate',
        model_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-haiku-20240307",
            temperature=0.7,
            max_tokens=100
        )
    )

    openai_node = Node(
        'generate',
        model_config=ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=100
        )
    )

    print("Testing Anthropic (real API):")
    result1 = await anthropic_node.execute("Write a short poem about technology")
    print(f"  Result: {result1[:60]}...")
    metrics1 = anthropic_node.get_last_metrics()
    if metrics1:
        print(f"  Tokens: {metrics1.token_usage.total_tokens}")
        print(f"  Cost: ${metrics1.cost_info.total_cost_usd:.6f}")
        print(f"  Provider: {metrics1.provider}")
    print()

    print("Testing OpenAI (fallback to mock):")
    result2 = await openai_node.execute("Write a short poem about innovation")
    print(f"  Result: {result2[:60]}...")
    metrics2 = openai_node.get_last_metrics()
    if metrics2:
        print(f"  Tokens: {metrics2.token_usage.total_tokens}")
        print(f"  Cost: ${metrics2.cost_info.total_cost_usd:.6f}")
        print(f"  Provider: {metrics2.provider}")

    print("\n" + "="*50 + "\n")


async def demo_network_mixed_providers():
    """Demonstrate network with mixed providers."""
    print("🟢 NETWORK WITH MIXED PROVIDERS")
    print("=" * 50)

    # Create a network with mixed providers using from_operators
    network = NoN.from_operators([
        'generate',                    # Layer 0: Single generate (will use default config)
        ['generate', 'generate']       # Layer 1: Parallel generates
    ])

    # Update the nodes with specific provider configurations
    # Layer 0: Anthropic (real API)
    network.layers[0].nodes[0].configure_model(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        max_tokens=50
    )

    # Layer 1: Mixed providers
    network.layers[1].nodes[0].configure_model(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        max_tokens=50
    )
    network.layers[1].nodes[1].configure_model(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        max_tokens=50
    )

    print("Network architecture:")
    print(network)
    print()

    # Execute the network
    print("Executing mixed provider network...")
    result = await network.forward("Explain the benefits of renewable energy")
    print()

    print("Network after execution:")
    print(network)
    print()

    # Show cost breakdown by provider
    anthropic_cost = 0.0
    openai_cost = 0.0
    anthropic_tokens = 0
    openai_tokens = 0

    print("📊 Cost Breakdown by Provider:")
    for i, layer in enumerate(network.layers):
        print(f"  Layer {i}:")
        for j, node in enumerate(layer.nodes):
            metrics = node.get_last_metrics()
            if metrics:
                cost = node.get_total_cost()
                tokens = node.get_total_tokens()
                print(f"    Node {j} ({metrics.provider}): {tokens} tokens, ${cost:.6f}")

                if metrics.provider == "anthropic":
                    anthropic_cost += cost
                    anthropic_tokens += tokens
                else:
                    openai_cost += cost
                    openai_tokens += tokens

    print()
    print("🏆 PROVIDER TOTALS:")
    print(f"  Anthropic (real): {anthropic_tokens} tokens, ${anthropic_cost:.6f}")
    print(f"  OpenAI (mock): {openai_tokens} tokens, ${openai_cost:.6f}")
    print(f"  Total: {anthropic_tokens + openai_tokens} tokens, ${anthropic_cost + openai_cost:.6f}")

    print("\n" + "="*50 + "\n")


async def main():
    """Run mixed provider demonstrations."""
    print("🌐 REAL API + MOCK PROVIDER DEMO")
    print("="*50)
    print("Testing with real Anthropic API and OpenAI mock fallback!")
    print("="*50)
    print()

    await demo_mixed_providers()
    await demo_network_mixed_providers()

    print("🎉 MIXED PROVIDER DEMO COMPLETED!")
    print("="*50)
    print("✨ Real Anthropic API calls with token/cost tracking!")
    print("✨ Graceful fallback to mock for invalid OpenAI keys!")
    print("✨ Mixed provider networks working seamlessly!")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())