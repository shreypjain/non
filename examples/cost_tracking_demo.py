#!/usr/bin/env python3
"""
Cost and Token Tracking Demo

This demo showcases the cost and token tracking capabilities of the NoN system
with mock LLM providers that simulate real API usage and costs.
"""

import asyncio
import sys
import os

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nons.core.node import Node
from nons.core.layer import Layer
from nons.core.network import NoN
from nons.core.types import (
    ModelConfig,
    ModelProvider,
    TokenUsage,
    CostInfo,
    ExecutionMetrics,
)
from nons.utils.providers import create_provider, test_provider
import nons.operators.base


async def demo_provider_integration():
    """Demonstrate LLM provider integration with cost tracking."""
    print("🔗 PROVIDER INTEGRATION DEMO")
    print("=" * 50)

    # Test different model configurations
    configs = [
        ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=1000,
        ),
        ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens=1000,
        ),
    ]

    for config in configs:
        print(f"\nTesting {config.provider.value} - {config.model_name}:")
        result = await test_provider(config)

        if result["status"] == "success":
            print(f"  ✅ Connection successful")
            print(f"  📄 Response: {result['completion'][:60]}...")
            print(f"  🔢 Tokens: {result['tokens']}")
            print(f"  💰 Cost: {result['cost']}")
            print(f"  ⏱️  Response time: {result['response_time_ms']:.1f}ms")
        else:
            print(f"  ❌ Connection failed: {result['error']}")
            print(f"  📝 Note: Using mock provider instead")

    print("\n" + "=" * 50 + "\n")


async def demo_node_cost_tracking():
    """Demonstrate cost tracking at the Node level."""
    print("🔵 NODE COST TRACKING")
    print("=" * 50)

    # Create a node (will use mock provider due to no API keys in demo)
    node = Node("generate")

    print("Before execution:")
    print(node)
    print()

    # Execute the node multiple times to accumulate costs
    inputs = [
        "Write a short story about AI",
        "Create a business plan for a startup",
        "Explain quantum computing",
    ]

    for i, input_text in enumerate(inputs, 1):
        print(f"Execution {i}: {input_text[:30]}...")
        result = await node.execute(input_text)
        print(f"  Result: {result[:50]}...")

        # Show latest metrics
        if node.get_last_metrics():
            metrics = node.get_last_metrics()
            print(f"  Tokens: {metrics.token_usage.total_tokens}")
            print(f"  Cost: ${metrics.cost_info.total_cost_usd:.6f}")
        print()

    print("After all executions:")
    print(node)
    print()

    # Show detailed statistics
    stats = node.get_execution_stats()
    print("📊 Detailed Statistics:")
    print(f"  Total executions: {stats['execution_count']}")
    print(f"  Total tokens: {stats['total_tokens']['total_tokens']:,}")
    print(f"  Total cost: ${stats['total_cost_usd']:.6f}")
    print(
        f"  Average tokens per execution: {stats['average_tokens_per_execution']:.1f}"
    )
    print(f"  Average cost per execution: ${stats['average_cost_per_execution']:.6f}")

    print("\n" + "=" * 50 + "\n")


async def demo_network_cost_aggregation():
    """Demonstrate cost aggregation across a network."""
    print("🟢 NETWORK COST AGGREGATION")
    print("=" * 50)

    # Create a network with multiple layers
    network = NoN.from_operators(
        [
            "generate",  # Layer 0
            ["generate", "condense"],  # Layer 1 (parallel)
            "generate",  # Layer 2
        ]
    )

    print("Network architecture:")
    print(network)
    print()

    # Execute the network
    print("Executing network...")
    result = await network.forward("Design a sustainable smart city of the future")
    print()

    print("Network after execution:")
    print(network)
    print()

    # Aggregate costs across all nodes
    total_cost = 0.0
    total_tokens = 0

    print("📊 Per-Node Cost Breakdown:")
    for i, layer in enumerate(network.layers):
        layer_cost = 0.0
        layer_tokens = 0

        print(f"  Layer {i}:")
        for j, node in enumerate(layer.nodes):
            node_cost = node.get_total_cost()
            node_tokens = node.get_total_tokens()

            total_cost += node_cost
            total_tokens += node_tokens
            layer_cost += node_cost
            layer_tokens += node_tokens

            print(
                f"    Node {j} ({node.operator_name}): {node_tokens:,} tokens, ${node_cost:.6f}"
            )

        print(f"    Layer {i} Total: {layer_tokens:,} tokens, ${layer_cost:.6f}")
        print()

    print(f"🏆 NETWORK TOTALS:")
    print(f"  Total Tokens: {total_tokens:,}")
    print(f"  Total Cost: ${total_cost:.6f}")
    print(f"  Nodes Executed: {sum(len(layer.nodes) for layer in network.layers)}")

    print("\n" + "=" * 50 + "\n")


async def demo_cost_optimization():
    """Demonstrate cost optimization strategies."""
    print("💡 COST OPTIMIZATION STRATEGIES")
    print("=" * 50)

    # Compare different model configurations
    configs = [
        (
            "GPT-4 (High-performance)",
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                temperature=0.7,
                max_tokens=500,
            ),
        ),
        (
            "GPT-3.5-Turbo (Cost-effective)",
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=500,
            ),
        ),
        (
            "Claude Haiku (Budget)",
            ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-haiku-20240307",
                temperature=0.7,
                max_tokens=500,
            ),
        ),
    ]

    prompt = "Explain the benefits of renewable energy in 100 words"

    print(f"📝 Test prompt: {prompt}")
    print()

    for name, config in configs:
        node = Node("generate", model_config=config)

        # Execute with the same prompt
        result = await node.execute(prompt)
        metrics = node.get_last_metrics()

        if metrics:
            print(f"🔸 {name}:")
            print(f"  Model: {config.model_name}")
            print(f"  Tokens: {metrics.token_usage.total_tokens}")
            print(f"  Cost: ${metrics.cost_info.total_cost_usd:.6f}")
            print(f"  Result: {result[:60]}...")
            print()

    print("💰 Cost Optimization Tips:")
    print("  • Use smaller models for simple tasks")
    print("  • Set appropriate max_tokens limits")
    print("  • Consider Claude Haiku for budget-conscious applications")
    print("  • Use GPT-3.5-Turbo as a good balance of cost and performance")
    print("  • Reserve GPT-4 for complex reasoning tasks")

    print("\n" + "=" * 50 + "\n")


async def main():
    """Run all cost tracking demonstrations."""
    print("💰 NoN COST & TOKEN TRACKING SHOWCASE")
    print("=" * 50)
    print("This demo showcases comprehensive cost and token tracking!")
    print("=" * 50)
    print()

    await demo_provider_integration()
    await demo_node_cost_tracking()
    await demo_network_cost_aggregation()
    await demo_cost_optimization()

    print("🎉 COST TRACKING DEMO COMPLETED!")
    print("=" * 50)
    print("✨ Comprehensive cost and token tracking at every level!")
    print("✨ Real-time cost monitoring for budget management!")
    print("✨ Detailed metrics for optimization and analysis!")
    print("✨ Support for OpenAI and Anthropic with accurate pricing!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
