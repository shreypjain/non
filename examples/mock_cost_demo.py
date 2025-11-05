#!/usr/bin/env python3
"""
Mock Cost and Token Tracking Demo

This demo showcases the cost and token tracking capabilities using mock providers
to demonstrate the system without requiring API keys.
"""

import asyncio
import sys
import os

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nons.core.node import Node
from nons.core.layer import Layer
from nons.core.network import NoN
from nons.core.types import ModelConfig, ModelProvider


async def demo_mock_cost_tracking():
    """Demonstrate cost tracking with mock providers."""
    print("🔵 MOCK COST TRACKING DEMO")
    print("=" * 50)

    # Create a node with OpenAI config (will fall back to mock)
    node = Node(
        "generate",
        model_config=ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=1000,
        ),
    )

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
    """Demonstrate cost aggregation across a network with mock providers."""
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


async def main():
    """Run mock cost tracking demonstrations."""
    print("💰 MOCK NoN COST & TOKEN TRACKING SHOWCASE")
    print("=" * 50)
    print("This demo showcases cost tracking using mock providers!")
    print("=" * 50)
    print()

    await demo_mock_cost_tracking()
    await demo_network_cost_aggregation()

    print("🎉 MOCK COST TRACKING DEMO COMPLETED!")
    print("=" * 50)
    print("✨ Cost and token tracking working with mock providers!")
    print("✨ Real-time cost monitoring ready for production APIs!")
    print("✨ Detailed metrics for optimization and analysis!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
