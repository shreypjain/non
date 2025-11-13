#!/usr/bin/env python3
"""
Minimal NoN Demo with Anthropic

This is a minimal example to demonstrate NoN working end-to-end.
"""

import asyncio
import sys
import os

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nons.core.node import Node
from nons.core.network import NoN
from nons.core.types import ModelConfig, ModelProvider
import nons.operators.base


async def main():
    """Run a minimal NoN example."""
    print("=" * 60)
    print("MINIMAL NON DEMO WITH ANTHROPIC")
    print("=" * 60)
    print()

    # Create a simple node with Anthropic
    print("1. Creating a generate node with Anthropic Claude...")
    node = Node(
        "generate",
        model_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-sonnet-4-5-20250929",
            max_tokens=100,
            temperature=0.7,
        ),
        additional_prompt_context="Be concise and creative.",
    )
    print(f"Node created: {node}")
    print()

    # Execute the node
    print("2. Executing node with a simple prompt...")
    result = await node.execute("Write a short haiku about AI")
    print(f"Result: {result}")
    print()

    # Create a simple network
    print("3. Creating a simple 2-layer network...")
    network = NoN.from_operators(
        ["generate", "condense"],
        model_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-sonnet-4-5-20250929",
            max_tokens=150,
        ),
    )
    print(f"Network created with {len(network.layers)} layers")
    print()

    # Execute network forward pass
    print("4. Running network forward pass...")
    network_result = await network.forward("Explain quantum computing")
    print()
    print("Network execution complete!")
    print(f"Final output: {network_result.final_output}")
    print(f"Execution time: {network_result.execution_time:.3f}s")
    print(f"Total layers: {network_result.total_layers}")
    print(f"Success rate: {network_result.layer_success_rate:.2%}")
    print()

    # Create a more complex network with parallel nodes
    print("5. Creating a network with parallel nodes...")
    generator = Node(
        "generate",
        model_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-sonnet-4-5-20250929",
            max_tokens=80,
        ),
    )

    # Use node multiplication to create parallel nodes
    parallel_generators = generator * 3

    network2 = NoN.from_operators(
        ["transform", parallel_generators, "condense"],
        model_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-sonnet-4-5-20250929",
            max_tokens=150,
        ),
    )
    print(f"Complex network created with {len(network2.layers)} layers")
    print()

    # Execute complex network
    print("6. Running complex network with parallel execution...")
    complex_result = await network2.forward("Write three different story ideas")
    print()
    print("Complex network execution complete!")
    print(f"Final output preview: {complex_result.final_output[:200]}...")
    print(f"Execution time: {complex_result.execution_time:.3f}s")
    print()

    print("=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("Key features demonstrated:")
    print("  - Single node execution")
    print("  - Sequential network execution")
    print("  - Parallel node execution with multiplication operator")
    print("  - Multiple operators (transform, generate, condense)")
    print("  - Full observability with execution statistics")


if __name__ == "__main__":
    asyncio.run(main())
