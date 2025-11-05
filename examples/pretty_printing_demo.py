#!/usr/bin/env python3
"""
Pretty Printing Demo for NoN Components

This demo showcases the beautiful string representations of all NoN components
including Nodes, Layers, Networks, and execution results with detailed outputs.
"""

import asyncio
import sys
import os

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nons.core.node import Node
from nons.core.layer import Layer
from nons.core.network import NoN
from nons.core.types import LayerConfig, ErrorPolicy
import nons.operators.base


async def demo_node_printing():
    """Demonstrate Node pretty printing."""
    print("🔵 NODE PRETTY PRINTING")
    print("=" * 50)

    # Create a node
    node = Node("generate", additional_prompt_context="Be creative and inspirational")

    print("Before execution:")
    print(node)
    print()

    # Execute the node to show stats
    await node.execute("Write a motivational quote")

    print("After execution:")
    print(node)
    print("\n" + "=" * 50 + "\n")


async def demo_layer_printing():
    """Demonstrate Layer pretty printing."""
    print("🟡 LAYER PRETTY PRINTING")
    print("=" * 50)

    # Create a layer with multiple nodes
    nodes = [Node("generate"), Node("condense"), Node("generate")]

    config = LayerConfig(
        error_policy=ErrorPolicy.SKIP_AND_CONTINUE,
        timeout_seconds=45.0,
        min_success_threshold=0.8,
    )

    layer = Layer(nodes, layer_config=config)

    print("Before execution:")
    print(layer)
    print()

    # Execute the layer
    result = await layer.execute_parallel("Create innovative solutions")

    print("After execution:")
    print(layer)
    print()

    print("Layer Result:")
    print(result)
    print("\n" + "=" * 50 + "\n")


async def demo_network_printing():
    """Demonstrate Network pretty printing."""
    print("🟢 NETWORK PRETTY PRINTING")
    print("=" * 50)

    # Create a complex network
    network = NoN.from_operators(
        [
            "generate",  # Single operator layer
            ["generate", "condense"],  # Parallel operators layer
            "condense",  # Single operator layer
            ["generate", "generate", "generate"],  # Parallel same operators layer
        ]
    )

    print("Before execution:")
    print(network)
    print()

    # Execute the network
    result = await network.forward("Design the future of sustainable transportation")

    print("After execution:")
    print(network)
    print()

    print("Network Result:")
    print(result)
    print("\n" + "=" * 50 + "\n")


async def demo_comprehensive_workflow():
    """Demonstrate a comprehensive workflow with pretty printing at each step."""
    print("🚀 COMPREHENSIVE WORKFLOW DEMO")
    print("=" * 50)

    print("Step 1: Creating Network Architecture")
    print("-" * 35)

    # Build a sophisticated network step by step
    network = NoN.from_operators(
        [
            "generate",  # Content generation
            ["expand", "condense"],  # Parallel processing
            "synthesize",  # Combining results
            ["classify", "extract"],  # Analysis
            "generate",  # Final output
        ]
    )

    print(network)
    print()

    print("Step 2: Executing Forward Pass")
    print("-" * 35)

    # Execute with progress tracking
    input_prompt = "Create a comprehensive business plan for a sustainable tech startup"

    print(f"Input: {input_prompt}")
    print("\nExecuting network...")

    result = await network.forward(input_prompt)

    print("\nStep 3: Execution Complete!")
    print("-" * 35)
    print(result)
    print()

    print("Step 4: Network State After Execution")
    print("-" * 35)
    print(network)
    print()

    print("Step 5: Individual Layer Analysis")
    print("-" * 35)

    for i, layer_result in enumerate(result.layer_results):
        print(f"Layer {i} Result:")
        print(layer_result)
        print()

    print("✅ Comprehensive workflow completed successfully!")
    print("\n" + "=" * 50 + "\n")


async def demo_error_handling_printing():
    """Demonstrate pretty printing with error scenarios."""
    print("⚠️  ERROR HANDLING PRETTY PRINTING")
    print("=" * 50)

    try:
        # Create a network that will have mixed success/failure
        network = NoN.from_operators(
            ["generate", "validate"]  # This will fail due to missing parameters
        )

        result = await network.forward("Test input")

    except Exception as e:
        print("Expected Error Caught:")
        print(f"❌ {type(e).__name__}: {e}")
        print()

        # Show the network state even after failure
        print("Network state after error:")
        print(network)
        print()

    print("✅ Error handling demonstration complete!")
    print("\n" + "=" * 50 + "\n")


async def main():
    """Run all pretty printing demonstrations."""
    print("🎨 NoN PRETTY PRINTING SHOWCASE")
    print("=" * 50)
    print("This demo showcases beautiful terminal output for all NoN components!")
    print("=" * 50)
    print()

    await demo_node_printing()
    await demo_layer_printing()
    await demo_network_printing()
    await demo_comprehensive_workflow()
    await demo_error_handling_printing()

    print("🎉 ALL DEMOS COMPLETED!")
    print("=" * 50)
    print("✨ Beautiful terminal output for debugging and monitoring!")
    print("✨ Clear visualization of network architecture and execution!")
    print("✨ Comprehensive statistics and result formatting!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
