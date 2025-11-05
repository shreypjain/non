#!/usr/bin/env python3
"""
Basic NoN (Network of Networks) Demo

This example demonstrates the core functionality of the NoN system:
- Creating operators, nodes, layers, and networks
- Sequential layer execution with forward pass
- Parallel node execution within layers
- Error handling with different policies
- Configuration management

Run this demo to see the NoN system in action!
"""

import asyncio
import sys
import os

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import NoN components
from nons.core.node import Node
from nons.core.layer import Layer, create_layer
from nons.core.network import NoN, create_network
from nons.core.types import LayerConfig, ErrorPolicy
import nons.operators.base  # This registers all the base operators


async def demo_basic_operators():
    """Demonstrate basic operator usage."""
    print("=== Basic Operator Demo ===")

    # Create individual nodes
    transform_node = Node("transform")
    generate_node = Node("generate")
    condense_node = Node("condense")

    print(f"Created nodes: {transform_node}, {generate_node}, {condense_node}")

    # Test individual operator execution
    result1 = await generate_node.execute("Write a short poem about AI")
    print(f"Generate result: {result1}")

    result2 = await condense_node.execute(result1)
    print(f"Condense result: {result2}")
    print()


async def demo_parallel_layer():
    """Demonstrate parallel execution within a layer."""
    print("=== Parallel Layer Demo ===")

    # Create multiple nodes for parallel execution
    nodes = [Node("generate"), Node("generate"), Node("generate")]

    # Create layer with skip_and_continue policy for graceful error handling
    config = LayerConfig(
        error_policy=ErrorPolicy.SKIP_AND_CONTINUE,
        min_success_threshold=0.7,  # Require 70% success rate
    )

    layer = Layer(nodes, layer_config=config)
    print(f"Created layer: {layer}")

    # Execute all nodes in parallel with the same input
    input_text = "Write different creative stories"
    result = await layer.execute_parallel(input_text)

    print(f"Layer execution result: {result}")
    print(f"Success rate: {result.success_rate:.2%}")
    print(f"Outputs generated: {len(result.outputs)}")
    print(f"Execution time: {result.execution_time:.3f}s")
    print()


async def demo_sequential_network():
    """Demonstrate sequential network execution."""
    print("=== Sequential Network Demo ===")

    # Create a network using the convenient factory method
    # This creates: generate -> condense -> generate pipeline
    network = NoN.from_operators(
        [
            "generate",  # Layer 1: Generate initial content
            "condense",  # Layer 2: Condense the generated content
            "generate",  # Layer 3: Generate something new based on condensed content
        ]
    )

    print(f"Created network: {network}")

    # Execute forward pass through the entire network
    initial_input = "Create an innovative app idea"
    result = await network.forward(initial_input)

    print(f"Network result: {result}")
    print(f"Final output: {result.final_output}")
    print(f"Layers executed: {result.total_layers}")
    print(f"Total nodes executed: {result.total_nodes_executed}")
    print(f"Layer success rate: {result.layer_success_rate:.2%}")
    print(f"Total execution time: {result.execution_time:.3f}s")
    print()


async def demo_complex_network():
    """Demonstrate a more complex network with parallel and sequential layers."""
    print("=== Complex Network Demo ===")

    # Create a network with both parallel and sequential execution
    # Structure:
    # Layer 1: Single generate node
    # Layer 2: Three parallel generate nodes
    # Layer 3: Single condense node
    network = NoN.from_operators(
        [
            "generate",  # Layer 1: Single node
            ["generate", "generate", "generate"],  # Layer 2: Parallel nodes
            "condense",  # Layer 3: Single node
        ]
    )

    print(f"Created complex network: {network}")

    # Execute the network
    result = await network.forward("Design a sustainable city")

    print(f"Complex network result: {result}")
    print(f"Final output: {result.final_output}")

    # Show detailed layer statistics
    stats = network.get_execution_stats()
    print("\nDetailed Network Statistics:")
    print(f"Network ID: {stats['network_id'][:8]}...")
    print(f"Total executions: {stats['execution_count']}")
    print(f"Average execution time: {stats['average_execution_time']:.3f}s")

    for layer_stat in stats["layer_stats"]:
        print(f"  Layer {layer_stat['layer_index']}: {layer_stat['node_count']} nodes")
    print()


async def demo_error_handling():
    """Demonstrate different error handling policies."""
    print("=== Error Handling Demo ===")

    # Create a layer with an operator that will fail (validate needs parameters)
    try:
        nodes = [Node("generate"), Node("validate")]  # validate will fail
        config = LayerConfig(error_policy=ErrorPolicy.SKIP_AND_CONTINUE)
        layer = Layer(nodes, layer_config=config)

        result = await layer.execute_parallel("Test input")
        print(f"Error handling result: {result}")
        print(f"Successful nodes: {result.successful_nodes}")
        print(f"Failed nodes: {result.failed_nodes}")

    except Exception as e:
        print(f"Expected error caught: {type(e).__name__}: {e}")
    print()


async def main():
    """Run all demos."""
    print("🚀 Network of Networks (NoN) System Demo")
    print("=========================================\n")

    # Import operators to register them
    print("Available operators:", list(nons.operators.registry.list_operators()))
    print()

    # Run all demonstrations
    await demo_basic_operators()
    await demo_parallel_layer()
    await demo_sequential_network()
    await demo_complex_network()
    await demo_error_handling()

    print("✅ Demo completed successfully!")
    print("\nThe NoN system provides:")
    print("• 10 base operators for any AI task composition")
    print("• Parallel execution within layers using asyncio")
    print("• Sequential execution across layers with forward pass")
    print("• Comprehensive error handling with multiple policies")
    print("• Configuration management with environment variable support")
    print("• Execution statistics and observability")


if __name__ == "__main__":
    asyncio.run(main())
