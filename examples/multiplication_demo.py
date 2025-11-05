#!/usr/bin/env python3
"""
Node Multiplication Operator Demo

This demo showcases the multiplication operator (*) for creating parallel node instances,
allowing for easy scaling and parallel execution patterns.
"""

import asyncio
import sys
import os

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import operators to ensure they're registered
import nons.operators.base

from nons.core.node import Node
from nons.core.layer import Layer, create_parallel_layer
from nons.core.network import NoN
from nons.core.types import ModelConfig, ModelProvider
from nons.core.scheduler import start_scheduler, stop_scheduler


async def demo_basic_multiplication():
    """Demonstrate basic node multiplication."""
    print("✖️  BASIC NODE MULTIPLICATION DEMO")
    print("=" * 60)

    # Create a base node
    base_node = Node(
        "generate",
        model_config=ModelConfig(
            provider=ModelProvider.MOCK, model_name="mock-model", max_tokens=30
        ),
    )

    print(f"📝 Base Node: {base_node.operator_name} ({base_node.node_id[:8]}...)")

    # Test multiplication operator
    print("\n🔢 Testing multiplication operators:")

    # Right multiplication: node * 3
    parallel_nodes_1 = base_node * 3
    print(f"  node * 3 = {len(parallel_nodes_1)} nodes")
    for i, node in enumerate(parallel_nodes_1):
        print(f"    [{i}]: {node.node_id[:8]}... ({node.operator_name})")

    # Left multiplication: 5 * node
    parallel_nodes_2 = 5 * base_node
    print(f"  5 * node = {len(parallel_nodes_2)} nodes")
    for i, node in enumerate(parallel_nodes_2):
        print(f"    [{i}]: {node.node_id[:8]}... ({node.operator_name})")

    # Verify all nodes are independent clones
    print(f"\n🔍 Verification:")
    print(
        f"  All node IDs unique: {len(set(n.node_id for n in parallel_nodes_1)) == len(parallel_nodes_1)}"
    )
    print(
        f"  Same operator name: {all(n.operator_name == base_node.operator_name for n in parallel_nodes_1)}"
    )
    print(
        f"  Same model config: {all(n.model_config.model_name == base_node.model_config.model_name for n in parallel_nodes_1)}"
    )

    print()


async def demo_layer_creation():
    """Demonstrate creating layers from multiplied nodes."""
    print("🏗️  LAYER CREATION FROM MULTIPLIED NODES")
    print("=" * 60)

    # Create base node
    base_node = Node(
        "generate",
        model_config=ModelConfig(
            provider=ModelProvider.MOCK, model_name="mock-model", max_tokens=25
        ),
    )

    # Create parallel nodes using multiplication
    parallel_nodes = base_node * 4
    print(f"📝 Created {len(parallel_nodes)} parallel nodes")

    # Create layer from multiplied nodes
    layer = create_parallel_layer(parallel_nodes)
    print(f"🏗️  Created layer with {len(layer)} nodes")
    print(f"   Layer ID: {layer.layer_id[:8]}...")

    # Execute the layer
    print(f"\n🚀 Executing parallel layer...")
    inputs = "Generate a creative response about parallel computing"

    result = await layer.execute_parallel(inputs)

    print(f"✅ Layer execution completed!")
    print(f"   Success Rate: {result.success_rate:.1%}")
    print(f"   Execution Time: {result.execution_time:.3f}s")
    print(f"   Outputs: {len(result.outputs)}")

    # Show sample outputs
    print(f"\n📋 Sample Outputs:")
    for i, output in enumerate(result.outputs[:3]):
        output_preview = (
            str(output)[:50] + "..." if len(str(output)) > 50 else str(output)
        )
        print(f"  [{i}]: {output_preview}")

    print()


async def demo_network_integration():
    """Demonstrate network creation with multiplied nodes."""
    print("🌐 NETWORK INTEGRATION WITH MULTIPLICATION")
    print("=" * 60)

    # Create nodes with multiplication
    generate_node1 = Node(
        "generate",
        model_config=ModelConfig(
            provider=ModelProvider.MOCK, model_name="mock-model", max_tokens=30
        ),
    )

    generate_node2 = Node(
        "generate",
        model_config=ModelConfig(
            provider=ModelProvider.MOCK, model_name="mock-model", max_tokens=35
        ),
    )

    # Create parallel processing layers using multiplication
    parallel_generators1 = generate_node1 * 2
    parallel_generators2 = generate_node2 * 3

    print(f"📝 Created parallel layers:")
    print(f"   Generate layer 1: {len(parallel_generators1)} nodes")
    print(f"   Generate layer 2: {len(parallel_generators2)} nodes")

    # Create network using the new from_operators enhancement
    network = NoN.from_operators(
        [
            "generate",  # Single generate node
            parallel_generators1,  # Parallel generate nodes from multiplication
            parallel_generators2,  # Parallel generate nodes from multiplication
            "generate",  # Single generate node
        ]
    )

    print(f"🏗️  Created network with {len(network.layers)} layers:")
    for i, layer in enumerate(network.layers):
        print(f"   Layer {i}: {len(layer)} nodes")

    # Execute the network
    print(f"\n🚀 Executing network with multiplied nodes...")
    input_text = "Analyze the benefits of parallel processing in AI systems"

    result = await network.forward(input_text)

    print(f"✅ Network execution completed!")
    print(f"   Total Layers: {len(network.layers)}")
    print(f"   Execution Time: {network._last_result.execution_time:.3f}s")
    print(f"   Successful Layers: {network._last_result.successful_layers}")
    print(f"   Total Nodes Executed: {network._last_result.total_nodes_executed}")

    # Show final result preview
    result_preview = str(result)[:80] + "..." if len(str(result)) > 80 else str(result)
    print(f"   Final Result: {result_preview}")

    print()


async def demo_advanced_patterns():
    """Demonstrate advanced multiplication patterns."""
    print("⚡ ADVANCED MULTIPLICATION PATTERNS")
    print("=" * 60)

    # Pattern 1: Different multiplications for different nodes
    generate_node1 = Node(
        "generate",
        model_config=ModelConfig(
            provider=ModelProvider.MOCK, model_name="mock-model", max_tokens=20
        ),
    )

    generate_node2 = Node(
        "generate",
        model_config=ModelConfig(
            provider=ModelProvider.MOCK, model_name="mock-model", max_tokens=25
        ),
    )

    # Create different numbers of parallel instances
    generators1 = generate_node1 * 2
    generators2 = generate_node2 * 4

    print(f"📝 Pattern 1 - Variable Multiplication:")
    print(f"   Generators1: {len(generators1)} nodes")
    print(f"   Generators2: {len(generators2)} nodes")

    # Pattern 2: Nested processing with multiplication
    generate_base = Node(
        "generate",
        model_config=ModelConfig(
            provider=ModelProvider.MOCK, model_name="mock-model", max_tokens=30
        ),
    )

    # Create a complex network with multiple parallel layers
    network = NoN.from_operators(
        [
            "generate",  # Input processing
            generate_base * 3,  # 3 parallel generators
            ["generate", "generate"],  # 2 different parallel operators
            generators1,  # 2 parallel generators from multiplication
            "generate",  # Final aggregation
        ]
    )

    print(f"\n🏗️  Pattern 2 - Complex Network:")
    print(f"   Total Layers: {len(network.layers)}")
    for i, layer in enumerate(network.layers):
        operator_names = [node.operator_name for node in layer.nodes]
        print(f"   Layer {i}: {operator_names}")

    # Execute the complex network
    print(f"\n🚀 Executing complex network...")
    input_text = "Create multiple perspectives on sustainable energy"

    result = await network.forward(input_text)

    print(f"✅ Complex network completed!")
    print(f"   Total Nodes Executed: {network._last_result.total_nodes_executed}")
    print(f"   Execution Time: {network._last_result.execution_time:.3f}s")

    print()


async def demo_error_handling():
    """Demonstrate error handling with multiplication."""
    print("🚨 ERROR HANDLING WITH MULTIPLICATION")
    print("=" * 60)

    # Test invalid multiplication values
    base_node = Node(
        "generate",
        model_config=ModelConfig(
            provider=ModelProvider.MOCK, model_name="mock-model", max_tokens=20
        ),
    )

    print("🔍 Testing error conditions:")

    # Test invalid multiplication count
    try:
        invalid_nodes = base_node * 0
        print("   ❌ Should have failed for count=0")
    except ValueError as e:
        print(f"   ✅ Correctly caught error for count=0: {e}")

    try:
        invalid_nodes = base_node * -1
        print("   ❌ Should have failed for count=-1")
    except ValueError as e:
        print(f"   ✅ Correctly caught error for count=-1: {e}")

    try:
        invalid_nodes = base_node * "three"
        print("   ❌ Should have failed for non-integer")
    except (TypeError, ValueError) as e:
        print(f"   ✅ Correctly caught error for non-integer: {e}")

    # Test create_parallel_layer error handling
    try:
        layer = create_parallel_layer([])
        print("   ❌ Should have failed for empty list")
    except ValueError as e:
        print(f"   ✅ Correctly caught error for empty list: {e}")

    try:
        layer = create_parallel_layer(["not", "nodes"])
        print("   ❌ Should have failed for non-Node objects")
    except TypeError as e:
        print(f"   ✅ Correctly caught error for non-Node objects: {e}")

    print()


async def demo_performance_comparison():
    """Demonstrate performance characteristics of multiplied nodes."""
    print("⚡ PERFORMANCE WITH MULTIPLICATION")
    print("=" * 60)

    base_node = Node(
        "generate",
        model_config=ModelConfig(
            provider=ModelProvider.MOCK, model_name="mock-model", max_tokens=20
        ),
    )

    # Test different scales of multiplication
    scales = [1, 2, 4, 8]
    input_text = "Quick performance test"

    print("🔍 Performance comparison across different scales:")

    for scale in scales:
        # Create nodes
        nodes = base_node * scale
        layer = create_parallel_layer(nodes)

        # Time execution
        import time

        start_time = time.time()
        result = await layer.execute_parallel(input_text)
        end_time = time.time()

        print(
            f"   Scale {scale:2d}: {result.execution_time:.3f}s "
            f"({len(result.outputs)} outputs, "
            f"{result.success_rate:.1%} success)"
        )

    print()


async def main():
    """Run comprehensive multiplication operator demonstration."""
    print("✖️  COMPREHENSIVE NODE MULTIPLICATION DEMONSTRATION")
    print("=" * 70)
    print("Showcasing parallel node creation with multiplication operators!")
    print("=" * 70)
    print()

    # Start scheduler for proper request handling
    await start_scheduler()

    await demo_basic_multiplication()
    await demo_layer_creation()
    await demo_network_integration()
    await demo_advanced_patterns()
    await demo_error_handling()
    await demo_performance_comparison()

    await stop_scheduler()

    print("🎉 MULTIPLICATION OPERATOR DEMO COMPLETED!")
    print("=" * 70)
    print("✨ Node multiplication enabling easy parallel scaling!")
    print("✨ Seamless integration with layers and networks!")
    print("✨ Flexible patterns for complex parallel processing!")
    print("✨ Robust error handling and validation!")
    print("✨ Production-ready parallel execution infrastructure!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
