#!/usr/bin/env python3
"""
Simple Pretty Printing Demo

This demo showcases the beautiful string representations without parameter conflicts.
"""

import asyncio
import sys
import os

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nons.core.node import Node
from nons.core.layer import Layer
from nons.core.network import NoN
from nons.core.types import LayerConfig, ErrorPolicy
import nons.operators.base


async def main():
    """Showcase beautiful NoN pretty printing."""
    print("🎨 NoN PRETTY PRINTING SHOWCASE")
    print("="*60)
    print()

    # 1. Node Pretty Printing
    print("🔵 NODE DEMONSTRATION")
    print("─" * 30)
    node = Node('generate', additional_prompt_context="Be creative and detailed")
    print("Before execution:")
    print(node)
    print()

    # Execute node
    await node.execute('Write a haiku about technology')
    print("After execution:")
    print(node)
    print("\n" + "="*60 + "\n")

    # 2. Layer Pretty Printing
    print("🟡 LAYER DEMONSTRATION")
    print("─" * 30)

    nodes = [Node('generate'), Node('generate'), Node('condense')]
    config = LayerConfig(
        error_policy=ErrorPolicy.SKIP_AND_CONTINUE,
        timeout_seconds=60.0,
        min_success_threshold=0.8
    )

    layer = Layer(nodes, layer_config=config)
    print("Layer configuration:")
    print(layer)
    print()

    # Execute layer
    result = await layer.execute_parallel("Create different stories about the future")
    print("Layer execution result:")
    print(result)
    print("\n" + "="*60 + "\n")

    # 3. Network Pretty Printing
    print("🟢 NETWORK DEMONSTRATION")
    print("─" * 30)

    # Create network with simple operators
    network = NoN.from_operators([
        'generate',                    # Layer 0: Single generate
        ['generate', 'generate'],      # Layer 1: Parallel generates
        'condense',                    # Layer 2: Condense results
        ['generate', 'condense']       # Layer 3: Final parallel processing
    ])

    print("Network architecture:")
    print(network)
    print()

    # Execute network
    print("Executing forward pass...")
    result = await network.forward("Design an eco-friendly smart city")
    print()

    print("Network after execution:")
    print(network)
    print()

    print("Network execution result:")
    print(result)
    print("\n" + "="*60 + "\n")

    # 4. Summary
    print("✨ PRETTY PRINTING FEATURES")
    print("─" * 40)
    print("✅ Beautiful boxed terminal output")
    print("✅ Comprehensive execution statistics")
    print("✅ Clear network architecture visualization")
    print("✅ Real-time status indicators (✅❌⚠️)")
    print("✅ Detailed configuration display")
    print("✅ Output previews with smart truncation")
    print("✅ Layer-by-layer execution tracking")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())