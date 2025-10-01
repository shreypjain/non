#!/usr/bin/env python3
"""
Document Synthesis Pipeline Demo

This example demonstrates a complex pipeline where:
1. We have 10 random documents
2. Split them into 3 groups (3, 3, 4 documents each)
3. Each group picks the most relevant document to a query
4. Final comparison picks the best of the 3 winners
"""

import asyncio
import sys
import os

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import operators to ensure they're registered
import nons.operators.base

from nons.core.node import Node
from nons.core.layer import Layer
from nons.core.network import NoN
from nons.core.types import ModelConfig, ModelProvider
from nons.core.scheduler import start_scheduler, stop_scheduler


# Sample random documents about various topics
DOCUMENTS = [
    "Artificial intelligence is transforming healthcare by enabling faster diagnosis and personalized treatment plans. Machine learning algorithms can analyze medical images with high accuracy.",
    "Climate change is causing rising sea levels and extreme weather events. Scientists warn that urgent action is needed to reduce carbon emissions and protect vulnerable ecosystems.",
    "Quantum computing promises to revolutionize cryptography and drug discovery. These computers use quantum bits that can exist in multiple states simultaneously.",
    "The Mediterranean diet, rich in olive oil, fish, and vegetables, has been linked to numerous health benefits including reduced risk of heart disease and improved cognitive function.",
    "Blockchain technology enables decentralized digital transactions without intermediaries. Smart contracts can automatically execute when predefined conditions are met.",
    "Space exploration has led to many technological innovations including GPS, memory foam, and water purification systems. Private companies are now competing in the space race.",
    "Renewable energy sources like solar and wind power are becoming more cost-effective. Energy storage technology is key to making these sources reliable for grid-scale deployment.",
    "The human microbiome contains trillions of bacteria that influence digestion, immunity, and even mental health. Probiotic research is uncovering new therapeutic possibilities.",
    "Virtual reality technology is being used for training simulations in fields like surgery, aviation, and military operations. VR therapy shows promise for treating PTSD and phobias.",
    "Ancient civilizations like the Maya and Egyptians developed sophisticated astronomical knowledge. Their calendars and monuments show remarkable precision in tracking celestial events."
]

RELEVANCY_QUERY = "Which document is most relevant to healthcare and medical technology?"


async def demo_document_synthesis_pipeline():
    """Demonstrate the document synthesis and comparison pipeline."""
    print("=" * 80)
    print("DOCUMENT SYNTHESIS AND RELEVANCY PIPELINE DEMO")
    print("=" * 80)
    print()

    print("📚 Starting with 10 random documents:")
    for i, doc in enumerate(DOCUMENTS, 1):
        preview = doc[:60] + "..." if len(doc) > 60 else doc
        print(f"  [{i}]: {preview}")
    print()

    print(f"🎯 Query: {RELEVANCY_QUERY}")
    print()

    # Create nodes for the pipeline
    # We'll use compare operator which can analyze multiple documents
    compare_node = Node('compare', model_config=ModelConfig(
        provider=ModelProvider.MOCK,
        model_name="mock-model",
        max_tokens=150
    ))

    # Create 3 parallel comparison nodes for groups (3, 3, 4)
    compare_group1 = compare_node.clone()  # Will handle docs 0-2
    compare_group2 = compare_node.clone()  # Will handle docs 3-5
    compare_group3 = compare_node.clone()  # Will handle docs 6-9

    # Create final comparison node
    final_compare = compare_node.clone()

    print("🏗️  Building pipeline:")
    print("  Layer 1: 3 parallel comparison nodes (groups of 3, 3, 4 documents)")
    print("  Layer 2: 1 final comparison node (best of 3)")
    print()

    # Create layers manually to have control over input distribution
    layer1 = Layer([compare_group1, compare_group2, compare_group3])
    layer2 = Layer([final_compare])

    print("🚀 Executing Layer 1: Comparing documents in 3 groups...")
    print()

    # Prepare inputs for layer 1 - split documents into 3 groups
    group1_docs = DOCUMENTS[0:3]
    group2_docs = DOCUMENTS[3:6]
    group3_docs = DOCUMENTS[6:10]

    print(f"  Group 1 (3 docs): Documents 1-3")
    print(f"  Group 2 (3 docs): Documents 4-6")
    print(f"  Group 3 (4 docs): Documents 7-10")
    print()

    # Execute layer 1 with different inputs for each node
    # We need to format the comparison inputs
    inputs_layer1 = [
        f"Query: {RELEVANCY_QUERY}\n\nCompare these documents and select the most relevant:\n\nDoc A: {group1_docs[0]}\n\nDoc B: {group1_docs[1]}\n\nDoc C: {group1_docs[2]}",
        f"Query: {RELEVANCY_QUERY}\n\nCompare these documents and select the most relevant:\n\nDoc A: {group2_docs[0]}\n\nDoc B: {group2_docs[1]}\n\nDoc C: {group2_docs[2]}",
        f"Query: {RELEVANCY_QUERY}\n\nCompare these documents and select the most relevant:\n\nDoc A: {group3_docs[0]}\n\nDoc B: {group3_docs[1]}\n\nDoc C: {group3_docs[2]}\n\nDoc D: {group3_docs[3]}"
    ]

    layer1_result = await layer1.execute_parallel(inputs_layer1)

    print(f"✅ Layer 1 completed!")
    print(f"  Success Rate: {layer1_result.success_rate:.1%}")
    print(f"  Execution Time: {layer1_result.execution_time:.3f}s")
    print()

    print("📋 Layer 1 Results (winners from each group):")
    for i, output in enumerate(layer1_result.outputs, 1):
        preview = str(output)[:80] + "..." if len(str(output)) > 80 else str(output)
        print(f"  Group {i} winner: {preview}")
    print()

    print("🚀 Executing Layer 2: Final comparison of 3 winners...")
    print()

    # Prepare input for layer 2 - compare the 3 winners
    final_input = f"Query: {RELEVANCY_QUERY}\n\nCompare these three candidates and select the single most relevant:\n\nCandidate 1 (from Group 1): {layer1_result.outputs[0]}\n\nCandidate 2 (from Group 2): {layer1_result.outputs[1]}\n\nCandidate 3 (from Group 3): {layer1_result.outputs[2]}"

    layer2_result = await layer2.execute_parallel(final_input)

    print(f"✅ Layer 2 completed!")
    print(f"  Success Rate: {layer2_result.success_rate:.1%}")
    print(f"  Execution Time: {layer2_result.execution_time:.3f}s")
    print()

    print("=" * 80)
    print("🏆 FINAL RESULT:")
    print("=" * 80)
    final_output = layer2_result.outputs[0]
    print(f"{final_output}")
    print("=" * 80)
    print()

    # Show pipeline statistics
    print("📊 Pipeline Statistics:")
    print(f"  Total documents processed: {len(DOCUMENTS)}")
    print(f"  Groups in Layer 1: 3 (sizes: 3, 3, 4)")
    print(f"  Layer 1 execution time: {layer1_result.execution_time:.3f}s")
    print(f"  Layer 2 execution time: {layer2_result.execution_time:.3f}s")
    print(f"  Total pipeline time: {layer1_result.execution_time + layer2_result.execution_time:.3f}s")
    print()


async def demo_alternative_with_network():
    """Alternative demo using NoN network structure."""
    print("=" * 80)
    print("ALTERNATIVE: USING NoN NETWORK STRUCTURE")
    print("=" * 80)
    print()

    print("🏗️  Building network with multiplication operator...")

    # Create a compare node and multiply for parallel execution
    compare_node = Node('compare', model_config=ModelConfig(
        provider=ModelProvider.MOCK,
        model_name="mock-model",
        max_tokens=150
    ))

    # Create network: 3 parallel comparisons -> 1 final comparison
    network = NoN.from_operators([
        compare_node * 3,  # Layer 1: 3 parallel compare nodes
        'compare'          # Layer 2: 1 final compare node
    ])

    print(f"✅ Network created with {len(network.layers)} layers:")
    for i, layer in enumerate(network.layers):
        print(f"  Layer {i}: {len(layer)} nodes")
    print()

    print("🚀 Executing network (simplified version)...")
    print("  Note: Network will broadcast same input to all nodes in Layer 1")
    print("  For true 3-3-4 split, use manual Layer execution (see above)")
    print()

    # This will broadcast the same input to all 3 nodes
    # For demonstration purposes
    sample_input = f"Query: {RELEVANCY_QUERY}\n\nAnalyze these sample documents: {DOCUMENTS[0]}, {DOCUMENTS[1]}, {DOCUMENTS[2]}"

    result = await network.forward(sample_input)

    print(f"✅ Network execution completed!")
    print(f"  Total Layers: {len(network.layers)}")
    print(f"  Execution Time: {result.execution_time:.3f}s")
    print(f"  Success Rate: {result.layer_success_rate:.1%}")
    print()

    print("📋 Final Result:")
    final_preview = str(result.final_output)[:100] + "..." if len(str(result.final_output)) > 100 else str(result.final_output)
    print(f"  {final_preview}")
    print()


async def main():
    """Run all demonstrations."""
    print()
    print("🚀 DOCUMENT SYNTHESIS PIPELINE DEMONSTRATION")
    print("=" * 80)
    print()

    # Start scheduler for proper request handling
    await start_scheduler()

    # Run the main pipeline demo
    await demo_document_synthesis_pipeline()

    # Run alternative network-based demo
    await demo_alternative_with_network()

    await stop_scheduler()

    print("✅ Demo completed successfully!")
    print()
    print("💡 Key Takeaways:")
    print("  • Manual Layer execution gives fine-grained control over input distribution")
    print("  • Can split documents into uneven groups (3, 3, 4)")
    print("  • Parallel comparison nodes process groups independently")
    print("  • Final comparison selects best from group winners")
    print("  • Network structure provides high-level pipeline abstraction")
    print()


if __name__ == "__main__":
    asyncio.run(main())