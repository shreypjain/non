#!/usr/bin/env python3
"""
Quick test to verify RLM works end-to-end with a simple example.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from examples.rlm.rlm_network import RLMNetwork


async def main():
    """Run a simple test"""
    print("=" * 60)
    print("RLM Quick Test")
    print("=" * 60)
    print()

    # Create a small test document
    document = """
    Section 1: Introduction
    This is a test document about AI and machine learning.

    Section 2: Methods
    We used various AI techniques including neural networks.

    Section 3: Results
    The AI model achieved 95% accuracy on the test set.

    Section 4: Conclusion
    AI technology shows great promise for future applications.
    """

    # Create RLM network
    print("Creating RLM network...")
    network = RLMNetwork(
        max_iterations=3,
        confidence_threshold=0.85,
        max_llm_calls_per_execution=10
    )

    # Run a simple task
    task = "Count how many sections mention 'AI'"
    print(f"Task: {task}")
    print()

    result = await network.run(task, document)

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Final Output: {result.final_output}")
    print(f"Total Iterations: {result.total_iterations}")
    print(f"Total LLM Calls: {result.total_llm_calls}")
    print(f"Final Confidence: {result.final_confidence:.2f}")
    print(f"Stop Reason: {result.stop_reason}")
    print()

    if result.success:
        print("✓ RLM test passed!")
    else:
        print("✗ RLM test failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
