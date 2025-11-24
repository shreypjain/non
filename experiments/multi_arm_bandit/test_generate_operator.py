#!/usr/bin/env python3
"""
Simple test to verify the generate operator works.
"""

import asyncio
import sys
import os

# Add both experiments and project root to path
experiments_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(experiments_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, experiments_dir)

from nons.core.network import NoN
from nons.core.node import Node
from nons.core.layer import Layer
from nons.core.types import ModelConfig, ModelProvider
import nons.operators.base  # Register operators


async def test_generate():
    """Test if a simple generate node works."""
    print("=" * 60)
    print("Testing Generate Operator")
    print("=" * 60)

    # Create a simple question
    question = """Question: What is 2 + 2?

Options:
A) 3
B) 4
C) 5
D) 6

Please select the correct answer and respond with ONLY the letter (A, B, C, etc.).
Answer:"""

    # Create simple network with generate operator
    print("\nCreating network with generate operator...")
    try:
        model_config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-haiku-4.5",
            temperature=0.3,
        )

        context = "Analyze the question carefully and select the correct answer. Respond with ONLY the letter."

        node = Node(
            operator_name="generate",
            model_config=model_config,
            additional_prompt_context=context,
        )

        layer = Layer([node])
        network = NoN(layers=[layer])

        print(f"✓ Network created with {len(network.layers)} layers")

    except Exception as e:
        print(f"✗ Failed to create network: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test network execution
    print(f"\nRunning network on question...")
    print(f"Question: {question[:60]}...")

    try:
        result = await network.forward(question)
        print(f"\n✓ Network executed successfully!")
        print(f"Output: {result.final_output}")

    except Exception as e:
        print(f"\n✗ Network execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_generate())
