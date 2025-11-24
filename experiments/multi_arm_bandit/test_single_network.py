#!/usr/bin/env python3
"""
Debug test to see if a single network can answer a single question.
"""

import asyncio
import sys
import os

# Add both experiments and project root to path
experiments_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(experiments_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, experiments_dir)

from multi_arm_bandit import (
    SuperGPQADataset,
    random_network_chromosome,
    format_supergpqa_prompt,
    evaluate_supergpqa_answer,
)
from nons.core.network import NoN
import nons.operators.base  # Register operators


async def test_single_network():
    """Test if a single network can answer a single question."""
    print("=" * 60)
    print("Testing Single Network Evaluation")
    print("=" * 60)

    # Load one question
    dataset = SuperGPQADataset.load_mock_dataset(5)
    question = dataset.examples[0]

    print(f"\nQuestion: {question.question[:80]}...")
    print(f"Correct answer: {question.answer}")
    print(f"Options: {len(question.options)}")

    # Create simple network using chromosome
    print(f"\nCreating simple 2-layer network using chromosome...")
    try:
        chromosome = random_network_chromosome(min_layers=2, max_layers=2)
        from multi_arm_bandit.supergpqa_fitness import build_network_from_chromosome

        network = build_network_from_chromosome(chromosome)
        print(f"✓ Network created with {len(network.layers)} layers")
        print(f"  Architecture: {[gene.operators for gene in chromosome.genes]}")
        print(f"  Models: {[(gene.provider.value, gene.model_name) for gene in chromosome.genes]}")
    except Exception as e:
        print(f"✗ Failed to create network: {e}")
        import traceback
        traceback.print_exc()
        return

    # Format prompt
    prompt = format_supergpqa_prompt(question)
    print(f"\nPrompt length: {len(prompt)} chars")
    print(f"\nFirst 200 chars of prompt:")
    print(prompt[:200] + "...")

    # Try to run network
    print(f"\nRunning network...")
    try:
        result = await network.forward(prompt)
        print(f"\n✓ Network output: {result[:100]}")

        # Check if answer is correct
        is_correct = evaluate_supergpqa_answer(result, question.answer)
        print(f"\nCorrect: {'✓ YES' if is_correct else '✗ NO'}")
        print(f"Expected: {question.answer}")

    except Exception as e:
        print(f"\n✗ Network execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_single_network())
