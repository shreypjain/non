"""
Simple example demonstrating network architecture evolution.

This script shows how to use genetic algorithms to evolve
NoN network architectures for reasoning tasks.
"""

import asyncio
import sys
import os

# Add both experiments and project root to path
experiments_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(experiments_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, experiments_dir)

from multi_arm_bandit.network_evolution import (
    NetworkGeneticAlgorithm,
    NetworkGAConfig,
)
from multi_arm_bandit.reasoning_tasks import (
    ARITHMETIC_TASK,
    LOGICAL_TASK,
    PATTERN_TASK,
    GPQA_TASK,
)
from multi_arm_bandit.network_encoding import describe_network


async def main():
    """Run a simple network evolution experiment."""

    print("\n" + "=" * 60)
    print("NoN Network Architecture Evolution Demo")
    print("=" * 60)

    # Step 1: Select reasoning tasks
    print("\n1. Selecting reasoning tasks...")
    tasks = [
        ARITHMETIC_TASK,
        LOGICAL_TASK,
        PATTERN_TASK,
        GPQA_TASK,
    ]

    print(f"   Tasks: {[task.name for task in tasks]}")
    print(f"   Total examples: {sum(len(task.examples) for task in tasks)}")

    # Step 2: Configure genetic algorithm
    print("\n2. Configuring genetic algorithm...")
    config = NetworkGAConfig(
        population_size=10,  # Small for demo
        num_generations=5,  # Few generations for demo
        mutation_rate=0.2,
        crossover_rate=0.7,
        elitism_count=2,
        min_layers=2,
        max_layers=4,
    )

    print(f"   Population size: {config.population_size}")
    print(f"   Generations: {config.num_generations}")
    print(f"   Mutation rate: {config.mutation_rate}")

    # Step 3: Initialize and run evolution
    print("\n3. Initializing evolution...")

    ga = NetworkGeneticAlgorithm(
        tasks=tasks,
        config=config,
        seed=42,
    )

    print("\n4. Running evolution...")
    print("-" * 60)

    result = await ga.run(verbose=True)

    # Step 5: Analyze results
    print("\n" + "=" * 60)
    print("Results Analysis")
    print("=" * 60)

    print(f"\nBest network fitness: {result.best_fitness:.2%}")
    print(f"\nBest network architecture:")
    print(describe_network(result.best_chromosome))

    # Evolution progress
    print(f"\nEvolution progress:")
    for i, stats in enumerate(result.fitness_history):
        improvement = ""
        if i > 0:
            prev_best = result.fitness_history[i - 1]["max"]
            curr_best = stats["max"]
            if curr_best > prev_best:
                improvement = f" (+{(curr_best - prev_best):.2%})"

        print(
            f"  Gen {stats['generation']:2d}: "
            f"Best={stats['max']:.2%}, "
            f"Mean={stats['mean']:.2%}, "
            f"Std={stats['std']:.3f}"
            f"{improvement}"
        )

    # Final statistics
    print(f"\nFinal population statistics:")
    print(f"  Best fitness: {result.best_fitness:.2%}")
    print(f"  Mean fitness: {sum(result.final_fitness_scores) / len(result.final_fitness_scores):.2%}")
    print(f"  Worst fitness: {min(result.final_fitness_scores):.2%}")

    # Network structure analysis
    print(f"\nBest network details:")
    print(f"  Number of layers: {len(result.best_chromosome.genes)}")
    print(f"  Total operators: {sum(len(gene.operators) for gene in result.best_chromosome.genes)}")

    operator_counts = {}
    for gene in result.best_chromosome.genes:
        for op in gene.operators:
            operator_counts[op] = operator_counts.get(op, 0) + 1

    print(f"  Operator usage:")
    for op, count in sorted(operator_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {op}: {count}")

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
