"""
Network architecture evolution using SuperGPQA fitness.

This script demonstrates evolving NoN networks to maximize accuracy
on the SuperGPQA graduate-level reasoning benchmark.
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
from multi_arm_bandit.network_encoding import describe_network
from multi_arm_bandit.supergpqa_loader import SuperGPQADataset
from multi_arm_bandit.supergpqa_fitness import (
    batch_evaluate_supergpqa_fitness,
    evaluate_network_on_supergpqa,
)


class SuperGPQANetworkGA(NetworkGeneticAlgorithm):
    """
    Genetic algorithm for evolving networks on SuperGPQA.

    Extends the base NetworkGeneticAlgorithm to use SuperGPQA
    as the fitness function.
    """

    def __init__(
        self,
        dataset: SuperGPQADataset,
        num_questions: int = 20,
        config: NetworkGAConfig = None,
        seed: int = None,
    ):
        """
        Initialize SuperGPQA network evolution.

        Args:
            dataset: SuperGPQA dataset
            num_questions: Number of questions per fitness evaluation
            config: GA configuration
            seed: Random seed
        """
        # Don't call super().__init__ yet since we need different setup
        self.dataset = dataset
        self.num_questions = num_questions
        self.tasks = []  # Compatibility with parent class (we use dataset instead)
        self.config = config if config is not None else NetworkGAConfig()
        self.seed = seed

        if seed is not None:
            import random
            import numpy as np

            random.seed(seed)
            np.random.seed(seed)

        # Initialize population
        from multi_arm_bandit.network_encoding import random_network_chromosome

        self.population = [
            random_network_chromosome(
                min_layers=self.config.min_layers,
                max_layers=self.config.max_layers,
            )
            for _ in range(self.config.population_size)
        ]

        self.fitness_scores = []
        self.generation = 0
        self.fitness_history = []

    async def evaluate_population(self, verbose: bool = False):
        """Evaluate population on SuperGPQA questions."""
        self.fitness_scores = await batch_evaluate_supergpqa_fitness(
            self.population,
            self.dataset,
            num_questions=self.num_questions,
            verbose=verbose,
        )


async def main():
    """Run SuperGPQA network evolution experiment."""

    print("\n" + "=" * 60)
    print("Network Evolution on SuperGPQA Benchmark")
    print("=" * 60)

    # Step 1: Load SuperGPQA dataset
    print("\n1. Loading SuperGPQA dataset...")
    dataset = SuperGPQADataset.load_mock_dataset(num_examples=50)

    print(f"   Dataset size: {len(dataset)} questions")
    subjects = set(ex.subject for ex in dataset.examples)
    print(f"   Subjects: {', '.join(sorted(subjects))}")

    # Show example question
    print(f"\n   Example question:")
    example = dataset.examples[0]
    print(f"   Subject: {example.subject}")
    print(f"   Question: {example.question[:100]}...")
    print(f"   Options: {len(example.options)} choices")
    print(f"   Correct: {example.answer}")

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
    print(f"   Questions per evaluation: 20")

    # Step 3: Run evolution
    print("\n3. Running evolution...")
    print("-" * 60)

    ga = SuperGPQANetworkGA(
        dataset=dataset,
        num_questions=20,
        config=config,
        seed=42,
    )

    result = await ga.run(verbose=True)

    # Step 4: Detailed evaluation of best network
    print("\n" + "=" * 60)
    print("Detailed Evaluation of Best Network")
    print("=" * 60)

    print(f"\nBest Network Architecture:")
    print(describe_network(result.best_chromosome))

    # Evaluate on larger test set
    print(f"\nEvaluating on 50 test questions...")
    test_questions = dataset.sample(50, seed=100)

    detailed_result = await evaluate_network_on_supergpqa(
        result.best_chromosome, test_questions, verbose=True
    )

    # Step 5: Analysis
    print("\n" + "=" * 60)
    print("Evolution Analysis")
    print("=" * 60)

    print(f"\nFinal Performance:")
    print(f"  Test Accuracy: {detailed_result['accuracy']:.2%}")
    print(f"  Correct: {detailed_result['num_correct']}/{detailed_result['num_total']}")

    print(f"\nEvolution Progress:")
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
            f"Mean={stats['mean']:.2%}"
            f"{improvement}"
        )

    # Best network structure
    print(f"\nBest Network Structure:")
    print(f"  Layers: {len(result.best_chromosome.genes)}")

    for i, gene in enumerate(result.best_chromosome.genes):
        ops_str = ", ".join(gene.operators)
        print(f"  Layer {i+1}: {ops_str}")
        print(f"    Model: {gene.provider.value}/{gene.model_name}")
        print(f"    Parallel: {gene.parallel}")

    # Subject performance
    if "subject_accuracy" in detailed_result:
        print(f"\nPerformance by Subject:")
        for subject, metrics in detailed_result["subject_accuracy"].items():
            print(
                f"  {subject}: {metrics['accuracy']:.2%} "
                f"({metrics['correct']}/{metrics['total']})"
            )

    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)

    print(f"\nKey Findings:")
    print(f"  - Evolved networks achieve {result.best_fitness:.2%} accuracy on SuperGPQA")
    print(f"  - Best architecture has {len(result.best_chromosome.genes)} layers")
    print(f"  - Evolution improved fitness by {(result.best_fitness - result.fitness_history[0]['mean']):.2%}")


if __name__ == "__main__":
    asyncio.run(main())
