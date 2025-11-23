"""
Main experiment runner for multi-arm bandit genetic algorithm.

This script runs genetic algorithm experiments on various bandit benchmarks
and provides analysis and visualization of results.
"""

import argparse
import json
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np

from .genetic_algorithm import GeneticAlgorithm, GAConfig, GAResult, run_multiple_trials
from .fitness import create_fitness_function
from .benchmarks import get_benchmark_by_name, list_benchmarks, get_benchmark_suite
from .encoding import decode_chromosome


def run_single_experiment(
    benchmark_name: str,
    population_size: int = 100,
    num_generations: int = 50,
    mutation_rate: float = 0.01,
    crossover_rate: float = 0.8,
    horizon: int = 100,
    num_trials: int = 100,
    fitness_type: str = "reward",
    seed: int = None,
) -> Dict[str, Any]:
    """
    Run a single genetic algorithm experiment on a benchmark.

    Args:
        benchmark_name: Name of the benchmark problem
        population_size: Size of GA population
        num_generations: Number of generations to evolve
        mutation_rate: Probability of bit flip mutation
        crossover_rate: Probability of crossover
        horizon: Number of time steps in strategies
        num_trials: Number of trials for fitness evaluation
        fitness_type: Type of fitness function ("reward", "regret", "diversity")
        seed: Random seed for reproducibility

    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {benchmark_name}")
    print(f"{'='*60}")

    # Get benchmark
    benchmark = get_benchmark_by_name(benchmark_name)
    print(f"\nBenchmark: {benchmark.description}")
    print(f"Optimal strategy: {benchmark.optimal_strategy}")
    print(f"Expected optimal reward: {benchmark.expected_optimal_reward}")

    # Create fitness function
    fitness_function = create_fitness_function(
        bandit=benchmark.bandit,
        num_arms=benchmark.bandit.num_arms,
        fitness_type=fitness_type,
        num_trials=num_trials,
        horizon=horizon,
    )

    # Configure GA
    config = GAConfig(
        population_size=population_size,
        num_generations=num_generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        elitism_count=2,
        tournament_size=3,
        crossover_type="single_point",
    )

    # Run GA
    print(f"\nRunning GA with:")
    print(f"  Population size: {population_size}")
    print(f"  Generations: {num_generations}")
    print(f"  Mutation rate: {mutation_rate}")
    print(f"  Crossover rate: {crossover_rate}")
    print(f"  Horizon: {horizon}")
    print(f"  Fitness type: {fitness_type}")

    ga = GeneticAlgorithm(
        num_arms=benchmark.bandit.num_arms,
        horizon=horizon,
        fitness_function=fitness_function,
        config=config,
        seed=seed,
    )

    result = ga.run()

    # Decode best strategy
    best_strategy = decode_chromosome(result.best_chromosome, benchmark.bandit.num_arms)

    # Calculate strategy statistics
    arm_counts = [0] * benchmark.bandit.num_arms
    for arm in best_strategy:
        arm_counts[arm] += 1

    print(f"\n{'='*60}")
    print("Results:")
    print(f"{'='*60}")
    print(f"Best fitness: {result.best_fitness:.4f}")
    print(f"Optimal expected reward: {benchmark.expected_optimal_reward:.4f}")
    print(f"\nBest strategy arm distribution:")
    for arm_idx, count in enumerate(arm_counts):
        percentage = (count / len(best_strategy)) * 100
        expected = benchmark.bandit.get_expected_reward(arm_idx)
        print(f"  Arm {arm_idx}: {count:3d} pulls ({percentage:5.1f}%) [E[R]={expected:.3f}]")

    return {
        "benchmark_name": benchmark_name,
        "best_fitness": result.best_fitness,
        "optimal_reward": benchmark.expected_optimal_reward,
        "fitness_history": result.fitness_history,
        "best_strategy": best_strategy,
        "arm_distribution": arm_counts,
        "config": {
            "population_size": population_size,
            "num_generations": num_generations,
            "mutation_rate": mutation_rate,
            "crossover_rate": crossover_rate,
            "horizon": horizon,
            "fitness_type": fitness_type,
        },
    }


def plot_fitness_evolution(results: Dict[str, Any], save_path: str = None):
    """
    Plot fitness evolution over generations.

    Args:
        results: Dictionary with experiment results
        save_path: Path to save plot (displays if None)
    """
    fitness_history = results["fitness_history"]

    generations = [entry["generation"] for entry in fitness_history]
    max_fitness = [entry["max"] for entry in fitness_history]
    mean_fitness = [entry["mean"] for entry in fitness_history]
    min_fitness = [entry["min"] for entry in fitness_history]

    plt.figure(figsize=(10, 6))
    plt.plot(generations, max_fitness, label="Best", linewidth=2, color="green")
    plt.plot(generations, mean_fitness, label="Mean", linewidth=2, color="blue")
    plt.plot(generations, min_fitness, label="Worst", linewidth=2, color="red")

    # Add optimal reward line
    optimal = results["optimal_reward"]
    plt.axhline(y=optimal, color="black", linestyle="--", label="Optimal")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"Fitness Evolution - {results['benchmark_name']}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_arm_distribution(results: Dict[str, Any], save_path: str = None):
    """
    Plot distribution of arm pulls in best strategy.

    Args:
        results: Dictionary with experiment results
        save_path: Path to save plot (displays if None)
    """
    arm_counts = results["arm_distribution"]
    num_arms = len(arm_counts)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(num_arms), arm_counts, color="steelblue", alpha=0.7)

    # Highlight optimal arm
    benchmark = get_benchmark_by_name(results["benchmark_name"])
    optimal_arm = benchmark.bandit.get_optimal_arm()
    bars[optimal_arm].set_color("green")
    bars[optimal_arm].set_alpha(1.0)

    plt.xlabel("Arm Index")
    plt.ylabel("Number of Pulls")
    plt.title(f"Arm Distribution in Best Strategy - {results['benchmark_name']}")
    plt.xticks(range(num_arms))
    plt.grid(True, alpha=0.3, axis="y")

    # Add legend
    plt.legend(
        [bars[optimal_arm], bars[0 if optimal_arm != 0 else 1]],
        ["Optimal Arm", "Other Arms"],
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def run_benchmark_suite(
    difficulty: str = "easy",
    num_generations: int = 50,
    population_size: int = 100,
):
    """
    Run experiments on a suite of benchmarks.

    Args:
        difficulty: Difficulty level ("easy", "medium", "hard", "all")
        num_generations: Number of generations per experiment
        population_size: Population size for GA
    """
    benchmarks = get_benchmark_suite(difficulty)

    print(f"\n{'='*60}")
    print(f"Running {difficulty.upper()} benchmark suite ({len(benchmarks)} problems)")
    print(f"{'='*60}")

    all_results = []

    for benchmark in benchmarks:
        results = run_single_experiment(
            benchmark_name=benchmark.name,
            population_size=population_size,
            num_generations=num_generations,
        )
        all_results.append(results)

    # Summary
    print(f"\n{'='*60}")
    print("SUITE SUMMARY")
    print(f"{'='*60}")

    for results in all_results:
        fitness_gap = results["optimal_reward"] - results["best_fitness"]
        gap_percent = (fitness_gap / results["optimal_reward"]) * 100 if results["optimal_reward"] != 0 else 0

        print(f"\n{results['benchmark_name']}:")
        print(f"  Best fitness: {results['best_fitness']:.4f}")
        print(f"  Optimal: {results['optimal_reward']:.4f}")
        print(f"  Gap: {fitness_gap:.4f} ({gap_percent:.1f}%)")

    return all_results


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run multi-arm bandit genetic algorithm experiments"
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default="easy_bernoulli",
        help="Benchmark name (or 'suite' to run multiple)",
    )

    parser.add_argument(
        "--suite-difficulty",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard", "all"],
        help="Difficulty level for benchmark suite",
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=100,
        help="GA population size",
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations",
    )

    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.01,
        help="Mutation rate",
    )

    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.8,
        help="Crossover rate",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=100,
        help="Strategy horizon (number of time steps)",
    )

    parser.add_argument(
        "--fitness-trials",
        type=int,
        default=100,
        help="Number of trials for fitness evaluation",
    )

    parser.add_argument(
        "--fitness-type",
        type=str,
        default="reward",
        choices=["reward", "regret", "diversity"],
        help="Type of fitness function",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )

    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List available benchmarks and exit",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots of results",
    )

    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Path to save results JSON",
    )

    args = parser.parse_args()

    # List benchmarks if requested
    if args.list_benchmarks:
        print("\nAvailable Benchmarks:")
        print("=" * 60)
        for info in list_benchmarks():
            print(f"\n{info['name']}:")
            print(f"  {info['description']}")
            print(f"  Optimal: {info['optimal_strategy']}")
        return

    # Run experiments
    if args.benchmark == "suite":
        results = run_benchmark_suite(
            difficulty=args.suite_difficulty,
            num_generations=args.generations,
            population_size=args.population_size,
        )
    else:
        results = run_single_experiment(
            benchmark_name=args.benchmark,
            population_size=args.population_size,
            num_generations=args.generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            horizon=args.horizon,
            num_trials=args.fitness_trials,
            fitness_type=args.fitness_type,
            seed=args.seed,
        )

        # Generate plots if requested
        if args.plot:
            plot_fitness_evolution(results)
            plot_arm_distribution(results)

        # Save results if requested
        if args.save_results:
            with open(args.save_results, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.save_results}")


if __name__ == "__main__":
    main()
