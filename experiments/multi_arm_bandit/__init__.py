"""
Multi-arm bandit genetic algorithm experiment.

This package provides a complete implementation of genetic algorithms
for solving multi-arm bandit problems, including:

- Binary encoding of strategies
- Fitness evaluation functions
- Genetic operators (selection, crossover, mutation)
- Multi-arm bandit environments
- Standard benchmark problems
"""

from .encoding import (
    encode_strategy,
    decode_chromosome,
    random_chromosome,
    bits_per_arm,
    chromosome_length,
)

from .bandit_environment import (
    MultiArmBandit,
    BernoulliBandit,
    GaussianBandit,
    NonStationaryBandit,
    ContextualBandit,
)

from .fitness import (
    evaluate_strategy_fitness,
    evaluate_regret_fitness,
    evaluate_diversity_fitness,
    create_fitness_function,
)

from .genetic_algorithm import (
    GeneticAlgorithm,
    GAConfig,
    GAResult,
    run_multiple_trials,
)

from .benchmarks import (
    BenchmarkProblem,
    get_benchmark_by_name,
    list_benchmarks,
    get_benchmark_suite,
    ALL_BENCHMARKS,
)

__all__ = [
    # Encoding
    "encode_strategy",
    "decode_chromosome",
    "random_chromosome",
    "bits_per_arm",
    "chromosome_length",
    # Bandit environments
    "MultiArmBandit",
    "BernoulliBandit",
    "GaussianBandit",
    "NonStationaryBandit",
    "ContextualBandit",
    # Fitness
    "evaluate_strategy_fitness",
    "evaluate_regret_fitness",
    "evaluate_diversity_fitness",
    "create_fitness_function",
    # Genetic algorithm
    "GeneticAlgorithm",
    "GAConfig",
    "GAResult",
    "run_multiple_trials",
    # Benchmarks
    "BenchmarkProblem",
    "get_benchmark_by_name",
    "list_benchmarks",
    "get_benchmark_suite",
    "ALL_BENCHMARKS",
]
