"""
Standard benchmark problems for multi-arm bandit genetic algorithms.

This module provides a collection of well-defined benchmark problems
for testing and comparing genetic algorithm strategies.
"""

from typing import Dict, Any
from .bandit_environment import (
    BernoulliBandit,
    GaussianBandit,
    NonStationaryBandit,
    MultiArmBandit,
)


class BenchmarkProblem:
    """
    A benchmark problem consists of a bandit environment and metadata.
    """

    def __init__(
        self,
        name: str,
        bandit: MultiArmBandit,
        description: str,
        optimal_strategy: str,
        expected_optimal_reward: float,
    ):
        """
        Initialize benchmark problem.

        Args:
            name: Identifier for the benchmark
            bandit: Multi-arm bandit environment
            description: Human-readable description
            optimal_strategy: Description of optimal strategy
            expected_optimal_reward: Expected reward of optimal strategy
        """
        self.name = name
        self.bandit = bandit
        self.description = description
        self.optimal_strategy = optimal_strategy
        self.expected_optimal_reward = expected_optimal_reward

    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark info to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "optimal_strategy": self.optimal_strategy,
            "expected_optimal_reward": self.expected_optimal_reward,
            "num_arms": self.bandit.num_arms,
        }


# Easy benchmark: Clear optimal arm
EASY_BERNOULLI = BenchmarkProblem(
    name="easy_bernoulli",
    bandit=BernoulliBandit(probabilities=[0.1, 0.3, 0.5, 0.9], seed=42),
    description="Bernoulli bandit with 4 arms and a clearly superior arm (p=0.9)",
    optimal_strategy="Always pull arm 3 (highest probability)",
    expected_optimal_reward=0.9,
)

# Medium benchmark: Close probabilities
MEDIUM_BERNOULLI = BenchmarkProblem(
    name="medium_bernoulli",
    bandit=BernoulliBandit(probabilities=[0.45, 0.48, 0.50, 0.47], seed=42),
    description="Bernoulli bandit with 4 arms having similar probabilities",
    optimal_strategy="Always pull arm 2 (p=0.50), but margin is small",
    expected_optimal_reward=0.50,
)

# Hard benchmark: Non-stationary
HARD_NON_STATIONARY = BenchmarkProblem(
    name="hard_non_stationary",
    bandit=NonStationaryBandit(
        initial_probabilities=[0.3, 0.5, 0.7, 0.4],
        change_interval=50,
        drift_amount=0.15,
        seed=42,
    ),
    description="Non-stationary bandit where arm probabilities drift over time",
    optimal_strategy="Adapt to changing best arm (requires exploration)",
    expected_optimal_reward=0.6,  # Approximate, varies with drift
)

# Gaussian benchmark: Continuous rewards
GAUSSIAN_STANDARD = BenchmarkProblem(
    name="gaussian_standard",
    bandit=GaussianBandit(
        means=[0.0, 0.5, 1.0, 1.5], stds=[1.0, 1.0, 1.0, 1.0], seed=42
    ),
    description="Gaussian bandit with 4 arms, different means, same variance",
    optimal_strategy="Always pull arm 3 (highest mean)",
    expected_optimal_reward=1.5,
)

# High variance benchmark: Noisy rewards
GAUSSIAN_HIGH_VARIANCE = BenchmarkProblem(
    name="gaussian_high_variance",
    bandit=GaussianBandit(
        means=[1.0, 1.2, 1.5, 1.3], stds=[2.0, 2.0, 2.0, 2.0], seed=42
    ),
    description="Gaussian bandit with high variance rewards (hard to distinguish)",
    optimal_strategy="Pull arm 2 (mean=1.5), but high noise makes it difficult",
    expected_optimal_reward=1.5,
)

# Many arms benchmark: Larger action space
MANY_ARMS_BERNOULLI = BenchmarkProblem(
    name="many_arms_bernoulli",
    bandit=BernoulliBandit(
        probabilities=[0.1, 0.2, 0.15, 0.3, 0.25, 0.4, 0.35, 0.5, 0.45, 0.6], seed=42
    ),
    description="Bernoulli bandit with 10 arms (larger search space)",
    optimal_strategy="Always pull arm 9 (p=0.6)",
    expected_optimal_reward=0.6,
)

# Uniform benchmark: All arms equal (exploration vs exploitation test)
UNIFORM_BERNOULLI = BenchmarkProblem(
    name="uniform_bernoulli",
    bandit=BernoulliBandit(probabilities=[0.5, 0.5, 0.5, 0.5], seed=42),
    description="All arms have equal probability (no optimal arm)",
    optimal_strategy="Any strategy works equally well",
    expected_optimal_reward=0.5,
)


# Collection of all benchmarks
ALL_BENCHMARKS = [
    EASY_BERNOULLI,
    MEDIUM_BERNOULLI,
    HARD_NON_STATIONARY,
    GAUSSIAN_STANDARD,
    GAUSSIAN_HIGH_VARIANCE,
    MANY_ARMS_BERNOULLI,
    UNIFORM_BERNOULLI,
]


def get_benchmark_by_name(name: str) -> BenchmarkProblem:
    """
    Retrieve a benchmark problem by name.

    Args:
        name: Name of the benchmark

    Returns:
        BenchmarkProblem instance

    Raises:
        ValueError: If benchmark name not found
    """
    for benchmark in ALL_BENCHMARKS:
        if benchmark.name == name:
            return benchmark

    available = [b.name for b in ALL_BENCHMARKS]
    raise ValueError(
        f"Unknown benchmark '{name}'. Available: {', '.join(available)}"
    )


def list_benchmarks() -> list[Dict[str, Any]]:
    """
    Get list of all available benchmarks with metadata.

    Returns:
        List of dictionaries with benchmark information
    """
    return [benchmark.to_dict() for benchmark in ALL_BENCHMARKS]


def create_custom_bernoulli_benchmark(
    probabilities: list[float], name: str = "custom", seed: int = None
) -> BenchmarkProblem:
    """
    Create a custom Bernoulli bandit benchmark.

    Args:
        probabilities: Success probability for each arm
        name: Name for the benchmark
        seed: Random seed

    Returns:
        BenchmarkProblem instance
    """
    bandit = BernoulliBandit(probabilities, seed=seed)
    optimal_arm = bandit.get_optimal_arm()
    optimal_prob = probabilities[optimal_arm]

    return BenchmarkProblem(
        name=name,
        bandit=bandit,
        description=f"Custom Bernoulli bandit with {len(probabilities)} arms",
        optimal_strategy=f"Always pull arm {optimal_arm} (p={optimal_prob})",
        expected_optimal_reward=optimal_prob,
    )


def create_custom_gaussian_benchmark(
    means: list[float],
    stds: list[float],
    name: str = "custom",
    seed: int = None,
) -> BenchmarkProblem:
    """
    Create a custom Gaussian bandit benchmark.

    Args:
        means: Mean reward for each arm
        stds: Standard deviation for each arm
        name: Name for the benchmark
        seed: Random seed

    Returns:
        BenchmarkProblem instance
    """
    bandit = GaussianBandit(means, stds, seed=seed)
    optimal_arm = bandit.get_optimal_arm()
    optimal_mean = means[optimal_arm]

    return BenchmarkProblem(
        name=name,
        bandit=bandit,
        description=f"Custom Gaussian bandit with {len(means)} arms",
        optimal_strategy=f"Always pull arm {optimal_arm} (mean={optimal_mean})",
        expected_optimal_reward=optimal_mean,
    )


def get_benchmark_suite(difficulty: str = "all") -> list[BenchmarkProblem]:
    """
    Get a suite of benchmarks filtered by difficulty.

    Args:
        difficulty: "easy", "medium", "hard", or "all"

    Returns:
        List of BenchmarkProblem instances

    Raises:
        ValueError: If difficulty level unknown
    """
    if difficulty == "easy":
        return [EASY_BERNOULLI, GAUSSIAN_STANDARD]
    elif difficulty == "medium":
        return [MEDIUM_BERNOULLI, GAUSSIAN_HIGH_VARIANCE]
    elif difficulty == "hard":
        return [HARD_NON_STATIONARY, MANY_ARMS_BERNOULLI]
    elif difficulty == "all":
        return ALL_BENCHMARKS
    else:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. Choose: easy, medium, hard, all"
        )
