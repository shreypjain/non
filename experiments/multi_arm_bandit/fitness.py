"""
Fitness evaluation functions for genetic algorithm strategies.

This module provides fitness functions to evaluate the quality of
binary-encoded arm-pulling strategies against multi-arm bandit environments.
"""

from typing import List, Callable
import numpy as np
from .encoding import decode_chromosome
from .bandit_environment import MultiArmBandit


def evaluate_strategy_fitness(
    chromosome: List[int],
    bandit: MultiArmBandit,
    num_arms: int,
    num_trials: int = 100,
    horizon: int = None,
) -> float:
    """
    Evaluate fitness of a strategy by simulating performance against bandit.

    The fitness is the average cumulative reward obtained over multiple trials.
    Multiple trials are used to account for stochastic rewards.

    Args:
        chromosome: Binary chromosome encoding the strategy
        bandit: Multi-arm bandit environment
        num_arms: Number of arms in the bandit
        num_trials: Number of simulation trials to average
        horizon: Number of time steps per trial (inferred from chromosome if None)

    Returns:
        Average cumulative reward (fitness score)
    """
    # Decode chromosome to strategy
    strategy = decode_chromosome(chromosome, num_arms)

    if not strategy:
        return 0.0  # Empty strategy has zero fitness

    # Infer horizon if not provided
    if horizon is None:
        horizon = len(strategy)

    # Run multiple trials and average rewards
    total_reward = 0.0

    for _ in range(num_trials):
        bandit.reset()
        trial_reward = 0.0

        for t in range(horizon):
            # Get arm to pull (cycle through strategy if horizon > strategy length)
            arm = strategy[t % len(strategy)]
            reward = bandit.pull(arm)
            trial_reward += reward

        total_reward += trial_reward

    # Return average reward per trial
    return total_reward / num_trials


def evaluate_regret_fitness(
    chromosome: List[int],
    bandit: MultiArmBandit,
    num_arms: int,
    num_trials: int = 100,
    horizon: int = None,
) -> float:
    """
    Evaluate fitness based on minimizing regret (negative regret as fitness).

    Regret is the difference between optimal reward and achieved reward.
    Lower regret means better strategy, so we return negative regret as fitness.

    Args:
        chromosome: Binary chromosome encoding the strategy
        bandit: Multi-arm bandit environment
        num_arms: Number of arms in the bandit
        num_trials: Number of simulation trials to average
        horizon: Number of time steps per trial

    Returns:
        Negative average regret (higher is better)
    """
    strategy = decode_chromosome(chromosome, num_arms)

    if not strategy:
        # Worst possible fitness (maximum regret)
        return -float("inf")

    if horizon is None:
        horizon = len(strategy)

    # Get optimal expected reward
    optimal_arm = bandit.get_optimal_arm()
    optimal_expected = bandit.get_expected_reward(optimal_arm)
    optimal_reward = optimal_expected * horizon

    # Run trials and measure regret
    total_regret = 0.0

    for _ in range(num_trials):
        bandit.reset()
        trial_reward = 0.0

        for t in range(horizon):
            arm = strategy[t % len(strategy)]
            reward = bandit.pull(arm)
            trial_reward += reward

        # Regret is optimal - actual
        trial_regret = optimal_reward - trial_reward
        total_regret += trial_regret

    avg_regret = total_regret / num_trials

    # Return negative regret (we want to minimize regret)
    return -avg_regret


def evaluate_diversity_fitness(
    chromosome: List[int],
    bandit: MultiArmBandit,
    num_arms: int,
    reward_weight: float = 0.8,
    diversity_weight: float = 0.2,
    num_trials: int = 100,
) -> float:
    """
    Evaluate fitness with bonus for strategy diversity (exploration).

    Combines cumulative reward with a diversity bonus to encourage
    strategies that explore multiple arms.

    Args:
        chromosome: Binary chromosome encoding the strategy
        bandit: Multi-arm bandit environment
        num_arms: Number of arms in the bandit
        reward_weight: Weight for reward component
        diversity_weight: Weight for diversity component
        num_trials: Number of simulation trials

    Returns:
        Weighted combination of reward and diversity
    """
    strategy = decode_chromosome(chromosome, num_arms)

    if not strategy:
        return 0.0

    # Calculate reward component
    total_reward = 0.0
    for _ in range(num_trials):
        bandit.reset()
        trial_reward = 0.0

        for arm in strategy:
            reward = bandit.pull(arm)
            trial_reward += reward

        total_reward += trial_reward

    avg_reward = total_reward / num_trials

    # Calculate diversity component (entropy of arm distribution)
    arm_counts = [0] * num_arms
    for arm in strategy:
        arm_counts[arm] += 1

    arm_probs = [count / len(strategy) for count in arm_counts]
    diversity = -sum(p * np.log(p + 1e-10) for p in arm_probs if p > 0)

    # Normalize diversity by maximum possible entropy
    max_diversity = np.log(num_arms)
    normalized_diversity = diversity / max_diversity if max_diversity > 0 else 0

    # Combine components
    fitness = reward_weight * avg_reward + diversity_weight * normalized_diversity

    return fitness


def create_fitness_function(
    bandit: MultiArmBandit,
    num_arms: int,
    fitness_type: str = "reward",
    num_trials: int = 100,
    horizon: int = None,
    **kwargs,
) -> Callable[[List[int]], float]:
    """
    Factory function to create a fitness evaluation function.

    Args:
        bandit: Multi-arm bandit environment
        num_arms: Number of arms
        fitness_type: Type of fitness ("reward", "regret", "diversity")
        num_trials: Number of simulation trials
        horizon: Number of time steps per trial
        **kwargs: Additional parameters for specific fitness types

    Returns:
        Fitness function that takes a chromosome and returns fitness score

    Raises:
        ValueError: If fitness_type is unknown
    """
    if fitness_type == "reward":
        return lambda chromosome: evaluate_strategy_fitness(
            chromosome, bandit, num_arms, num_trials, horizon
        )

    elif fitness_type == "regret":
        return lambda chromosome: evaluate_regret_fitness(
            chromosome, bandit, num_arms, num_trials, horizon
        )

    elif fitness_type == "diversity":
        reward_weight = kwargs.get("reward_weight", 0.8)
        diversity_weight = kwargs.get("diversity_weight", 0.2)

        return lambda chromosome: evaluate_diversity_fitness(
            chromosome,
            bandit,
            num_arms,
            reward_weight,
            diversity_weight,
            num_trials,
        )

    else:
        raise ValueError(
            f"Unknown fitness type: {fitness_type}. "
            f"Choose from: reward, regret, diversity"
        )


def batch_evaluate_fitness(
    population: List[List[int]],
    fitness_function: Callable[[List[int]], float],
) -> List[float]:
    """
    Evaluate fitness for a batch of chromosomes.

    Args:
        population: List of chromosomes to evaluate
        fitness_function: Function to compute fitness

    Returns:
        List of fitness scores corresponding to each chromosome
    """
    return [fitness_function(chromosome) for chromosome in population]


def get_population_statistics(fitness_scores: List[float]) -> dict:
    """
    Calculate statistics about population fitness.

    Args:
        fitness_scores: List of fitness values

    Returns:
        Dictionary with statistics (mean, max, min, std)
    """
    if not fitness_scores:
        return {
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "std": 0.0,
            "median": 0.0,
        }

    return {
        "mean": float(np.mean(fitness_scores)),
        "max": float(np.max(fitness_scores)),
        "min": float(np.min(fitness_scores)),
        "std": float(np.std(fitness_scores)),
        "median": float(np.median(fitness_scores)),
    }
