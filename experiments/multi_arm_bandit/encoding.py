"""
Binary encoding and decoding utilities for multi-arm bandit strategies.

This module provides functions to convert between binary chromosomes and
arm-pulling sequences, enabling genetic algorithm optimization of bandit strategies.
"""

import math
from typing import List


def bits_per_arm(num_arms: int) -> int:
    """
    Calculate the number of bits needed to represent an arm choice.

    Args:
        num_arms: Number of arms in the bandit problem

    Returns:
        Number of bits required to encode arm indices
    """
    if num_arms <= 0:
        raise ValueError("Number of arms must be positive")
    return math.ceil(math.log2(num_arms))


def chromosome_length(num_arms: int, horizon: int) -> int:
    """
    Calculate the total chromosome length for a strategy.

    Args:
        num_arms: Number of arms in the bandit problem
        horizon: Number of time steps in the strategy

    Returns:
        Total number of bits in the chromosome
    """
    return bits_per_arm(num_arms) * horizon


def encode_strategy(strategy: List[int], num_arms: int) -> List[int]:
    """
    Encode an arm-pulling strategy as a binary chromosome.

    Args:
        strategy: List of arm indices to pull at each time step
        num_arms: Number of arms in the bandit problem

    Returns:
        Binary chromosome representation (list of 0s and 1s)

    Example:
        >>> encode_strategy([0, 1, 2, 3], num_arms=4)
        [0, 0, 0, 1, 1, 0, 1, 1]  # 2 bits per arm: 00, 01, 10, 11
    """
    bits_per = bits_per_arm(num_arms)
    chromosome = []

    for arm in strategy:
        if arm < 0 or arm >= num_arms:
            raise ValueError(f"Invalid arm index {arm} for {num_arms} arms")

        # Convert arm index to binary representation
        arm_bits = [(arm >> i) & 1 for i in range(bits_per - 1, -1, -1)]
        chromosome.extend(arm_bits)

    return chromosome


def decode_chromosome(chromosome: List[int], num_arms: int) -> List[int]:
    """
    Decode a binary chromosome into an arm-pulling strategy.

    Args:
        chromosome: Binary chromosome (list of 0s and 1s)
        num_arms: Number of arms in the bandit problem

    Returns:
        List of arm indices to pull at each time step

    Example:
        >>> decode_chromosome([0, 0, 0, 1, 1, 0, 1, 1], num_arms=4)
        [0, 1, 2, 3]
    """
    bits_per = bits_per_arm(num_arms)
    strategy = []

    # Process chromosome in chunks of bits_per
    for i in range(0, len(chromosome), bits_per):
        arm_bits = chromosome[i : i + bits_per]

        # Handle incomplete chunks at the end
        if len(arm_bits) < bits_per:
            break

        # Convert binary to arm index
        arm = 0
        for bit in arm_bits:
            arm = (arm << 1) | bit

        # Map to valid arm index using modulo
        # This handles cases where binary value >= num_arms
        arm = arm % num_arms
        strategy.append(arm)

    return strategy


def random_chromosome(num_arms: int, horizon: int) -> List[int]:
    """
    Generate a random binary chromosome for a strategy.

    Args:
        num_arms: Number of arms in the bandit problem
        horizon: Number of time steps in the strategy

    Returns:
        Random binary chromosome
    """
    import random

    length = chromosome_length(num_arms, horizon)
    return [random.randint(0, 1) for _ in range(length)]


def validate_chromosome(chromosome: List[int], num_arms: int) -> bool:
    """
    Validate that a chromosome is well-formed.

    Args:
        chromosome: Binary chromosome to validate
        num_arms: Number of arms in the bandit problem

    Returns:
        True if valid, False otherwise
    """
    # Check all values are binary
    if not all(bit in [0, 1] for bit in chromosome):
        return False

    # Check length is multiple of bits_per_arm
    bits_per = bits_per_arm(num_arms)
    if len(chromosome) % bits_per != 0:
        return False

    return True


def get_horizon(chromosome: List[int], num_arms: int) -> int:
    """
    Calculate the horizon (number of time steps) from chromosome length.

    Args:
        chromosome: Binary chromosome
        num_arms: Number of arms in the bandit problem

    Returns:
        Number of time steps in the encoded strategy
    """
    bits_per = bits_per_arm(num_arms)
    return len(chromosome) // bits_per


def mutate_chromosome(chromosome: List[int], mutation_rate: float) -> List[int]:
    """
    Apply bit-flip mutation to a chromosome.

    Args:
        chromosome: Binary chromosome to mutate
        mutation_rate: Probability of flipping each bit

    Returns:
        Mutated chromosome (new list)
    """
    import random

    mutated = chromosome.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]  # Flip bit

    return mutated


def crossover_single_point(
    parent1: List[int], parent2: List[int], crossover_point: int = None
) -> tuple[List[int], List[int]]:
    """
    Perform single-point crossover between two parent chromosomes.

    Args:
        parent1: First parent chromosome
        parent2: Second parent chromosome
        crossover_point: Position to split chromosomes (random if None)

    Returns:
        Tuple of two offspring chromosomes
    """
    import random

    if len(parent1) != len(parent2):
        raise ValueError("Parents must have equal length")

    if crossover_point is None:
        crossover_point = random.randint(1, len(parent1) - 1)

    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]

    return offspring1, offspring2


def crossover_uniform(
    parent1: List[int], parent2: List[int], swap_prob: float = 0.5
) -> tuple[List[int], List[int]]:
    """
    Perform uniform crossover between two parent chromosomes.

    Args:
        parent1: First parent chromosome
        parent2: Second parent chromosome
        swap_prob: Probability of swapping each bit

    Returns:
        Tuple of two offspring chromosomes
    """
    import random

    if len(parent1) != len(parent2):
        raise ValueError("Parents must have equal length")

    offspring1 = []
    offspring2 = []

    for bit1, bit2 in zip(parent1, parent2):
        if random.random() < swap_prob:
            offspring1.append(bit2)
            offspring2.append(bit1)
        else:
            offspring1.append(bit1)
            offspring2.append(bit2)

    return offspring1, offspring2
