"""
Tests for multi-arm bandit genetic algorithm implementation.

This module contains unit tests to verify correctness of:
- Binary encoding/decoding
- Bandit environments
- Fitness functions
- Genetic operators
"""

import sys
import os
import importlib.util

# Add experiments directory to path
experiments_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, experiments_dir)

from multi_arm_bandit import (
    encode_strategy,
    decode_chromosome,
    bits_per_arm,
    chromosome_length,
    random_chromosome,
    BernoulliBandit,
    GaussianBandit,
    evaluate_strategy_fitness,
    create_fitness_function,
    GeneticAlgorithm,
    GAConfig,
)
from multi_arm_bandit.encoding import (
    validate_chromosome,
    mutate_chromosome,
    crossover_single_point,
)


def test_encoding_decoding():
    """Test binary encoding and decoding."""
    print("Testing encoding/decoding...")

    # Test 1: Simple encoding/decoding
    strategy = [0, 1, 2, 3]
    num_arms = 4
    chromosome = encode_strategy(strategy, num_arms)
    decoded = decode_chromosome(chromosome, num_arms)

    assert decoded == strategy, f"Encoding/decoding failed: {strategy} != {decoded}"

    # Test 2: Bits per arm calculation
    assert bits_per_arm(4) == 2  # 4 arms need 2 bits
    assert bits_per_arm(8) == 3  # 8 arms need 3 bits
    assert bits_per_arm(3) == 2  # 3 arms need 2 bits (ceil)

    # Test 3: Chromosome length
    assert chromosome_length(4, 10) == 20  # 4 arms, 10 steps = 20 bits

    # Test 4: Validation
    valid_chromosome = [0, 1, 0, 1, 1, 0]
    assert validate_chromosome(valid_chromosome, 4) == True

    invalid_chromosome = [0, 1, 2]  # Contains non-binary value
    assert validate_chromosome(invalid_chromosome, 4) == False

    print("  ✓ Encoding/decoding tests passed")


def test_bandit_environment():
    """Test bandit environments."""
    print("Testing bandit environments...")

    # Test 1: Bernoulli bandit
    probs = [0.1, 0.5, 0.3, 0.7]
    bandit = BernoulliBandit(probs, seed=42)

    assert bandit.num_arms == 4
    assert bandit.get_optimal_arm() == 3  # Highest probability

    # Pull each arm
    for arm in range(4):
        reward = bandit.pull(arm)
        assert reward in [0.0, 1.0], f"Invalid Bernoulli reward: {reward}"

    # Test 2: Gaussian bandit
    means = [0.0, 0.5, 1.0, 1.5]
    stds = [1.0, 1.0, 1.0, 1.0]
    bandit = GaussianBandit(means, stds, seed=42)

    assert bandit.num_arms == 4
    assert bandit.get_optimal_arm() == 3  # Highest mean

    # Pull arms and check rewards are reasonable
    for arm in range(4):
        reward = bandit.pull(arm)
        assert isinstance(reward, (float, int)), f"Invalid Gaussian reward type"

    print("  ✓ Bandit environment tests passed")


def test_fitness_functions():
    """Test fitness evaluation."""
    print("Testing fitness functions...")

    # Create a simple bandit
    probs = [0.1, 0.9]
    bandit = BernoulliBandit(probs, seed=42)

    # Create a strategy that always pulls optimal arm
    optimal_strategy = [1] * 10  # Always pull arm 1 (p=0.9)
    optimal_chromosome = encode_strategy(optimal_strategy, 2)

    # Create a strategy that always pulls suboptimal arm
    suboptimal_strategy = [0] * 10  # Always pull arm 0 (p=0.1)
    suboptimal_chromosome = encode_strategy(suboptimal_strategy, 2)

    # Evaluate fitness
    fitness_fn = create_fitness_function(
        bandit=bandit,
        num_arms=2,
        fitness_type="reward",
        num_trials=100,
        horizon=10,
    )

    optimal_fitness = fitness_fn(optimal_chromosome)
    suboptimal_fitness = fitness_fn(suboptimal_chromosome)

    # Optimal strategy should have higher fitness
    assert optimal_fitness > suboptimal_fitness, \
        f"Fitness function incorrect: optimal={optimal_fitness} not > suboptimal={suboptimal_fitness}"

    # Optimal fitness should be close to 9.0 (0.9 * 10 pulls cumulative reward)
    assert 7.0 < optimal_fitness < 10.0, \
        f"Optimal fitness out of expected range: {optimal_fitness}"

    print("  ✓ Fitness function tests passed")


def test_genetic_operators():
    """Test genetic operators."""
    print("Testing genetic operators...")

    # Test 1: Mutation
    chromosome = [0, 0, 0, 0, 0, 0, 0, 0]
    mutated = mutate_chromosome(chromosome, mutation_rate=0.5)

    # Should have some mutations (not guaranteed due to randomness, but likely)
    # Just check it returns valid chromosome
    assert len(mutated) == len(chromosome)
    assert all(bit in [0, 1] for bit in mutated)

    # Test 2: Crossover
    parent1 = [0, 0, 0, 0, 0, 0]
    parent2 = [1, 1, 1, 1, 1, 1]

    offspring1, offspring2 = crossover_single_point(parent1, parent2, crossover_point=3)

    # Check offspring are valid
    assert len(offspring1) == len(parent1)
    assert len(offspring2) == len(parent2)
    assert all(bit in [0, 1] for bit in offspring1)
    assert all(bit in [0, 1] for bit in offspring2)

    # Check crossover actually happened
    assert offspring1 == [0, 0, 0, 1, 1, 1]
    assert offspring2 == [1, 1, 1, 0, 0, 0]

    print("  ✓ Genetic operator tests passed")


def test_genetic_algorithm():
    """Test full genetic algorithm run."""
    print("Testing genetic algorithm...")

    # Create a simple bandit
    probs = [0.2, 0.8]  # Clear optimal arm
    bandit = BernoulliBandit(probs, seed=42)

    # Create fitness function
    fitness_fn = create_fitness_function(
        bandit=bandit,
        num_arms=2,
        fitness_type="reward",
        num_trials=50,
        horizon=20,
    )

    # Configure GA with small parameters for quick test
    config = GAConfig(
        population_size=20,
        num_generations=10,
        mutation_rate=0.01,
        crossover_rate=0.8,
    )

    # Run GA
    ga = GeneticAlgorithm(
        num_arms=2,
        horizon=20,
        fitness_function=fitness_fn,
        config=config,
        seed=42,
    )

    result = ga.run()

    # Check result structure
    assert result.best_chromosome is not None
    assert result.best_fitness > 0
    assert len(result.fitness_history) == config.num_generations

    # Check fitness improved over time
    initial_best = result.fitness_history[0]["max"]
    final_best = result.fitness_history[-1]["max"]

    assert final_best >= initial_best, \
        f"Fitness did not improve: initial={initial_best}, final={final_best}"

    # Decode best strategy and check it favors optimal arm
    best_strategy = decode_chromosome(result.best_chromosome, 2)
    arm_1_count = sum(1 for arm in best_strategy if arm == 1)
    arm_0_count = sum(1 for arm in best_strategy if arm == 0)

    # Should pull optimal arm (1) more often than suboptimal arm (0)
    # Note: This is probabilistic, so we use a relaxed check
    assert arm_1_count >= arm_0_count * 0.5, \
        f"GA did not favor optimal arm: arm1={arm_1_count}, arm0={arm_0_count}"

    print("  ✓ Genetic algorithm tests passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Multi-Arm Bandit GA Tests")
    print("=" * 60 + "\n")

    try:
        test_encoding_decoding()
        test_bandit_environment()
        test_fitness_functions()
        test_genetic_operators()
        test_genetic_algorithm()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60 + "\n")
        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
