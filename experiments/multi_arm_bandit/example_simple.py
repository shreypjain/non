"""
Simple example demonstrating genetic algorithm for multi-arm bandits.

This script shows how to:
1. Set up a bandit environment
2. Create a fitness function
3. Run a genetic algorithm
4. Analyze results
"""

from multi_arm_bandit import (
    BernoulliBandit,
    GeneticAlgorithm,
    GAConfig,
    create_fitness_function,
    decode_chromosome,
)


def main():
    """Run a simple genetic algorithm experiment."""

    print("Multi-Arm Bandit Genetic Algorithm Example")
    print("=" * 60)

    # Step 1: Create a bandit environment
    print("\n1. Creating bandit environment...")
    probabilities = [0.1, 0.5, 0.3, 0.7]
    bandit = BernoulliBandit(probabilities, seed=42)

    print(f"   Bandit with {len(probabilities)} arms")
    print(f"   Success probabilities: {probabilities}")
    print(f"   Optimal arm: {bandit.get_optimal_arm()} (p={max(probabilities)})")

    # Step 2: Set up genetic algorithm
    print("\n2. Configuring genetic algorithm...")
    num_arms = len(probabilities)
    horizon = 50  # Strategy length (number of arm pulls)

    config = GAConfig(
        population_size=50,
        num_generations=30,
        mutation_rate=0.01,
        crossover_rate=0.8,
        elitism_count=2,
    )

    print(f"   Population size: {config.population_size}")
    print(f"   Generations: {config.num_generations}")
    print(f"   Strategy horizon: {horizon} pulls")

    # Step 3: Create fitness function
    print("\n3. Creating fitness function...")
    fitness_function = create_fitness_function(
        bandit=bandit,
        num_arms=num_arms,
        fitness_type="reward",
        num_trials=50,  # Average over 50 trials
        horizon=horizon,
    )

    # Step 4: Run genetic algorithm
    print("\n4. Running genetic algorithm...")
    print("-" * 60)

    ga = GeneticAlgorithm(
        num_arms=num_arms,
        horizon=horizon,
        fitness_function=fitness_function,
        config=config,
        seed=42,
    )

    result = ga.run()

    # Step 5: Analyze results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nBest fitness achieved: {result.best_fitness:.4f}")
    print(f"Optimal possible reward: {max(probabilities):.4f}")

    # Decode and analyze best strategy
    best_strategy = decode_chromosome(result.best_chromosome, num_arms)

    print(f"\nBest strategy analysis:")
    print(f"  Total pulls: {len(best_strategy)}")

    # Count arm pulls
    arm_counts = [0] * num_arms
    for arm in best_strategy:
        arm_counts[arm] += 1

    print(f"\n  Arm distribution:")
    for arm_idx, count in enumerate(arm_counts):
        percentage = (count / len(best_strategy)) * 100
        marker = " <- OPTIMAL" if arm_idx == bandit.get_optimal_arm() else ""
        print(f"    Arm {arm_idx}: {count:2d} pulls ({percentage:5.1f}%) [p={probabilities[arm_idx]:.1f}]{marker}")

    # Evolution statistics
    print(f"\n  Evolution:")
    initial_best = result.fitness_history[0]["max"]
    final_best = result.fitness_history[-1]["max"]
    improvement = ((final_best - initial_best) / initial_best) * 100

    print(f"    Initial best fitness: {initial_best:.4f}")
    print(f"    Final best fitness: {final_best:.4f}")
    print(f"    Improvement: {improvement:.1f}%")

    # Convergence analysis
    print(f"\n  Convergence:")
    last_10_best = [entry["max"] for entry in result.fitness_history[-10:]]
    variance = sum((x - final_best) ** 2 for x in last_10_best) / len(last_10_best)

    print(f"    Final 10 generations variance: {variance:.6f}")
    print(f"    Converged: {'Yes' if variance < 0.001 else 'No'}")

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
