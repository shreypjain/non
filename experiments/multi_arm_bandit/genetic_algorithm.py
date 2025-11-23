"""
Core genetic algorithm implementation for evolving bandit strategies.

This module provides the genetic algorithm framework with selection, crossover,
mutation, and population management for optimizing multi-arm bandit strategies.
"""

import random
from typing import List, Callable, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from .encoding import (
    random_chromosome,
    mutate_chromosome,
    crossover_single_point,
    crossover_uniform,
    chromosome_length,
)


@dataclass
class GAConfig:
    """Configuration parameters for genetic algorithm."""

    population_size: int = 100
    num_generations: int = 50
    mutation_rate: float = 0.01
    crossover_rate: float = 0.8
    elitism_count: int = 2
    tournament_size: int = 3
    crossover_type: str = "single_point"  # "single_point" or "uniform"


@dataclass
class GAResult:
    """Results from genetic algorithm run."""

    best_chromosome: List[int]
    best_fitness: float
    fitness_history: List[dict]  # Per-generation statistics
    final_population: List[List[int]]
    final_fitness_scores: List[float]


class GeneticAlgorithm:
    """
    Genetic algorithm for evolving multi-arm bandit strategies.

    This class implements a standard genetic algorithm with:
    - Tournament selection
    - Single-point or uniform crossover
    - Bit-flip mutation
    - Elitism (preserve best individuals)
    """

    def __init__(
        self,
        num_arms: int,
        horizon: int,
        fitness_function: Callable[[List[int]], float],
        config: Optional[GAConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize genetic algorithm.

        Args:
            num_arms: Number of arms in the bandit problem
            horizon: Number of time steps in strategies
            fitness_function: Function to evaluate chromosome fitness
            config: GA configuration parameters
            seed: Random seed for reproducibility
        """
        self.num_arms = num_arms
        self.horizon = horizon
        self.fitness_function = fitness_function
        self.config = config if config is not None else GAConfig()
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize population
        self.population = self._initialize_population()
        self.fitness_scores = []
        self.generation = 0
        self.fitness_history = []

    def _initialize_population(self) -> List[List[int]]:
        """
        Create initial random population.

        Returns:
            List of random chromosomes
        """
        return [
            random_chromosome(self.num_arms, self.horizon)
            for _ in range(self.config.population_size)
        ]

    def evaluate_population(self):
        """Evaluate fitness for entire population."""
        self.fitness_scores = [
            self.fitness_function(chromosome) for chromosome in self.population
        ]

    def tournament_selection(self) -> List[int]:
        """
        Select an individual using tournament selection.

        Returns:
            Selected chromosome
        """
        # Randomly sample tournament_size individuals
        tournament_indices = random.sample(
            range(len(self.population)), self.config.tournament_size
        )

        # Select the one with best fitness
        best_idx = max(tournament_indices, key=lambda i: self.fitness_scores[i])

        return self.population[best_idx].copy()

    def crossover(
        self, parent1: List[int], parent2: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Perform crossover between two parents.

        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome

        Returns:
            Tuple of two offspring chromosomes
        """
        if random.random() < self.config.crossover_rate:
            if self.config.crossover_type == "single_point":
                return crossover_single_point(parent1, parent2)
            elif self.config.crossover_type == "uniform":
                return crossover_uniform(parent1, parent2)
            else:
                raise ValueError(f"Unknown crossover type: {self.config.crossover_type}")
        else:
            # No crossover, return copies of parents
            return parent1.copy(), parent2.copy()

    def mutate(self, chromosome: List[int]) -> List[int]:
        """
        Apply mutation to a chromosome.

        Args:
            chromosome: Chromosome to mutate

        Returns:
            Mutated chromosome
        """
        return mutate_chromosome(chromosome, self.config.mutation_rate)

    def evolve_generation(self):
        """Evolve population by one generation."""
        # Evaluate current population
        self.evaluate_population()

        # Record statistics
        stats = self._calculate_statistics()
        self.fitness_history.append(stats)

        # Elitism: preserve best individuals
        elite_indices = sorted(
            range(len(self.fitness_scores)),
            key=lambda i: self.fitness_scores[i],
            reverse=True,
        )[: self.config.elitism_count]

        elite = [self.population[i].copy() for i in elite_indices]

        # Create new population
        new_population = elite.copy()

        # Generate offspring to fill remaining population
        while len(new_population) < self.config.population_size:
            # Select parents
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            # Crossover
            offspring1, offspring2 = self.crossover(parent1, parent2)

            # Mutation
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)

            # Add to new population
            new_population.append(offspring1)
            if len(new_population) < self.config.population_size:
                new_population.append(offspring2)

        # Replace old population
        self.population = new_population[: self.config.population_size]
        self.generation += 1

    def run(self) -> GAResult:
        """
        Run genetic algorithm for specified number of generations.

        Returns:
            GAResult containing best solution and evolution history
        """
        for gen in range(self.config.num_generations):
            self.evolve_generation()

            # Optional: Print progress
            if (gen + 1) % 10 == 0:
                stats = self.fitness_history[-1]
                print(
                    f"Generation {gen + 1}/{self.config.num_generations}: "
                    f"Best={stats['max']:.4f}, Avg={stats['mean']:.4f}"
                )

        # Final evaluation
        self.evaluate_population()

        # Get best individual
        best_idx = int(np.argmax(self.fitness_scores))
        best_chromosome = self.population[best_idx]
        best_fitness = self.fitness_scores[best_idx]

        return GAResult(
            best_chromosome=best_chromosome,
            best_fitness=best_fitness,
            fitness_history=self.fitness_history,
            final_population=self.population,
            final_fitness_scores=self.fitness_scores,
        )

    def _calculate_statistics(self) -> dict:
        """
        Calculate population statistics for current generation.

        Returns:
            Dictionary with fitness statistics
        """
        return {
            "generation": self.generation,
            "mean": float(np.mean(self.fitness_scores)),
            "max": float(np.max(self.fitness_scores)),
            "min": float(np.min(self.fitness_scores)),
            "std": float(np.std(self.fitness_scores)),
            "median": float(np.median(self.fitness_scores)),
        }

    def get_best_individual(self) -> Tuple[List[int], float]:
        """
        Get the best individual from current population.

        Returns:
            Tuple of (best_chromosome, best_fitness)
        """
        if not self.fitness_scores:
            self.evaluate_population()

        best_idx = int(np.argmax(self.fitness_scores))
        return self.population[best_idx], self.fitness_scores[best_idx]

    def get_population_diversity(self) -> float:
        """
        Calculate population diversity (average Hamming distance).

        Returns:
            Average Hamming distance between all pairs of chromosomes
        """
        if len(self.population) < 2:
            return 0.0

        total_distance = 0.0
        comparisons = 0

        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Hamming distance
                distance = sum(
                    a != b
                    for a, b in zip(self.population[i], self.population[j])
                )
                total_distance += distance
                comparisons += 1

        return total_distance / comparisons if comparisons > 0 else 0.0


def run_multiple_trials(
    num_arms: int,
    horizon: int,
    fitness_function: Callable[[List[int]], float],
    num_trials: int = 10,
    config: Optional[GAConfig] = None,
) -> List[GAResult]:
    """
    Run genetic algorithm multiple times with different random seeds.

    Args:
        num_arms: Number of arms in the bandit problem
        horizon: Number of time steps in strategies
        fitness_function: Function to evaluate chromosome fitness
        num_trials: Number of independent GA runs
        config: GA configuration parameters

    Returns:
        List of GAResult objects, one per trial
    """
    results = []

    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        ga = GeneticAlgorithm(
            num_arms=num_arms,
            horizon=horizon,
            fitness_function=fitness_function,
            config=config,
            seed=trial,  # Use trial number as seed
        )

        result = ga.run()
        results.append(result)

    return results


def get_aggregated_results(results: List[GAResult]) -> dict:
    """
    Aggregate statistics across multiple GA runs.

    Args:
        results: List of GAResult objects

    Returns:
        Dictionary with aggregated statistics
    """
    best_fitnesses = [result.best_fitness for result in results]

    return {
        "mean_best_fitness": float(np.mean(best_fitnesses)),
        "std_best_fitness": float(np.std(best_fitnesses)),
        "min_best_fitness": float(np.min(best_fitnesses)),
        "max_best_fitness": float(np.max(best_fitnesses)),
        "num_trials": len(results),
    }
