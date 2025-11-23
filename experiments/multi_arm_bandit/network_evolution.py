"""
Genetic algorithm for evolving NoN network architectures.

This module implements the evolutionary algorithm that optimizes
network structures based on reasoning task performance.
"""

import asyncio
import random
from typing import List, Optional
from dataclasses import dataclass
import numpy as np

from .network_encoding import (
    NetworkChromosome,
    random_network_chromosome,
    mutate_network,
    crossover_networks,
    describe_network,
)
from .network_fitness import (
    evaluate_network_fitness,
    batch_evaluate_fitness,
    get_fitness_statistics,
)
from .reasoning_tasks import ReasoningTask


@dataclass
class NetworkGAConfig:
    """Configuration for network evolution genetic algorithm."""

    population_size: int = 20
    num_generations: int = 20
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    elitism_count: int = 2
    tournament_size: int = 3
    min_layers: int = 2
    max_layers: int = 5


@dataclass
class NetworkGAResult:
    """Results from network evolution."""

    best_chromosome: NetworkChromosome
    best_fitness: float
    fitness_history: List[dict]
    final_population: List[NetworkChromosome]
    final_fitness_scores: List[float]


class NetworkGeneticAlgorithm:
    """
    Genetic algorithm for evolving NoN network architectures.

    Evolves networks to maximize accuracy on reasoning tasks through:
    - Tournament selection
    - Network crossover
    - Architecture mutation
    - Elitism
    """

    def __init__(
        self,
        tasks: List[ReasoningTask],
        config: Optional[NetworkGAConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize network evolution GA.

        Args:
            tasks: List of reasoning tasks for fitness evaluation
            config: GA configuration parameters
            seed: Random seed for reproducibility
        """
        self.tasks = tasks
        self.config = config if config is not None else NetworkGAConfig()
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize population
        self.population = self._initialize_population()
        self.fitness_scores = []
        self.generation = 0
        self.fitness_history = []

    def _initialize_population(self) -> List[NetworkChromosome]:
        """Create initial random population of networks."""
        return [
            random_network_chromosome(
                min_layers=self.config.min_layers,
                max_layers=self.config.max_layers,
            )
            for _ in range(self.config.population_size)
        ]

    async def evaluate_population(self, verbose: bool = False):
        """Evaluate fitness for entire population."""
        self.fitness_scores = await batch_evaluate_fitness(
            self.population, self.tasks, verbose=verbose
        )

    def tournament_selection(self) -> NetworkChromosome:
        """Select an individual using tournament selection."""
        tournament_indices = random.sample(
            range(len(self.population)), self.config.tournament_size
        )

        best_idx = max(tournament_indices, key=lambda i: self.fitness_scores[i])
        return self.population[best_idx]

    def crossover(
        self, parent1: NetworkChromosome, parent2: NetworkChromosome
    ) -> tuple[NetworkChromosome, NetworkChromosome]:
        """Perform crossover between two parents."""
        if random.random() < self.config.crossover_rate:
            return crossover_networks(parent1, parent2)
        else:
            return parent1.copy(), parent2.copy()

    def mutate(self, chromosome: NetworkChromosome) -> NetworkChromosome:
        """Apply mutation to a network chromosome."""
        return mutate_network(chromosome, self.config.mutation_rate)

    async def evolve_generation(self, verbose: bool = False):
        """Evolve population by one generation."""
        # Evaluate current population
        if verbose:
            print(f"\nGeneration {self.generation + 1}/{self.config.num_generations}")
            print("-" * 60)

        await self.evaluate_population(verbose=False)

        # Record statistics
        stats = self._calculate_statistics()
        self.fitness_history.append(stats)

        if verbose:
            print(f"  Best fitness: {stats['max']:.2%}")
            print(f"  Mean fitness: {stats['mean']:.2%}")
            print(f"  Population diversity: {self._get_diversity():.2f}")

        # Elitism: preserve best individuals
        elite_indices = sorted(
            range(len(self.fitness_scores)),
            key=lambda i: self.fitness_scores[i],
            reverse=True,
        )[: self.config.elitism_count]

        elite = [self.population[i].copy() for i in elite_indices]

        # Create new population
        new_population = elite.copy()

        # Generate offspring
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

    async def run(self, verbose: bool = True) -> NetworkGAResult:
        """
        Run genetic algorithm for specified number of generations.

        Args:
            verbose: Whether to print progress

        Returns:
            NetworkGAResult with best network and evolution history
        """
        if verbose:
            print("\n" + "=" * 60)
            print("Network Architecture Evolution")
            print("=" * 60)
            print(f"Population size: {self.config.population_size}")
            print(f"Generations: {self.config.num_generations}")
            print(f"Tasks: {[task.name for task in self.tasks]}")

        for gen in range(self.config.num_generations):
            await self.evolve_generation(verbose=verbose)

        # Final evaluation
        await self.evaluate_population(verbose=False)

        # Get best individual
        best_idx = int(np.argmax(self.fitness_scores))
        best_chromosome = self.population[best_idx]
        best_fitness = self.fitness_scores[best_idx]

        if verbose:
            print("\n" + "=" * 60)
            print("Evolution Complete")
            print("=" * 60)
            print(f"\nBest Network (Fitness: {best_fitness:.2%}):")
            print(describe_network(best_chromosome))

        return NetworkGAResult(
            best_chromosome=best_chromosome,
            best_fitness=best_fitness,
            fitness_history=self.fitness_history,
            final_population=self.population,
            final_fitness_scores=self.fitness_scores,
        )

    def _calculate_statistics(self) -> dict:
        """Calculate population statistics for current generation."""
        return {
            "generation": self.generation,
            "mean": float(np.mean(self.fitness_scores)),
            "max": float(np.max(self.fitness_scores)),
            "min": float(np.min(self.fitness_scores)),
            "std": float(np.std(self.fitness_scores)),
        }

    def _get_diversity(self) -> float:
        """
        Calculate population diversity.

        Measures average structural difference between networks.
        """
        if len(self.population) < 2:
            return 0.0

        total_diff = 0.0
        comparisons = 0

        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Compare network structures
                net1 = self.population[i]
                net2 = self.population[j]

                # Simple diversity metric: difference in number of layers
                # and operator types
                diff = abs(len(net1.genes) - len(net2.genes))

                # Count operator differences
                for k in range(min(len(net1.genes), len(net2.genes))):
                    if net1.genes[k].operators != net2.genes[k].operators:
                        diff += 1

                total_diff += diff
                comparisons += 1

        return total_diff / comparisons if comparisons > 0 else 0.0
