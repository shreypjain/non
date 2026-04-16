"""
Fitness evaluation for NoN network architectures.

This module evaluates network performance on reasoning tasks
to guide genetic algorithm optimization.
"""

import asyncio
from typing import List, Dict, Any, Optional
from nons.core.network import NoN
from nons.core.node import Node
from nons.core.types import ModelConfig

# Import operators to register them
import nons.operators.base  # noqa: F401
import nons.operators.deterministic  # noqa: F401

from .network_encoding import NetworkChromosome, describe_network
from .reasoning_tasks import ReasoningTask, calculate_task_accuracy


async def evaluate_network_on_task(
    chromosome: NetworkChromosome,
    task: ReasoningTask,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a network architecture on a reasoning task.

    Args:
        chromosome: Network chromosome to evaluate
        task: Reasoning task to evaluate on
        verbose: Whether to print detailed output

    Returns:
        Dictionary with accuracy and other metrics
    """
    if verbose:
        print(f"\nEvaluating network on task: {task.name}")
        print(describe_network(chromosome))

    try:
        # Build network from chromosome
        network = build_network_from_chromosome(chromosome)

        # Run network on each example
        predictions = []
        for i, example in enumerate(task.examples):
            if verbose:
                print(f"\n  Example {i+1}/{len(task.examples)}")
                print(f"    Input: {example.input[:60]}...")

            try:
                # Run network
                result = await network.forward(example.input)

                if verbose:
                    print(f"    Output: {result[:60]}...")
                    print(f"    Expected: {example.expected_output}")

                predictions.append(result)

            except Exception as e:
                if verbose:
                    print(f"    Error: {e}")
                predictions.append("")  # Empty prediction on error

        # Calculate accuracy
        accuracy = calculate_task_accuracy(predictions, task)

        if verbose:
            print(f"\n  Task Accuracy: {accuracy:.2%}")

        return {
            "accuracy": accuracy,
            "task_name": task.name,
            "num_correct": int(accuracy * len(task.examples)),
            "num_total": len(task.examples),
            "predictions": predictions,
        }

    except Exception as e:
        if verbose:
            print(f"  Network evaluation failed: {e}")

        return {
            "accuracy": 0.0,
            "task_name": task.name,
            "num_correct": 0,
            "num_total": len(task.examples),
            "error": str(e),
        }


async def evaluate_network_fitness(
    chromosome: NetworkChromosome,
    tasks: List[ReasoningTask],
    verbose: bool = False,
) -> float:
    """
    Calculate overall fitness score for a network.

    Fitness is the average accuracy across all tasks.

    Args:
        chromosome: Network chromosome to evaluate
        tasks: List of reasoning tasks
        verbose: Whether to print detailed output

    Returns:
        Fitness score (0.0 to 1.0)
    """
    total_accuracy = 0.0

    for task in tasks:
        result = await evaluate_network_on_task(chromosome, task, verbose=verbose)
        total_accuracy += result["accuracy"]

    fitness = total_accuracy / len(tasks) if tasks else 0.0

    if verbose:
        print(f"\nOverall Fitness: {fitness:.2%}")

    return fitness


def build_network_from_chromosome(
    chromosome: NetworkChromosome,
) -> NoN:
    """
    Build a NoN network from a chromosome encoding.

    Args:
        chromosome: Network chromosome

    Returns:
        Constructed NoN network
    """
    # Get operator specification
    operator_spec = chromosome.to_operator_spec()

    # For now, use simple operator names
    # TODO: Add model configuration per layer
    network = NoN.from_operators(operator_spec)

    return network


async def batch_evaluate_fitness(
    population: List[NetworkChromosome],
    tasks: List[ReasoningTask],
    verbose: bool = False,
) -> List[float]:
    """
    Evaluate fitness for a batch of network chromosomes.

    Args:
        population: List of network chromosomes
        tasks: Reasoning tasks for evaluation
        verbose: Whether to print progress

    Returns:
        List of fitness scores
    """
    fitness_scores = []

    for i, chromosome in enumerate(population):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating individual {i+1}/{len(population)}")
            print(f"{'='*60}")

        fitness = await evaluate_network_fitness(chromosome, tasks, verbose=False)
        fitness_scores.append(fitness)

        if verbose:
            print(f"Fitness: {fitness:.2%}")

    return fitness_scores


def get_fitness_statistics(fitness_scores: List[float]) -> Dict[str, float]:
    """
    Calculate statistics about population fitness.

    Args:
        fitness_scores: List of fitness values

    Returns:
        Dictionary with statistics
    """
    if not fitness_scores:
        return {
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "std": 0.0,
        }

    import numpy as np

    return {
        "mean": float(np.mean(fitness_scores)),
        "max": float(np.max(fitness_scores)),
        "min": float(np.min(fitness_scores)),
        "std": float(np.std(fitness_scores)),
    }
