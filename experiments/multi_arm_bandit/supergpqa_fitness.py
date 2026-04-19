"""
SuperGPQA fitness evaluation for network evolution.

This module evaluates NoN network architectures on the SuperGPQA benchmark,
using accuracy on graduate-level multiple-choice questions as fitness.
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
from .supergpqa_loader import (
    SuperGPQADataset,
    SuperGPQAExample,
    evaluate_supergpqa_answer,
)


def format_supergpqa_prompt(example: SuperGPQAExample) -> str:
    """
    Format a SuperGPQA question as a prompt for the network.

    Args:
        example: SuperGPQA question example

    Returns:
        Formatted prompt string
    """
    prompt_parts = [
        f"Question: {example.question}",
        "",
        "Options:",
    ]

    # Add all options
    for option in example.options:
        prompt_parts.append(option)

    prompt_parts.extend(
        [
            "",
            "Please select the correct answer and respond with ONLY the letter (A, B, C, etc.).",
            "Answer:",
        ]
    )

    return "\n".join(prompt_parts)


async def evaluate_network_on_supergpqa(
    chromosome: NetworkChromosome,
    examples: List[SuperGPQAExample],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a network architecture on SuperGPQA examples.

    Args:
        chromosome: Network chromosome to evaluate
        examples: List of SuperGPQA examples to test on
        verbose: Whether to print detailed output

    Returns:
        Dictionary with accuracy and metrics
    """
    if verbose:
        print(f"\nEvaluating network on {len(examples)} SuperGPQA questions")
        print(describe_network(chromosome))

    try:
        # Build network from chromosome
        network = build_network_from_chromosome(chromosome)

        # Run network on each example
        correct = 0
        predictions = []

        for i, example in enumerate(examples):
            if verbose and i % 10 == 0:
                print(f"\n  Progress: {i}/{len(examples)}")

            try:
                # Format question as prompt
                prompt = format_supergpqa_prompt(example)

                # Run network
                result = await network.forward(prompt)

                # Evaluate answer
                is_correct = evaluate_supergpqa_answer(result, example.answer)

                if is_correct:
                    correct += 1

                predictions.append(
                    {
                        "question": example.question[:60] + "...",
                        "predicted": result[:50],
                        "correct_answer": example.answer,
                        "is_correct": is_correct,
                        "subject": example.subject,
                    }
                )

                if verbose and i < 5:  # Show first few examples
                    print(f"    Q: {example.question[:60]}...")
                    print(f"    Predicted: {result[:50]}")
                    print(f"    Correct: {example.answer}")
                    print(f"    Result: {'✓' if is_correct else '✗'}")

            except Exception as e:
                if verbose:
                    print(f"    Error on question {i}: {e}")
                predictions.append(
                    {
                        "question": example.question[:60] + "...",
                        "predicted": "",
                        "correct_answer": example.answer,
                        "is_correct": False,
                        "subject": example.subject,
                        "error": str(e),
                    }
                )

        # Calculate metrics
        accuracy = correct / len(examples) if examples else 0.0

        # Subject-wise accuracy
        subject_accuracy = {}
        for pred in predictions:
            subj = pred["subject"]
            if subj not in subject_accuracy:
                subject_accuracy[subj] = {"correct": 0, "total": 0}
            subject_accuracy[subj]["total"] += 1
            if pred["is_correct"]:
                subject_accuracy[subj]["correct"] += 1

        for subj in subject_accuracy:
            total = subject_accuracy[subj]["total"]
            subject_accuracy[subj]["accuracy"] = (
                subject_accuracy[subj]["correct"] / total if total > 0 else 0.0
            )

        if verbose:
            print(f"\n  Overall Accuracy: {accuracy:.2%} ({correct}/{len(examples)})")
            print(f"\n  Subject-wise Accuracy:")
            for subj, metrics in subject_accuracy.items():
                print(
                    f"    {subj}: {metrics['accuracy']:.2%} "
                    f"({metrics['correct']}/{metrics['total']})"
                )

        return {
            "accuracy": accuracy,
            "num_correct": correct,
            "num_total": len(examples),
            "subject_accuracy": subject_accuracy,
            "predictions": predictions,
        }

    except Exception as e:
        if verbose:
            print(f"  Network evaluation failed: {e}")

        return {
            "accuracy": 0.0,
            "num_correct": 0,
            "num_total": len(examples),
            "error": str(e),
        }


def build_network_from_chromosome(chromosome: NetworkChromosome) -> NoN:
    """
    Build a NoN network from a chromosome encoding.

    For SuperGPQA evaluation, we use the 'generate' operator for all layers.
    This is the correct approach because:

    1. SuperGPQA is a question-answering task that requires reasoning and generation
    2. The 'generate' operator only needs a generation_specification parameter,
       which we provide via additional_prompt_context
    3. Other operators (transform, extract, classify, etc.) require specific
       parameters (transformation_type, extraction_criteria, etc.) that are
       task-specific and don't apply to general question answering
    4. Using generate with different models per layer allows the GA to optimize
       model selection while keeping the operator interface simple

    The genetic algorithm optimizes:
    - Which model provider to use per layer (Anthropic/OpenAI/Google)
    - Which specific model to use (Claude/GPT/Gemini variants)
    - Network depth (number of layers)

    Args:
        chromosome: Network chromosome with model configurations per layer

    Returns:
        Constructed NoN network using generate operator with varied models
    """
    from nons.core.layer import Layer

    # Use generate operator for all layers with different model configurations
    # This is appropriate for SuperGPQA question answering task
    layers = []

    for i, gene in enumerate(chromosome.genes):
        # Create model config for this layer from chromosome gene
        model_config = ModelConfig(
            provider=gene.provider,
            model_name=gene.model_name,
            temperature=0.3,  # Lower temperature for more deterministic reasoning
        )

        # Create node with generate operator and model config
        # The additional_prompt_context serves as the generation_specification
        context = "Analyze the question and select the correct answer. Respond with ONLY the letter (A, B, C, etc.)."

        node = Node(
            operator_name="generate",
            model_config=model_config,
            additional_prompt_context=context,
        )

        # Create layer with this node
        layer = Layer([node])
        layers.append(layer)

    # Build network from layers
    network = NoN(layers=layers)

    return network


async def batch_evaluate_supergpqa_fitness(
    population: List[NetworkChromosome],
    dataset: SuperGPQADataset,
    num_questions: int = 20,
    verbose: bool = False,
) -> List[float]:
    """
    Evaluate fitness for a batch of network chromosomes on SuperGPQA.

    Args:
        population: List of network chromosomes
        dataset: SuperGPQA dataset
        num_questions: Number of questions to evaluate on per network
        verbose: Whether to print progress

    Returns:
        List of fitness scores (accuracy values)
    """
    # Sample questions for this evaluation
    questions = dataset.sample(num_questions)

    if verbose:
        print(f"\nEvaluating {len(population)} networks on {num_questions} SuperGPQA questions")
        subjects = set(q.subject for q in questions)
        print(f"Subjects covered: {', '.join(sorted(subjects))}")

    fitness_scores = []

    for i, chromosome in enumerate(population):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Network {i+1}/{len(population)}")
            print(f"{'='*60}")

        result = await evaluate_network_on_supergpqa(
            chromosome, questions, verbose=verbose
        )

        fitness_scores.append(result["accuracy"])

        if verbose:
            print(f"Fitness: {result['accuracy']:.2%}")

    return fitness_scores


def create_supergpqa_fitness_function(
    dataset: SuperGPQADataset,
    num_questions: int = 20,
):
    """
    Create a fitness function based on SuperGPQA evaluation.

    Args:
        dataset: SuperGPQA dataset
        num_questions: Number of questions to evaluate per individual

    Returns:
        Async fitness function that takes a chromosome and returns accuracy
    """

    async def fitness_function(chromosome: NetworkChromosome) -> float:
        """Evaluate network on SuperGPQA questions."""
        questions = dataset.sample(num_questions)
        result = await evaluate_network_on_supergpqa(
            chromosome, questions, verbose=False
        )
        return result["accuracy"]

    return fitness_function
