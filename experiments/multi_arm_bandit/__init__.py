"""
Network Architecture Evolution with Genetic Algorithms.

This package provides genetic algorithm-based optimization for NoN network
architectures, evolving networks to maximize reasoning task performance.

Main Components:
- Network encoding and genetic operators
- Reasoning task benchmarks
- Fitness evaluation based on task accuracy
- Evolutionary algorithm for architecture search
"""

# Network encoding
from .network_encoding import (
    NetworkGene,
    NetworkChromosome,
    random_network_chromosome,
    mutate_network,
    crossover_networks,
    describe_network,
    AVAILABLE_OPERATORS,
    MODEL_CONFIGS,
)

# Reasoning tasks
from .reasoning_tasks import (
    ReasoningExample,
    ReasoningTask,
    ARITHMETIC_TASK,
    LOGICAL_TASK,
    PATTERN_TASK,
    COMMONSENSE_TASK,
    MULTISTEP_TASK,
    GPQA_TASK,
    ALL_TASKS,
    get_task_by_name,
    create_task_suite,
)

# Fitness evaluation
from .network_fitness import (
    evaluate_network_on_task,
    evaluate_network_fitness,
    build_network_from_chromosome,
    batch_evaluate_fitness,
)

# Genetic algorithm
from .network_evolution import (
    NetworkGAConfig,
    NetworkGAResult,
    NetworkGeneticAlgorithm,
)

# SuperGPQA integration
from .supergpqa_loader import (
    SuperGPQAExample,
    SuperGPQADataset,
    extract_answer_letter,
    evaluate_supergpqa_answer,
)

from .supergpqa_fitness import (
    format_supergpqa_prompt,
    evaluate_network_on_supergpqa,
    batch_evaluate_supergpqa_fitness,
    create_supergpqa_fitness_function,
)

__all__ = [
    # Network evolution (primary interface)
    "NetworkGene",
    "NetworkChromosome",
    "random_network_chromosome",
    "mutate_network",
    "crossover_networks",
    "describe_network",
    "AVAILABLE_OPERATORS",
    "MODEL_CONFIGS",
    # Reasoning tasks
    "ReasoningExample",
    "ReasoningTask",
    "ARITHMETIC_TASK",
    "LOGICAL_TASK",
    "PATTERN_TASK",
    "COMMONSENSE_TASK",
    "MULTISTEP_TASK",
    "GPQA_TASK",
    "ALL_TASKS",
    "get_task_by_name",
    "create_task_suite",
    # Fitness
    "evaluate_network_on_task",
    "evaluate_network_fitness",
    "build_network_from_chromosome",
    "batch_evaluate_fitness",
    # GA
    "NetworkGAConfig",
    "NetworkGAResult",
    "NetworkGeneticAlgorithm",
    # SuperGPQA
    "SuperGPQAExample",
    "SuperGPQADataset",
    "extract_answer_letter",
    "evaluate_supergpqa_answer",
    "format_supergpqa_prompt",
    "evaluate_network_on_supergpqa",
    "batch_evaluate_supergpqa_fitness",
    "create_supergpqa_fitness_function",
]
