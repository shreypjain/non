"""
Reasoning task benchmarks for network fitness evaluation.

This module provides simple reasoning tasks to evaluate NoN network
performance. Tasks are designed to test different reasoning capabilities.
"""

from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import re


@dataclass
class ReasoningExample:
    """A single reasoning task example."""

    input: str
    expected_output: str
    task_type: str


@dataclass
class ReasoningTask:
    """A reasoning task consisting of multiple examples."""

    name: str
    description: str
    examples: List[ReasoningExample]
    evaluation_fn: Callable[[str, str], bool]


def exact_match_eval(predicted: str, expected: str) -> bool:
    """Exact string match evaluation."""
    return predicted.strip().lower() == expected.strip().lower()


def contains_eval(predicted: str, expected: str) -> bool:
    """Check if predicted contains expected answer."""
    return expected.strip().lower() in predicted.strip().lower()


def numeric_eval(predicted: str, expected: str) -> bool:
    """Extract and compare numeric answers."""
    try:
        # Extract first number from predicted
        pred_match = re.search(r'-?\d+\.?\d*', predicted)
        exp_match = re.search(r'-?\d+\.?\d*', expected)

        if pred_match and exp_match:
            return float(pred_match.group()) == float(exp_match.group())
        return False
    except (ValueError, AttributeError):
        return False


# Simple arithmetic reasoning
ARITHMETIC_TASK = ReasoningTask(
    name="arithmetic",
    description="Basic arithmetic word problems",
    examples=[
        ReasoningExample(
            input="If John has 5 apples and Mary gives him 3 more, how many apples does John have?",
            expected_output="8",
            task_type="arithmetic",
        ),
        ReasoningExample(
            input="A store has 20 books. They sell 7 books. How many books are left?",
            expected_output="13",
            task_type="arithmetic",
        ),
        ReasoningExample(
            input="There are 4 boxes with 6 items each. What is the total number of items?",
            expected_output="24",
            task_type="arithmetic",
        ),
        ReasoningExample(
            input="If a car travels 60 miles in 2 hours, how many miles per hour is that?",
            expected_output="30",
            task_type="arithmetic",
        ),
        ReasoningExample(
            input="Sarah has 15 candies. She gives 5 to Tom and 3 to Lisa. How many candies does Sarah have left?",
            expected_output="7",
            task_type="arithmetic",
        ),
    ],
    evaluation_fn=numeric_eval,
)


# Logical reasoning
LOGICAL_TASK = ReasoningTask(
    name="logical",
    description="Simple logical deduction problems",
    examples=[
        ReasoningExample(
            input="All cats are animals. Fluffy is a cat. Is Fluffy an animal? Answer: Yes or No",
            expected_output="Yes",
            task_type="logical",
        ),
        ReasoningExample(
            input="If it rains, the ground gets wet. The ground is wet. Did it rain? Answer: Yes, No, or Maybe",
            expected_output="Maybe",
            task_type="logical",
        ),
        ReasoningExample(
            input="All birds can fly. Penguins are birds. Can penguins fly? Answer: Yes or No (be careful!)",
            expected_output="No",
            task_type="logical",
        ),
        ReasoningExample(
            input="If A is greater than B, and B is greater than C, is A greater than C? Answer: Yes or No",
            expected_output="Yes",
            task_type="logical",
        ),
        ReasoningExample(
            input="All roses are flowers. Some flowers are red. Are all roses red? Answer: Yes or No",
            expected_output="No",
            task_type="logical",
        ),
    ],
    evaluation_fn=contains_eval,
)


# Pattern recognition
PATTERN_TASK = ReasoningTask(
    name="pattern",
    description="Number and sequence pattern recognition",
    examples=[
        ReasoningExample(
            input="What comes next in this sequence: 2, 4, 6, 8, ?",
            expected_output="10",
            task_type="pattern",
        ),
        ReasoningExample(
            input="Complete the pattern: 1, 1, 2, 3, 5, 8, ?",
            expected_output="13",
            task_type="pattern",
        ),
        ReasoningExample(
            input="What number continues this sequence: 10, 20, 30, 40, ?",
            expected_output="50",
            task_type="pattern",
        ),
        ReasoningExample(
            input="Find the next number: 3, 6, 12, 24, ?",
            expected_output="48",
            task_type="pattern",
        ),
        ReasoningExample(
            input="Complete: 100, 90, 80, 70, ?",
            expected_output="60",
            task_type="pattern",
        ),
    ],
    evaluation_fn=numeric_eval,
)


# Simple commonsense reasoning
COMMONSENSE_TASK = ReasoningTask(
    name="commonsense",
    description="Basic commonsense reasoning",
    examples=[
        ReasoningExample(
            input="If you drop a glass on a hard floor, what will likely happen? Answer in one word.",
            expected_output="break",
            task_type="commonsense",
        ),
        ReasoningExample(
            input="Is fire hot or cold? Answer in one word.",
            expected_output="hot",
            task_type="commonsense",
        ),
        ReasoningExample(
            input="Can humans breathe underwater without equipment? Answer: Yes or No",
            expected_output="No",
            task_type="commonsense",
        ),
        ReasoningExample(
            input="What happens to ice when it gets warm? Answer in one word.",
            expected_output="melts",
            task_type="commonsense",
        ),
        ReasoningExample(
            input="Do trees grow upward or downward? Answer in one word.",
            expected_output="upward",
            task_type="commonsense",
        ),
    ],
    evaluation_fn=contains_eval,
)


# Multi-step reasoning
MULTISTEP_TASK = ReasoningTask(
    name="multistep",
    description="Problems requiring multiple reasoning steps",
    examples=[
        ReasoningExample(
            input="A pizza is cut into 8 slices. John eats 2 slices, Mary eats 3 slices. How many slices remain?",
            expected_output="3",
            task_type="multistep",
        ),
        ReasoningExample(
            input="A train leaves at 3:00 PM and takes 2 hours to reach its destination. What time does it arrive?",
            expected_output="5:00 PM",
            task_type="multistep",
        ),
        ReasoningExample(
            input="You have $50. You buy a book for $12 and lunch for $18. How much money do you have left?",
            expected_output="20",
            task_type="multistep",
        ),
        ReasoningExample(
            input="There are 3 boxes with 5 red balls each and 2 boxes with 4 blue balls each. How many balls total?",
            expected_output="23",
            task_type="multistep",
        ),
        ReasoningExample(
            input="If Jane is twice as old as Tom, and Tom is 8 years old, how old is Jane?",
            expected_output="16",
            task_type="multistep",
        ),
    ],
    evaluation_fn=numeric_eval,
)


# GPQA-style graduate-level reasoning
GPQA_TASK = ReasoningTask(
    name="gpqa",
    description="Graduate-level science and reasoning questions (GPQA-style)",
    examples=[
        ReasoningExample(
            input="In thermodynamics, which law states that energy cannot be created or destroyed, only converted from one form to another?",
            expected_output="first law",
            task_type="gpqa",
        ),
        ReasoningExample(
            input="What is the term for a molecule that has both hydrophilic and hydrophobic regions?",
            expected_output="amphipathic",
            task_type="gpqa",
        ),
        ReasoningExample(
            input="In quantum mechanics, what principle states that you cannot simultaneously know both the exact position and momentum of a particle?",
            expected_output="Heisenberg uncertainty principle",
            task_type="gpqa",
        ),
        ReasoningExample(
            input="What type of chemical bond involves the sharing of electron pairs between atoms?",
            expected_output="covalent",
            task_type="gpqa",
        ),
        ReasoningExample(
            input="In genetics, what is the term for alternative forms of a gene that arise by mutation and are found at the same place on a chromosome?",
            expected_output="allele",
            task_type="gpqa",
        ),
    ],
    evaluation_fn=contains_eval,
)


# Collection of all tasks
ALL_TASKS = [
    ARITHMETIC_TASK,
    LOGICAL_TASK,
    PATTERN_TASK,
    COMMONSENSE_TASK,
    MULTISTEP_TASK,
    GPQA_TASK,
]


def get_task_by_name(name: str) -> ReasoningTask:
    """Get a reasoning task by name."""
    for task in ALL_TASKS:
        if task.name == name:
            return task
    raise ValueError(f"Unknown task: {name}")


def evaluate_prediction(
    predicted: str,
    expected: str,
    evaluation_fn: Callable[[str, str], bool],
) -> bool:
    """
    Evaluate a prediction against expected output.

    Args:
        predicted: Model's predicted answer
        expected: Expected correct answer
        evaluation_fn: Function to evaluate correctness

    Returns:
        True if correct, False otherwise
    """
    return evaluation_fn(predicted, expected)


def calculate_task_accuracy(
    predictions: List[str],
    task: ReasoningTask,
) -> float:
    """
    Calculate accuracy on a reasoning task.

    Args:
        predictions: List of predicted answers
        task: Reasoning task with expected answers

    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if len(predictions) != len(task.examples):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) does not match "
            f"number of examples ({len(task.examples)})"
        )

    correct = 0
    for pred, example in zip(predictions, task.examples):
        if evaluate_prediction(pred, example.expected_output, task.evaluation_fn):
            correct += 1

    return correct / len(task.examples)


def create_task_suite(task_names: List[str] = None) -> List[ReasoningTask]:
    """
    Create a suite of reasoning tasks.

    Args:
        task_names: List of task names to include (all if None)

    Returns:
        List of ReasoningTask objects
    """
    if task_names is None:
        return ALL_TASKS

    return [get_task_by_name(name) for name in task_names]
