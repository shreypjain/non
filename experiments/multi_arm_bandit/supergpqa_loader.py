"""
SuperGPQA dataset loader and evaluator.

This module handles loading the SuperGPQA dataset and evaluating
network predictions against multiple-choice answers.

SuperGPQA Format (based on https://huggingface.co/datasets/m-a-p/SuperGPQA):
- 26,529 graduate-level questions across 285 disciplines
- Average of 9.67 options per question (much harder than 4-option)
- 42.33% require mathematical calculation or formal reasoning
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import re
import random


@dataclass
class SuperGPQAExample:
    """A single SuperGPQA question."""

    question: str
    options: List[str]  # List of answer options
    answer: str  # Correct answer (letter or text)
    subject: str  # Academic discipline
    requires_math: bool = False  # Whether question requires calculation


class SuperGPQADataset:
    """
    SuperGPQA dataset loader and manager.

    Handles loading, sampling, and evaluation of SuperGPQA questions.
    """

    def __init__(self, examples: List[SuperGPQAExample]):
        """
        Initialize dataset.

        Args:
            examples: List of SuperGPQA examples
        """
        self.examples = examples

    @classmethod
    def load_mock_dataset(cls, num_examples: int = 50) -> "SuperGPQADataset":
        """
        Create a mock SuperGPQA dataset for testing.

        This creates graduate-level questions similar to the real dataset
        format. Replace with actual dataset loading when available.

        Args:
            num_examples: Number of mock examples to create

        Returns:
            SuperGPQADataset instance
        """
        examples = []

        # Physics questions
        examples.extend(
            [
                SuperGPQAExample(
                    question="In quantum field theory, what is the significance of the Feynman propagator?",
                    options=[
                        "A) It describes the probability amplitude for a particle to travel from one point to another",
                        "B) It measures the energy of virtual particles",
                        "C) It calculates the speed of light in vacuum",
                        "D) It determines the mass of fundamental particles",
                        "E) It represents the wave function collapse",
                        "F) It quantifies entanglement strength",
                        "G) It measures quantum decoherence",
                        "H) It describes electromagnetic field strength",
                    ],
                    answer="A",
                    subject="Physics",
                    requires_math=False,
                ),
                SuperGPQAExample(
                    question="What is the Schwarzschild radius of a black hole with mass M equal to 10 solar masses? (Use G = 6.67×10^-11 m^3/kg/s^2, c = 3×10^8 m/s, solar mass = 2×10^30 kg)",
                    options=[
                        "A) 2.96 km",
                        "B) 29.6 km",
                        "C) 296 km",
                        "D) 2.96 m",
                        "E) 14.8 km",
                        "F) 148 km",
                        "G) 59.2 km",
                        "H) 5.92 km",
                    ],
                    answer="B",
                    subject="Physics",
                    requires_math=True,
                ),
            ]
        )

        # Chemistry questions
        examples.extend(
            [
                SuperGPQAExample(
                    question="Which of the following best describes the Hammond postulate in physical organic chemistry?",
                    options=[
                        "A) Exothermic reactions have transition states that resemble products",
                        "B) Endothermic reactions have transition states that resemble reactants",
                        "C) The transition state resembles the nearest stable species in energy",
                        "D) All transition states have equal energy regardless of reaction",
                        "E) Transition states always resemble the most stable intermediate",
                        "F) The activation energy is proportional to reaction enthalpy",
                        "G) Transition states are always planar geometries",
                    ],
                    answer="C",
                    subject="Chemistry",
                    requires_math=False,
                ),
                SuperGPQAExample(
                    question="Calculate the pH of a 0.1 M solution of acetic acid (Ka = 1.8×10^-5). Assume complete dissociation of the weak acid is negligible.",
                    options=[
                        "A) pH = 1.0",
                        "B) pH = 2.87",
                        "C) pH = 4.74",
                        "D) pH = 5.0",
                        "E) pH = 3.87",
                        "F) pH = 2.37",
                        "G) pH = 6.0",
                        "H) pH = 3.0",
                    ],
                    answer="B",
                    subject="Chemistry",
                    requires_math=True,
                ),
            ]
        )

        # Biology questions
        examples.extend(
            [
                SuperGPQAExample(
                    question="In molecular biology, what is the primary function of the CRISPR-Cas9 system in its natural bacterial context?",
                    options=[
                        "A) DNA replication and repair",
                        "B) Adaptive immune defense against viral DNA",
                        "C) Protein synthesis regulation",
                        "D) Metabolic pathway control",
                        "E) Cell division checkpoint regulation",
                        "F) Membrane transport",
                        "G) Energy production via ATP synthesis",
                        "H) Signal transduction",
                    ],
                    answer="B",
                    subject="Biology",
                    requires_math=False,
                ),
                SuperGPQAExample(
                    question="During meiosis, if a diploid cell with 2n=16 chromosomes undergoes normal division, how many chromatids are present during metaphase I?",
                    options=[
                        "A) 8 chromatids",
                        "B) 16 chromatids",
                        "C) 32 chromatids",
                        "D) 64 chromatids",
                        "E) 4 chromatids",
                        "F) 24 chromatids",
                        "G) 48 chromatids",
                    ],
                    answer="C",
                    subject="Biology",
                    requires_math=True,
                ),
            ]
        )

        # Computer Science questions
        examples.extend(
            [
                SuperGPQAExample(
                    question="In computational complexity theory, which of the following best characterizes the relationship between P, NP, and NP-complete?",
                    options=[
                        "A) P ⊂ NP and NP-complete ⊂ NP",
                        "B) P = NP = NP-complete",
                        "C) NP-complete ⊂ P ⊂ NP",
                        "D) P ⊂ NP-complete ⊂ NP",
                        "E) All three classes are equivalent",
                        "F) P and NP are disjoint sets",
                        "G) NP-complete ⊄ NP",
                    ],
                    answer="A",
                    subject="Computer Science",
                    requires_math=False,
                ),
            ]
        )

        # Mathematics questions
        examples.extend(
            [
                SuperGPQAExample(
                    question="What is the Hausdorff dimension of the Cantor set?",
                    options=[
                        "A) 0",
                        "B) log(2)/log(3)",
                        "C) 1",
                        "D) log(3)/log(2)",
                        "E) 0.5",
                        "F) 2/3",
                        "G) 1/2",
                        "H) sqrt(2)",
                    ],
                    answer="B",
                    subject="Mathematics",
                    requires_math=True,
                ),
            ]
        )

        # Pad with additional questions if needed
        while len(examples) < num_examples:
            # Cycle through subjects and create variations
            subjects = ["Physics", "Chemistry", "Biology", "Mathematics", "Computer Science"]
            subject = random.choice(subjects)

            examples.append(
                SuperGPQAExample(
                    question=f"This is a graduate-level {subject} question requiring advanced knowledge.",
                    options=[
                        f"{chr(65+i)}) Option {i+1}"
                        for i in range(random.randint(4, 10))
                    ],
                    answer=random.choice(["A", "B", "C", "D"]),
                    subject=subject,
                    requires_math=random.choice([True, False]),
                )
            )

        return cls(examples[:num_examples])

    @classmethod
    def load_from_huggingface(cls, subset: Optional[str] = None, limit: Optional[int] = None) -> "SuperGPQADataset":
        """
        Load SuperGPQA dataset from Hugging Face.

        Args:
            subset: Optional subset name (e.g., "physics", "chemistry")
            limit: Optional limit on number of examples

        Returns:
            SuperGPQADataset instance
        """
        try:
            from datasets import load_dataset

            # Load dataset
            dataset = load_dataset("m-a-p/SuperGPQA", split="train")

            if subset:
                # Filter by subject if specified
                dataset = dataset.filter(lambda x: x.get("subject", "").lower() == subset.lower())

            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))

            # Convert to our format
            examples = []
            for item in dataset:
                examples.append(
                    SuperGPQAExample(
                        question=item["question"],
                        options=item["options"],
                        answer=item["answer"],
                        subject=item.get("subject", "Unknown"),
                        requires_math=item.get("requires_math", False),
                    )
                )

            return cls(examples)

        except Exception as e:
            print(f"Warning: Could not load from Hugging Face: {e}")
            print("Falling back to mock dataset...")
            return cls.load_mock_dataset(num_examples=limit or 50)

    def sample(self, n: int, seed: Optional[int] = None) -> List[SuperGPQAExample]:
        """
        Sample n examples from the dataset.

        Args:
            n: Number of examples to sample
            seed: Random seed for reproducibility

        Returns:
            List of sampled examples
        """
        if seed is not None:
            random.seed(seed)

        return random.sample(self.examples, min(n, len(self.examples)))

    def get_by_subject(self, subject: str) -> List[SuperGPQAExample]:
        """Get all examples for a specific subject."""
        return [ex for ex in self.examples if ex.subject.lower() == subject.lower()]

    def __len__(self) -> int:
        """Return number of examples in dataset."""
        return len(self.examples)


def extract_answer_letter(response: str) -> Optional[str]:
    """
    Extract answer letter (A, B, C, etc.) from model response.

    Looks for patterns like:
    - "A)" or "A."
    - "The answer is A"
    - "Option A"
    - Just "A" on its own

    Args:
        response: Model's response text

    Returns:
        Extracted letter or None if not found
    """
    # Remove extra whitespace
    response = response.strip()

    # Pattern 1: Letter followed by ) or . at start
    match = re.search(r"^([A-J])[).]", response)
    if match:
        return match.group(1)

    # Pattern 2: "The answer is X", "Answer: X", "X is correct", etc.
    match = re.search(r"(?:answer|option|choice|think|believe)\s+(?:is\s+)?([A-J])\s+(?:is|are)?", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-J])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 3: Just the letter (with word boundaries)
    match = re.search(r"\b([A-J])\b", response)
    if match:
        return match.group(1)

    # Pattern 4: First capital letter in response
    match = re.search(r"([A-J])", response)
    if match:
        return match.group(1)

    return None


def evaluate_supergpqa_answer(prediction: str, correct_answer: str) -> bool:
    """
    Evaluate if a prediction matches the correct answer.

    Args:
        prediction: Model's predicted answer
        correct_answer: Correct answer letter

    Returns:
        True if correct, False otherwise
    """
    extracted = extract_answer_letter(prediction)
    if extracted is None:
        return False

    return extracted.upper() == correct_answer.upper()
