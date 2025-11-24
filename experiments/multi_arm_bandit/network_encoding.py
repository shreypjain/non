"""
Network encoding for genetic algorithm optimization.

This module defines how to represent NoN network architectures as
genetic algorithm chromosomes for evolutionary optimization.
"""

from typing import List, Dict, Any, Optional
import random
from dataclasses import dataclass
from nons.core.types import ModelProvider


# Available operators for network construction
AVAILABLE_OPERATORS = [
    "transform",
    "generate",
    "classify",
    "extract",
    "condense",
    "expand",
    "compare",
    "validate",
    "synthesize",
]

# Available model providers
AVAILABLE_PROVIDERS = [
    ModelProvider.ANTHROPIC,
    ModelProvider.OPENAI,
    ModelProvider.GOOGLE,
]

# Model configurations for each provider (November 2025 latest models)
MODEL_CONFIGS = {
    ModelProvider.ANTHROPIC: [
        "claude-haiku-4.5",
        "claude-sonnet-4.5",
        "claude-opus-4.1",
    ],
    ModelProvider.OPENAI: [
        "gpt-4o",
        "gpt-5.1",
        "gpt-5.1-codex-max",
    ],
    ModelProvider.GOOGLE: [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3-pro",
        "gemini-3-deep-think",
    ],
}


@dataclass
class NetworkGene:
    """
    A single gene representing one layer/operator in the network.

    For simplicity, each gene represents either:
    - A single operator (sequential layer)
    - Multiple operators in parallel (parallel layer)
    """

    operators: List[str]  # Operator name(s) for this layer
    provider: ModelProvider  # Model provider to use
    model_name: str  # Specific model name
    parallel: bool = False  # Whether operators run in parallel

    def to_dict(self) -> Dict[str, Any]:
        """Convert gene to dictionary representation."""
        return {
            "operators": self.operators,
            "provider": self.provider.value if isinstance(self.provider, ModelProvider) else self.provider,
            "model_name": self.model_name,
            "parallel": self.parallel,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkGene":
        """Create gene from dictionary."""
        provider = data["provider"]
        if isinstance(provider, str):
            provider = ModelProvider(provider)

        return cls(
            operators=data["operators"],
            provider=provider,
            model_name=data["model_name"],
            parallel=data.get("parallel", False),
        )


class NetworkChromosome:
    """
    Chromosome encoding a complete NoN network architecture.

    A network is encoded as a sequence of genes, where each gene
    represents a layer with its operators and model configuration.
    """

    def __init__(self, genes: List[NetworkGene]):
        """
        Initialize network chromosome.

        Args:
            genes: List of network genes (layers)
        """
        self.genes = genes

    def __len__(self) -> int:
        """Return number of layers in network."""
        return len(self.genes)

    def to_operator_spec(self) -> List:
        """
        Convert chromosome to NoN operator specification.

        Returns:
            List that can be passed to NoN.from_operators()
        """
        spec = []
        for gene in self.genes:
            if len(gene.operators) == 1 and not gene.parallel:
                # Single sequential operator
                spec.append(gene.operators[0])
            else:
                # Parallel operators
                spec.append(gene.operators)
        return spec

    def to_dict(self) -> Dict[str, Any]:
        """Convert chromosome to dictionary."""
        return {
            "genes": [gene.to_dict() for gene in self.genes],
            "num_layers": len(self.genes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkChromosome":
        """Create chromosome from dictionary."""
        genes = [NetworkGene.from_dict(g) for g in data["genes"]]
        return cls(genes)

    def copy(self) -> "NetworkChromosome":
        """Create a deep copy of this chromosome."""
        genes_copy = [
            NetworkGene(
                operators=gene.operators.copy(),
                provider=gene.provider,
                model_name=gene.model_name,
                parallel=gene.parallel,
            )
            for gene in self.genes
        ]
        return NetworkChromosome(genes_copy)


def random_network_chromosome(
    min_layers: int = 2,
    max_layers: int = 5,
    allow_parallel: bool = True,
) -> NetworkChromosome:
    """
    Generate a random network chromosome.

    Args:
        min_layers: Minimum number of layers
        max_layers: Maximum number of layers
        allow_parallel: Whether to allow parallel operator layers

    Returns:
        Random NetworkChromosome
    """
    num_layers = random.randint(min_layers, max_layers)
    genes = []

    for _ in range(num_layers):
        # Decide if this layer is parallel
        is_parallel = allow_parallel and random.random() < 0.3

        if is_parallel:
            # Create parallel layer with 2-3 operators
            num_parallel = random.randint(2, 3)
            operators = random.sample(AVAILABLE_OPERATORS, num_parallel)
        else:
            # Single operator
            operators = [random.choice(AVAILABLE_OPERATORS)]

        # Choose model configuration
        provider = random.choice(AVAILABLE_PROVIDERS)
        model_name = random.choice(MODEL_CONFIGS[provider])

        gene = NetworkGene(
            operators=operators,
            provider=provider,
            model_name=model_name,
            parallel=is_parallel,
        )
        genes.append(gene)

    return NetworkChromosome(genes)


def mutate_network(
    chromosome: NetworkChromosome,
    mutation_rate: float = 0.1,
) -> NetworkChromosome:
    """
    Mutate a network chromosome.

    Mutations can:
    - Change an operator in a layer
    - Change the model/provider
    - Add a new layer
    - Remove a layer
    - Toggle parallel execution

    Args:
        chromosome: Network chromosome to mutate
        mutation_rate: Probability of mutation per gene

    Returns:
        Mutated chromosome (new copy)
    """
    mutated = chromosome.copy()

    for i, gene in enumerate(mutated.genes):
        if random.random() < mutation_rate:
            mutation_type = random.choice([
                "change_operator",
                "change_model",
                "change_provider",
                "toggle_parallel",
            ])

            if mutation_type == "change_operator":
                # Replace one operator with another
                if gene.operators:
                    idx = random.randint(0, len(gene.operators) - 1)
                    gene.operators[idx] = random.choice(AVAILABLE_OPERATORS)

            elif mutation_type == "change_model":
                # Change to different model from same provider
                gene.model_name = random.choice(MODEL_CONFIGS[gene.provider])

            elif mutation_type == "change_provider":
                # Switch to different provider
                gene.provider = random.choice(AVAILABLE_PROVIDERS)
                gene.model_name = random.choice(MODEL_CONFIGS[gene.provider])

            elif mutation_type == "toggle_parallel":
                # Toggle parallel execution
                if len(gene.operators) == 1 and not gene.parallel:
                    # Add another operator for parallel execution
                    gene.operators.append(random.choice(AVAILABLE_OPERATORS))
                    gene.parallel = True
                elif gene.parallel and len(gene.operators) > 1:
                    # Remove parallel, keep first operator
                    gene.operators = [gene.operators[0]]
                    gene.parallel = False

    # Structural mutations (add/remove layers)
    if random.random() < mutation_rate / 2:
        if random.random() < 0.5 and len(mutated.genes) > 1:
            # Remove a random layer
            idx = random.randint(0, len(mutated.genes) - 1)
            mutated.genes.pop(idx)
        else:
            # Add a new layer at random position
            provider = random.choice(AVAILABLE_PROVIDERS)
            new_gene = NetworkGene(
                operators=[random.choice(AVAILABLE_OPERATORS)],
                provider=provider,
                model_name=random.choice(MODEL_CONFIGS[provider]),
                parallel=False,
            )
            idx = random.randint(0, len(mutated.genes))
            mutated.genes.insert(idx, new_gene)

    return mutated


def crossover_networks(
    parent1: NetworkChromosome,
    parent2: NetworkChromosome,
) -> tuple[NetworkChromosome, NetworkChromosome]:
    """
    Perform crossover between two network chromosomes.

    Uses single-point crossover on the gene sequence.

    Args:
        parent1: First parent chromosome
        parent2: Second parent chromosome

    Returns:
        Tuple of two offspring chromosomes
    """
    # Copy parents
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    # Single-point crossover
    if len(parent1.genes) > 1 and len(parent2.genes) > 1:
        # Choose crossover point
        point1 = random.randint(1, len(parent1.genes) - 1)
        point2 = random.randint(1, len(parent2.genes) - 1)

        # Create offspring
        offspring1.genes = parent1.genes[:point1] + parent2.genes[point2:]
        offspring2.genes = parent2.genes[:point2] + parent1.genes[point1:]

    return offspring1, offspring2


def describe_network(chromosome: NetworkChromosome) -> str:
    """
    Generate human-readable description of network architecture.

    Args:
        chromosome: Network chromosome to describe

    Returns:
        String description
    """
    lines = [f"Network with {len(chromosome)} layers:"]

    for i, gene in enumerate(chromosome.genes):
        if gene.parallel:
            ops_str = f"[{', '.join(gene.operators)}]"
        else:
            ops_str = gene.operators[0]

        lines.append(
            f"  Layer {i+1}: {ops_str} ({gene.provider.value}/{gene.model_name})"
        )

    return "\n".join(lines)
