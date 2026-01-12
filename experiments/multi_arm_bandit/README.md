# Network Architecture Evolution with Genetic Algorithms

This experiment uses genetic algorithms to evolve optimal NoN network architectures for reasoning tasks. Rather than manually designing network structures, we let evolution discover effective operator compositions and model configurations.

## Problem Overview

Designing compound AI systems requires selecting:
- Which operators to use (transform, generate, classify, etc.)
- How to arrange them in layers (sequential vs parallel)
- Which models to use for each operation
- How many layers the network should have

This is a combinatorial optimization problem perfect for genetic algorithms.

## Approach

### Network Encoding

Each network architecture is encoded as a chromosome containing genes:

```python
NetworkGene:
  - operators: ["transform", "generate"]  # Operations in this layer
  - provider: ModelProvider.ANTHROPIC     # Model provider
  - model_name: "claude-sonnet-4.5"       # Specific model
  - parallel: True                        # Parallel execution
```

A complete network is a sequence of genes representing layers.

### Fitness Function

Networks are evaluated on reasoning task accuracy:

```python
fitness = average_accuracy_across_tasks([
    ARITHMETIC_TASK,    # Basic math problems
    LOGICAL_TASK,       # Logical deduction
    PATTERN_TASK,       # Sequence completion
    COMMONSENSE_TASK,   # Common sense reasoning
    MULTISTEP_TASK,     # Multi-step problems
    GPQA_TASK,          # Graduate-level science questions
])
```

### Genetic Operators

**Mutation** can:
- Change an operator in a layer
- Swap model/provider
- Add or remove layers
- Toggle parallel execution

**Crossover**:
- Single-point crossover on gene sequences
- Combines parent network structures

**Selection**:
- Tournament selection based on task accuracy

## Implementation

### Core Modules

**network_encoding.py**: Network chromosome representation
- `NetworkGene`: Single layer configuration
- `NetworkChromosome`: Complete network encoding
- `random_network_chromosome()`: Generate random architectures
- `mutate_network()`: Apply mutations
- `crossover_networks()`: Combine parent networks

**reasoning_tasks.py**: Benchmark reasoning problems
- `ARITHMETIC_TASK`: Basic arithmetic word problems
- `LOGICAL_TASK`: Logical deduction
- `PATTERN_TASK`: Number sequences
- `COMMONSENSE_TASK`: Common sense reasoning
- `MULTISTEP_TASK`: Multi-step reasoning
- `GPQA_TASK`: Graduate-level science questions

Each task has examples with expected outputs and evaluation functions.

**network_fitness.py**: Fitness evaluation
- `evaluate_network_on_task()`: Test network on one task
- `evaluate_network_fitness()`: Aggregate across tasks
- `build_network_from_chromosome()`: Construct NoN from chromosome

**network_evolution.py**: Genetic algorithm
- `NetworkGeneticAlgorithm`: Main evolution loop
- Tournament selection
- Elitism (preserve best networks)
- Population diversity tracking

## Usage

### Basic Example

```python
import asyncio
from multi_arm_bandit.network_evolution import (
    NetworkGeneticAlgorithm,
    NetworkGAConfig,
)
from multi_arm_bandit.reasoning_tasks import (
    ARITHMETIC_TASK,
    LOGICAL_TASK,
    GPQA_TASK,
)

async def evolve_network():
    # Configure evolution
    config = NetworkGAConfig(
        population_size=20,
        num_generations=10,
        mutation_rate=0.2,
        min_layers=2,
        max_layers=5,
    )

    # Run evolution
    ga = NetworkGeneticAlgorithm(
        tasks=[ARITHMETIC_TASK, LOGICAL_TASK, GPQA_TASK],
        config=config,
    )

    result = await ga.run()

    print(f"Best fitness: {result.best_fitness:.2%}")
    print(result.best_chromosome.to_operator_spec())

asyncio.run(evolve_network())
```

### Run Demo

```bash
cd experiments/multi_arm_bandit
uv run python example_network_evolution.py
```

### Available Models (November 2025)

The evolution can select from:

**Anthropic:**
- claude-haiku-4.5 (fast, efficient)
- claude-sonnet-4.5 (balanced)
- claude-opus-4.1 (most capable)

**OpenAI:**
- gpt-4o (multimodal)
- gpt-5.1 (latest foundation model)
- gpt-5.1-codex-max (coding specialist)

**Google:**
- gemini-2.5-flash (fast)
- gemini-2.5-pro (advanced thinking)
- gemini-3-pro (latest multimodal)
- gemini-3-deep-think (complex problems)

## Network Architecture Search Space

The GA explores:

**Operators (9 types):**
- transform, generate, classify, extract
- condense, expand, compare, validate, synthesize

**Layer structures:**
- Sequential: Single operator per layer
- Parallel: Multiple operators executing concurrently

**Depth:**
- 2-5 layers (configurable)

**Model selection:**
- 3 providers × 3-4 models each

**Total search space:** ~10^12 possible architectures

## Reasoning Tasks

### Arithmetic (5 examples)
```
Input: "John has 5 apples, Mary gives him 3 more..."
Expected: "8"
Evaluation: Numeric extraction
```

### Logical (5 examples)
```
Input: "All cats are animals. Fluffy is a cat..."
Expected: "Yes"
Evaluation: Contains match
```

### Pattern (5 examples)
```
Input: "What comes next: 2, 4, 6, 8, ?"
Expected: "10"
Evaluation: Numeric match
```

### Commonsense (5 examples)
```
Input: "If you drop a glass on hard floor..."
Expected: "break"
Evaluation: Contains match
```

### Multistep (5 examples)
```
Input: "Pizza has 8 slices, John eats 2, Mary eats 3..."
Expected: "3"
Evaluation: Numeric extraction
```

### GPQA (Graduate-Level Questions)
```
Input: "In thermodynamics, which law states that energy cannot be created or destroyed?"
Expected: "first law"
Evaluation: Contains match
```

## Future Directions

### Immediate Extensions
1. Layer-specific model configuration
2. More sophisticated crossover (layer-aware)
3. Adaptive mutation rates based on fitness progress
4. Multi-objective optimization (accuracy + cost + latency)

### Advanced Features
1. Hierarchical encoding for deeper networks
2. Context-specific operator selection
3. Co-evolution of prompts and architectures
4. Transfer learning across task domains

### Real-World Benchmarks
1. Full GPQA dataset integration
2. GSM8K math reasoning
3. ARC challenge
4. Domain-specific task suites

## Key Insights

**Why Genetic Algorithms for Network Search?**
- No gradient information needed
- Explores discrete architecture space naturally
- Handles non-differentiable objectives (task accuracy)
- Population diversity prevents local optima
- Interpretable solutions (explicit network structures)

**What We're Optimizing:**
Unlike traditional neural architecture search that optimizes layer connections and neuron counts, we're optimizing:
- Semantic operator composition
- Model-operator matching
- Layer parallelization strategy
- Network depth for reasoning tasks

This is a higher-level architectural search tailored for compound AI systems.

## References

- Genetic Algorithms: Holland (1975), Goldberg (1989)
- Neural Architecture Search: Zoph & Le (2017)
- Compound AI Systems: Berkeley AI Research
- GPQA: Rein et al. (2023) - Graduate-Level Google-Proof Q&A Benchmark
- Reasoning Benchmarks: GSM8K, ARC, DROP
