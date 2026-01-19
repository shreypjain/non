# NoN: Network of Networks

NoN is a framework for building the next generation of compound AI systems. As AI systems move away from single-shot inference and toward orchestrated, multi-step reasoning across diverse model modalities and providers, the limits of individual model inference are becoming clear. The future of engineering frontier AI systems lies in the composition and coordination of many specialized model calls, each tuned for a specific role within a larger, adaptive workflow.

NoNs are designed to make this kind of systems engineering tractable, scalable, and robust. Heavily inspired by Jared Quincy Davis, Ember, and (Are LLM Calls All You Need?)[https://arxiv.org/abs/2403.02419]

At its core, NoN provides the abstractions and infrastructure to construct, schedule, and observe networks of "operators". Each a distinct, reusable function with functional AI model use cases. These operators are composed into nodes, layers, and ultimately networks, enabling parallel execution, intelligent routing, and fine-grained control over model "parameters" and prompt context. The result is a platform for experimenting with and deploying complex, high-performance AI workflows that can outperform monolithic models on challenging, real-world tasks where both cost, performance, and interpretability are the most important levers to have control over.

---

## Vocabulary

- **NoN**: Network of Networks – a flexible, directed network of interconnected nodes passing information forward, generalizing the concept of neural networks for compound AI systems.
- **Forward Pass**: A single execution of all sequential and layered model calls through the network, analogous to a forward pass in deep learning.
- **Operator**: A function encapsulating a specific AI model operation, with fixed inputs and outputs, registered for reuse and composition.
- **Node**: An instance of an operator with specific configuration (prompt context, model, hyperparameters), positioned within a NoN.
- **Layer**: An array of nodes executed in parallel within a network.
- **Registry**: The dynamic collection of all available operators, supporting runtime registration and reuse.

## Key Features

- **Modular Operators**: 14+ built-in operators (Transform, Generate, Classify, Extract, etc.) plus heuristic-based ensemble operators
- **Network Composition**: Chain operators into sequential layers and parallel nodes
- **Async Execution**: Improve performance using concurrent execution with asyncio
- **Smart Scheduling**: Request scheduling with rate limiting and queue management across model providers
- **Full Observability**: Full network tracing, structured logging, and metrics collection
- **Multi-Provider Support**: OpenAI, Anthropic, Google Gemini with automatic fallbacks
- **Robust Error Handling**: Multiple error policies and retry strategies
- **Parallel Scaling**: Node multiplication operator for easy parallel processing
- **Database-Ready**: All telemetry data exportable for database storage

## High Level Architecture

```
Network (NoN)
├── Layer 1: [Node A] ──┐
├── Layer 2: [Node B, Node C, Node D] ──┐  (Parallel Execution)
└── Layer 3: [Node E] ──┘
```

- **Operators**: Functional units that transform content (Transform, Generate, Classify, Extract, etc.)
- **Nodes**: Configured instances of operators with model settings and context
- **Layers**: Collections of nodes that execute in parallel
- **Networks**: Sequential chains of layers that process data through a pipeline

## Installation

### Prerequisites
- Python 3.9+ (required for Google GenAI support)
- API keys for at least one LLM provider

### Quick Install

```bash
# Clone the repository
git clone https://github.com/shreypjain/non
cd non

# Install with uv (recommended - fastest)
uv sync

# Alternative: Install with pip – not recommended
pip install -e .

# Verify installation
uv run python -c "import nons; print('✅ NoN installed successfully!')"
```

### API Key Setup

```bash
# Set your API keys (choose one or more)
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"

# Optional: Add to your shell profile for persistence
echo 'export ANTHROPIC_API_KEY="your-anthropic-api-key"' >> ~/.bashrc
```

### Installation Testing

```bash
# Test your installation
cd examples
uv run python basic_usage.py
```

## Quick Start

### Basic Network Creation

```python
import asyncio
from nons.core.network import NoN
from nons.core.types import ModelProvider

# Create a simple network
async def main():
    # Method 1: From operator specifications
    network = NoN.from_operators([
        'transform',                    # Single operator
        ['classify', 'extract'],        # Parallel operators
        'generate'                      # Final operator
    ])

    # Execute the network
    result = await network.forward("Analyze this text about renewable energy...")
    print(result)

asyncio.run(main())
```

### Advanced Network with Node Multiplication

```python
from nons.core.node import Node
from nons.core.types import ModelConfig, ModelProvider

async def advanced_example():
    # Create a base node
    generator = Node('generate', model_config=ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-haiku-4-5",
        max_tokens=100
    ))

    # Create multiple parallel instances
    parallel_generators = generator * 3  # 3 parallel nodes

    # Build network with multiplied nodes
    network = NoN.from_operators([
        'transform',           # Preprocessing
        parallel_generators,   # Parallel generation
        'condense'            # Aggregation
    ])

    result = await network.forward("Create multiple perspectives on AI safety")
    print(result)

asyncio.run(advanced_example())
```

## 🎛️ Configuration

### Environment Variables

```bash
# LLM Provider API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"

# Optional: Custom configurations
export NON_LOG_LEVEL="INFO"
export NON_ENABLE_TRACING="true"
```

### Model Configuration

```python
from nons.core.types import ModelConfig, ModelProvider

config = ModelConfig(
    provider=ModelProvider.GOOGLE,
    model_name="gemini-3-flash",
    temperature=0.7,
    max_tokens=150,
    top_p=0.9
)
```

## Observability

NoN provides comprehensive observability out of the box:

```python
from nons.observability.integration import get_observability

# Get observability data
obs = get_observability()
data = obs.export_all_data()

print(f"Spans: {len(data['spans'])}")
print(f"Logs: {len(data['logs'])}")
print(f"Metrics: {len(data['metrics'])}")

# Get trace summary
trace_summary = obs.get_trace_summary(trace_id)
print(f"Success rate: {trace_summary['success']}")
print(f"Total cost: ${trace_summary['total_cost_usd']:.6f}")
```

## Request Scheduling

Built-in intelligent request scheduling with rate limiting:

```python
from nons.core.scheduler import configure_scheduler, RateLimitConfig, QueueStrategy
from nons.core.types import ModelProvider

# Configure custom rate limits
scheduler = configure_scheduler(
    rate_limits={
        ModelProvider.OPENAI: RateLimitConfig(
            requests_per_minute=100,
            max_concurrent=10
        ),
        ModelProvider.ANTHROPIC: RateLimitConfig(
            requests_per_minute=200,
            max_concurrent=15
        )
    },
    queue_strategy=QueueStrategy.PRIORITY
)
```

## Available Operators

The bottom four operators (`pack_candidates`, `extract_winners`, `majority`, and `select_by_id`) are all non deterministic

| Operator | Description | Example Use Case |
|----------|-------------|------------------|
| `transform` | Transform content format/style | Convert JSON to text |
| `generate` | Generate new content | Create stories, responses |
| `classify` | Categorize content | Sentiment, topic classification |
| `extract` | Extract specific information | Pull dates, names, facts |
| `condense` | Summarize/compress content | Create summaries |
| `expand` | Add detail/context | Elaborate on points |
| `compare` | Find similarities/differences | Compare documents |
| `validate` | Check correctness/quality | Fact-checking |
| `route` | Determine next action | Workflow routing |
| `synthesize` | Combine multiple inputs | Merge perspectives |
| `pack_candidates` | Structure data for ensemble voting | Pack LLM outputs for consensus |
| `extract_winners` | Select top candidates by score | Best-of-N sampling |
| `majority` | Perform voting on candidates | Ensemble decision making |
| `select_by_id` | Select specific candidates | Custom filtering logic |

## Advanced Compound AI Patterns

NoN enables sophisticated compound AI patterns that dramatically improve output quality, reliability, and reasoning capabilities beyond single-shot inference.

### Best-of-N Sampling

Generate multiple candidates and select the highest quality response.

```python
# Create generator with high diversity
generator = Node('generate', model_config=ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-sonnet-4-5",
    temperature=0.9  # High diversity
))

# Generate 5 candidates in parallel
generators = generator * 5

# Score each candidate
evaluator = Node('classify', context="Score 1-10 on quality. Return only number.")
evaluators = evaluator * 5

# Select best
selector = Node('extract', context="Return the highest-scored response")

network = NoN.from_operators([generators, evaluators, selector])
result = await network.forward("Write a creative story premise")
```

### Verifier-Prover Structure

Iteratively generate and verify solutions until correctness is achieved.

```python
async def solve_with_verification(problem: str, max_attempts: int = 3):
    prover = Node('generate', context="Solve step-by-step with detailed reasoning")
    verifier = Node('validate', context="Check for errors. Reply: VERIFIED or ERROR: [details]")
    refiner = Node('transform', context="Fix solution based on feedback")

    current_input = problem

    for attempt in range(max_attempts):
        # Generate solution
        solution = await NoN.from_operators([prover]).forward(current_input)

        # Verify correctness
        verification = await NoN.from_operators([verifier]).forward(
            f"Problem: {problem}\nSolution: {solution}"
        )

        if "VERIFIED" in verification:
            return solution

        # Refine for next iteration
        current_input = f"{problem}\nPrevious: {solution}\nFeedback: {verification}"

    return solution
```

### Majority Voting Ensembles

Combine multiple independent predictions to reduce bias and increase reliability.

```python
# Multi-provider ensemble for maximum diversity
claude = Node('classify', model_config=ModelConfig(
    provider=ModelProvider.ANTHROPIC, model_name="claude-3-5-haiku-20241022"
))
gpt = Node('classify', model_config=ModelConfig(
    provider=ModelProvider.OPENAI, model_name="gpt-4o-mini"
))
gemini = Node('classify', model_config=ModelConfig(
    provider=ModelProvider.GOOGLE, model_name="gemini-2.0-flash"
))

# Using deterministic operators for consensus
network = NoN.from_operators([
    [claude, gpt, gemini],  # Parallel voting
    'pack_candidates',      # Structure responses
    'majority'              # Find consensus
])

result = await network.forward("Classify sentiment: The product is okay but shipping was slow")
```

### Pattern Selection Guide

| Pattern | Best For |
|---------|----------|
| **Best-of-N** | Creative tasks, open-ended generation |
| **Verifier-Prover** | Math, code, logical reasoning |
| **Majority Voting** | Classification, decision making |

### Hybrid Patterns

Combine patterns for maximum effectiveness:

```python
# Best-of-N + Verification: Generate multiple solutions, verify each, select best verified
generators = Node('generate', temperature=0.8) * 3
verifiers = Node('validate', context="Check correctness, score 1-10") * 3
selector = Node('extract', context="Return highest-scored verified solution")

network = NoN.from_operators([generators, verifiers, selector])
```

## Error Handling

Multiple error handling strategies:

```python
from nons.core.types import ErrorPolicy, LayerConfig

# Configure error handling
layer_config = LayerConfig(
    error_policy=ErrorPolicy.RETRY_WITH_BACKOFF,
    max_retries=3,
    min_success_threshold=0.7
)

network = NoN.from_operators(
    ['transform', 'generate', 'validate'],
    layer_config=layer_config
)
```

## Project Structure

```
nons/
├── README.md              # Core package guide with examples
├── core/                     # Core system components
│   ├── README.md         # Node, Layer, Network examples
│   ├── node.py              # Node implementation
│   ├── layer.py             # Layer with parallel execution
│   ├── network.py           # Network orchestration
│   ├── scheduler.py         # Request scheduling
│   └── types.py             # Type definitions
├── operators/               # Operator implementations
│   ├── README.md         # 14 operators + deterministic + custom operator guide
│   ├── base.py              # Base operators
│   ├── deterministic.py     # Deterministic ensemble operators
│   └── registry.py          # Operator registry
├── observability/           # Monitoring and tracing
│   ├── README.md         # Tracing, logging, metrics guide
│   ├── tracing.py           # Distributed tracing
│   ├── logging.py           # Structured logging
│   ├── metrics.py           # Metrics collection
│   └── integration.py       # Unified observability
├── utils/                   # Utilities and helpers
│   ├── README.md         # Provider integration guide
│   └── providers.py         # LLM provider adapters
├── docs/                    # Comprehensive documentation
│   ├── getting-started.md   # Step-by-step tutorial
│   ├── architecture.md      # Technical architecture
│   ├── api-reference.md     # Complete API docs
│   ├── advanced-features.md # Advanced patterns
│   └── faq.md              # Common questions
├── examples/                # Usage examples and demos
├── tests/                   # Comprehensive test suite (235+ tests)
└── LICENSE               # MIT License
```

### Documentation Guide

Each module has its own README with progressively complex examples:

1. **[`nons/README.md`](nons/README.md)** - Start here! 7 progressive examples from "Hello World" to full observability
2. **[`nons/core/README.md`](nons/core/README.md)** - Core components with hands-on examples
3. **[`nons/operators/README.md`](nons/operators/README.md)** - All 10 operators + custom operator creation
4. **[`nons/observability/README.md`](nons/observability/README.md)** - Monitoring, tracing, and database exports
5. **[`nons/utils/README.md`](nons/utils/README.md)** - Multi-provider strategies and configurations

### 📚 Complete Documentation

- **[Getting Started](docs/getting-started.md)** - Installation to first network in 5 minutes
- **[Architecture Guide](docs/architecture.md)** - Deep dive into system design
- **[API Reference](docs/api-reference.md)** - Complete method documentation
- **[Advanced Features](docs/advanced-features.md)** - Production patterns and custom components
- **[FAQ](docs/faq.md)** - Common questions and troubleshooting

## Examples

Explore the `examples/` directory for comprehensive demonstrations:

- `basic_usage.py` - Simple network creation and execution
- `observability_demo.py` - Full observability showcase
- `scheduler_demo.py` - Request scheduling and rate limiting
- `multiplication_demo.py` - Node multiplication and parallel processing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: See `/docs` directory
- Issues: GitHub Issues
- Discussions: GitHub Discussions

## Acknowledgments

SWYgeW91IGZpbmQgYmVhdXR5IGluIHNwcmV6emF0dXJhIGVtYWlsIG1lOiBzaHJleUBsdW5jaGxpbmVwYXJ0bmVycy5jb20=

---

**NoN**: Enabling the next generation of compound AI applications through composable network architectures.