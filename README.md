# NoN (Network of Networks)

A powerful Python framework for building compound AI systems using networks of interconnected language model operators.

## 🚀 Overview

NoN enables you to create sophisticated AI workflows by composing multiple language model calls into networks of operators. Each network consists of layers of nodes that can execute in parallel, with intelligent request scheduling, comprehensive observability, and production-ready infrastructure.

## ✨ Key Features

- **🧩 Modular Operators**: 10+ built-in operators (Transform, Generate, Classify, Extract, etc.)
- **🔗 Network Composition**: Chain operators into sequential layers and parallel nodes
- **⚡ Async Execution**: High-performance concurrent execution with asyncio
- **🎯 Smart Scheduling**: Request scheduling with rate limiting and queue management
- **📊 Full Observability**: Distributed tracing, structured logging, and metrics collection
- **🔄 Multi-Provider Support**: OpenAI, Anthropic, Google Gemini with automatic fallbacks
- **🛡️ Robust Error Handling**: Multiple error policies and retry strategies
- **📈 Parallel Scaling**: Node multiplication operator for easy parallel processing
- **💾 Database-Ready**: All telemetry data exportable for database storage

## 🏗️ Architecture

```
Network (NoN)
├── Layer 1: [Node A] ──┐
├── Layer 2: [Node B, Node C, Node D] ──┐  (Parallel Execution)
└── Layer 3: [Node E] ──┘
```

- **Operators**: Functional units that transform content (transform, generate, classify, etc.)
- **Nodes**: Configured instances of operators with model settings and context
- **Layers**: Collections of nodes that execute in parallel
- **Networks**: Sequential chains of layers that process data through a pipeline

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd non

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## 🔧 Quick Start

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
        model_name="claude-3-haiku-20240307",
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
    model_name="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=150,
    top_p=0.9
)
```

## 📊 Observability

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

## ⚡ Request Scheduling

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

## 🧩 Available Operators

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

## 🔄 Error Handling

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

## 📁 Project Structure

```
nons/
├── core/           # Core system components
│   ├── node.py     # Node implementation
│   ├── layer.py    # Layer with parallel execution
│   ├── network.py  # Network orchestration
│   ├── scheduler.py # Request scheduling
│   └── types.py    # Type definitions
├── operators/      # Operator implementations
│   ├── base.py     # Base operators
│   └── registry.py # Operator registry
├── observability/  # Monitoring and tracing
│   ├── tracing.py  # Distributed tracing
│   ├── logging.py  # Structured logging
│   ├── metrics.py  # Metrics collection
│   └── integration.py # Unified observability
├── utils/          # Utilities and helpers
│   └── providers.py # LLM provider adapters
└── examples/       # Usage examples and demos
```

## 🧪 Examples

Explore the `examples/` directory for comprehensive demonstrations:

- `basic_usage.py` - Simple network creation and execution
- `observability_demo.py` - Full observability showcase
- `scheduler_demo.py` - Request scheduling and rate limiting
- `multiplication_demo.py` - Node multiplication and parallel processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📖 Documentation: See `/docs` directory
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions

## 🌟 Acknowledgments

Built with modern Python async/await patterns and production-ready observability infrastructure for compound AI systems.

---

**NoN**: Enabling the next generation of compound AI applications through composable network architectures.