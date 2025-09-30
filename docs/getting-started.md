# Getting Started with NoN

This guide will help you get up and running with NoN (Network of Networks) for building compound AI systems.

## Installation

### Prerequisites

- Python 3.9+
- API keys for at least one LLM provider (OpenAI, Anthropic, or Google)

### Install NoN

```bash
# Clone the repository
git clone <repository-url>
cd non

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Set up API Keys

```bash
# Set your API keys as environment variables
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

## Core Concepts

### 1. Operators

Operators are the building blocks of NoN networks. They represent functional transformations:

```python
# Available operators
operators = [
    'transform',   # Transform content format/style
    'generate',    # Generate new content
    'classify',    # Categorize content
    'extract',     # Extract specific information
    'condense',    # Summarize content
    'expand',      # Add detail/context
    'compare',     # Find similarities/differences
    'validate',    # Check correctness
    'route',       # Determine next action
    'synthesize'   # Combine multiple inputs
]
```

### 2. Nodes

Nodes are configured instances of operators with specific model settings:

```python
from nons.core.node import Node
from nons.core.types import ModelConfig, ModelProvider

# Create a node
node = Node(
    operator_name='generate',
    model_config=ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        temperature=0.7,
        max_tokens=100
    ),
    additional_prompt_context="You are a helpful assistant."
)
```

### 3. Layers

Layers contain multiple nodes that execute in parallel:

```python
from nons.core.layer import Layer

# Create nodes
node1 = Node('classify')
node2 = Node('extract')
node3 = Node('generate')

# Create a layer with parallel execution
layer = Layer([node1, node2, node3])

# Execute the layer
result = await layer.execute_parallel("Input text")
```

### 4. Networks

Networks chain layers together for sequential processing:

```python
from nons.core.network import NoN

# Method 1: From operator specifications
network = NoN.from_operators([
    'transform',                    # Layer 1: Single node
    ['classify', 'extract'],        # Layer 2: Two parallel nodes
    'generate'                      # Layer 3: Single node
])

# Method 2: From pre-built layers
network = NoN([layer1, layer2, layer3])

# Execute the network
result = await network.forward("Your input text here")
```

## Your First Network

Let's build a simple content analysis network:

```python
import asyncio
from nons.core.network import NoN

async def analyze_content():
    # Create a content analysis pipeline
    network = NoN.from_operators([
        'transform',    # Clean and normalize input
        'classify',     # Classify the content
        'extract',      # Extract key information
        'generate'      # Generate a summary
    ])

    # Analyze some content
    content = """
    Renewable energy sources like solar and wind power are becoming
    increasingly cost-effective and efficient. Many countries are
    investing heavily in these technologies to reduce carbon emissions.
    """

    result = await network.forward(content)
    print("Analysis Result:", result)

# Run the example
asyncio.run(analyze_content())
```

## Working with Multiple Providers

NoN automatically handles multiple LLM providers:

```python
from nons.core.node import Node
from nons.core.types import ModelConfig, ModelProvider

# Create nodes with different providers
openai_node = Node('generate', model_config=ModelConfig(
    provider=ModelProvider.OPENAI,
    model_name="gpt-4",
    max_tokens=150
))

anthropic_node = Node('generate', model_config=ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-3-haiku-20240307",
    max_tokens=150
))

google_node = Node('generate', model_config=ModelConfig(
    provider=ModelProvider.GOOGLE,
    model_name="gemini-2.5-flash",
    max_tokens=150
))

# Use in a network with mixed providers
network = NoN.from_operators([
    'transform',
    [openai_node, anthropic_node, google_node],  # Parallel execution
    'condense'
])
```

## Error Handling

Configure error handling strategies:

```python
from nons.core.types import ErrorPolicy, LayerConfig

# Configure retry behavior
layer_config = LayerConfig(
    error_policy=ErrorPolicy.RETRY_WITH_BACKOFF,
    max_retries=3,
    timeout_seconds=30,
    min_success_threshold=0.7
)

# Apply to network creation
network = NoN.from_operators(
    ['transform', 'generate', 'validate'],
    layer_config=layer_config
)
```

## Observability

Monitor your networks with built-in observability:

```python
from nons.observability.integration import get_observability

async def monitored_execution():
    # Execute your network
    result = await network.forward("Input text")

    # Get observability data
    obs = get_observability()
    stats = obs.get_stats()

    print(f"Active spans: {stats['tracing']['active_spans']}")
    print(f"Total logs: {stats['logging']['total_entries']}")
    print(f"Metrics points: {stats['metrics']['total_points']}")

    # Export for database storage
    all_data = obs.export_all_data()
    # Process spans, logs, and metrics...
```

## Node Multiplication

Scale processing with the multiplication operator:

```python
# Create multiple parallel instances
base_node = Node('generate')
parallel_nodes = base_node * 5  # Creates 5 parallel instances

# Use in networks
network = NoN.from_operators([
    'transform',
    parallel_nodes,    # 5 parallel generators
    'condense'
])
```

## Best Practices

### 1. Model Selection

- Use faster models (like Claude Haiku or GPT-3.5) for simple operations
- Use more powerful models (like Claude Opus or GPT-4) for complex reasoning
- Leverage parallel execution for redundancy and speed

### 2. Error Handling

- Set appropriate retry policies for your use case
- Use `SKIP_AND_CONTINUE` for non-critical operations
- Use `FAIL_FAST` for critical validations

### 3. Performance

- Use node multiplication for CPU-bound parallel processing
- Configure appropriate rate limits for your API quotas
- Monitor execution times and optimize bottlenecks

### 4. Cost Management

- Track costs with built-in metrics collection
- Use cheaper models for preprocessing steps
- Set reasonable token limits

## Next Steps

- Read the [Architecture Guide](architecture.md) for deeper understanding
- Explore [Advanced Features](advanced-features.md) for complex use cases
- Check out [Examples](../examples/) for real-world implementations
- Review [API Reference](api-reference.md) for detailed documentation

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your API keys are set correctly
2. **Import Errors**: Make sure you've imported operators with `import nons.operators.base`
3. **Async Errors**: Always use `await` with async operations
4. **Rate Limiting**: Configure appropriate rate limits for your API quotas

### Getting Help

- Check the [FAQ](faq.md)
- Review [Examples](../examples/)
- Open an issue on GitHub