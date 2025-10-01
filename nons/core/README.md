# NoN Core Components

The core module contains the fundamental building blocks of the NoN system: Nodes, Layers, Networks, and supporting infrastructure.

## 🧩 Components

### Node (`node.py`)
Individual execution units that wrap operators with configuration.

**Simple Example:**
```python
from nons.core.node import Node
from nons.core.types import ModelConfig, ModelProvider

# Create a basic node
node = Node('generate', model_config=ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-3-haiku-20240307"
))

# Execute the node
result = await node.execute(prompt="Write a haiku about coding")
print(result)
```

**Advanced Example with Multiplication:**
```python
# Create multiple parallel instances
base_node = Node('generate')
parallel_nodes = base_node * 5  # Creates 5 independent copies

# Use in a layer
from nons.core.layer import create_parallel_layer
layer = create_parallel_layer(parallel_nodes)
```

### Layer (`layer.py`)
Manages parallel execution of multiple nodes using asyncio.

**Basic Parallel Execution:**
```python
from nons.core.layer import Layer
from nons.core.node import Node

# Create nodes for parallel execution
nodes = [
    Node('classify'),
    Node('extract'),
    Node('condense')
]

layer = Layer(nodes)

# Execute all nodes in parallel
result = await layer.execute_parallel("Analyze this text simultaneously")
print(f"Parallel results: {result.outputs}")
print(f"Success rate: {result.success_rate}")
```

**Error Handling Configuration:**
```python
from nons.core.types import LayerConfig, ErrorPolicy

# Configure resilient layer
resilient_config = LayerConfig(
    error_policy=ErrorPolicy.RETRY_WITH_BACKOFF,
    max_retries=3,
    min_success_threshold=0.6  # Continue if 60% succeed
)

layer = Layer(nodes, layer_config=resilient_config)
```

### Network (`network.py`)
Orchestrates sequential execution of layers.

**Sequential Pipeline:**
```python
from nons.core.network import NoN

# Create a processing pipeline
network = NoN.from_operators([
    'transform',                    # Layer 1: Preprocessing
    ['classify', 'extract'],        # Layer 2: Parallel analysis
    'synthesize'                    # Layer 3: Combine results
])

result = await network.forward("Input data")
```

**Complex Network with Custom Nodes:**
```python
# Mix operator strings with custom nodes
custom_node = Node('generate', model_config=ModelConfig(
    provider=ModelProvider.OPENAI,
    model_name="gpt-4",
    temperature=0.1
))

network = NoN.from_operators([
    'transform',
    [custom_node, 'classify'],      # Mix custom and string specs
    custom_node * 3,                # 3 parallel custom nodes
    'validate'
])
```

### Types (`types.py`)
Core type definitions and configurations.

**Model Configuration:**
```python
from nons.core.types import ModelConfig, ModelProvider

# Configure different models
fast_config = ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-3-haiku-20240307",
    temperature=0.1,
    max_tokens=50
)

powerful_config = ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-3-opus-20240229",
    temperature=0.7,
    max_tokens=500
)
```

**Error Policies:**
```python
from nons.core.types import ErrorPolicy, LayerConfig

# Different error handling strategies
fail_fast = LayerConfig(error_policy=ErrorPolicy.FAIL_FAST)
retry_config = LayerConfig(
    error_policy=ErrorPolicy.RETRY_WITH_BACKOFF,
    max_retries=5,
    retry_delay_seconds=2.0
)
skip_errors = LayerConfig(error_policy=ErrorPolicy.SKIP_AND_CONTINUE)
```

### Scheduler (`scheduler.py`)
Intelligent request scheduling with rate limiting.

**Basic Scheduling:**
```python
from nons.core.scheduler import configure_scheduler, RateLimitConfig
from nons.core.types import ModelProvider

# Configure rate limits
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
    }
)
```

## 🔄 Complete Workflow Example

Here's how all core components work together:

```python
import asyncio
from nons.core.network import NoN
from nons.core.node import Node
from nons.core.types import ModelConfig, ModelProvider, LayerConfig, ErrorPolicy
from nons.core.scheduler import configure_scheduler, start_scheduler, stop_scheduler
import nons.operators.base

async def complete_workflow():
    # Configure scheduler
    configure_scheduler()
    await start_scheduler()

    try:
        # Create custom nodes
        fast_node = Node('generate', ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-haiku-20240307"
        ))

        # Create resilient layer config
        layer_config = LayerConfig(
            error_policy=ErrorPolicy.RETRY_WITH_BACKOFF,
            max_retries=2
        )

        # Build complex network
        network = NoN.from_operators([
            'transform',                # Preprocessing
            fast_node * 3,              # 3 parallel fast processors
            ['classify', 'extract'],    # Parallel analysis
            'synthesize'                # Final synthesis
        ], layer_config=layer_config)

        # Execute with comprehensive processing
        result = await network.forward(
            "Analyze the impact of artificial intelligence on modern education systems"
        )

        print(f"Complete Analysis: {result}")

        # Check execution metrics
        total_cost = sum(
            sum(node.get_total_cost() for node in layer.nodes)
            for layer in network.layers
        )
        print(f"Total Cost: ${total_cost:.6f}")

    finally:
        await stop_scheduler()

asyncio.run(complete_workflow())
```

## 🎯 Key Features

- **Async-First**: All operations use async/await for maximum concurrency
- **Composable**: Mix and match components freely
- **Observable**: Built-in tracing, logging, and metrics
- **Fault-Tolerant**: Multiple error handling strategies
- **Scalable**: Node multiplication and parallel execution
- **Cost-Aware**: Automatic cost and token tracking

## 🔗 Related Modules

- [`../operators/`](../operators/README.md) - Operator implementations
- [`../observability/`](../observability/README.md) - Monitoring and tracing
- [`../utils/`](../utils/README.md) - Provider adapters and utilities