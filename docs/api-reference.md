# API Reference

Complete reference for all NoN classes, methods, and functions.

## Core Components

### Node

Main execution unit that wraps an operator with configuration.

```python
class Node:
    def __init__(
        self,
        operator_name: str,
        model_config: Optional[ModelConfig] = None,
        additional_prompt_context: str = "",
        node_id: Optional[str] = None
    )
```

#### Parameters

- `operator_name` (str): Name of the registered operator
- `model_config` (ModelConfig, optional): Model configuration
- `additional_prompt_context` (str): Additional context for prompts
- `node_id` (str, optional): Unique identifier (auto-generated if not provided)

#### Methods

##### `async execute(*args, execution_context=None, **kwargs) -> Any`

Execute the node's operator with given inputs.

**Parameters:**
- `*args`: Positional arguments for the operator
- `execution_context` (ExecutionContext, optional): Execution context
- `**kwargs`: Keyword arguments for the operator

**Returns:** Result from operator execution

**Raises:**
- `OperatorError`: If execution fails
- `ValidationError`: If input validation fails

##### `configure_model(provider=None, model_name=None, temperature=None, **kwargs)`

Update model configuration for this node.

**Parameters:**
- `provider` (ModelProvider, optional): Model provider
- `model_name` (str, optional): Model name
- `temperature` (float, optional): Temperature setting
- `**kwargs`: Additional model parameters

##### `clone(new_node_id=None) -> Node`

Create a clone of this node with a new ID.

**Parameters:**
- `new_node_id` (str, optional): ID for the cloned node

**Returns:** Cloned Node instance

##### `__mul__(count: int) -> List[Node]`

Create multiple clones using multiplication operator.

**Parameters:**
- `count` (int): Number of clones to create

**Returns:** List of cloned nodes

**Example:**
```python
node = Node('generate')
parallel_nodes = node * 3  # Creates 3 clones
```

#### Properties

- `node_id` (str): Unique node identifier
- `operator_name` (str): Name of the operator
- `model_config` (ModelConfig): Current model configuration
- `execution_count` (int): Number of times executed
- `total_cost` (float): Total cost in USD
- `total_tokens` (int): Total tokens used

### Layer

Container for parallel node execution.

```python
class Layer:
    def __init__(
        self,
        nodes: List[Node],
        layer_config: Optional[LayerConfig] = None,
        layer_id: Optional[str] = None
    )
```

#### Parameters

- `nodes` (List[Node]): Nodes to execute in parallel
- `layer_config` (LayerConfig, optional): Layer configuration
- `layer_id` (str, optional): Unique identifier

#### Methods

##### `async execute_parallel(inputs, execution_context=None) -> LayerResult`

Execute all nodes in parallel.

**Parameters:**
- `inputs` (Union[List[Any], Any]): Input data for nodes
- `execution_context` (ExecutionContext, optional): Execution context

**Returns:** LayerResult with outputs and metadata

##### `from_operators(operator_names, layer_config=None, **node_kwargs) -> Layer`

Create layer from operator names.

**Parameters:**
- `operator_names` (List[str]): Names of operators
- `layer_config` (LayerConfig, optional): Layer configuration
- `**node_kwargs`: Additional node configuration

**Returns:** New Layer instance

### Network (NoN)

Orchestrates sequential execution of layers.

```python
class NoN:
    def __init__(
        self,
        layers: List[Layer],
        network_config: Optional[NetworkConfig] = None,
        network_id: Optional[str] = None
    )
```

#### Parameters

- `layers` (List[Layer]): Layers to execute sequentially
- `network_config` (NetworkConfig, optional): Network configuration
- `network_id` (str, optional): Unique identifier

#### Methods

##### `async forward(initial_input, execution_context=None) -> Any`

Execute the network with given input.

**Parameters:**
- `initial_input` (Any): Input for the first layer
- `execution_context` (ExecutionContext, optional): Execution context

**Returns:** Final output from the last layer

##### `from_operators(operator_specs, network_config=None, **node_kwargs) -> NoN`

Create network from operator specifications.

**Parameters:**
- `operator_specs` (List[Union[str, List[str], List[Node]]]): Layer specifications
- `network_config` (NetworkConfig, optional): Network configuration
- `**node_kwargs`: Additional node configuration

**Returns:** New NoN instance

**Example:**
```python
network = NoN.from_operators([
    'transform',                    # Single node layer
    ['classify', 'extract'],        # Parallel nodes layer
    'generate'                      # Single node layer
])
```

## Configuration Types

### ModelConfig

Configuration for LLM providers and parameters.

```python
@dataclass
class ModelConfig:
    provider: ModelProvider
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
```

### LayerConfig

Configuration for layer behavior.

```python
@dataclass
class LayerConfig:
    error_policy: ErrorPolicy = ErrorPolicy.FAIL_FAST
    max_retries: int = 3
    timeout_seconds: float = 60.0
    min_success_threshold: float = 1.0
    retry_delay_seconds: float = 1.0
    retry_backoff_multiplier: float = 2.0
    max_retry_delay: float = 60.0
```

### NetworkConfig

Configuration for network behavior.

```python
@dataclass
class NetworkConfig:
    max_concurrent_layers: int = 1
    layer_timeout_seconds: float = 300.0
    enable_layer_caching: bool = False
    default_layer_config: Optional[LayerConfig] = None
```

## Enums

### ModelProvider

Supported LLM providers.

```python
class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MOCK = "mock"
```

### ErrorPolicy

Error handling strategies.

```python
class ErrorPolicy(str, Enum):
    FAIL_FAST = "fail_fast"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    SKIP_AND_CONTINUE = "skip_and_continue"
    FALLBACK_MODEL = "fallback_model"
    RETURN_PARTIAL = "return_partial"
```

### SpanKind

Types of operations for tracing.

```python
class SpanKind(str, Enum):
    NETWORK = "network"
    LAYER = "layer"
    NODE = "node"
    OPERATOR = "operator"
    LLM_CALL = "llm_call"
    RETRY = "retry"
    FALLBACK = "fallback"
```

## Observability

### ObservabilityManager

Unified observability management.

```python
class ObservabilityManager:
    def __init__(
        self,
        enable_tracing: bool = True,
        enable_logging: bool = True,
        enable_metrics: bool = True
    )
```

#### Methods

##### `export_all_data(since=None, trace_id=None) -> Dict[str, List[Dict]]`

Export all observability data.

**Parameters:**
- `since` (float, optional): Timestamp to filter from
- `trace_id` (str, optional): Specific trace to export

**Returns:** Dictionary with 'spans', 'logs', and 'metrics' keys

##### `get_trace_summary(trace_id: str) -> Dict[str, Any]`

Get comprehensive summary for a trace.

**Parameters:**
- `trace_id` (str): Trace identifier

**Returns:** Summary with timing, costs, and success metrics

##### `get_stats() -> Dict[str, Any]`

Get comprehensive observability statistics.

**Returns:** Dictionary with tracing, logging, and metrics stats

### Functions

#### `get_observability() -> ObservabilityManager`

Get the global observability manager.

#### `configure_observability(enable_tracing=True, enable_logging=True, enable_metrics=True) -> ObservabilityManager`

Configure global observability settings.

## Request Scheduling

### RequestScheduler

Intelligent request scheduling with rate limiting.

```python
class RequestScheduler:
    def __init__(
        self,
        default_rate_limits: Optional[Dict[ModelProvider, RateLimitConfig]] = None,
        queue_strategy: QueueStrategy = QueueStrategy.PRIORITY,
        enable_observability: bool = True
    )
```

#### Methods

##### `async schedule_request(operation, provider, model_config=None, priority=0, estimated_tokens=0, component_type="", component_id="", *args, **kwargs) -> Any`

Schedule a request for execution.

**Parameters:**
- `operation` (Callable): Async operation to execute
- `provider` (ModelProvider): LLM provider to use
- `model_config` (ModelConfig, optional): Model configuration
- `priority` (int): Request priority (higher = more urgent)
- `estimated_tokens` (int): Estimated token usage
- `component_type` (str): Component type for observability
- `component_id` (str): Component ID for observability

**Returns:** Result of the operation

##### `async start()`

Start the request scheduler.

##### `async stop()`

Stop the request scheduler.

##### `get_stats() -> Dict[str, Any]`

Get scheduler statistics.

### RateLimitConfig

Configuration for rate limiting.

```python
@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    requests_per_second: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    max_concurrent: int = 10
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_base_delay: float = 1.0
    backoff_max_delay: float = 60.0
    backoff_multiplier: float = 2.0
```

### QueueStrategy

Request queue strategies.

```python
class QueueStrategy(str, Enum):
    FIFO = "fifo"
    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
```

## Operators

### Decorator

Register new operators.

```python
@operator(
    input_schema: Dict[str, Any],
    output_schema: Optional[Dict[str, Any]] = None,
    description: str = ""
)
def operator_function(*args, **kwargs):
    pass
```

### Registry Functions

#### `get_operator(name: str) -> RegisteredOperator`

Get a registered operator by name.

#### `list_operators() -> List[str]`

List all registered operator names.

#### `get_operator_info(name: str) -> Dict[str, Any]`

Get detailed information about an operator.

## Utility Functions

### Layer Creation

#### `create_layer(*nodes: Node, **kwargs) -> Layer`

Create a layer from nodes.

#### `create_parallel_layer(node_list: List[Node], **kwargs) -> Layer`

Create a layer from a list of nodes (e.g., from multiplication).

### Network Creation

#### `create_network(*layers: Layer, **kwargs) -> NoN`

Create a network from layers.

### Node Creation

#### `create_node(operator_name: str, **kwargs) -> Node`

Create a node from an operator name.

## Exception Classes

### OperatorError

Raised when operator execution fails.

### ValidationError

Raised when input validation fails.

### NetworkError

Raised when network execution fails.

### ConfigurationError

Raised when configuration is invalid.

## Type Aliases

```python
Content = Union[str, Dict[str, Any]]
```

Basic content type for operator inputs/outputs.

## Examples

### Basic Usage

```python
import asyncio
from nons.core.network import NoN

async def main():
    network = NoN.from_operators(['transform', 'generate'])
    result = await network.forward("Input text")
    print(result)

asyncio.run(main())
```

### Advanced Configuration

```python
from nons.core.node import Node
from nons.core.types import ModelConfig, ModelProvider, LayerConfig, ErrorPolicy

# Create configured node
node = Node(
    'generate',
    model_config=ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        temperature=0.8,
        max_tokens=200
    )
)

# Configure layer with error handling
layer_config = LayerConfig(
    error_policy=ErrorPolicy.RETRY_WITH_BACKOFF,
    max_retries=3,
    timeout_seconds=30
)

# Create network
network = NoN.from_operators(
    ['transform', [node], 'validate'],
    layer_config=layer_config
)
```

### Observability

```python
from nons.observability.integration import get_observability

async def monitored_execution():
    result = await network.forward("Input")

    obs = get_observability()
    data = obs.export_all_data()

    print(f"Spans: {len(data['spans'])}")
    print(f"Logs: {len(data['logs'])}")
    print(f"Metrics: {len(data['metrics'])}")
```