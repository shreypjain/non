# NoN Architecture Guide

This document provides an in-depth look at the architecture and design principles of NoN (Network of Networks).

## Overview

NoN is built around a hierarchical composition model that enables complex AI workflows through the combination of simple, reusable components.

```
┌─────────────────────────────────────────┐
│                Network                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │ Layer 1 │─▶│ Layer 2 │─▶│ Layer 3 │  │
│  └─────────┘  └─────────┘  └─────────┘  │
└─────────────────────────────────────────┘
              │
              ▼
    ┌─────────────────┐
    │     Layer       │
    │ ┌─────┐ ┌─────┐ │ ◀── Parallel Execution
    │ │Node │ │Node │ │
    │ └─────┘ └─────┘ │
    └─────────────────┘
              │
              ▼
      ┌─────────────┐
      │    Node     │
      │ ┌─────────┐ │
      │ │Operator │ │ ◀── Function + Configuration
      │ └─────────┘ │
      └─────────────┘
```

## Core Components

### 1. Operators

Operators are pure functional units that define transformations on content.

#### Operator Registry

```python
# operators/registry.py
class OperatorRegistry:
    def __init__(self):
        self._operators: Dict[str, RegisteredOperator] = {}

    def register(self, name: str, function: Callable, schema: Dict[str, Any]):
        """Register an operator with validation schema"""
        self._operators[name] = RegisteredOperator(
            name=name,
            function=function,
            input_schema=schema.get('input', {}),
            output_schema=schema.get('output', {}),
            description=schema.get('description', '')
        )
```

#### Operator Decorator

```python
@operator(
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "target_style": {"type": "string"}
        },
        "required": ["text", "target_style"]
    },
    description="Transform text to a different style or format"
)
async def transform(text: str, target_style: str) -> str:
    """Transform content to specified style"""
    # Implementation here
    pass
```

### 2. Nodes

Nodes wrap operators with configuration and execution context.

#### Node Architecture

```python
class Node:
    def __init__(self, operator_name: str, model_config: ModelConfig):
        self.operator = get_operator(operator_name)
        self.model_config = model_config
        self.execution_metrics = []

    async def execute(self, *args, **kwargs):
        # Validation → Execution → Metrics Collection
        self.operator.validate_inputs(*args, **kwargs)
        result = await self._execute_with_provider(*args, **kwargs)
        self._update_metrics(result)
        return result
```

#### Model Configuration

```python
@dataclass
class ModelConfig:
    provider: ModelProvider
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
```

### 3. Layers

Layers manage parallel execution of multiple nodes using asyncio.

#### Layer Execution Flow

```python
class Layer:
    async def execute_parallel(self, inputs, execution_context):
        # 1. Prepare inputs for each node
        node_inputs = self._prepare_inputs(inputs)

        # 2. Create execution contexts
        node_contexts = self._create_node_contexts(execution_context)

        # 3. Execute with error policy
        outputs, results = await self._execute_with_policy(
            node_inputs, node_contexts
        )

        # 4. Aggregate results
        return LayerResult(outputs, execution_time, ...)
```

#### Error Policies

```python
class ErrorPolicy(str, Enum):
    FAIL_FAST = "fail_fast"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    SKIP_AND_CONTINUE = "skip_and_continue"
    FALLBACK_MODEL = "fallback_model"
    RETURN_PARTIAL = "return_partial"
```

### 4. Networks

Networks orchestrate sequential execution of layers.

#### Network Execution

```python
class NoN:
    async def forward(self, initial_input):
        current_output = initial_input

        for layer_index, layer in enumerate(self.layers):
            # Execute layer with current output as input
            layer_result = await layer.execute_parallel(
                current_output, execution_context
            )

            # Pass output to next layer
            current_output = layer_result.outputs

        return current_output
```

## Advanced Architecture Components

### 1. Request Scheduler

Intelligent request scheduling with rate limiting across providers.

```python
class RequestScheduler:
    def __init__(self):
        self.request_queues = {}  # Per-provider queues
        self.rate_limits = {}     # Per-provider limits
        self.provider_stats = {}  # Performance tracking

    async def schedule_request(self, operation, provider, priority=0):
        # 1. Add to appropriate queue
        # 2. Check rate limits
        # 3. Execute when slot available
        # 4. Update statistics
```

#### Queue Strategies

- **FIFO**: First-in, first-out processing
- **PRIORITY**: Priority-based scheduling
- **ROUND_ROBIN**: Distribute across providers
- **LEAST_LOADED**: Route to least busy provider

### 2. Observability System

Comprehensive monitoring with distributed tracing, logging, and metrics.

#### Tracing Architecture

```python
class Span:
    trace_id: str           # Links related operations
    span_id: str           # Unique span identifier
    parent_span_id: str    # Hierarchical relationships
    operation_name: str    # What operation is being traced
    kind: SpanKind        # Type of operation

    # Timing
    start_time: float
    end_time: float
    duration_ms: float

    # Metadata
    tags: Dict[str, Any]
    logs: List[Dict]

    # NoN-specific
    component_type: str    # "network", "layer", "node"
    component_id: str

    # Resource tracking
    token_usage: TokenUsage
    cost_info: CostInfo
```

#### Automatic Correlation

```python
class ObservabilityManager:
    def start_operation(self, operation_name, kind):
        # 1. Create span with automatic parent relationship
        # 2. Set trace context for logging
        # 3. Record start metrics

    def finish_operation(self, span, result=None, error=None):
        # 1. Finish span with timing
        # 2. Log completion/error
        # 3. Record metrics
        # 4. Clear trace context
```

### 3. Provider Adapters

Unified interface for multiple LLM providers.

```python
class BaseLLMProvider:
    async def generate_completion(self, prompt: str) -> Tuple[str, ExecutionMetrics]:
        # Standard interface for all providers

class OpenAIProvider(BaseLLMProvider):
    async def generate_completion(self, prompt: str):
        # OpenAI-specific implementation

class AnthropicProvider(BaseLLMProvider):
    async def generate_completion(self, prompt: str):
        # Anthropic-specific implementation
```

## Data Flow

### 1. Request Flow

```
User Input
    ↓
Network.forward()
    ↓
Layer.execute_parallel()
    ↓
Node.execute() (parallel)
    ↓
Scheduler.schedule_request()
    ↓
Provider.generate_completion()
    ↓
Response Aggregation
    ↓
Final Output
```

### 2. Observability Flow

```
Operation Start
    ↓
Create Span
    ↓
Set Trace Context
    ↓
Log Operation Start
    ↓
Record Start Metrics
    ↓
Execute Operation
    ↓
Record Execution Metrics
    ↓
Log Completion/Error
    ↓
Finish Span
    ↓
Clear Trace Context
```

## Design Principles

### 1. Composability

Every component can be combined with others:

```python
# Operators combine into nodes
node = Node('transform', config)

# Nodes combine into layers
layer = Layer([node1, node2, node3])

# Layers combine into networks
network = NoN([layer1, layer2])

# Networks can be components of larger systems
meta_network = MetaNoN([network1, network2])
```

### 2. Async-First

All operations are async to maximize concurrency:

```python
# Parallel layer execution
async def execute_parallel(self, inputs):
    tasks = [node.execute(input) for node, input in zip(self.nodes, inputs)]
    return await asyncio.gather(*tasks)
```

### 3. Observable by Default

Every operation generates telemetry automatically:

```python
@traced_operation("node_execution", SpanKind.NODE)
async def execute(self, *args):
    # Automatic span creation, timing, and correlation
    result = await self.operator.function(*args)
    return result
```

### 4. Fault Tolerant

Multiple strategies for handling failures:

```python
class LayerConfig:
    error_policy: ErrorPolicy
    max_retries: int
    timeout_seconds: float
    min_success_threshold: float
    fallback_providers: List[ModelProvider]
```

### 5. Resource Aware

Track and optimize resource usage:

```python
class ExecutionMetrics:
    token_usage: TokenUsage
    cost_info: CostInfo
    execution_time: float
    provider: ModelProvider
    model_name: str
```

## Extensibility

### Adding New Operators

```python
@operator(
    input_schema={"type": "object", "properties": {...}},
    description="Custom operator description"
)
async def my_custom_operator(input_data: str) -> str:
    # Custom logic here
    return processed_data
```

### Adding New Providers

```python
class CustomProvider(BaseLLMProvider):
    async def generate_completion(self, prompt: str):
        # Custom provider implementation
        response = await self.custom_api_call(prompt)
        metrics = self._calculate_metrics(response)
        return response.text, metrics
```

### Custom Error Policies

```python
class CustomErrorPolicy:
    async def handle_error(self, error: Exception, context: ExecutionContext):
        # Custom error handling logic
        if isinstance(error, RateLimitError):
            return await self.backoff_and_retry()
        # ...
```

## Performance Characteristics

### Concurrency

- **Layer Level**: Nodes execute in parallel within layers
- **Network Level**: Layers execute sequentially
- **Request Level**: Multiple requests scheduled concurrently
- **Provider Level**: Rate limiting prevents quota exhaustion

### Memory Usage

- **Streaming**: Large inputs can be processed in chunks
- **Garbage Collection**: Automatic cleanup of completed spans/logs
- **Resource Limits**: Configurable memory limits for buffers

### Scalability

- **Horizontal**: Multiple networks can run in parallel
- **Vertical**: Node multiplication for CPU-bound scaling
- **Provider**: Automatic load balancing across providers
- **Geographic**: Provider selection can consider latency

This architecture provides a robust foundation for building sophisticated compound AI systems with production-ready reliability, observability, and performance characteristics.