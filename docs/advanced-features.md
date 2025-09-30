# Advanced Features

This guide covers advanced features and patterns for building sophisticated compound AI systems with NoN.

## Custom Operators

### Creating Custom Operators

```python
from nons.operators.registry import operator

@operator(
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "style": {"type": "string"},
            "format": {"type": "string"}
        },
        "required": ["text", "style"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "transformed_text": {"type": "string"},
            "metadata": {"type": "object"}
        }
    },
    description="Advanced text transformation with metadata"
)
async def advanced_transform(text: str, style: str, format: str = "markdown") -> dict:
    """Advanced transformation operator with structured output."""
    # Custom transformation logic
    transformed = f"[{style.upper()}] {text}"

    metadata = {
        "original_length": len(text),
        "transformed_length": len(transformed),
        "style": style,
        "format": format
    }

    return {
        "transformed_text": transformed,
        "metadata": metadata
    }
```

### Operator Validation

```python
# Input validation is automatic based on schema
@operator(
    input_schema={
        "type": "object",
        "properties": {
            "numbers": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 1
            },
            "operation": {
                "type": "string",
                "enum": ["sum", "mean", "max", "min"]
            }
        },
        "required": ["numbers", "operation"]
    }
)
async def calculate(numbers: list, operation: str) -> float:
    """Mathematical calculations with validation."""
    if operation == "sum":
        return sum(numbers)
    elif operation == "mean":
        return sum(numbers) / len(numbers)
    elif operation == "max":
        return max(numbers)
    elif operation == "min":
        return min(numbers)
```

## Advanced Network Patterns

### Conditional Routing

```python
from nons.core.node import Node
from nons.core.network import NoN

async def create_conditional_network():
    # Route node determines the path
    router = Node('route', additional_prompt_context="""
    Based on the input, respond with one of: 'technical', 'creative', 'analytical'
    """)

    # Specialized processing nodes
    technical_node = Node('generate', additional_prompt_context="Technical expert response")
    creative_node = Node('generate', additional_prompt_context="Creative storytelling response")
    analytical_node = Node('generate', additional_prompt_context="Data-driven analytical response")

    # Dynamic network creation based on routing
    def create_specialized_network(route: str):
        if route == 'technical':
            return NoN.from_operators([technical_node, 'validate'])
        elif route == 'creative':
            return NoN.from_operators([creative_node, 'expand'])
        else:
            return NoN.from_operators([analytical_node, 'condense'])

    return router, create_specialized_network
```

### Hierarchical Networks

```python
async def create_hierarchical_network():
    # Sub-networks for specialized tasks
    preprocessing_network = NoN.from_operators([
        'transform',
        ['extract', 'classify'],
        'validate'
    ])

    analysis_network = NoN.from_operators([
        'compare',
        ['generate', 'generate', 'generate'],  # Multiple perspectives
        'synthesize'
    ])

    postprocessing_network = NoN.from_operators([
        'condense',
        'validate'
    ])

    # Main network orchestrating sub-networks
    async def hierarchical_forward(input_data):
        # Step 1: Preprocessing
        preprocessed = await preprocessing_network.forward(input_data)

        # Step 2: Analysis
        analyzed = await analysis_network.forward(preprocessed)

        # Step 3: Postprocessing
        final_result = await postprocessing_network.forward(analyzed)

        return final_result

    return hierarchical_forward
```

## Advanced Configuration

### Custom Error Policies

```python
from nons.core.types import ErrorPolicy, LayerConfig
import asyncio

class CustomErrorPolicy:
    """Custom error handling with circuit breaker pattern."""

    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_open = False

    async def handle_error(self, error, context):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.circuit_open = True
            raise CircuitBreakerError("Circuit breaker activated")

        # Exponential backoff
        delay = min(2 ** self.failure_count, 30)
        await asyncio.sleep(delay)

        return await context.retry()

    def can_execute(self):
        if not self.circuit_open:
            return True

        if time.time() - self.last_failure_time > self.recovery_timeout:
            self.circuit_open = False
            self.failure_count = 0
            return True

        return False
```

### Dynamic Model Selection

```python
from nons.core.types import ModelConfig, ModelProvider

class AdaptiveModelSelector:
    """Dynamically select models based on context and performance."""

    def __init__(self):
        self.performance_history = {}
        self.cost_limits = {}
        self.latency_requirements = {}

    def select_model(self, task_type: str, complexity: str, budget: float) -> ModelConfig:
        if complexity == "high" and budget > 0.01:
            return ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-opus-20240229",
                temperature=0.3
            )
        elif complexity == "medium":
            return ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-sonnet-20240229",
                temperature=0.5
            )
        else:
            return ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-haiku-20240307",
                temperature=0.7
            )

    async def update_performance(self, model_config: ModelConfig,
                               execution_time: float, cost: float, success: bool):
        """Update performance metrics for model selection."""
        key = f"{model_config.provider}:{model_config.model_name}"

        if key not in self.performance_history:
            self.performance_history[key] = {
                "avg_time": execution_time,
                "avg_cost": cost,
                "success_rate": 1.0 if success else 0.0,
                "count": 1
            }
        else:
            history = self.performance_history[key]
            count = history["count"]

            # Update running averages
            history["avg_time"] = (history["avg_time"] * count + execution_time) / (count + 1)
            history["avg_cost"] = (history["avg_cost"] * count + cost) / (count + 1)
            history["success_rate"] = (history["success_rate"] * count + (1.0 if success else 0.0)) / (count + 1)
            history["count"] = count + 1
```

## Performance Optimization

### Batching and Streaming

```python
from typing import AsyncIterable
import asyncio

class BatchProcessor:
    """Process inputs in batches for improved efficiency."""

    def __init__(self, batch_size: int = 10, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []

    async def process_stream(self, inputs: AsyncIterable,
                           network: NoN) -> AsyncIterable:
        """Process a stream of inputs in batches."""
        batch = []

        async for input_item in inputs:
            batch.append(input_item)

            if len(batch) >= self.batch_size:
                # Process full batch
                results = await self._process_batch(batch, network)
                for result in results:
                    yield result
                batch = []

        # Process remaining items
        if batch:
            results = await self._process_batch(batch, network)
            for result in results:
                yield result

    async def _process_batch(self, batch: list, network: NoN) -> list:
        """Process a batch of inputs concurrently."""
        tasks = [network.forward(item) for item in batch]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

### Memory Management

```python
import gc
import weakref
from typing import Optional

class MemoryOptimizedNetwork:
    """Network with automatic memory management."""

    def __init__(self, network: NoN, max_cache_size: int = 100):
        self.network = network
        self.result_cache = {}
        self.max_cache_size = max_cache_size
        self.cache_order = []

    async def forward_with_caching(self, input_data, cache_key: Optional[str] = None):
        """Execute with result caching and memory management."""
        if cache_key and cache_key in self.result_cache:
            return self.result_cache[cache_key]

        # Execute network
        result = await self.network.forward(input_data)

        # Cache result if key provided
        if cache_key:
            self._add_to_cache(cache_key, result)

        # Trigger garbage collection periodically
        if len(self.result_cache) % 50 == 0:
            gc.collect()

        return result

    def _add_to_cache(self, key: str, result):
        """Add result to cache with LRU eviction."""
        if len(self.result_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = self.cache_order.pop(0)
            del self.result_cache[oldest_key]

        self.result_cache[key] = result
        self.cache_order.append(key)
```

## Advanced Observability

### Custom Metrics

```python
from nons.observability.metrics import get_metrics_collector

class BusinessMetricsCollector:
    """Collect business-specific metrics."""

    def __init__(self):
        self.metrics = get_metrics_collector()
        self.business_events = []

    async def track_user_interaction(self, user_id: str, network_id: str,
                                   satisfaction_score: float):
        """Track user interaction metrics."""
        self.metrics.record_gauge(
            "user_satisfaction",
            satisfaction_score,
            tags={"user_id": user_id, "network_id": network_id}
        )

        self.business_events.append({
            "event_type": "user_interaction",
            "user_id": user_id,
            "network_id": network_id,
            "satisfaction_score": satisfaction_score,
            "timestamp": time.time()
        })

    async def track_business_outcome(self, outcome_type: str, value: float):
        """Track business outcome metrics."""
        self.metrics.record_counter(
            f"business_outcome_{outcome_type}",
            1,
            tags={"outcome_type": outcome_type}
        )

        self.metrics.record_gauge(
            f"business_value_{outcome_type}",
            value,
            tags={"outcome_type": outcome_type}
        )
```

### Custom Spans

```python
from nons.observability.tracing import get_tracer
from contextlib import asynccontextmanager

@asynccontextmanager
async def custom_span(operation_name: str, **tags):
    """Create custom spans for business logic."""
    tracer = get_tracer()

    span = tracer.start_span(
        operation_name=operation_name,
        kind=SpanKind.CUSTOM,
        tags=tags
    )

    try:
        yield span
        span.finish(status=SpanStatus.SUCCESS)
    except Exception as e:
        span.add_log({"error": str(e), "error_type": type(e).__name__})
        span.finish(status=SpanStatus.ERROR)
        raise

# Usage
async def business_workflow(data):
    async with custom_span("data_validation", data_type="customer") as span:
        # Validation logic
        validated_data = validate_customer_data(data)
        span.add_log({"validation_result": "success"})

    async with custom_span("ai_processing", model="claude-3") as span:
        # AI processing
        result = await network.forward(validated_data)
        span.add_log({"tokens_used": result.metadata.total_tokens})

    return result
```

## Integration Patterns

### Database Integration

```python
import asyncpg
from typing import Dict, Any

class DatabaseIntegratedNetwork:
    """Network with automatic database logging."""

    def __init__(self, network: NoN, db_pool: asyncpg.Pool):
        self.network = network
        self.db_pool = db_pool

    async def forward_with_logging(self, input_data, user_id: str = None):
        """Execute network with database logging."""
        execution_id = str(uuid.uuid4())

        # Log execution start
        await self._log_execution_start(execution_id, input_data, user_id)

        try:
            result = await self.network.forward(input_data)
            await self._log_execution_success(execution_id, result)
            return result
        except Exception as e:
            await self._log_execution_error(execution_id, str(e))
            raise

    async def _log_execution_start(self, execution_id: str,
                                 input_data: Any, user_id: str):
        """Log execution start to database."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO network_executions
                (execution_id, user_id, input_data, status, created_at)
                VALUES ($1, $2, $3, 'started', NOW())
            """, execution_id, user_id, str(input_data))

    async def _log_execution_success(self, execution_id: str, result: Any):
        """Log successful execution to database."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE network_executions
                SET status = 'completed', result_data = $2, completed_at = NOW()
                WHERE execution_id = $1
            """, execution_id, str(result))
```

### Event-Driven Architecture

```python
from asyncio import Queue
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class NetworkEvent:
    event_type: str
    network_id: str
    data: Dict[str, Any]
    timestamp: float

class EventDrivenNetwork:
    """Network with event publishing."""

    def __init__(self, network: NoN):
        self.network = network
        self.event_queue = Queue()
        self.subscribers: Dict[str, List[Callable]] = {}

    async def forward_with_events(self, input_data):
        """Execute network with event publishing."""
        # Publish start event
        await self.publish_event("execution_started", {
            "input_preview": str(input_data)[:100]
        })

        try:
            result = await self.network.forward(input_data)

            # Publish success event
            await self.publish_event("execution_completed", {
                "result_preview": str(result)[:100],
                "success": True
            })

            return result
        except Exception as e:
            # Publish error event
            await self.publish_event("execution_failed", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise

    async def publish_event(self, event_type: str, data: Dict[str, Any]):
        """Publish an event to subscribers."""
        event = NetworkEvent(
            event_type=event_type,
            network_id=self.network.network_id,
            data=data,
            timestamp=time.time()
        )

        await self.event_queue.put(event)

        # Notify subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    await callback(event)
                except Exception as e:
                    print(f"Subscriber error: {e}")

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to network events."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
```

## Best Practices

### Error Recovery Strategies

1. **Circuit Breaker Pattern**: Prevent cascading failures
2. **Bulkhead Pattern**: Isolate failures to specific components
3. **Timeout Pattern**: Prevent hanging operations
4. **Retry with Exponential Backoff**: Handle transient failures
5. **Graceful Degradation**: Provide fallback responses

### Performance Optimization

1. **Use appropriate model sizes** for task complexity
2. **Implement caching** for repeated operations
3. **Batch similar requests** when possible
4. **Monitor and optimize** token usage
5. **Use async/await** properly for concurrency

### Cost Management

1. **Track costs** at node level
2. **Set budget limits** per execution
3. **Use cheaper models** for preprocessing
4. **Implement cost-based routing**
5. **Monitor and alert** on cost thresholds

This guide covers advanced patterns and techniques for building production-ready compound AI systems with NoN.