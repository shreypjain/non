# Frequently Asked Questions

## General Questions

### What is NoN (Network of Networks)?

NoN is a Python framework for building compound AI systems using networks of interconnected language model operators. It allows you to create sophisticated AI workflows by composing multiple LLM calls into sequential layers with parallel execution.

### How is NoN different from other AI frameworks?

NoN focuses specifically on:
- **Operator-based composition**: Mathematical primitives for AI transformations
- **Built-in observability**: Comprehensive tracing, logging, and metrics
- **Multi-provider support**: Seamless integration with OpenAI, Anthropic, and Google
- **Parallel scaling**: Easy node multiplication for concurrent processing
- **Production-ready**: Request scheduling, rate limiting, and error handling

### What are the system requirements?

- Python 3.9+
- API keys for at least one LLM provider (OpenAI, Anthropic, or Google)
- Internet connection for LLM API calls

## Installation and Setup

### Why does installation fail with Python 3.8?

NoN requires Python 3.9+ due to dependencies on newer packages like `google-genai`. Update to Python 3.9 or higher:

```bash
# Check your Python version
python --version

# Install Python 3.9+ if needed
# On macOS with Homebrew:
brew install python@3.9

# On Ubuntu:
sudo apt update && sudo apt install python3.9
```

### How do I set up API keys?

Set environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

Or set them in your code:

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "your-key"
```

### What if I don't have all API keys?

NoN works with any single provider. Just set the key(s) you have. The framework will automatically use available providers and fall back to mock providers for testing.

## Usage Questions

### How do I create my first network?

```python
import asyncio
from nons.core.network import NoN

async def main():
    # Import operators first
    import nons.operators.base

    # Create a simple network
    network = NoN.from_operators([
        'transform',
        'generate'
    ])

    result = await network.forward("Hello, world!")
    print(result)

asyncio.run(main())
```

### Why do I get "Operator not found" errors?

Make sure to import the operators module before using them:

```python
# Always import this first
import nons.operators.base

# Then create your network
network = NoN.from_operators(['transform', 'generate'])
```

### How do I handle rate limiting?

Configure rate limits for your API quotas:

```python
from nons.core.scheduler import configure_scheduler, RateLimitConfig
from nons.core.types import ModelProvider

scheduler = configure_scheduler(
    rate_limits={
        ModelProvider.OPENAI: RateLimitConfig(
            requests_per_minute=100,
            max_concurrent=10
        )
    }
)
```

### How do I track costs?

Costs are automatically tracked on every node:

```python
# After execution
print(f"Total cost: ${node.total_cost:.6f}")
print(f"Total tokens: {node.total_tokens}")

# Get detailed execution metrics
for execution in node.execution_metrics:
    print(f"Cost: ${execution.cost:.6f}, Tokens: {execution.tokens}")
```

## Error Handling

### What does "ValidationError" mean?

Validation errors occur when input doesn't match the operator's expected schema. Check the operator documentation:

```python
from nons.operators.registry import get_operator_info

# Check what inputs an operator expects
info = get_operator_info('transform')
print(info['input_schema'])
```

### How do I handle network failures?

Configure error policies:

```python
from nons.core.types import ErrorPolicy, LayerConfig

layer_config = LayerConfig(
    error_policy=ErrorPolicy.RETRY_WITH_BACKOFF,
    max_retries=3,
    timeout_seconds=30
)

network = NoN.from_operators(
    ['transform', 'generate'],
    layer_config=layer_config
)
```

### What if a provider is down?

NoN automatically handles provider failures:

1. **Retry with backoff**: Temporary failures are retried
2. **Fallback providers**: Configure multiple providers
3. **Graceful degradation**: Return partial results when possible

```python
# Configure multiple providers for redundancy
node1 = Node('generate', model_config=ModelConfig(provider=ModelProvider.OPENAI))
node2 = Node('generate', model_config=ModelConfig(provider=ModelProvider.ANTHROPIC))

# Use in parallel for redundancy
layer = Layer([node1, node2])
```

## Performance

### How do I improve execution speed?

1. **Use parallel execution**:
```python
# Parallel nodes
network = NoN.from_operators([
    'transform',
    ['generate', 'generate', 'generate'],  # 3 parallel nodes
    'condense'
])
```

2. **Node multiplication**:
```python
base_node = Node('generate')
parallel_nodes = base_node * 5  # 5 parallel instances
```

3. **Choose appropriate models**:
```python
# Fast model for simple tasks
fast_config = ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-3-haiku-20240307"
)

# Powerful model for complex tasks
powerful_config = ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-3-opus-20240229"
)
```

### How do I optimize costs?

1. **Use cheaper models for preprocessing**
2. **Set token limits**:
```python
config = ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-3-haiku-20240307",
    max_tokens=100  # Limit output
)
```

3. **Monitor costs**:
```python
from nons.observability.integration import get_observability

obs = get_observability()
trace_summary = obs.get_trace_summary(trace_id)
print(f"Total cost: ${trace_summary['total_cost_usd']:.6f}")
```

### Why is my network slow?

Common causes and solutions:

1. **Sequential execution**: Use parallel layers
2. **Large token limits**: Reduce `max_tokens`
3. **Heavy models**: Use lighter models for simple tasks
4. **Rate limiting**: Check your API quotas
5. **Network latency**: Consider geographic provider selection

## Advanced Usage

### How do I create custom operators?

```python
from nons.operators.registry import operator

@operator(
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "language": {"type": "string"}
        },
        "required": ["text"]
    },
    description="Translate text to specified language"
)
async def translate(text: str, language: str = "Spanish") -> str:
    # Your translation logic here
    return f"Translated '{text}' to {language}"
```

### How do I integrate with databases?

```python
import asyncpg
from nons.observability.integration import get_observability

async def store_execution_data():
    obs = get_observability()
    data = obs.export_all_data()

    # Connect to database
    conn = await asyncpg.connect("postgresql://user:pass@localhost/db")

    # Store spans
    for span in data['spans']:
        await conn.execute("""
            INSERT INTO execution_spans (trace_id, span_id, operation_name, duration_ms)
            VALUES ($1, $2, $3, $4)
        """, span['trace_id'], span['span_id'], span['operation_name'], span['duration_ms'])

    await conn.close()
```

### How do I implement custom error handling?

```python
from nons.core.types import ErrorPolicy

class CustomErrorHandler:
    async def handle_error(self, error, context):
        if isinstance(error, RateLimitError):
            # Custom rate limit handling
            await asyncio.sleep(60)
            return await context.retry()
        elif isinstance(error, ValidationError):
            # Log and skip
            logger.warning(f"Validation error: {error}")
            return None
        else:
            # Re-raise unknown errors
            raise error
```

## Troubleshooting

### Import errors with operators

**Problem**: `OperatorError: Operator 'transform' not found`

**Solution**: Import operators before using them:
```python
import nons.operators.base  # This registers all operators
```

### Async/await errors

**Problem**: `RuntimeError: coroutine 'forward' was never awaited`

**Solution**: Always use `await` with async methods:
```python
# Wrong
result = network.forward("input")

# Correct
result = await network.forward("input")
```

### API key issues

**Problem**: `AuthenticationError: Invalid API key`

**Solution**:
1. Check your API key is correct
2. Verify environment variables are set
3. Test with a simple API call first

### Memory issues with large networks

**Problem**: High memory usage or OutOfMemory errors

**Solutions**:
1. Use streaming for large inputs
2. Implement result caching with LRU eviction
3. Set appropriate token limits
4. Process in batches

### Performance debugging

Use observability tools to identify bottlenecks:

```python
from nons.observability.integration import get_observability

# After execution
obs = get_observability()
stats = obs.get_stats()

print(f"Active spans: {stats['tracing']['active_spans']}")
print(f"Average span duration: {stats['tracing']['avg_duration']}")

# Get detailed trace
trace_summary = obs.get_trace_summary(trace_id)
for span in trace_summary['spans']:
    print(f"{span['operation_name']}: {span['duration_ms']}ms")
```

## Getting Help

### Where can I find more examples?

Check the `examples/` directory in the repository:
- `basic_usage.py` - Simple network creation
- `observability_demo.py` - Monitoring and metrics
- `scheduler_demo.py` - Request scheduling
- `multiplication_demo.py` - Parallel processing

### How do I report bugs?

1. Check existing issues on GitHub
2. Create a minimal reproducible example
3. Include your Python version, OS, and NoN version
4. Provide error messages and stack traces

### Where can I discuss features?

- GitHub Discussions for feature requests
- GitHub Issues for bug reports
- Documentation for usage questions

### Performance optimization help

1. Share your network structure
2. Include execution time measurements
3. Provide input/output sizes
4. Mention your hardware specifications

Still have questions? Check the documentation or open an issue on GitHub!