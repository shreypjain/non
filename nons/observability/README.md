# NoN Observability

Comprehensive monitoring, tracing, and metrics system for production NoN deployments. Get complete visibility into your compound AI systems with distributed tracing, structured logging, and detailed metrics.

## 🔍 Overview

The observability system automatically tracks:
- **Distributed Tracing**: Request flows across networks, layers, and nodes
- **Structured Logging**: Contextual logs with automatic trace correlation
- **Metrics Collection**: Performance, costs, tokens, and business metrics
- **Database-Ready Exports**: All data formatted for easy database storage

## 📊 Components

### Tracing (`tracing.py`)
Distributed tracing with automatic span relationships and timing.

### Logging (`logging.py`)
Structured logging with automatic trace context injection.

### Metrics (`metrics.py`)
Comprehensive metrics collection for performance and business insights.

### Integration (`integration.py`)
Unified observability manager that coordinates all systems.

## 🚀 Quick Start Examples

### 1. Basic Observability
```python
import asyncio
from nons.core.network import NoN
from nons.observability.integration import get_observability
import nons.operators.base

async def basic_monitoring():
    # Create network (observability is automatic)
    network = NoN.from_operators(['generate', 'validate'])

    # Execute with automatic tracing
    result = await network.forward("Write about space exploration")

    # Get observability data
    obs = get_observability()
    stats = obs.get_stats()

    print(f"Result: {result}")
    print(f"Spans created: {stats['tracing']['total_spans']}")
    print(f"Log entries: {stats['logging']['total_entries']}")

asyncio.run(basic_monitoring())
```

### 2. Custom Trace Context
```python
from nons.core.types import ExecutionContext

async def custom_trace_context():
    network = NoN.from_operators(['transform', 'generate'])

    # Create custom execution context
    context = ExecutionContext(
        trace_id="custom-trace-123",
        user_id="user-456",
        session_id="session-789"
    )

    result = await network.forward(
        "Analyze market trends",
        execution_context=context
    )

    # Get trace-specific data
    obs = get_observability()
    trace_data = obs.export_all_data(trace_id="custom-trace-123")

    print(f"Trace spans: {len(trace_data['spans'])}")
    print(f"Trace logs: {len(trace_data['logs'])}")
```

### 3. Export for Database Storage
```python
async def database_export():
    network = NoN.from_operators(['classify', 'extract', 'synthesize'])

    # Execute multiple operations
    inputs = [
        "Analyze customer feedback",
        "Process sales data",
        "Review market research"
    ]

    for input_text in inputs:
        await network.forward(input_text)

    # Export all data for database storage
    obs = get_observability()
    all_data = obs.export_all_data()

    # Data is ready for database insertion
    spans = all_data['spans']
    logs = all_data['logs']
    metrics = all_data['metrics']

    print(f"Ready for DB: {len(spans)} spans, {len(logs)} logs, {len(metrics)} metrics")

    # Example: Insert into database (pseudocode)
    # for span in spans:
    #     db.insert_span(span)
    # for log in logs:
    #     db.insert_log(log)
    # for metric in metrics:
    #     db.insert_metric(metric)
```

## 📈 Advanced Monitoring

### 4. Performance Analysis
```python
async def performance_analysis():
    obs = get_observability()

    # Create network with varying complexity
    simple_network = NoN.from_operators(['generate'])
    complex_network = NoN.from_operators([
        'transform',
        ['classify', 'extract', 'condense'],
        'synthesize'
    ])

    # Test simple network
    await simple_network.forward("Simple task")

    # Test complex network
    await complex_network.forward("Complex analysis task")

    # Analyze performance differences
    all_spans = obs.export_all_data()['spans']

    # Group by network complexity
    simple_spans = [s for s in all_spans if len(s.get('tags', {})) < 3]
    complex_spans = [s for s in all_spans if len(s.get('tags', {})) >= 3]

    print(f"Simple network average duration: {sum(s['duration_ms'] for s in simple_spans) / len(simple_spans):.2f}ms")
    print(f"Complex network average duration: {sum(s['duration_ms'] for s in complex_spans) / len(complex_spans):.2f}ms")
```

### 5. Cost and Token Tracking
```python
async def cost_analysis():
    from nons.core.node import Node
    from nons.core.types import ModelConfig, ModelProvider

    # Create nodes with different models
    fast_node = Node('generate', ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307"
    ))

    powerful_node = Node('generate', ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-opus-20240229"
    ))

    # Test both models
    test_prompt = "Write a detailed analysis of renewable energy trends"

    await fast_node.execute(prompt=test_prompt)
    await powerful_node.execute(prompt=test_prompt)

    # Analyze costs
    obs = get_observability()
    spans = obs.export_all_data()['spans']

    total_cost = sum(
        span.get('cost_info', {}).get('cost_usd', 0)
        for span in spans
    )

    total_tokens = sum(
        span.get('token_usage', {}).get('total_tokens', 0)
        for span in spans
    )

    print(f"Total cost: ${total_cost:.6f}")
    print(f"Total tokens: {total_tokens:,}")

    # Model-specific analysis
    for span in spans:
        if 'model_name' in span:
            model = span['model_name']
            cost = span.get('cost_info', {}).get('cost_usd', 0)
            tokens = span.get('token_usage', {}).get('total_tokens', 0)
            print(f"{model}: ${cost:.6f} for {tokens} tokens")
```

### 6. Custom Business Metrics
```python
from nons.observability.metrics import get_metrics_collector

async def business_metrics():
    metrics = get_metrics_collector()
    network = NoN.from_operators(['classify', 'generate'])

    # Simulate business operations
    user_requests = [
        {"user_id": "user1", "request": "Generate marketing copy"},
        {"user_id": "user2", "request": "Analyze customer sentiment"},
        {"user_id": "user1", "request": "Create product description"}
    ]

    for req in user_requests:
        # Process request
        result = await network.forward(req["request"])

        # Record business metrics
        metrics.record_counter("user_requests", 1, {
            "user_id": req["user_id"],
            "request_type": "ai_generation"
        })

        # Record satisfaction (simulated)
        satisfaction = 4.5  # Out of 5
        metrics.record_gauge("user_satisfaction", satisfaction, {
            "user_id": req["user_id"]
        })

        # Record processing time
        metrics.record_histogram("request_processing_time", 1.23, {
            "request_type": "ai_generation"
        })

    # Get business insights
    obs = get_observability()
    all_metrics = obs.export_all_data()['metrics']

    print(f"Business metrics collected: {len(all_metrics)}")
    print("Sample metrics:")
    for metric in all_metrics[:3]:
        print(f"  {metric['name']}: {metric['value']} ({metric['type']})")
```

## 🔧 Configuration

### 7. Custom Observability Configuration
```python
from nons.observability.integration import configure_observability

async def custom_config():
    # Configure observability settings
    obs = configure_observability(
        enable_tracing=True,
        enable_logging=True,
        enable_metrics=True
    )

    # Tracing can be configured further
    obs.tracing.max_spans = 1000
    obs.logging.max_entries = 5000
    obs.metrics.max_points = 2000

    # Use configured observability
    network = NoN.from_operators(['transform', 'generate'])
    result = await network.forward("Test with custom config")

    print(f"Configured observability used: {result}")
```

### 8. Selective Observability
```python
async def selective_observability():
    # Disable specific observability features
    obs = configure_observability(
        enable_tracing=True,   # Keep tracing
        enable_logging=False,  # Disable logging
        enable_metrics=True    # Keep metrics
    )

    network = NoN.from_operators(['classify', 'extract'])
    await network.forward("Test selective observability")

    stats = obs.get_stats()
    print(f"Tracing enabled: {stats['tracing']['total_spans'] > 0}")
    print(f"Logging disabled: {stats['logging']['total_entries'] == 0}")
    print(f"Metrics enabled: {stats['metrics']['total_points'] > 0}")
```

## 💾 Database Integration Examples

### 9. PostgreSQL Export
```python
import asyncpg
import json

async def postgresql_export():
    # Execute some network operations
    network = NoN.from_operators(['transform', 'generate', 'validate'])
    await network.forward("Sample data for database export")

    # Get all observability data
    obs = get_observability()
    data = obs.export_all_data()

    # Connect to PostgreSQL (example)
    # conn = await asyncpg.connect("postgresql://user:pass@localhost/db")

    print("Data ready for PostgreSQL export:")
    print(f"Spans: {len(data['spans'])}")
    print(f"Logs: {len(data['logs'])}")
    print(f"Metrics: {len(data['metrics'])}")

    # Example span structure for database
    if data['spans']:
        sample_span = data['spans'][0]
        print("\nSample span structure:")
        for key, value in sample_span.items():
            print(f"  {key}: {type(value).__name__}")

    # SQL table creation example
    create_spans_table = """
    CREATE TABLE IF NOT EXISTS spans (
        trace_id VARCHAR(255),
        span_id VARCHAR(255) PRIMARY KEY,
        parent_span_id VARCHAR(255),
        operation_name VARCHAR(255),
        kind VARCHAR(50),
        status VARCHAR(50),
        start_time TIMESTAMP,
        end_time TIMESTAMP,
        duration_ms FLOAT,
        tags JSONB,
        logs JSONB,
        component_type VARCHAR(100),
        component_id VARCHAR(255),
        token_usage JSONB,
        cost_info JSONB
    );
    """
    print(f"\nSQL for spans table:\n{create_spans_table}")
```

### 10. Real-time Monitoring Dashboard
```python
async def monitoring_dashboard():
    """Simulate real-time monitoring for a dashboard."""
    import time

    # Create different types of networks for monitoring
    networks = {
        "simple": NoN.from_operators(['generate']),
        "analysis": NoN.from_operators(['classify', 'extract']),
        "complex": NoN.from_operators([
            'transform',
            ['classify', 'extract', 'condense'],
            'synthesize'
        ])
    }

    # Simulate continuous operations
    start_time = time.time()
    operations = 0

    for network_type, network in networks.items():
        # Execute multiple operations
        for i in range(3):
            await network.forward(f"Test operation {i} for {network_type}")
            operations += 1

    # Get comprehensive stats for dashboard
    obs = get_observability()
    stats = obs.get_stats()
    all_data = obs.export_all_data()

    # Dashboard metrics
    total_time = time.time() - start_time
    avg_operation_time = total_time / operations if operations > 0 else 0

    print("🖥️  MONITORING DASHBOARD")
    print("=" * 50)
    print(f"📊 Operations: {operations}")
    print(f"⏱️  Total Time: {total_time:.2f}s")
    print(f"📈 Avg Operation Time: {avg_operation_time:.2f}s")
    print(f"🔍 Active Spans: {stats['tracing']['active_spans']}")
    print(f"📝 Total Logs: {stats['logging']['total_entries']}")
    print(f"📊 Total Metrics: {stats['metrics']['total_points']}")

    # Cost analysis
    total_cost = sum(
        span.get('cost_info', {}).get('cost_usd', 0)
        for span in all_data['spans']
    )
    print(f"💰 Total Cost: ${total_cost:.6f}")

    # Token usage
    total_tokens = sum(
        span.get('token_usage', {}).get('total_tokens', 0)
        for span in all_data['spans']
    )
    print(f"🎯 Total Tokens: {total_tokens:,}")

    # Success rate
    successful_spans = len([
        span for span in all_data['spans']
        if span.get('status') == 'success'
    ])
    success_rate = successful_spans / len(all_data['spans']) if all_data['spans'] else 0
    print(f"✅ Success Rate: {success_rate:.1%}")
```

## 🔗 Integration Points

- **Database Storage**: All data is export-ready for PostgreSQL, MongoDB, ClickHouse
- **Monitoring Tools**: Compatible with Prometheus, Grafana, DataDog
- **Alerting**: Custom metrics can trigger alerts based on thresholds
- **Analytics**: Span and log data perfect for business intelligence tools

## 🎯 Best Practices

1. **Always Export**: Regularly export observability data to persistent storage
2. **Monitor Costs**: Track token usage and costs for budget management
3. **Set Alerts**: Configure alerts for high error rates or costs
4. **Analyze Trends**: Use trace data to identify performance bottlenecks
5. **Business Metrics**: Collect custom metrics for business insights

## 🔗 Next Steps

- Learn about [Core Components](../core/README.md)
- Explore [Operators](../operators/README.md)
- Check [Utilities](../utils/README.md)
- See [Complete Examples](../../examples/)