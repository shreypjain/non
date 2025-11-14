# NoN Production Deployment Guide

**Version**: 0.1.0
**Last Updated**: 2025-11-14
**Status**: Production Ready

## Table of Contents

1. [Overview](#overview)
2. [Provider Selection](#provider-selection)
3. [Environment Configuration](#environment-configuration)
4. [Deployment Options](#deployment-options)
5. [Performance Optimization](#performance-optimization)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Error Handling](#error-handling)
8. [Security Best Practices](#security-best-practices)
9. [Scaling Guidelines](#scaling-guidelines)
10. [Cost Optimization](#cost-optimization)

---

## Overview

NoN is production-ready following comprehensive latency testing and infrastructure hardening. This guide covers best practices for deploying NoN-based applications in production environments.

### System Requirements

- Python 3.9+
- At least one LLM provider API key (Anthropic, Google, or OpenAI)
- Sufficient memory for concurrent request handling (recommended: 2GB+ per worker)
- Network connectivity to LLM provider endpoints

---

## Provider Selection

Based on comprehensive latency testing (see `FINAL_LATENCY_REPORT.md`), here are production recommendations:

### Primary Provider: Anthropic Claude Sonnet 4.5

**Recommended for**: Production systems requiring high reliability

**Characteristics**:
- Average latency: 3-5 seconds per LLM call
- Reliability: Excellent (100% success rate in testing)
- Tool execution: Working perfectly
- Multi-agent support: Fully functional
- Cost: Moderate (input: $3/MTok, output: $15/MTok)

**Configuration**:
```python
from nons.core.types import ModelConfig, ModelProvider

config = ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-sonnet-4-5-20250929",
    temperature=0.7,
    max_tokens=2048
)
```

### Speed-Optimized Provider: Google Gemini 2.0 Flash

**Recommended for**: Latency-critical applications with proper prompt engineering

**Characteristics**:
- Average latency: 0.12-0.15 seconds per LLM call (25-30x faster than Claude!)
- Reliability: Requires improved prompting for tool selection
- Speed advantage: Exceptional
- Cost: Very low (input: $0.075/MTok, output: $0.30/MTok)

**Configuration**:
```python
config = ModelConfig(
    provider=ModelProvider.GOOGLE,
    model_name="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=2048
)
```

**Note**: Gemini requires more detailed prompts for reliable tool selection. Consider using Claude for complex routing and Gemini for fast generation tasks.

### Alternative Provider: OpenAI GPT-4o-mini

**Recommended for**: Mixed workloads requiring balance of speed and reliability

**Characteristics**:
- Average latency: ~0.4-0.5 seconds per LLM call (8-10x faster than Claude)
- Reliability: Good (based on limited testing)
- Cost: Low (input: $0.15/MTok, output: $0.60/MTok)

### Multi-Provider Strategy

For production resilience, use multiple providers with fallback logic:

```python
from nons.core.scheduler import configure_scheduler, RateLimitConfig
from nons.core.types import ModelProvider

scheduler = configure_scheduler(
    rate_limits={
        ModelProvider.ANTHROPIC: RateLimitConfig(
            requests_per_minute=100,
            max_concurrent=10
        ),
        ModelProvider.GOOGLE: RateLimitConfig(
            requests_per_minute=300,  # Gemini supports higher throughput
            max_concurrent=20
        ),
        ModelProvider.OPENAI: RateLimitConfig(
            requests_per_minute=200,
            max_concurrent=15
        )
    }
)
```

---

## Environment Configuration

### Step 1: Copy Environment Template

```bash
cp .env.example .env
```

### Step 2: Configure API Keys

```bash
# Edit .env and add your API keys
export ANTHROPIC_API_KEY="your-actual-key"
export GOOGLE_API_KEY="your-actual-key"
export OPENAI_API_KEY="your-actual-key"
```

### Step 3: Production Settings

For production, set these environment variables:

```bash
# Environment
export NON_ENVIRONMENT=production
export NON_DEBUG=false
export NON_LOG_LEVEL=INFO

# Observability
export NON_ENABLE_TRACING=true
export NON_ENABLE_METRICS=true

# Error handling
export NON_ERROR_POLICY=retry_with_backoff
export NON_MAX_RETRIES=3

# Performance
export NON_MAX_CONCURRENT_REQUESTS=10
export NON_REQUESTS_PER_MINUTE=100
export NON_REQUEST_TIMEOUT=300
```

---

## Deployment Options

### Option 1: Docker Deployment (Recommended)

See `Dockerfile` and `docker-compose.yml` for containerized deployment.

```bash
# Build image
docker build -t non-app:latest .

# Run with environment variables
docker run -d \
  --name non-app \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e NON_LOG_LEVEL=INFO \
  -p 8000:8000 \
  non-app:latest
```

### Option 2: Direct Python Deployment

```bash
# Install dependencies
uv sync

# Run your application
uv run python your_app.py
```

### Option 3: Kubernetes Deployment

Create a ConfigMap for non-sensitive configuration:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: non-config
data:
  NON_LOG_LEVEL: "INFO"
  NON_ENVIRONMENT: "production"
  NON_ENABLE_TRACING: "true"
```

Create a Secret for API keys:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: non-secrets
type: Opaque
stringData:
  ANTHROPIC_API_KEY: "your-key"
  GOOGLE_API_KEY: "your-key"
```

---

## Performance Optimization

### 1. Scheduler Initialization

**CRITICAL**: Always start the scheduler before agent execution:

```python
from nons.core.scheduler import start_scheduler

async def main():
    # MUST call this first!
    await start_scheduler()

    # Now safe to run agents
    result = await agent.run(task)
```

**Why**: Without starting the scheduler, requests will queue indefinitely causing infinite hangs. This was the primary latency issue discovered during testing.

### 2. Parallel Node Execution

Use node multiplication for parallel processing:

```python
from nons.core.node import Node
from nons.core.network import create_network

# Create 5 parallel generators
generator = Node("generate", model_config=fast_config)
parallel_generators = generator * 5

network = create_network(
    layers=[parallel_generators, "condense"]
)
```

### 3. Request Batching

Batch similar requests together for better throughput:

```python
import asyncio

async def process_batch(requests):
    tasks = [agent.run(req) for req in requests]
    results = await asyncio.gather(*tasks)
    return results
```

### 4. Caching Strategy

Implement response caching for repeated queries:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
async def cached_forward(network_id: str, input_hash: str):
    # Cache by network ID and input hash
    return await network.forward(input_data)

# Usage
input_hash = hashlib.sha256(input_text.encode()).hexdigest()
result = await cached_forward(network.id, input_hash)
```

---

## Monitoring and Observability

### 1. Enable Full Observability

```python
from nons.observability.integration import get_observability

# Get observability data
obs = get_observability()
data = obs.export_all_data()

# Monitor key metrics
print(f"Total spans: {len(data['spans'])}")
print(f"Total cost: ${sum(s.get('cost_usd', 0) for s in data['spans']):.6f}")
print(f"Success rate: {sum(1 for s in data['spans'] if s.get('success', False)) / len(data['spans'])}")
```

### 2. Export to Database

For production monitoring, export telemetry to a database:

```python
import psycopg2
import json

def export_to_postgres(data):
    conn = psycopg2.connect(os.environ['NON_DATABASE_URL'])
    cursor = conn.cursor()

    for span in data['spans']:
        cursor.execute("""
            INSERT INTO spans (trace_id, span_id, operation, duration_ms, cost_usd, success)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            span['trace_id'],
            span['span_id'],
            span['operation_name'],
            span.get('duration_ms', 0),
            span.get('cost_usd', 0),
            span.get('success', False)
        ))

    conn.commit()
    conn.close()
```

### 3. Prometheus Metrics

Export metrics for Prometheus scraping:

```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
request_counter = Counter('non_requests_total', 'Total requests', ['provider', 'operation'])
latency_histogram = Histogram('non_request_duration_seconds', 'Request latency', ['provider'])

# Instrument your code
request_counter.labels(provider='anthropic', operation='generate').inc()
latency_histogram.labels(provider='anthropic').observe(duration)

# Start metrics server
start_http_server(9090)
```

### 4. Health Checks

Implement health check endpoints:

```python
async def health_check():
    """Check if NoN system is healthy"""
    from nons.core.scheduler import get_scheduler

    scheduler = get_scheduler()

    return {
        "status": "healthy",
        "scheduler_running": scheduler.is_running,
        "queue_size": len(scheduler.queue),
        "timestamp": datetime.utcnow().isoformat()
    }
```

---

## Error Handling

### 1. Configure Error Policies

```python
from nons.core.types import ErrorPolicy, LayerConfig

production_config = LayerConfig(
    error_policy=ErrorPolicy.RETRY_WITH_BACKOFF,
    max_retries=3,
    min_success_threshold=0.7  # Allow 30% failures in parallel layers
)

network = create_network(
    layers=["generate", "validate"],
    layer_config=production_config
)
```

### 2. Handle Agent Exceptions

```python
async def safe_agent_run(agent, task):
    """Run agent with proper error handling"""
    try:
        async for result in agent.run(task):
            if result.get("success", False):
                return result
            else:
                logging.warning(f"Agent step failed: {result.get('error')}")

        return {"success": False, "error": "Max steps reached"}

    except Exception as e:
        logging.error(f"Agent execution failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
```

### 3. Circuit Breaker Pattern

Implement circuit breaker for provider failures:

```python
from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout_seconds=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timedelta(seconds=timeout_seconds)
        self.last_failure_time = None
        self.is_open = False

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self.is_open = True

    def record_success(self):
        self.failure_count = 0
        self.is_open = False

    def can_attempt(self):
        if not self.is_open:
            return True

        if datetime.utcnow() - self.last_failure_time > self.timeout:
            self.is_open = False
            self.failure_count = 0
            return True

        return False
```

---

## Security Best Practices

### 1. API Key Management

- Never commit API keys to version control
- Use environment variables or secret management services (AWS Secrets Manager, HashiCorp Vault)
- Rotate API keys regularly (quarterly recommended)
- Use separate keys for development, staging, and production

### 2. Input Validation

Always validate and sanitize user inputs:

```python
from pydantic import BaseModel, validator

class TaskInput(BaseModel):
    content: str

    @validator('content')
    def validate_content(cls, v):
        if len(v) > 50000:  # Limit input size
            raise ValueError("Input too large")
        if not v.strip():
            raise ValueError("Input cannot be empty")
        return v.strip()
```

### 3. Rate Limiting

Implement per-user rate limiting:

```python
from collections import defaultdict
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, max_requests_per_minute=60):
        self.max_requests = max_requests_per_minute
        self.user_requests = defaultdict(list)

    def is_allowed(self, user_id):
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=1)

        # Clean old requests
        self.user_requests[user_id] = [
            req_time for req_time in self.user_requests[user_id]
            if req_time > cutoff
        ]

        # Check limit
        if len(self.user_requests[user_id]) >= self.max_requests:
            return False

        self.user_requests[user_id].append(now)
        return True
```

### 4. Secure Network Communication

Use HTTPS for all external API calls (handled automatically by provider SDKs).

---

## Scaling Guidelines

### Horizontal Scaling

NoN agents are stateless and can be scaled horizontally:

```yaml
# Kubernetes Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: non-app
spec:
  replicas: 5  # Scale to 5 instances
  selector:
    matchLabels:
      app: non-app
  template:
    metadata:
      labels:
        app: non-app
    spec:
      containers:
      - name: non-app
        image: non-app:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### Vertical Scaling

For compute-intensive workloads, increase resources per instance:

- Memory: 2-4GB per worker minimum
- CPU: 1-2 cores per worker for optimal performance

### Load Balancing

Use load balancer for request distribution:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: non-app-service
spec:
  type: LoadBalancer
  selector:
    app: non-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

---

## Cost Optimization

### 1. Provider Cost Comparison (per 1M tokens)

| Provider | Input Cost | Output Cost | Speed | Best For |
|----------|-----------|-------------|-------|----------|
| **Gemini 2.0 Flash** | $0.075 | $0.30 | 0.12s | High-volume, speed-critical |
| **GPT-4o-mini** | $0.15 | $0.60 | 0.4s | Balanced workloads |
| **Claude Sonnet 4.5** | $3.00 | $15.00 | 3.75s | Complex reasoning, reliability |

### 2. Cost Optimization Strategies

**Use Gemini for bulk operations**:
```python
# Fast preprocessing with Gemini
gemini_node = Node("transform", model_config=ModelConfig(
    provider=ModelProvider.GOOGLE,
    model_name="gemini-2.0-flash"
))

# High-quality reasoning with Claude
claude_node = Node("generate", model_config=ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-sonnet-4-5-20250929"
))

# Pipeline: Fast transform → Quality generation
network = create_network(layers=[gemini_node, claude_node])
```

**Implement response caching**:
```python
# Cache responses for repeated queries
from functools import lru_cache

@lru_cache(maxsize=10000)
async def cached_generation(prompt_hash):
    return await network.forward(prompt)
```

**Optimize token usage**:
```python
# Set appropriate max_tokens
config = ModelConfig(
    max_tokens=500,  # Only request what you need
    temperature=0.7
)

# Trim unnecessary context
def optimize_prompt(prompt):
    # Remove redundant information
    # Keep only essential context
    return prompt[:2000]  # Limit context size
```

### 3. Cost Monitoring

Track costs in real-time:

```python
from nons.observability.integration import get_observability

def calculate_costs():
    obs = get_observability()
    data = obs.export_all_data()

    total_cost = sum(
        span.get('cost_usd', 0)
        for span in data['spans']
    )

    return {
        "total_cost_usd": total_cost,
        "total_requests": len(data['spans']),
        "avg_cost_per_request": total_cost / len(data['spans']) if data['spans'] else 0
    }
```

---

## Production Checklist

Before deploying to production, verify:

- [ ] Scheduler initialization (`await start_scheduler()`) in all agent code
- [ ] API keys configured in environment variables
- [ ] Logging level set to INFO or WARNING
- [ ] Observability enabled (tracing, metrics)
- [ ] Error handling configured (retry with backoff)
- [ ] Rate limiting configured per provider
- [ ] Health check endpoint implemented
- [ ] Cost monitoring enabled
- [ ] Security best practices followed (input validation, secret management)
- [ ] Load testing completed
- [ ] Rollback plan documented
- [ ] Monitoring and alerting configured

---

## Troubleshooting

### Issue: Infinite Hangs

**Symptom**: Requests queue but never complete
**Cause**: Scheduler not started
**Fix**: Call `await start_scheduler()` before agent execution

### Issue: High Latency

**Symptom**: Slow response times
**Cause**: Using Claude for all operations
**Fix**: Use Gemini for speed-critical paths, Claude for complex reasoning

### Issue: Tool Selection Failures

**Symptom**: Agents select wrong tools or generic "tool_name"
**Cause**: Insufficient prompt engineering (especially with Gemini)
**Fix**: Add detailed tool descriptions and examples in system prompts

### Issue: Rate Limit Errors

**Symptom**: HTTP 429 errors from providers
**Cause**: Exceeding provider rate limits
**Fix**: Configure scheduler rate limits appropriately

---

## Support and Resources

- **Latency Report**: See `FINAL_LATENCY_REPORT.md` for comprehensive performance analysis
- **Architecture**: See `docs/architecture.md`
- **API Reference**: See `docs/api-reference.md`
- **Examples**: Check `examples/` directory

---

**NoN is production-ready**. Follow this guide for reliable, scalable deployments.
