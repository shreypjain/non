# Quick Deployment Guide

Get NoN running in production in under 5 minutes.

## Prerequisites

- Python 3.9+ OR Docker
- API key for at least one LLM provider (Anthropic, Google, or OpenAI)

## Option 1: PyPI Installation (Recommended)

### Step 1: Install from PyPI

```bash
pip install nons
```

### Step 2: Set API Keys

```bash
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

### Step 3: Run Your Application

```python
import asyncio
from nons.core.network import create_network
from nons.core.node import Node
from nons.core.scheduler import start_scheduler

async def main():
    # Critical: Start scheduler first!
    await start_scheduler()

    # Create your network
    network = create_network(
        layers=[Node("generate")],
        provider="anthropic",
        model="claude-sonnet-4-5-20250929"
    )

    # Run inference
    result = await network.forward("Hello, NoN!")
    print(result)

asyncio.run(main())
```

## Option 2: Docker Deployment

### Step 1: Clone Repository

```bash
git clone https://github.com/shreypjain/non.git
cd non
```

### Step 2: Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Step 3: Run with Docker Compose

```bash
docker-compose up -d
```

Your application will be available in the container.

## Option 3: From Source

### Step 1: Clone and Install

```bash
git clone https://github.com/shreypjain/non.git
cd non
uv sync
```

### Step 2: Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Step 3: Run

```bash
uv run python your_app.py
```

## Provider Recommendations

Based on comprehensive latency testing (see `FINAL_LATENCY_REPORT.md`):

### For Production Reliability
Use **Anthropic Claude Sonnet 4.5**:
- Average latency: 3-5 seconds
- Reliability: Excellent (100% success rate)
- Tool execution: Working perfectly
- Multi-agent support: Fully functional

```python
config = ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-sonnet-4-5-20250929"
)
```

### For Speed-Critical Applications
Use **Google Gemini 2.0 Flash**:
- Average latency: 0.12-0.15 seconds (25-30x faster!)
- Best for: High-volume, latency-sensitive workloads
- Note: Requires good prompt engineering

```python
config = ModelConfig(
    provider=ModelProvider.GOOGLE,
    model_name="gemini-2.0-flash"
)
```

### For Balanced Workloads
Use **OpenAI GPT-4o-mini**:
- Average latency: ~0.4-0.5 seconds
- Good balance of speed and capability

```python
config = ModelConfig(
    provider=ModelProvider.OPENAI,
    model_name="gpt-4o-mini"
)
```

## Critical Production Requirements

### 1. Always Start the Scheduler

```python
from nons.core.scheduler import start_scheduler

async def main():
    # MUST call this before any agent execution!
    await start_scheduler()

    # Now safe to run agents
    result = await agent.run(task)
```

**Why**: Without starting the scheduler, requests will queue indefinitely causing infinite hangs.

### 2. Configure Rate Limits

```python
from nons.core.scheduler import configure_scheduler, RateLimitConfig

scheduler = configure_scheduler(
    rate_limits={
        ModelProvider.ANTHROPIC: RateLimitConfig(
            requests_per_minute=100,
            max_concurrent=10
        )
    }
)
```

### 3. Enable Observability

```python
from nons.observability.integration import get_observability

# Get telemetry data
obs = get_observability()
data = obs.export_all_data()

# Monitor costs and performance
print(f"Total cost: ${sum(s.get('cost_usd', 0) for s in data['spans']):.6f}")
```

## Kubernetes Deployment

### Step 1: Create Secret

```bash
kubectl create secret generic non-secrets \
  --from-literal=ANTHROPIC_API_KEY=your-key \
  --from-literal=GOOGLE_API_KEY=your-key
```

### Step 2: Deploy

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: non-app
spec:
  replicas: 3
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
        image: ghcr.io/shreypjain/non:latest
        envFrom:
        - secretRef:
            name: non-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### Step 3: Apply

```bash
kubectl apply -f deployment.yaml
```

## Health Checks

Implement health checks for monitoring:

```python
async def health_check():
    from nons.core.scheduler import get_scheduler

    scheduler = get_scheduler()
    return {
        "status": "healthy" if scheduler.is_running else "unhealthy",
        "queue_size": len(scheduler.queue)
    }
```

## Monitoring

### Prometheus Metrics

Run with monitoring:

```bash
docker-compose --profile monitoring up -d
```

Access:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Database Export

For persistent storage of observability data:

```python
export NON_EXPORT_TO_DATABASE=true
export NON_DATABASE_URL=postgresql://user:pass@localhost:5432/non_observability
```

## Scaling

### Horizontal Scaling

NoN agents are stateless - scale horizontally:

```bash
# Kubernetes
kubectl scale deployment non-app --replicas=10

# Docker Compose
docker-compose up -d --scale non-app=5
```

### Vertical Scaling

Recommended resources per instance:
- Memory: 2-4GB
- CPU: 1-2 cores
- Concurrent requests: 10-20

## Troubleshooting

### Issue: Requests hang indefinitely

**Solution**: Ensure `await start_scheduler()` is called before agent execution.

### Issue: High latency

**Solution**: Switch to Gemini for speed-critical paths, use Claude for complex reasoning.

### Issue: Rate limit errors

**Solution**: Configure appropriate rate limits in scheduler.

## Support

- **Full Guide**: See `PRODUCTION_GUIDE.md` for comprehensive documentation
- **Latency Report**: See `FINAL_LATENCY_REPORT.md` for performance analysis
- **Architecture**: See `docs/architecture.md`
- **Issues**: https://github.com/shreypjain/non/issues

## Next Steps

1. Read the full production guide: `PRODUCTION_GUIDE.md`
2. Review latency testing results: `FINAL_LATENCY_REPORT.md`
3. Explore examples: `examples/` directory
4. Set up monitoring and observability

---

**NoN is production-ready**. Get started in under 5 minutes!
