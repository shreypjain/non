# Skill: NoN (Network of Networks) Development

## When to Use

Use this skill when working on the `nons` package -- building, extending, debugging, or demonstrating network orchestration pipelines. This includes:

- Creating or modifying nodes, layers, and networks
- Building agent workflows with tool registries
- Adding or updating operators
- Configuring observability, scheduling, or rate limiting
- Writing examples or tests for the NoN framework

---

## Project Layout

```
nons/                    # Main package (import as `nons`)
  core/                  # Node, Layer, NoN, types, config, scheduler, agents
    agents/              # Agent + ToolRegistry for agentic workflows
  operators/             # Built-in operators (base.py) and deterministic ops
  observability/         # Tracing, logging, metrics
  runtime/               # Runtime stubs (currently empty)
  utils/                 # Provider abstraction, helpers
examples/                # Runnable demo scripts
tests/                   # pytest test suite
```

---

## Setup and Execution

All Python execution uses `uv`:

```bash
# Install dependencies
uv sync

# Run a script
uv run python examples/basic_non_demo.py

# Run tests
uv run pytest

# Code quality
uv run ruff check .
uv run black .
```

Required environment variables (at least one provider key):

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
```

---

## Core Concepts

### 1. Node

A `Node` wraps a single operator with model configuration. It is the smallest execution unit.

```python
from nons import Node, ModelConfig, ModelProvider

# Default config (auto-detects provider from env)
node = Node("generate")

# Explicit model config
node = Node("generate", model_config=ModelConfig(
    provider=ModelProvider.OPENAI,
    model_name="gpt-4o-mini",
    temperature=0.1,
    max_tokens=500,
))

# With additional prompt context
node = Node("classify", additional_prompt_context="Focus on sentiment analysis.")

# Parallel cloning via multiplication
parallel_nodes = Node("generate") * 5  # Returns List[Node]
```

**Execution:**

```python
result = await node.execute("input text")
```

### 2. Layer

A `Layer` holds nodes that execute in parallel via `asyncio.gather`.

```python
from nons import Layer, create_layer, create_parallel_layer, LayerConfig, ErrorPolicy

# From nodes
layer = Layer([Node("classify"), Node("extract")])

# Factory helpers
layer = create_layer(node_a, node_b)
layer = Layer.from_operators(["classify", "extract"])

# With error handling
layer = Layer(nodes, layer_config=LayerConfig(
    error_policy=ErrorPolicy.SKIP_AND_CONTINUE,
    min_success_threshold=0.7,
    max_retries=3,
    timeout_seconds=30.0,
))
```

**Execution:**

```python
result = await layer.execute_parallel("input text")
# result.outputs, result.success_rate, result.execution_time
```

### 3. NoN (Network)

A `NoN` chains layers sequentially. Each layer's output feeds the next layer's input.

```python
from nons import NoN, create_network

# Recommended: from_operators factory
network = NoN.from_operators([
    "transform",                    # Layer 0: single node
    ["classify", "extract"],        # Layer 1: parallel nodes
    "synthesize",                   # Layer 2: single node
])

# With node multiplication
parallel = Node("generate") * 3
network = NoN.from_operators([
    "transform",
    parallel,       # Layer 1: three parallel generate nodes
    "condense",
])

# Manual construction
network = NoN.from_layers(layer1, layer2, layer3)

# create_network helper (used in agent examples)
network = create_network(
    layers=[Node("route")],
    provider="openai",
    model="gpt-4o-mini",
    system_prompt="You are a planning agent.",
)
```

**Execution:**

```python
import asyncio

async def main():
    network = NoN.from_operators(["generate", "condense"])
    result = await network.forward("Explain quantum computing")
    print(result.final_output)
    print(result.execution_time, result.total_nodes_executed)

asyncio.run(main())
```

**Dynamic modification:**

```python
network.add_layer(new_layer)
network.insert_layer(index=1, layer=new_layer)
network.remove_layer(layer_id="...")
```

---

## Agents: Tool-Augmented Networks

For multi-step, tool-using workflows, use `Agent` with a `ToolRegistry`.

### ToolRegistry

```python
from nons.core.agents.registry import ToolRegistry
from pydantic import BaseModel, Field

class AddParams(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

registry = ToolRegistry()

@registry.tool(name="add", description="Add two numbers", param_model=AddParams)
async def add(params: AddParams, **context):
    return {"result": params.a + params.b}

# Stop tool (terminates agent loop)
class StopParams(BaseModel):
    final_answer: str = Field(description="The final answer")

@registry.tool(name="stop", description="Return final answer", param_model=StopParams, is_stop_tool=True)
async def stop(params: StopParams, **context):
    return {"final_answer": params.final_answer}
```

### Agent

```python
from nons.core.agents.agent import Agent
from nons import NoN, create_network

network = create_network(
    layers=[Node("route")],
    provider="openai",
    model="gpt-4o-mini",
    system_prompt="You are a calculator agent.",
)

agent = Agent(
    network=network,
    registry=registry,
    max_steps=10,
    max_llm_retry=3,
)

# Streaming execution (async generator)
async for result in agent.run("What is 15 + 7?"):
    print(result["step"], result["tool"], result["reasoning"])
    if result.get("is_stop"):
        print("Final:", result["result"]["final_answer"])
```

### Multi-Agent Pattern

Sub-agents are composed by registering delegation tools on a planner agent:

```python
@planner_registry.tool(name="delegate_to_coder", description="...", param_model=DelegateParams)
async def delegate(params, **context):
    results = []
    async for r in sub_agents["coder"].run(params.task):
        results.append(r)
    return {"sub_agent": "coder", "results": results[-1]}
```

---

## Built-in Operators

| Operator | Purpose |
|----------|---------|
| `transform` | Reformat or clean content |
| `generate` | Generate new content (uses LLM provider) |
| `classify` | Categorize content |
| `extract` | Pull out specific information |
| `condense` | Summarize/compress |
| `expand` | Add detail and context |
| `compare` | Find similarities/differences |
| `validate` | Check correctness or quality |
| `route` | Determine next workflow step (used in agents) |
| `synthesize` | Combine multiple inputs |
| `pack_candidates` | Structure outputs for ensemble voting |
| `extract_winners` | Best-of-N selection |
| `majority` | Majority vote across candidates |
| `select_by_id` | Filter specific candidates |

Register a custom operator with the `@operator` decorator from `nons.operators.registry`.

---

## Configuration

### ModelConfig

```python
from nons import ModelConfig, ModelProvider

config = ModelConfig(
    provider=ModelProvider.OPENAI,     # OPENAI | ANTHROPIC | GOOGLE
    model_name="gpt-4o-mini",
    temperature=0.7,
    max_tokens=4000,
    top_p=None,
    frequency_penalty=None,
    presence_penalty=None,
    extra_params={},
)
```

### LayerConfig

```python
from nons import LayerConfig, ErrorPolicy

config = LayerConfig(
    error_policy=ErrorPolicy.RETRY_WITH_BACKOFF,  # FAIL_FAST | RETRY_WITH_BACKOFF | SKIP_AND_CONTINUE | RETURN_PARTIAL
    min_success_threshold=1.0,
    max_retries=3,
    retry_delay_seconds=1.0,
    timeout_seconds=30.0,
)
```

### Environment Variable Overrides

| Variable | Effect |
|----------|--------|
| `NON_DEFAULT_OPENAI_MODEL` | Override default OpenAI model |
| `NON_DEFAULT_TEMPERATURE` | Override default temperature |
| `NON_DEFAULT_MAX_TOKENS` | Override default max tokens |
| `NON_LAYER_TIMEOUT` | Override layer timeout |
| `NON_MAX_RETRIES` | Override max retries |
| `NON_ENABLE_TRACING` | Toggle tracing (`true`/`false`) |
| `NON_ENABLE_METRICS` | Toggle metrics (`true`/`false`) |
| `NON_LOG_LEVEL` | Set log level |

---

## Observability

```python
from nons import configure_observability, get_observability
from nons import trace_network_operation, trace_layer_operation, trace_node_operation

# Configure
obs = configure_observability(
    enable_tracing=True,
    enable_logging=True,
    enable_metrics=True,
)

# Use trace decorators on async functions
@trace_network_operation("my_pipeline", network_id="net-001")
async def run_pipeline(prompt):
    ...

# Export all observability data
data = obs.export_all_data()
# data == {"spans": [...], "logs": [...], "metrics": [...]}

# Get trace summary
summary = obs.get_trace_summary(trace_id="...")
```

---

## Scheduler and Rate Limiting

```python
from nons import configure_scheduler, get_scheduler, RateLimitConfig, QueueStrategy, ModelProvider

scheduler = configure_scheduler(
    rate_limits={
        ModelProvider.OPENAI: RateLimitConfig(
            requests_per_minute=100,
            requests_per_second=5,
            max_concurrent=10,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
        ),
    },
    queue_strategy=QueueStrategy.PRIORITY,  # FIFO | PRIORITY | ROUND_ROBIN | LEAST_LOADED
    enable_observability=True,
)

# Start/stop (required for scheduling to process the queue)
from nons.core.scheduler import start_scheduler, stop_scheduler
await start_scheduler()
# ... run networks ...
await stop_scheduler()
```

---

## Deterministic Operators

For ensemble/voting patterns without LLM calls:

```python
from nons.operators.deterministic import (
    PackCandidates, ExtractWinners, Majority, SelectById,
    Candidate, PackedCandidates,
)

# Pack raw outputs into candidates
packer = PackCandidates()
packed = packer(["output1", "output2", "output3"])

# Extract top-k winners
extractor = ExtractWinners(strategy="top_k", k=1)
winners = extractor(packed)

# Majority vote
voter = Majority(strategy="simple", min_consensus=0.5)
result = voter(packed)
```

---

## Testing Patterns

Tests live in `/tests/` and use pytest with async support:

```python
import pytest
from nons import NoN, Node

@pytest.mark.asyncio
async def test_network_forward():
    network = NoN.from_operators(["transform", "condense"])
    result = await network.forward("test input")
    assert result.final_output is not None
    assert result.total_layers == 2
```

Run tests:

```bash
uv run pytest
uv run pytest tests/test_node.py -v
uv run pytest -k "test_network" -v
```

---

## Common Patterns

### Pipeline with Cost Tracking

```python
async def tracked_pipeline():
    node = Node("generate", model_config=ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini",
    ))
    network = NoN.from_operators([node, "condense"])
    result = await network.forward("Write a poem")

    # Per-node cost and token tracking
    print(node.get_total_cost())
    print(node.get_total_tokens())
    print(node.get_execution_stats())
```

### Best-of-N with Voting

```python
generator = Node("generate") * 5
network = NoN.from_operators([
    "transform",
    generator,           # 5 parallel generations
    "pack_candidates",   # Pack into candidates
    "majority",          # Vote on best
])
result = await network.forward("Solve this problem...")
```

---

## Key Rules When Developing

1. Always use `uv run` for executing Python files.
2. Format code with `black` before committing.
3. Every directory must have a `README.md`.
4. Every file must have a docstring. Functions and classes must have docstrings with parameter and return type info.
5. Search for existing functions before writing new ones.
6. Check for circular dependencies before adding new modules.
7. Do not use emojis in documentation.
8. All execution is async -- use `asyncio.run(main())` as the entry point.
