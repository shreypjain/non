# RLM (Recursive Language Model) Example

Process long documents (10M+ tokens) using LLM-augmented Python REPL with iterative refinement.

## Overview

The RLM example demonstrates how to process extremely long documents that exceed typical LLM context windows. Instead of putting context in the prompt, we:

1. Store context in a Python REPL execution environment
2. Generate Python code to process the context
3. Inject `llm_query(prompt, context_chunk)` function for semantic operations
4. Use Plan → Execute → Verify → Refine loop for iterative improvement

## Architecture

```
User Task + Document
        ↓
    Planner (Node)
        ↓
  Python Code
        ↓
  RLMOperator (execute code)
  - If error → Fixer (Node) → retry
        ↓
  Execution Result
        ↓
   Verifier (Node)
        ↓
  Confidence < 0.8?
    → Refine and repeat
    → Else: Return result
```

## Key Features

1. **Context Storage**: Document stored in REPL, not in prompt
2. **LLM Augmentation**: `llm_query()` function for semantic operations
3. **Automatic Error Fixing**: Single retry with fixer node
4. **Iterative Refinement**: Loop until confidence threshold or max iterations
5. **Rate Limiting**: Max 50 LLM calls per code execution
6. **Full Observability**: Iteration trace with confidence scores

## Usage

### Basic Example

```python
import asyncio
from examples.rlm.rlm_network import RLMNetwork

async def main():
    # Create network
    network = RLMNetwork(
        max_iterations=5,
        confidence_threshold=0.85,
        max_llm_calls_per_execution=50
    )

    # Prepare long document
    document = "..." # Your long document here

    # Run task
    task = "Find all Q3 2022 revenue figures and return the average."
    result = await network.run(task, document)

    # Check result
    print(f"Success: {result.success}")
    print(f"Output: {result.final_output}")
    print(f"Iterations: {result.total_iterations}")
    print(f"Confidence: {result.final_confidence}")

asyncio.run(main())
```

### Run Demos

```bash
cd examples/rlm
uv run python demo.py
```

The demo includes 3 benchmark problems:
1. Find specific information (Q3 revenues)
2. Count/aggregate data (mentions of "AI")
3. Multi-hop reasoning (who said what about what)

## How It Works

### 1. Planner Node

Generates Python code that:
- Accesses `context` variable (the long document)
- Uses `llm_query(prompt, context_chunk)` for semantic operations
- Stores final answer in `result` variable
- Uses `print()` for debugging

### 2. RLM Operator

Executes Python code with:
- Persistent execution environment
- Injected `llm_query()` async function
- Rate limiting (max 50 LLM calls)
- Output capture via StringIO
- Natural error handling (no pre-validation)

### 3. Fixer Node

If execution fails:
- Analyzes error message
- Fixes code (common issues: forgot await, chunks too large)
- Retries execution once

### 4. Verifier Node

Analyzes execution result:
- Returns confidence score 0-1
- Provides reasoning
- Informs refinement loop

### 5. Loop Control

Stops when:
- Confidence > threshold (0.85 default)
- Max iterations reached (5 default)
- Persistent error after fix

## Implementation Details

### LLM Query Function

The injected `llm_query()` function:

```python
async def llm_query(prompt: str, context_chunk: str = "") -> str:
    """
    Query LLM with prompt and optional context chunk.

    Use this for semantic operations like:
    - "Is this section about X?"
    - "Summarize this paragraph"
    - "Extract key points from this text"

    Args:
        prompt: Your question/instruction
        context_chunk: Small piece of context (keep under 10k tokens)

    Returns:
        LLM response as string
    """
```

### Example Code Pattern

Generated code typically follows this pattern:

```python
# Split long context into manageable chunks
chunks = context.split('\n\n')

# Filter and process chunks
results = []
for chunk in chunks:
    # Use regular Python for filtering
    if len(chunk) > 100 and 'Q3 2022' in chunk:
        # Use llm_query for semantic operations
        response = await llm_query(
            "Extract the revenue figure from this section",
            chunk
        )
        results.append(response)

# Aggregate results
result = sum(float(r.replace('$', '').replace('M', '')) for r in results) / len(results)
print(f"Average Q3 2022 revenue: ${result}M")
```

## Configuration

### RLMNetwork Parameters

```python
RLMNetwork(
    max_iterations=5,           # Max refinement iterations
    confidence_threshold=0.8,   # Stop if confidence exceeds this
    max_llm_calls_per_execution=50  # Rate limit per code execution
)
```

### Model Configuration

All nodes use `gpt-5.2` by default (cheap and fast). You can override:

```python
from nons.core.types import ModelConfig, ModelProvider

custom_config = ModelConfig(
    provider=ModelProvider.OPENAI,
    model_name="gpt-4o",
    temperature=0.7,
    max_tokens=2000
)

operator = RLMOperator(model_config=custom_config)
```

## Testing

Run tests:

```bash
uv run pytest tests/test_rlm.py -v
```

Tests cover:
- Basic code execution
- LLM query injection
- Error handling and fixing
- Rate limiting
- Iteration logic
- Confidence-based stopping

## Troubleshooting

### Common Issues

1. **"Exceeded maximum LLM calls"**
   - Reduce chunk size or filter more before calling llm_query
   - Increase max_llm_calls_per_execution

2. **"NameError: name 'llm_query' is not defined"**
   - Forgot to use `await` with llm_query (it's async)
   - Fixer should catch this automatically

3. **Low confidence scores**
   - Task may be too complex for simple code
   - Try increasing max_iterations
   - Check iteration trace to see what's failing

4. **Slow execution**
   - Processing very long documents takes time
   - Reduce max_llm_calls_per_execution for faster testing
   - Use smaller document for debugging

## Design Philosophy

RLM demonstrates the "computation over memorization" paradigm:

- **Traditional LLM**: Put entire document in context window (limited to ~1M tokens)
- **RLM**: Store document in REPL, use code to compute over it (unlimited size)

### Benefits
- Process arbitrarily long documents
- Explicit, debuggable processing logic (it's Python code)
- Efficient token usage (only pass small chunks to LLM)
- Iterative refinement until high confidence

### Trade-offs
- More LLM calls (planning + verification + queries)
- Slower than single-shot inference
- Requires code generation capability

### Best for
- Documents > 1M tokens
- Tasks requiring multiple passes over data
- Need for transparency and debugging
- Cost-sensitive applications (process once, cache results)

## Components

- `rlm_operator.py`: Python REPL with llm_query() injection
- `rlm_network.py`: Plan-Execute-Verify-Refine loop
- `demo.py`: OOLONG-style benchmark demonstrations
- `README.md`: This file
