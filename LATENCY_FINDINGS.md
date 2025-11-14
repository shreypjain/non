# Multi-Agent Latency Investigation - Comprehensive Findings

## Executive Summary

Successfully identified and partially fixed the critical hang issue in the multi-agent demo. The root cause was **the scheduler not being started**. Additional issues were discovered with API access and agent parsing.

## Critical Findings

### 1. ROOT CAUSE: Scheduler Not Running

**Issue**: The `RequestScheduler` must be explicitly started with `await start_scheduler()` before any agent can process requests.

**Evidence**:
- Scheduler was initialized with `is_running = False` (line 142 in scheduler.py)
- `schedule_request()` adds requests to a queue and waits for `request_item.future` to resolve
- The future is only resolved by `_scheduler_loop()` which only runs if `is_running = True`
- Result: Infinite hang waiting for a future that will never resolve

**Fix Applied**: Agent initialization must call `await start_scheduler()` before use

**Test Results**:
- Before fix: Hung indefinitely after scheduler logs "Completed request_scheduled"
- After fix: Request execution completes in **1-2ms**

**Latency Measurements** (with MockProvider):
- Request scheduling: ~0.15-0.40ms
- Request execution: ~1.7-2.1ms
- Total per LLM call: ~2-3ms (mock provider overhead)

### 2. OpenAI API Access Denied

**Issue**: The OpenAI API key exists but returns "Access denied" errors

**Error**:
```
openai.PermissionDeniedError: Access denied
```

**Impact**: Cannot measure real API latency or test with actual OpenAI models

**Recommendations**:
- Verify API key is valid and not expired
- Check if key has sufficient credits/permissions
- Check for rate limits or IP restrictions
- Test with a fresh API key if needed

### 3. Agent Parsing Issues

**Issue**: Agent's `_llm_forward()` method had multiple problems:

1. **Type Mismatch**: Tried to call `RouteDecision.model_validate()` but `RouteDecision` is a `TypedDict`, not a Pydantic `BaseModel`

2. **JSON Extraction**: MockProvider wraps output in `[MOCK MODEL: ... ]` format, making JSON extraction difficult

**Fixes Applied**:
- Changed from `model_validate()` to manual dict parsing
- Added robust JSON extraction from wrapped text
- Added balanced brace extraction for nested JSON
- Better error messages showing actual output

**Remaining Issues**:
- MockProvider returns the input prompt, not structured JSON responses
- For proper testing, need either:  - Real API access, OR
  - Smarter mock provider that generates valid JSON, OR
  - Pre-canned JSON responses for testing

## Architecture Insights

### How the Scheduler Works

1. **Initialization**: Global scheduler created but not started
2. **Request Flow**:
   ```
   Node.execute()
   → scheduler.schedule_request(operation)
   → Add to queue + create Future
   → Wait for future.result()  ← HANGS HERE if scheduler not running
   ```

3. **Processing** (only if started):
   ```
   _scheduler_loop() runs continuously
   → _process_all_queues()
   → _process_provider_queue()
   → Execute queued operations
   → Resolve futures
   ```

4. **Required**:  - Must call `await start_scheduler()` ONCE at app startup
   - Scheduler runs as background task
   - Handles rate limiting, queueing, and concurrent request management

### Agent Execution Flow

```
Agent.run(input)
→ Agent.step(state)
  → Agent._llm_forward(state)
    → network.forward(state)
      → Layer.execute_parallel()
        → Node.execute()
          → Node._execute_with_provider()
            → scheduler.schedule_request()  ← Needs running scheduler
              → provider.generate_completion()
            → Return result
      → Parse as RouteDecision
    → Validate decision
  → Agent._execute_tool(decision)
    → registry.execute(tool_name, params)
  → Return step result
→ Yield result to caller
```

## Test Infrastructure Created

### 1. `test_real_api_latency.py`
- Tests provider selection (verifies real vs mock)
- Tests simple agent execution
- Measures actual latency with real API calls
- **Status**: Cannot run due to API access denied

### 2. `test_simple_agent_no_hang.py`
- Progressive diagnosis test
- Tests each layer: MockProvider → Node → Network → Agent
- Identified hang at Node level (scheduler not started)
- **Result**: Confirmed scheduler as root cause

### 3. `test_scheduler_fix.py`
- Tests agent execution WITH scheduler started
- Measures latency with mock provider
- Tests both simple agent and multi-agent planner
- **Result**: Executions complete, but JSON parsing issues remain

### 4. `test_multi_agent_latency.py` (from earlier)
- Comprehensive latency measurement framework
- Tracks per-step, per-agent, and per-tool timing
- Supports multiple test runs with statistics
- **Status**: Ready to use once all issues resolved

## Latency Baseline (MockProvider)

With scheduler properly started:

| Metric | Time |
|--------|------|
| Scheduler startup | ~0.3-0.5ms |
| Request scheduling | ~0.15-0.4ms |
| Request execution | ~1.7-2.1ms |
| **Total per call** | **~2-3ms** |

**Note**: These are mock provider times. Real API latency will be 100-1000x higher (typical OpenAI latency: 500-2000ms).

## Fixes Applied

### Files Modified:

1. **nons/core/agents/agent.py**
   - Fixed `_llm_forward()` to handle TypedDict instead of Pydantic model
   - Added robust JSON extraction from wrapped text
   - Added balanced brace extraction
   - Improved error messages

2. **nons/core/network.py** (from earlier commits)
   - Extended `create_network()` to support provider/model/system_prompt parameters
   - Added automatic ModelConfig creation
   - Fixed layers as keyword argument handling

3. **examples/*.py** (from earlier commits)
   - Changed from `Node("route")` to `Node("generate")`
   - Added `import nons.operators.base` to register operators

## Remaining Work

### High Priority

1. **Scheduler Auto-Start**: Add automatic scheduler startup when first agent is created
   - Check if scheduler is running before `schedule_request()`
   - Auto-start if not running
   - OR require explicit initialization in docs

2. **Fix OpenAI API Access**: Resolve the "Access denied" error
   - Verify/replace API key
   - Test with different key/account
   - Check quotas and limits

3. **Agent JSON Generation**: Ensure agents generate proper structured output
   - Test with real API (once access fixed)
   - OR create smarter MockProvider for testing
   - OR add response templates for common patterns

### Medium Priority

4. **Add Scheduler Documentation**: Document that scheduler must be started

5. **Add Helper Functions**: Create convenience functions that handle scheduler startup

6. **Improve MockProvider**: Make it generate realistic structured responses for testing

## Recommendations

### For Immediate Testing

1. Fix OpenAI API access to enable real latency measurements
2. OR improve MockProvider to generate proper JSON responses
3. Add `await start_scheduler()` to all example scripts
4. Update documentation to mention scheduler requirements

### For Production

1. Add automatic scheduler startup (with option to disable)
2. Add clear error messages if scheduler not running
3. Add scheduler health checks
4. Consider making scheduler optional for simple use cases

### For Latency Optimization

Once working:
1. Measure baseline with real API
2. Test concurrent request handling
3. Profile scheduler overhead
4. Optimize queue processing
5. Test different rate limit configurations

## Files Created

- `LATENCY_FINDINGS.md` (this file)
- `MULTI_AGENT_STATUS.md` (from earlier)
- `test_real_api_latency.py`
- `test_scheduler_fix.py`
- `test_simple_agent_no_hang.py`
- `test_multi_agent_latency.py` (from earlier)
- `test_multi_agent_mock.py` (from earlier)

## Conclusion

The **main latency issue (hang) is SOLVED** - it was caused by the scheduler not being started. The fix is simple: call `await start_scheduler()` before using agents.

**Remaining issues** are secondary:
- API access (external problem)
- JSON parsing (can be worked around)
- MockProvider limitations (for testing only)

The infrastructure for comprehensive latency measurement is in place and ready to use once API access is restored.
