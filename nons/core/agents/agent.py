from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Optional,
)
from pydantic import ValidationError
import time

from nons.core.network import NoN
from .registry import ToolRegistry
from nons.core.types import RouteDecision

AsyncCallable = Callable[..., Awaitable[Any]]


class Agent:
    """
    Orchestrates LLM reasoning with tool execution.
    Handles nested agent dispatch and state management.
    """

    def __init__(
        self,
        network: NoN,
        registry: ToolRegistry,
        max_steps: int = 100,
        max_llm_retry: int = 3,
        state_adapter: Optional[Callable[[Any], Dict[str, Any]]] = None,
        merge_fn: Optional[
            Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
        ] = None,
    ):
        self.network = network
        self.registry = registry
        self.max_steps = max_steps
        self.max_llm_retry = max_llm_retry
        self.state_adapter = state_adapter or (
            lambda x: x if isinstance(x, dict) else {"content": x}
        )
        self.merge_fn = merge_fn or self._default_merge

    async def _llm_forward(self, state: Dict[str, Any]) -> RouteDecision:
        """
        Pure LLM forward pass. Returns routing decision.
        No side effects - just network execution.
        """
        import json
        import re

        for attempt in range(self.max_llm_retry):
            try:
                result = await self.network.forward(state)
                # RouteDecision is a TypedDict, parse from result
                output = result.final_output

                if isinstance(output, dict):
                    decision = output
                elif isinstance(output, str):
                    # Try to find JSON in the string (handle wrapped JSON)
                    # First try direct parsing
                    try:
                        decision = json.loads(output)
                    except json.JSONDecodeError:
                        # Try to extract JSON object from text (handles nested braces better)
                        json_match = re.search(r'\{[^{}]*"selected_path"[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', output, re.DOTALL)
                        if json_match:
                            try:
                                decision = json.loads(json_match.group())
                            except json.JSONDecodeError:
                                # More aggressive extraction - find balanced braces
                                start = output.find('{')
                                if start != -1:
                                    brace_count = 0
                                    for i in range(start, len(output)):
                                        if output[i] == '{':
                                            brace_count += 1
                                        elif output[i] == '}':
                                            brace_count -= 1
                                            if brace_count == 0:
                                                decision = json.loads(output[start:i+1])
                                                break
                                else:
                                    raise ValueError(f"Could not find valid JSON in output: {output[:200]}")
                        else:
                            raise ValueError(f"Could not find valid JSON in output: {output[:200]}")
                else:
                    raise ValueError(f"Unexpected output type: {type(output)}")

                # Validate required fields
                required_fields = ["selected_path", "routing_confidence", "reasoning"]
                missing = [f for f in required_fields if f not in decision]
                if missing:
                    raise ValidationError(f"Missing required fields: {missing}")

                return decision
            except (ValidationError, ValueError, KeyError, json.JSONDecodeError) as e:
                if attempt == self.max_llm_retry - 1:
                    raise ValueError(
                        f"LLM failed to return valid routing decision after {self.max_llm_retry} attempts: {e}\nLast output: {output[:500] if 'output' in locals() else 'N/A'}"
                    )
                continue

    async def _execute_tool(
        self, decision: RouteDecision, state: Dict[str, Any], **context
    ) -> Dict[str, Any]:
        """
        Execute and validate tool call.
        Returns standardized tool result with success/error status.
        """
        tool_result = await self.registry.execute(
            decision["selected_path"],  # Dict access, not attribute
            decision.get("params", {}),  # Dict access with default
            state=state,
            **context,
        )

        return {
            "tool": decision["selected_path"],
            "reasoning": decision.get("reasoning", ""),
            "confidence": decision.get("routing_confidence", 0.0),
            "result": tool_result,
            "success": tool_result.get("success", False),
            "is_stop": tool_result.get("is_stop_tool", False),
        }

    def _default_merge(
        self, state: Dict[str, Any], tool_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Default state merge and append strategy"""
        if isinstance(tool_output.get("output"), dict):
            return {**state, **tool_output["output"]}
        else:
            return {**state, "last_tool_output": tool_output.get("output")}

    async def step(self, state: Dict[str, Any], **context) -> Dict[str, Any]:
        """
        Single agent step: LLM forward + tool execution.
        Returns complete step result.
        """
        # LLM reasoning
        decision = await self._llm_forward(state)

        # Tool execution
        tool_result = await self._execute_tool(decision, state, **context)

        return tool_result

    async def run(
        self, initial_input: Any, **context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main orchestration loop.
        Yields incremental results: reasoning + tool outputs.
        """
        state = self.state_adapter(initial_input)

        for step_num in range(self.max_steps):
            try:
                # Execute single step
                step_result = await self.step(state, **context)

                # Yield result with step metadata
                yield {"step": step_num + 1, "timestamp": time.time(), **step_result}

                if step_result["is_stop"]:
                    return

                if not step_result["success"]:
                    # Could implement retry logic here
                    yield {
                        "step": step_num + 1,
                        "error": step_result["result"].get("error"),
                        "retry_available": True,
                    }
                    continue

                state = self.merge_fn(state, step_result["result"])

            except Exception as e:
                yield {"step": step_num + 1, "fatal_error": str(e), "state": state}
                raise

        yield {
            "warning": f"Reached max_steps ({self.max_steps}) without termination",
            "final_state": state,
        }
