"""
RLM Operator - Python REPL with LLM Query Injection

This operator executes Python code with an injected llm_query() function
that allows semantic operations on stored context. The execution environment
persists across calls, enabling iterative refinement.

Key Patterns Demonstrated:
1. Python REPL with persistent state
2. LLM function injection into execution namespace
3. Rate limiting for LLM calls (max 50 per execution)
4. Natural error handling (no pre-validation)
5. StringIO output capture

Architecture:
- RLMOperator: Executes code with llm_query() injection
- Execution environment: Dict storing variables and context
- Rate limiting: Track LLM calls per execution

Safety Note: This uses exec() for demonstration purposes. In production,
consider using restricted execution environments or containers.
"""

from typing import Dict, Any, Optional, Callable
from pydantic import BaseModel, Field
from io import StringIO
import contextlib
import sys
import os
import traceback
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from nons.core.types import ModelConfig, ModelProvider
from nons.utils.providers import create_provider


class RLMExecutionResult(BaseModel):
    """Result of RLM code execution"""

    output: str = Field(description="Captured stdout output")
    result: Any = Field(default=None, description="Final result value from execution")
    llm_calls: int = Field(description="Number of LLM queries made")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    execution_time: float = Field(description="Execution time in seconds")


class RLMOperator:
    """
    Execute Python code with llm_query() function injection.

    The operator maintains a global execution environment where:
    - context variable is stored
    - llm_query(prompt, context_chunk) is available
    - All variables persist across calls
    - LLM calls are tracked and rate-limited

    Architecture:
    - execution_environment: Dict storing all variables
    - llm_call_count: Track LLM usage (max 50)
    - model_config: gpt-5.2 configuration
    """

    def __init__(self, model_config: Optional[ModelConfig] = None, max_llm_calls: int = 50):
        """
        Initialize RLM operator.

        Args:
            model_config: Model configuration (defaults to gpt-5.2)
            max_llm_calls: Maximum LLM calls per execution
        """
        if model_config is None:
            model_config = ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4o-mini",
                temperature=0.7,
                max_tokens=2000
            )

        self.model_config = model_config
        self.max_llm_calls = max_llm_calls

        # Pattern: Global execution environment (from python_interpreter_agent.py)
        self.execution_environment: Dict[str, Any] = {
            "__builtins__": __builtins__,
        }

        # Reset call count for each execution
        self.llm_call_count = 0

    def _create_llm_query_function(self) -> Callable:
        """
        Create the llm_query function to inject into REPL.

        This function will be available in the execution environment and
        allows the code to make semantic queries on context chunks.

        Returns:
            Async function that can be called from executed code
        """
        async def llm_query(prompt: str, context_chunk: str = "") -> str:
            """
            Query LLM with prompt and optional context chunk.

            This function is injected into the REPL execution environment.
            It allows the executed code to make semantic operations on chunks.

            Args:
                prompt: The question/instruction for the LLM
                context_chunk: Optional context chunk to include

            Returns:
                LLM response as string

            Raises:
                RuntimeError: If max LLM calls exceeded
            """
            self.llm_call_count += 1

            if self.llm_call_count > self.max_llm_calls:
                raise RuntimeError(f"Exceeded maximum LLM calls ({self.max_llm_calls})")

            # Create provider and generate completion
            provider = create_provider(self.model_config)

            # Construct full prompt with context
            full_prompt = f"{prompt}\n\nContext:\n{context_chunk}" if context_chunk else prompt

            result, metrics = await provider.generate_completion(full_prompt)
            return result

        return llm_query

    async def execute(self, code: str, context: Optional[Any] = None) -> RLMExecutionResult:
        """
        Execute Python code with llm_query() function available.

        Pattern: Use exec() with StringIO to capture output (from python_interpreter_agent.py)

        Args:
            code: Python code to execute
            context: Optional context to store in execution environment

        Returns:
            RLMExecutionResult with output, result, llm_calls, and error
        """
        start_time = time.time()

        # Reset LLM call count for this execution
        self.llm_call_count = 0

        # Store context in execution environment if provided
        if context is not None:
            self.execution_environment["context"] = context

        # Inject llm_query function
        self.execution_environment["llm_query"] = self._create_llm_query_function()

        # Add asyncio for handling await statements
        import asyncio
        self.execution_environment["asyncio"] = asyncio

        try:
            # Pattern: Capture stdout/stderr (from python_interpreter_agent.py:97-107)
            output_capture = StringIO()
            error_capture = StringIO()

            with contextlib.redirect_stdout(output_capture), \
                 contextlib.redirect_stderr(error_capture):

                # Check if code contains await - if so, wrap in async function and await it
                if "await" in code:
                    # Compile async code and execute
                    # Important: Use globals() to ensure variables are set in execution_environment
                    wrapped_code = f"""
async def __rlm_async_exec():
    global result
{chr(10).join('    ' + line for line in code.split(chr(10)))}
"""
                    # Execute the function definition
                    exec(wrapped_code, self.execution_environment)
                    # Get the function and await it
                    async_func = self.execution_environment["__rlm_async_exec"]
                    await async_func()
                else:
                    # Execute synchronous code directly
                    exec(code, self.execution_environment)

            stdout = output_capture.getvalue()
            stderr = error_capture.getvalue()

            # Try to get result variable if it exists
            result = self.execution_environment.get("result", None)

            execution_time = time.time() - start_time

            return RLMExecutionResult(
                output=stdout if stdout else "No output",
                result=result,
                llm_calls=self.llm_call_count,
                error=None,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time

            # Pattern: Return error in result dict (no pre-validation)
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

            return RLMExecutionResult(
                output="",
                result=None,
                llm_calls=self.llm_call_count,
                error=error_msg,
                execution_time=execution_time
            )

    def reset_environment(self) -> None:
        """Reset execution environment, clearing all variables."""
        self.execution_environment = {
            "__builtins__": __builtins__,
        }
        self.llm_call_count = 0

    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about current execution environment."""
        user_vars = {
            k: type(v).__name__
            for k, v in self.execution_environment.items()
            if not k.startswith("__") and k != "__builtins__" and k != "asyncio"
        }

        return {
            "variables": user_vars,
            "variable_count": len(user_vars),
            "llm_calls": self.llm_call_count,
            "max_llm_calls": self.max_llm_calls
        }
