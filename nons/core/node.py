"""
Node implementation for NoN (Network of Networks).

A Node is a singular instance of an Operator placed in a specific position
of a NoN with additional configuration like prompt context, model settings,
and execution parameters.
"""

import uuid
import time
from typing import Any, Dict, Optional, Union, List
from ..operators.registry import RegisteredOperator, get_operator
from .types import (
    Content,
    ModelConfig,
    ExecutionContext,
    ErrorPolicy,
    OperatorError,
    ValidationError,
    ModelProvider,
    ExecutionMetrics,
    TokenUsage,
    CostInfo,
)
from .config import get_default_model_config
from ..utils.providers import create_provider
import os


class Node:
    """
    A singular instance of an Operator with specific configuration.

    Nodes have additional prompt context, model name, temperature, and other
    decoding hyperparameters injected into their specific instance.
    """

    def __init__(
        self,
        operator_name: str,
        model_config: Optional[ModelConfig] = None,
        additional_prompt_context: str = "",
        node_id: Optional[str] = None,
    ):
        """
        Initialize a Node with an operator and configuration.

        Args:
            operator_name: Name of the registered operator to use
            model_config: Model configuration for this node
            additional_prompt_context: Additional context to inject into prompts
            node_id: Optional node ID (auto-generated if not provided)
        """
        self.node_id = node_id or str(uuid.uuid4())
        self.operator_name = operator_name
        self.additional_prompt_context = additional_prompt_context

        # Get the registered operator
        try:
            self.operator: RegisteredOperator = get_operator(operator_name)
        except ValidationError as e:
            raise ValidationError(f"Failed to create node: {e}")

        # Set model config with defaults if not provided
        self.model_config = model_config or get_default_model_config()

        # Runtime state
        self._execution_count = 0
        self._last_execution_time: Optional[float] = None
        self._last_error: Optional[Exception] = None

        # Cost and token tracking
        self._total_tokens = TokenUsage()
        self._total_cost = CostInfo()
        self._execution_metrics: List[ExecutionMetrics] = []
        self._last_metrics: Optional[ExecutionMetrics] = None

    def __repr__(self) -> str:
        return f"Node(id={self.node_id[:8]}, operator={self.operator_name}, model={self.model_config.model_name})"

    def __str__(self) -> str:
        """Detailed string representation with execution stats and cost/token info."""
        lines = [
            f"┌─ Node: {self.operator_name} ─┐",
            f"│ ID: {self.node_id[:8]}...    │",
            f"│ Model: {self.model_config.model_name} │",
            f"│ Provider: {self.model_config.provider.value} │",
            f"│ Temperature: {self.model_config.temperature} │",
            f"│ Executions: {self._execution_count} │",
        ]

        if self._last_execution_time is not None:
            lines.append(f"│ Last Time: {self._last_execution_time:.3f}s │")

        # Add cost and token information
        if self._execution_count > 0:
            lines.extend(
                [
                    "├─ Usage & Cost ─┤",
                    f"│ Total Tokens: {self._total_tokens.total_tokens:,} │",
                    f"│ Total Cost: ${self._total_cost.total_cost_usd:.6f} │",
                ]
            )

            if self._last_metrics:
                lines.extend(
                    [
                        f"│ Last Tokens: {self._last_metrics.token_usage.total_tokens} │",
                        f"│ Last Cost: ${self._last_metrics.cost_info.total_cost_usd:.6f} │",
                    ]
                )

        if self._last_error:
            lines.append(f"│ Last Error: ❌ │")
        else:
            lines.append(f"│ Status: ✅ │")

        if self.additional_prompt_context:
            context_preview = (
                self.additional_prompt_context[:20] + "..."
                if len(self.additional_prompt_context) > 20
                else self.additional_prompt_context
            )
            lines.append(f"│ Context: {context_preview} │")

        lines.append("└─────────────────────┘")
        return "\n".join(lines)

    def _format_output(self, output: Any, max_length: int = 100) -> str:
        """Format output for display with truncation."""
        output_str = str(output)
        if len(output_str) > max_length:
            return output_str[:max_length] + "..."
        return output_str

    async def execute(
        self, *args, execution_context: Optional[ExecutionContext] = None, **kwargs
    ) -> Any:
        """
        Execute the node's operator with the given inputs.

        Args:
            *args: Positional arguments for the operator
            execution_context: Optional execution context
            **kwargs: Keyword arguments for the operator

        Returns:
            Result from the operator execution

        Raises:
            OperatorError: If execution fails
            ValidationError: If input validation fails
        """
        start_time = time.time()
        self._execution_count += 1

        # Create execution context if not provided
        if execution_context is None:
            execution_context = ExecutionContext(
                request_id=str(uuid.uuid4()),
                trace_id=str(uuid.uuid4()),
                layer_index=0,
                node_index=0,
                start_time=start_time,
                metadata={"node_id": self.node_id},
            )

        try:
            # Validate inputs using the operator's schema
            self.operator.validate_inputs(*args, **kwargs)

            # For 'generate' operator, use LLM provider directly to capture metrics
            if self.operator_name == "generate":
                result, metrics = await self._execute_with_provider(*args, **kwargs)
                self._update_metrics(metrics)
            else:
                # For other operators, execute normally (no LLM calls)
                result = await self.operator.function(*args, **kwargs)

            self._last_execution_time = time.time() - start_time
            self._last_error = None

            return result

        except Exception as e:
            self._last_error = e
            self._last_execution_time = time.time() - start_time

            # Wrap in OperatorError with additional context
            error_msg = f"Node {self.node_id} ({self.operator_name}) failed: {str(e)}"
            raise OperatorError(error_msg) from e

    async def _execute_with_provider(
        self, *args, **kwargs
    ) -> tuple[str, ExecutionMetrics]:
        """
        Execute using LLM provider with request scheduling for rate limiting.
        Falls back to mock provider if API calls fail.

        Args:
            *args: Arguments passed to the operator
            **kwargs: Keyword arguments passed to the operator

        Returns:
            tuple: (result_text, execution_metrics)
        """
        from .scheduler import get_scheduler

        # Create provider instance
        provider = create_provider(self.model_config)

        # Extract prompt from first argument (assuming first arg is the prompt)
        prompt = str(args[0]) if args else ""

        # Add additional prompt context if configured
        if self.additional_prompt_context:
            prompt = f"{self.additional_prompt_context}\n\n{prompt}"

        # Estimate token count for scheduling (rough approximation)
        estimated_tokens = len(prompt.split()) * 1.3  # Average tokens per word
        if self.model_config.max_tokens:
            estimated_tokens += self.model_config.max_tokens

        async def _execute_request():
            """Internal function to execute the LLM request."""
            try:
                # Generate completion with metrics
                result, metrics = await provider.generate_completion(prompt)
                return result, metrics
            except Exception as e:
                # If real API fails, fall back to mock provider
                from ..utils.providers import MockProvider

                mock_provider = MockProvider(self.model_config)
                result, metrics = await mock_provider.generate_completion(prompt)
                return result, metrics

        # Schedule the request through the global scheduler
        scheduler = get_scheduler()

        try:
            result, metrics = await scheduler.schedule_request(
                operation=_execute_request,
                provider=self.model_config.provider,
                model_config=self.model_config,
                priority=0,  # Default priority
                estimated_tokens=int(estimated_tokens),
                component_type="node",
                component_id=self.node_id,
            )
            return result, metrics
        except Exception as e:
            # If scheduler fails, execute directly as fallback
            return await _execute_request()

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics for this node."""
        return {
            "node_id": self.node_id,
            "operator_name": self.operator_name,
            "execution_count": self._execution_count,
            "last_execution_time": self._last_execution_time,
            "has_error": self._last_error is not None,
            "last_error": str(self._last_error) if self._last_error else None,
            "total_tokens": self._total_tokens.model_dump(),
            "total_cost_usd": self._total_cost.total_cost_usd,
            "average_tokens_per_execution": (
                self._total_tokens.total_tokens / self._execution_count
                if self._execution_count > 0
                else 0
            ),
            "average_cost_per_execution": (
                self._total_cost.total_cost_usd / self._execution_count
                if self._execution_count > 0
                else 0.0
            ),
            "last_metrics": (
                self._last_metrics.model_dump() if self._last_metrics else None
            ),
        }

    def reset_stats(self) -> None:
        """Reset execution statistics and cost/token tracking."""
        self._execution_count = 0
        self._last_execution_time = None
        self._last_error = None
        self._total_tokens = TokenUsage()
        self._total_cost = CostInfo()
        self._execution_metrics = []
        self._last_metrics = None

    def get_total_cost(self) -> float:
        """Get total cost in USD for all executions."""
        return self._total_cost.total_cost_usd

    def get_total_tokens(self) -> int:
        """Get total tokens used across all executions."""
        return self._total_tokens.total_tokens

    def get_last_metrics(self) -> Optional[ExecutionMetrics]:
        """Get metrics from the last execution."""
        return self._last_metrics

    def get_all_metrics(self) -> List[ExecutionMetrics]:
        """Get all execution metrics."""
        return self._execution_metrics.copy()

    def _update_metrics(self, metrics: ExecutionMetrics) -> None:
        """Update internal metrics tracking."""
        self._last_metrics = metrics
        self._execution_metrics.append(metrics)

        # Update totals
        self._total_tokens += metrics.token_usage
        self._total_cost += metrics.cost_info

    def configure_model(
        self,
        provider: Optional[ModelProvider] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Update model configuration for this node.

        Args:
            provider: Model provider to use
            model_name: Name of the model
            temperature: Temperature for generation
            **kwargs: Additional model parameters
        """
        if provider is not None:
            self.model_config.provider = provider
        if model_name is not None:
            self.model_config.model_name = model_name
        if temperature is not None:
            self.model_config.temperature = temperature

        # Update extra params
        for key, value in kwargs.items():
            if hasattr(self.model_config, key):
                setattr(self.model_config, key, value)
            else:
                self.model_config.extra_params[key] = value

    def clone(self, new_node_id: Optional[str] = None) -> "Node":
        """
        Create a clone of this node with a new ID.

        Args:
            new_node_id: Optional new node ID

        Returns:
            Cloned node instance
        """
        return Node(
            operator_name=self.operator_name,
            model_config=ModelConfig(**self.model_config.model_dump()),
            additional_prompt_context=self.additional_prompt_context,
            node_id=new_node_id,
        )

    def __mul__(self, count: int) -> List["Node"]:
        """
        Create multiple clones of this node for parallel execution.

        Args:
            count: Number of node clones to create

        Returns:
            List of cloned nodes

        Example:
            >>> node = Node('generate')
            >>> parallel_nodes = node * 3  # Creates 3 parallel nodes
        """
        if not isinstance(count, int) or count < 1:
            raise ValueError(
                f"Multiplication count must be a positive integer, got {count}"
            )

        return [self.clone() for _ in range(count)]

    def __rmul__(self, count: int) -> List["Node"]:
        """
        Right multiplication operator for creating multiple node clones.

        Args:
            count: Number of node clones to create

        Returns:
            List of cloned nodes

        Example:
            >>> node = Node('generate')
            >>> parallel_nodes = 3 * node  # Creates 3 parallel nodes
        """
        return self.__mul__(count)

    @classmethod
    def from_operator(cls, operator_name: str, **config_kwargs) -> "Node":
        """
        Factory method to create a node from an operator name.

        Args:
            operator_name: Name of the registered operator
            **config_kwargs: Configuration parameters

        Returns:
            New Node instance
        """
        return cls(operator_name=operator_name, **config_kwargs)


def create_node(operator_name: str, **kwargs) -> Node:
    """
    Convenience function to create a Node.

    Args:
        operator_name: Name of the registered operator
        **kwargs: Additional configuration parameters

    Returns:
        New Node instance
    """
    return Node(operator_name=operator_name, **kwargs)
