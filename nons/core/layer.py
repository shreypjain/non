"""
Layer implementation for NoN (Network of Networks).

A Layer is a single array of all Nodes requiring parallel execution.
By default, all Layers are executed with concurrent coroutines to reduce
latency and improve compute utilization.
"""

import asyncio
import time
import uuid
from typing import List, Any, Dict, Optional, Union
from .node import Node
from .types import (
    Content,
    LayerConfig,
    ExecutionContext,
    ErrorPolicy,
    OperatorError,
    NetworkError,
)
from .config import get_default_layer_config


class LayerResult:
    """Container for layer execution results with metadata."""

    def __init__(
        self,
        outputs: List[Any],
        execution_time: float,
        successful_nodes: int,
        failed_nodes: int,
        node_results: Dict[str, Any],
    ):
        self.outputs = outputs
        self.execution_time = execution_time
        self.successful_nodes = successful_nodes
        self.failed_nodes = failed_nodes
        self.node_results = node_results

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        total = self.successful_nodes + self.failed_nodes
        return (self.successful_nodes / total) if total > 0 else 0.0

    def __repr__(self) -> str:
        return f"LayerResult(outputs={len(self.outputs)}, success_rate={self.success_rate:.2%}, time={self.execution_time:.3f}s)"

    def __str__(self) -> str:
        """Detailed string representation with outputs and node results."""
        lines = [
            "┌─ Layer Execution Result ─┐",
            f"│ Success Rate: {self.success_rate:.1%} │",
            f"│ Successful: {self.successful_nodes} │",
            f"│ Failed: {self.failed_nodes} │",
            f"│ Execution Time: {self.execution_time:.3f}s │",
            f"│ Outputs: {len(self.outputs)} │",
            "├─ Outputs Preview ─┤",
        ]

        # Show output previews
        for i, output in enumerate(self.outputs[:3]):  # Show first 3 outputs
            output_preview = (
                str(output)[:40] + "..." if len(str(output)) > 40 else str(output)
            )
            lines.append(f"│ [{i}]: {output_preview} │")

        if len(self.outputs) > 3:
            lines.append(f"│ ... and {len(self.outputs) - 3} more │")

        lines.append("└─────────────────────────┘")
        return "\n".join(lines)


class Layer:
    """
    A single array of all Nodes requiring parallel execution.

    Manages concurrent execution of multiple nodes with error handling,
    timeout management, and result aggregation.
    """

    def __init__(
        self,
        nodes: List[Node],
        layer_config: Optional[LayerConfig] = None,
        layer_id: Optional[str] = None,
    ):
        """
        Initialize a Layer with nodes and configuration.

        Args:
            nodes: List of nodes to execute in parallel
            layer_config: Configuration for layer behavior
            layer_id: Optional layer ID (auto-generated if not provided)
        """
        if not nodes:
            raise ValueError("Layer must contain at least one node")

        self.layer_id = layer_id or str(uuid.uuid4())
        self.nodes = nodes
        self.layer_config = layer_config or get_default_layer_config()

        # Runtime state
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._last_result: Optional[LayerResult] = None

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"Layer(id={self.layer_id[:8]}, nodes={len(self.nodes)}, policy={self.layer_config.error_policy})"

    def __str__(self) -> str:
        """Detailed string representation with nodes and statistics."""
        lines = [
            f"┌─ Layer: {self.layer_id[:8]}... ─┐",
            f"│ Nodes: {len(self.nodes)} │",
            f"│ Policy: {self.layer_config.error_policy.value} │",
            f"│ Timeout: {self.layer_config.timeout_seconds}s │",
            f"│ Max Retries: {self.layer_config.max_retries} │",
            f"│ Min Success: {self.layer_config.min_success_threshold:.1%} │",
            f"│ Executions: {self._execution_count} │",
        ]

        if self._last_result:
            lines.extend(
                [
                    "├─ Last Result ─┤",
                    f"│ Success Rate: {self._last_result.success_rate:.1%} │",
                    f"│ Exec Time: {self._last_result.execution_time:.3f}s │",
                ]
            )

        lines.extend(["├─ Nodes ─┤"])

        # Show operator names for each node
        for i, node in enumerate(self.nodes[:5]):  # Show first 5 nodes
            lines.append(f"│ [{i}]: {node.operator_name} │")

        if len(self.nodes) > 5:
            lines.append(f"│ ... and {len(self.nodes) - 5} more │")

        lines.append("└─────────────────┘")
        return "\n".join(lines)

    async def execute_parallel(
        self,
        inputs: Union[List[Any], Any],
        execution_context: Optional[ExecutionContext] = None,
    ) -> LayerResult:
        """
        Execute all nodes in parallel with the given inputs.

        Args:
            inputs: Input data for the nodes (broadcast if single value, or distributed if list)
            execution_context: Optional execution context

        Returns:
            LayerResult containing outputs and execution metadata

        Raises:
            NetworkError: If layer execution fails based on error policy
        """
        start_time = time.time()
        self._execution_count += 1

        # Prepare inputs for each node
        node_inputs = self._prepare_inputs(inputs)

        # Create execution contexts for each node
        node_contexts = self._create_node_contexts(execution_context)

        # Execute nodes based on error policy
        try:
            outputs, node_results = await self._execute_with_policy(
                node_inputs, node_contexts
            )

            execution_time = time.time() - start_time
            self._total_execution_time += execution_time

            # Count successful and failed executions
            successful_nodes = sum(
                1 for result in node_results.values() if result.get("success", False)
            )
            failed_nodes = len(self.nodes) - successful_nodes

            result = LayerResult(
                outputs=outputs,
                execution_time=execution_time,
                successful_nodes=successful_nodes,
                failed_nodes=failed_nodes,
                node_results=node_results,
            )

            self._last_result = result

            # Validate success threshold
            if result.success_rate < self.layer_config.min_success_threshold:
                raise NetworkError(
                    f"Layer {self.layer_id} failed: success rate {result.success_rate:.2%} "
                    f"below threshold {self.layer_config.min_success_threshold:.2%}"
                )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self._total_execution_time += execution_time

            if isinstance(e, NetworkError):
                raise
            else:
                raise NetworkError(
                    f"Layer {self.layer_id} execution failed: {str(e)}"
                ) from e

    def _prepare_inputs(self, inputs: Union[List[Any], Any]) -> List[Any]:
        """Prepare inputs for each node."""
        if isinstance(inputs, list) and len(inputs) == len(self.nodes):
            # Distribute inputs one-to-one with nodes
            return inputs
        else:
            # Broadcast single input to all nodes
            return [inputs] * len(self.nodes)

    def _create_node_contexts(
        self, base_context: Optional[ExecutionContext]
    ) -> List[ExecutionContext]:
        """Create execution contexts for each node."""
        contexts = []

        for i, node in enumerate(self.nodes):
            if base_context:
                context = ExecutionContext(
                    request_id=base_context.request_id,
                    trace_id=base_context.trace_id,
                    layer_index=base_context.layer_index,
                    node_index=i,
                    start_time=time.time(),
                    metadata={**base_context.metadata, "node_id": node.node_id},
                )
            else:
                context = ExecutionContext(
                    request_id=str(uuid.uuid4()),
                    trace_id=str(uuid.uuid4()),
                    layer_index=0,
                    node_index=i,
                    start_time=time.time(),
                    metadata={"node_id": node.node_id},
                )
            contexts.append(context)

        return contexts

    async def _execute_with_policy(
        self, node_inputs: List[Any], node_contexts: List[ExecutionContext]
    ) -> tuple[List[Any], Dict[str, Any]]:
        """Execute nodes according to the configured error policy."""

        if self.layer_config.error_policy == ErrorPolicy.FAIL_FAST:
            return await self._execute_fail_fast(node_inputs, node_contexts)
        elif self.layer_config.error_policy == ErrorPolicy.RETRY_WITH_BACKOFF:
            return await self._execute_with_retry(node_inputs, node_contexts)
        elif self.layer_config.error_policy == ErrorPolicy.SKIP_AND_CONTINUE:
            return await self._execute_skip_continue(node_inputs, node_contexts)
        elif self.layer_config.error_policy == ErrorPolicy.RETURN_PARTIAL:
            return await self._execute_return_partial(node_inputs, node_contexts)
        else:
            # Default to fail fast
            return await self._execute_fail_fast(node_inputs, node_contexts)

    async def _execute_fail_fast(
        self, node_inputs: List[Any], node_contexts: List[ExecutionContext]
    ) -> tuple[List[Any], Dict[str, Any]]:
        """Execute with fail-fast policy."""
        tasks = []
        for i, (node, node_input, context) in enumerate(
            zip(self.nodes, node_inputs, node_contexts)
        ):
            task = asyncio.create_task(
                node.execute(node_input, execution_context=context),
                name=f"node_{i}_{node.node_id}",
            )
            tasks.append(task)

        try:
            outputs = await asyncio.gather(*tasks)
            node_results = {
                node.node_id: {"success": True, "output": output}
                for node, output in zip(self.nodes, outputs)
            }
            return outputs, node_results
        except Exception as e:
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise

    async def _execute_with_retry(
        self, node_inputs: List[Any], node_contexts: List[ExecutionContext]
    ) -> tuple[List[Any], Dict[str, Any]]:
        """Execute with retry and backoff policy."""
        outputs = []
        node_results = {}

        for node, node_input, context in zip(self.nodes, node_inputs, node_contexts):
            success = False
            last_error = None

            for attempt in range(self.layer_config.max_retries + 1):
                try:
                    output = await node.execute(node_input, execution_context=context)
                    outputs.append(output)
                    node_results[node.node_id] = {"success": True, "output": output}
                    success = True
                    break
                except Exception as e:
                    last_error = e
                    if attempt < self.layer_config.max_retries:
                        # Wait with exponential backoff
                        delay = self.layer_config.retry_delay_seconds * (2**attempt)
                        await asyncio.sleep(delay)

            if not success:
                node_results[node.node_id] = {
                    "success": False,
                    "error": str(last_error),
                }
                if self.layer_config.error_policy == ErrorPolicy.RETRY_WITH_BACKOFF:
                    # Still fail if retries exhausted
                    raise OperatorError(
                        f"Node {node.node_id} failed after {self.layer_config.max_retries} retries: {last_error}"
                    )

        return outputs, node_results

    async def _execute_skip_continue(
        self, node_inputs: List[Any], node_contexts: List[ExecutionContext]
    ) -> tuple[List[Any], Dict[str, Any]]:
        """Execute with skip and continue policy."""
        tasks = []
        for i, (node, node_input, context) in enumerate(
            zip(self.nodes, node_inputs, node_contexts)
        ):
            task = asyncio.create_task(
                self._safe_execute(node, node_input, context),
                name=f"node_{i}_{node.node_id}",
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        outputs = []
        node_results = {}

        for node, result in zip(self.nodes, results):
            if isinstance(result, Exception):
                outputs.append(None)
                node_results[node.node_id] = {"success": False, "error": str(result)}
            else:
                outputs.append(result)
                node_results[node.node_id] = {"success": True, "output": result}

        return outputs, node_results

    async def _execute_return_partial(
        self, node_inputs: List[Any], node_contexts: List[ExecutionContext]
    ) -> tuple[List[Any], Dict[str, Any]]:
        """Execute with return partial results policy."""
        return await self._execute_skip_continue(node_inputs, node_contexts)

    async def _safe_execute(
        self, node: Node, node_input: Any, context: ExecutionContext
    ) -> Any:
        """Safely execute a node, catching exceptions."""
        try:
            return await node.execute(node_input, execution_context=context)
        except Exception as e:
            # Log error but don't raise
            return e

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for this layer."""
        avg_execution_time = (
            self._total_execution_time / self._execution_count
            if self._execution_count > 0
            else 0.0
        )

        return {
            "layer_id": self.layer_id,
            "node_count": len(self.nodes),
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": avg_execution_time,
            "last_result": self._last_result.__dict__ if self._last_result else None,
            "error_policy": self.layer_config.error_policy.value,
        }

    def add_node(self, node: Node) -> None:
        """Add a node to this layer."""
        self.nodes.append(node)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node by ID. Returns True if removed, False if not found."""
        for i, node in enumerate(self.nodes):
            if node.node_id == node_id:
                self.nodes.pop(i)
                return True
        return False

    @classmethod
    def from_nodes(cls, *nodes: Node, **kwargs) -> "Layer":
        """Factory method to create a layer from multiple nodes."""
        return cls(nodes=list(nodes), **kwargs)

    @classmethod
    def from_operators(
        cls,
        operator_names: List[str],
        layer_config: Optional[LayerConfig] = None,
        **node_kwargs,
    ) -> "Layer":
        """Factory method to create a layer from operator names."""
        nodes = [Node(name, **node_kwargs) for name in operator_names]
        return cls(nodes=nodes, layer_config=layer_config)


def create_layer(*nodes: Node, **kwargs) -> Layer:
    """
    Convenience function to create a Layer from nodes.

    Args:
        *nodes: Nodes to include in the layer
        **kwargs: Additional layer configuration

    Returns:
        New Layer instance
    """
    return Layer(nodes=list(nodes), **kwargs)


def create_parallel_layer(node_list: List[Node], **kwargs) -> Layer:
    """
    Create a Layer from a list of nodes (e.g., from node multiplication).

    This function is specifically designed to work with the output of
    node multiplication operators like `node * 3`.

    Args:
        node_list: List of nodes to execute in parallel
        **kwargs: Additional layer configuration

    Returns:
        New Layer instance

    Example:
        >>> node = Node('generate')
        >>> parallel_nodes = node * 3
        >>> layer = create_parallel_layer(parallel_nodes)
    """
    if not isinstance(node_list, list):
        raise TypeError(f"Expected list of nodes, got {type(node_list)}")

    if not node_list:
        raise ValueError("Cannot create layer from empty node list")

    if not all(isinstance(node, Node) for node in node_list):
        raise TypeError("All items in node_list must be Node instances")

    return Layer(nodes=node_list, **kwargs)
