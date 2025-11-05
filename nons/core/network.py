"""
Network of Networks (NoN) implementation.

The NoN class represents a complete network structure composed of sequential
layers with forward pass execution. This is the main interface for building
and executing compound AI systems.
"""

import asyncio
import time
import uuid
from typing import List, Any, Dict, Optional, Union
from .layer import Layer, LayerResult
from .node import Node
from .types import (
    Content,
    NetworkConfig,
    ExecutionContext,
    NetworkError,
    ValidationError,
)
from .config import get_default_network_config


class NetworkResult:
    """Container for network execution results with comprehensive metadata."""

    def __init__(
        self,
        final_output: Any,
        layer_results: List[LayerResult],
        execution_time: float,
        total_nodes_executed: int,
        successful_layers: int,
        failed_layers: int,
    ):
        self.final_output = final_output
        self.layer_results = layer_results
        self.execution_time = execution_time
        self.total_nodes_executed = total_nodes_executed
        self.successful_layers = successful_layers
        self.failed_layers = failed_layers

    @property
    def layer_success_rate(self) -> float:
        """Calculate layer success rate as a percentage."""
        total = self.successful_layers + self.failed_layers
        return (self.successful_layers / total) if total > 0 else 0.0

    @property
    def total_layers(self) -> int:
        """Total number of layers executed."""
        return len(self.layer_results)

    def __repr__(self) -> str:
        return (
            f"NetworkResult(layers={self.total_layers}, "
            f"success_rate={self.layer_success_rate:.2%}, "
            f"time={self.execution_time:.3f}s)"
        )

    def __str__(self) -> str:
        """Detailed string representation with layer results and final output."""
        lines = [
            "┌─ Network Execution Result ─┐",
            f"│ Total Layers: {self.total_layers} │",
            f"│ Success Rate: {self.layer_success_rate:.1%} │",
            f"│ Successful: {self.successful_layers} │",
            f"│ Failed: {self.failed_layers} │",
            f"│ Total Nodes: {self.total_nodes_executed} │",
            f"│ Execution Time: {self.execution_time:.3f}s │",
            "├─ Final Output ─┤",
        ]

        # Format final output with truncation
        output_str = str(self.final_output)
        if len(output_str) > 60:
            output_preview = output_str[:60] + "..."
        else:
            output_preview = output_str

        lines.append(f"│ {output_preview} │")

        # Show layer execution summary
        lines.extend(["├─ Layer Summary ─┤"])

        for i, layer_result in enumerate(self.layer_results):
            status = (
                "✅"
                if layer_result.success_rate == 1.0
                else "⚠️" if layer_result.success_rate > 0 else "❌"
            )
            lines.append(
                f"│ L{i}: {status} {layer_result.success_rate:.0%} ({layer_result.execution_time:.3f}s) │"
            )

        lines.append("└───────────────────────────┘")
        return "\n".join(lines)


class NoN:
    """
    Network of Networks - Complete network structure for compound AI systems.

    Manages sequential execution of layers with forward pass semantics,
    optimization capabilities, and comprehensive error handling.
    """

    def __init__(
        self,
        layers: List[Layer],
        network_config: Optional[NetworkConfig] = None,
        network_id: Optional[str] = None,
    ):
        """
        Initialize a NoN with layers and configuration.

        Args:
            layers: List of layers to execute sequentially
            network_config: Configuration for network behavior
            network_id: Optional network ID (auto-generated if not provided)
        """
        if not layers:
            raise ValueError("Network must contain at least one layer")

        self.network_id = network_id or str(uuid.uuid4())
        self.layers = layers
        self.network_config = network_config or get_default_network_config()

        # Runtime state
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._last_result: Optional[NetworkResult] = None

        # Validate network structure
        self._validate_network()

    def __len__(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        total_nodes = sum(len(layer) for layer in self.layers)
        return f"NoN(id={self.network_id[:8]}, layers={len(self.layers)}, nodes={total_nodes})"

    def __str__(self) -> str:
        """Detailed string representation with network architecture and statistics."""
        total_nodes = sum(len(layer) for layer in self.layers)

        lines = [
            f"┌─ Network of Networks ─┐",
            f"│ ID: {self.network_id[:8]}... │",
            f"│ Layers: {len(self.layers)} │",
            f"│ Total Nodes: {total_nodes} │",
            f"│ Executions: {self._execution_count} │",
        ]

        if self._last_result:
            lines.extend(
                [
                    "├─ Last Execution ─┤",
                    f"│ Success Rate: {self._last_result.layer_success_rate:.1%} │",
                    f"│ Exec Time: {self._last_result.execution_time:.3f}s │",
                    f"│ Nodes Run: {self._last_result.total_nodes_executed} │",
                ]
            )

        # Show network architecture
        lines.extend(["├─ Architecture ─┤"])

        for i, layer in enumerate(self.layers):
            if len(layer.nodes) == 1:
                # Single node layer
                node_info = layer.nodes[0].operator_name
            else:
                # Parallel nodes layer
                operators = [node.operator_name for node in layer.nodes]
                if len(set(operators)) == 1:
                    # All same operator
                    node_info = f"{operators[0]} × {len(operators)}"
                else:
                    # Different operators
                    node_info = f"[{', '.join(operators[:3])}{'...' if len(operators) > 3 else ''}]"

            lines.append(f"│ L{i}: {node_info} │")

        # Show configuration
        lines.extend(
            [
                "├─ Configuration ─┤",
                f"│ Tracing: {'✅' if self.network_config.enable_tracing else '❌'} │",
                f"│ Metrics: {'✅' if self.network_config.enable_metrics else '❌'} │",
                f"│ Timeout: {self.network_config.global_timeout_seconds}s │",
            ]
        )

        lines.append("└─────────────────────┘")
        return "\n".join(lines)

    def _validate_network(self) -> None:
        """Validate network structure and configuration."""
        if not self.layers:
            raise ValidationError("Network must have at least one layer")

        # Validate each layer has nodes
        for i, layer in enumerate(self.layers):
            if len(layer) == 0:
                raise ValidationError(f"Layer {i} has no nodes")

    async def forward(
        self, initial_input: Any, trace_id: Optional[str] = None
    ) -> NetworkResult:
        """
        Execute forward pass through all layers sequentially.

        Args:
            initial_input: Initial input to the first layer
            trace_id: Optional trace ID for observability

        Returns:
            NetworkResult containing final output and execution metadata

        Raises:
            NetworkError: If network execution fails
        """
        start_time = time.time()
        self._execution_count += 1

        # Create base execution context
        execution_context = ExecutionContext(
            request_id=str(uuid.uuid4()),
            trace_id=trace_id or str(uuid.uuid4()),
            layer_index=0,
            node_index=0,
            start_time=start_time,
            metadata={"network_id": self.network_id, "total_layers": len(self.layers)},
        )

        try:
            current_output = initial_input
            layer_results: List[LayerResult] = []
            successful_layers = 0
            failed_layers = 0
            total_nodes_executed = 0

            # Execute each layer sequentially
            for layer_index, layer in enumerate(self.layers):
                try:
                    # Update execution context for current layer
                    layer_context = ExecutionContext(
                        request_id=execution_context.request_id,
                        trace_id=execution_context.trace_id,
                        layer_index=layer_index,
                        node_index=0,
                        start_time=time.time(),
                        metadata={
                            **execution_context.metadata,
                            "layer_id": layer.layer_id,
                        },
                    )

                    # Execute layer with current output as input
                    layer_result = await layer.execute_parallel(
                        current_output, execution_context=layer_context
                    )

                    layer_results.append(layer_result)
                    successful_layers += 1
                    total_nodes_executed += len(layer)

                    # Use first output as input to next layer if multiple outputs
                    if isinstance(layer_result.outputs, list) and layer_result.outputs:
                        if len(layer_result.outputs) == 1:
                            current_output = layer_result.outputs[0]
                        else:
                            # For multiple outputs, pass the list to next layer
                            current_output = layer_result.outputs
                    else:
                        current_output = layer_result.outputs

                except Exception as e:
                    failed_layers += 1
                    # Decide whether to continue or stop based on network policy
                    if self.network_config.max_concurrent_layers == 1:
                        # Sequential execution - stop on first failure
                        raise NetworkError(
                            f"Network {self.network_id} failed at layer {layer_index}: {str(e)}"
                        ) from e
                    else:
                        # Could implement more sophisticated error handling here
                        raise

            execution_time = time.time() - start_time
            self._total_execution_time += execution_time

            result = NetworkResult(
                final_output=current_output,
                layer_results=layer_results,
                execution_time=execution_time,
                total_nodes_executed=total_nodes_executed,
                successful_layers=successful_layers,
                failed_layers=failed_layers,
            )

            self._last_result = result
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self._total_execution_time += execution_time

            if isinstance(e, NetworkError):
                raise
            else:
                raise NetworkError(
                    f"Network {self.network_id} execution failed: {str(e)}"
                ) from e

    async def optimize(self) -> Dict[str, Any]:
        """
        Placeholder for network optimization functionality.

        This would implement structural and node-level improvements
        as described in the research direction of the spec.

        Returns:
            Dictionary with optimization results
        """
        # Placeholder implementation
        return {
            "optimization_status": "not_implemented",
            "message": "Optimization framework is part of future research direction",
            "current_structure": {
                "layers": len(self.layers),
                "total_nodes": sum(len(layer) for layer in self.layers),
            },
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics for this network."""
        avg_execution_time = (
            self._total_execution_time / self._execution_count
            if self._execution_count > 0
            else 0.0
        )

        layer_stats = []
        for i, layer in enumerate(self.layers):
            layer_stats.append(
                {
                    "layer_index": i,
                    "layer_id": layer.layer_id,
                    "node_count": len(layer),
                    "stats": layer.get_execution_stats(),
                }
            )

        return {
            "network_id": self.network_id,
            "layer_count": len(self.layers),
            "total_nodes": sum(len(layer) for layer in self.layers),
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": avg_execution_time,
            "last_result": self._last_result.__dict__ if self._last_result else None,
            "layer_stats": layer_stats,
            "network_config": self.network_config.model_dump(),
        }

    def add_layer(self, layer: Layer) -> None:
        """Add a layer to the end of the network."""
        self.layers.append(layer)

    def insert_layer(self, index: int, layer: Layer) -> None:
        """Insert a layer at a specific position."""
        self.layers.insert(index, layer)

    def remove_layer(self, layer_id: str) -> bool:
        """Remove a layer by ID. Returns True if removed, False if not found."""
        for i, layer in enumerate(self.layers):
            if layer.layer_id == layer_id:
                self.layers.pop(i)
                return True
        return False

    def get_layer(self, layer_id: str) -> Optional[Layer]:
        """Get a layer by ID."""
        for layer in self.layers:
            if layer.layer_id == layer_id:
                return layer
        return None

    def clone(self, new_network_id: Optional[str] = None) -> "NoN":
        """
        Create a clone of this network with new IDs.

        Args:
            new_network_id: Optional new network ID

        Returns:
            Cloned network instance
        """
        # Note: This is a shallow clone of layers
        # For deep clone, would need to clone each layer and node
        return NoN(
            layers=self.layers.copy(),
            network_config=NetworkConfig(**self.network_config.model_dump()),
            network_id=new_network_id,
        )

    @classmethod
    def from_layers(cls, *layers: Layer, **kwargs) -> "NoN":
        """Factory method to create a network from multiple layers."""
        return cls(layers=list(layers), **kwargs)

    @classmethod
    def from_operators(
        cls,
        operator_specs: List[Union[str, List[str], List["Node"]]],
        network_config: Optional[NetworkConfig] = None,
        **node_kwargs,
    ) -> "NoN":
        """
        Factory method to create a network from operator specifications.

        Args:
            operator_specs: List where each element is either:
                           - A string (single operator = single node layer)
                           - A list of strings (multiple operators = parallel layer)
                           - A list of Node objects (parallel nodes from multiplication)
            network_config: Network configuration
            **node_kwargs: Additional node configuration

        Returns:
            New NoN instance

        Example:
            # Create network: transform -> [classify, extract] -> validate
            network = NoN.from_operators([
                'transform',                    # Single operator layer
                ['classify', 'extract'],        # Parallel operators layer
                'validate'                      # Single operator layer
            ])

            # Or using node multiplication:
            node = Node('generate')
            parallel_nodes = node * 3           # Create 3 parallel nodes
            network = NoN.from_operators([
                'transform',                    # Single operator layer
                parallel_nodes,                 # Parallel nodes from multiplication
                'validate'                      # Single operator layer
            ])
        """
        from .node import Node  # Import here to avoid circular imports

        layers = []

        for spec in operator_specs:
            if isinstance(spec, str):
                # Single operator -> single node layer
                node = Node(spec, **node_kwargs)
                layer = Layer([node])
            elif isinstance(spec, list):
                if all(isinstance(item, str) for item in spec):
                    # Multiple operators -> parallel nodes layer
                    nodes = [Node(op_name, **node_kwargs) for op_name in spec]
                    layer = Layer(nodes)
                elif all(isinstance(item, Node) for item in spec):
                    # List of Node objects (from multiplication) -> parallel layer
                    layer = Layer(spec)
                else:
                    raise ValueError(
                        f"Invalid operator spec: {spec}. List must contain all strings or all Node objects."
                    )
            else:
                raise ValueError(
                    f"Invalid operator spec: {spec}. Must be string, list of strings, or list of Node objects."
                )

            layers.append(layer)

        return cls(layers=layers, network_config=network_config)


def create_network(*layers: Layer, **kwargs) -> NoN:
    """
    Convenience function to create a NoN from layers.

    Args:
        *layers: Layers to include in the network
        **kwargs: Additional network configuration

    Returns:
        New NoN instance
    """
    return NoN(layers=list(layers), **kwargs)
