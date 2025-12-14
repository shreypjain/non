"""
NoN: Network of Networks

A framework for building the next generation of compound AI systems through
composable network architectures.
"""

__version__ = "0.1.1"

# Core components
from nons.core.node import Node, create_node
from nons.core.layer import Layer, LayerResult, create_layer, create_parallel_layer
from nons.core.network import NoN, NetworkResult, create_network

# Type definitions
from nons.core.types import (
    Content,
    ModelProvider,
    ModelConfig,
    LayerConfig,
    NetworkConfig,
    ErrorPolicy,
    StructuredOutput,
    Classification,
    ValidationResult,
    ComparisonAnalysis,
    RouteDecision,
    ExecutionContext,
    TokenUsage,
    CostInfo,
    ExecutionMetrics,
    # Exceptions
    NoNError,
    OperatorError,
    ValidationError,
    ConfigurationError,
    RateLimitError,
    NetworkError,
)

# Configuration management
from nons.core.config import (
    ConfigManager,
    get_config_manager,
    get_default_model_config,
    get_default_layer_config,
    get_default_network_config,
    get_api_key,
    validate_api_keys,
)

# Scheduler
from nons.core.scheduler import (
    RequestScheduler,
    get_scheduler,
    configure_scheduler,
    QueueStrategy,
    BackoffStrategy,
    RateLimitConfig,
)

# Observability
from nons.observability.integration import (
    get_observability,
    configure_observability,
    trace_network_operation,
    trace_layer_operation,
    trace_node_operation,
)

from nons.observability.tracing import SpanKind

# Operator registry
from nons.operators.registry import (
    OperatorRegistry,
    get_registry,
    list_operators,
    get_operator,
    operator,
)

# Import base operators to register them
import nons.operators.base
import nons.operators.deterministic

__all__ = [
    # Version
    "__version__",
    # Core
    "Node",
    "Layer",
    "NoN",
    "NetworkResult",
    "LayerResult",
    "create_node",
    "create_layer",
    "create_parallel_layer",
    "create_network",
    # Types
    "Content",
    "ModelProvider",
    "ModelConfig",
    "LayerConfig",
    "NetworkConfig",
    "ErrorPolicy",
    "StructuredOutput",
    "Classification",
    "ValidationResult",
    "ComparisonAnalysis",
    "RouteDecision",
    "ExecutionContext",
    "TokenUsage",
    "CostInfo",
    "ExecutionMetrics",
    # Exceptions
    "NoNError",
    "OperatorError",
    "ValidationError",
    "ConfigurationError",
    "RateLimitError",
    "NetworkError",
    # Configuration
    "ConfigManager",
    "get_config_manager",
    "get_default_model_config",
    "get_default_layer_config",
    "get_default_network_config",
    "get_api_key",
    "validate_api_keys",
    # Scheduler
    "RequestScheduler",
    "get_scheduler",
    "configure_scheduler",
    "QueueStrategy",
    "BackoffStrategy",
    "RateLimitConfig",
    # Observability
    "get_observability",
    "configure_observability",
    "trace_network_operation",
    "trace_layer_operation",
    "trace_node_operation",
    "SpanKind",
    # Registry
    "OperatorRegistry",
    "get_registry",
    "list_operators",
    "get_operator",
    "operator",
]
