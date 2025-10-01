"""
Core type definitions for NoN (Network of Networks) system.

This module defines all the fundamental types used throughout the NoN package,
including content types, structured outputs, and operator-specific return types.
"""

from typing import Any, Dict, List, Optional, Union, TypeVar, Generic, TypedDict
from typing_extensions import Literal
from pydantic import BaseModel, ConfigDict
from enum import Enum
import json


# Core Content Types
Content = Union[str, Dict[str, Any]]
"""
Core content type that can be either plain text or structured data.
All operator chains must maintain type compatibility.
"""


# Generic Structured Output
T = TypeVar('T')

class StructuredOutput(BaseModel, Generic[T]):
    """
    Validated structured data container for complex operator outputs.
    Used when operators return lists, dicts, or complex objects.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: T
    metadata: Dict[str, Any] = {}

    def get_data(self) -> T:
        return self.data

    def to_content(self) -> Content:
        """Convert structured output back to Content type for chaining."""
        if isinstance(self.data, str):
            return self.data
        elif isinstance(self.data, dict):
            return self.data
        else:
            # For other types, serialize to JSON string
            return json.dumps(self.data, default=str)


# Operator-specific return types
class Classification(TypedDict):
    """Classification result with confidence and reasoning."""
    category: str
    confidence: float
    reasoning: Optional[str]


class ValidationResult(TypedDict):
    """Validation result with boolean outcome and reasoning."""
    is_valid: bool
    validation_reasoning: str
    confidence: float


class ComparisonAnalysis(TypedDict):
    """Structured comparison analysis between content pieces."""
    differences: List[str]
    similarities: List[str]
    conclusion: str


class RouteDecision(TypedDict):
    """Routing decision with selected path and confidence."""
    selected_path: str
    routing_confidence: float
    reasoning: str


# Spec Types (string-based specifications for operator behavior)
TransformSpec = str
"""String specification defining how content should be transformed."""

ExtractionCriteria = str
"""String specification defining what information to extract."""

ClassificationSchema = str
"""String specification defining the classification categories and rules."""

GenerationSpec = str
"""String specification defining what content to generate."""

ValidationCriteria = str
"""String specification defining validation rules."""

ExpansionTopic = str
"""String specification defining the topic or angle for expansion."""

SynthesisFocus = str
"""String specification defining how to focus the synthesis."""

ComparisonDimensions = str
"""String specification defining dimensions for comparison."""

RoutingLogic = str
"""String specification defining routing decision logic."""

# Enum types for configuration
class ErrorPolicy(str, Enum):
    """Error handling policies for nodes and layers."""
    FAIL_FAST = "fail_fast"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK_MODEL = "fallback_model"
    SKIP_AND_CONTINUE = "skip_and_continue"
    RETURN_PARTIAL = "return_partial"


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MOCK = "mock"


# Configuration types
class ModelConfig(BaseModel):
    """Configuration for model provider and parameters."""
    provider: ModelProvider
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None

    # Provider-specific parameters
    extra_params: Dict[str, Any] = {}


class LayerConfig(BaseModel):
    """Configuration for layer-level behavior."""
    timeout_seconds: float = 30.0
    error_policy: ErrorPolicy = ErrorPolicy.RETRY_WITH_BACKOFF
    min_success_threshold: float = 1.0  # For RETURN_PARTIAL policy
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


class NetworkConfig(BaseModel):
    """Configuration for network-level behavior."""
    max_concurrent_layers: int = 1  # Sequential by default
    global_timeout_seconds: float = 300.0
    enable_tracing: bool = True
    enable_metrics: bool = True


# Rate limiting types
class RateLimitConfig(BaseModel):
    """Rate limit configuration for model providers."""
    requests_per_minute: int = 60
    tokens_per_minute: int = 150000
    concurrent_requests: int = 10


# Input/Output schemas for operators
class InputSchema(BaseModel):
    """Schema defining required and optional inputs for operators."""
    required_params: List[str]
    optional_params: List[str] = []
    param_types: Dict[str, str] = {}


class OutputSchema(BaseModel):
    """Schema defining expected output types for operators."""
    return_type: str
    description: str


class OperatorMetadata(BaseModel):
    """Metadata for operator documentation and registry."""
    name: str
    description: str
    examples: List[str] = []
    tags: List[str] = []


# Execution context types
class ExecutionContext(BaseModel):
    """Context passed through the execution pipeline."""
    request_id: str
    trace_id: str
    layer_index: int
    node_index: int
    start_time: float
    metadata: Dict[str, Any] = {}


# Cost and token tracking types
class TokenUsage(BaseModel):
    """Token usage information from LLM API calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add two TokenUsage objects together."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )


class CostInfo(BaseModel):
    """Cost information for LLM API calls."""
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0

    def __add__(self, other: 'CostInfo') -> 'CostInfo':
        """Add two CostInfo objects together."""
        return CostInfo(
            input_cost_usd=self.input_cost_usd + other.input_cost_usd,
            output_cost_usd=self.output_cost_usd + other.output_cost_usd,
            total_cost_usd=self.total_cost_usd + other.total_cost_usd
        )


class ExecutionMetrics(BaseModel):
    """Comprehensive execution metrics for a single API call."""
    token_usage: TokenUsage = TokenUsage()
    cost_info: CostInfo = CostInfo()
    model_name: str = ""
    provider: str = ""
    request_id: Optional[str] = None
    response_time_ms: float = 0.0

    def __add__(self, other: 'ExecutionMetrics') -> 'ExecutionMetrics':
        """Aggregate two ExecutionMetrics objects."""
        return ExecutionMetrics(
            token_usage=self.token_usage + other.token_usage,
            cost_info=self.cost_info + other.cost_info,
            model_name=f"{self.model_name},{other.model_name}" if self.model_name != other.model_name else self.model_name,
            provider=f"{self.provider},{other.provider}" if self.provider != other.provider else self.provider,
            response_time_ms=self.response_time_ms + other.response_time_ms
        )


# Pricing information for cost calculation
PRICING_INFO = {
    # OpenAI pricing (per 1M tokens) as of 2025
    "openai": {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    },
    # Anthropic pricing (per 1M tokens) as of 2025
    "anthropic": {
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-5-haiku-20241022": {"input": 1.0, "output": 5.0},
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    },
    # Google pricing (per 1M tokens) as of 2024-2025
    "google": {
        "gemini-2.5-flash": {"input": 0.075, "output": 0.3},
        "gemini-2.5-pro": {"input": 1.25, "output": 5.0},
        "gemini-2.0-flash": {"input": 0.075, "output": 0.3},
        "gemini-2.0-flash-001": {"input": 0.075, "output": 0.3},
        "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.3},
        "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.3},
    }
}


def calculate_cost(token_usage: TokenUsage, model_name: str, provider: str) -> CostInfo:
    """Calculate cost based on token usage and model pricing."""
    provider_pricing = PRICING_INFO.get(provider.lower(), {})
    model_pricing = provider_pricing.get(model_name.lower(), {"input": 0.0, "output": 0.0})

    input_cost = (token_usage.prompt_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (token_usage.completion_tokens / 1_000_000) * model_pricing["output"]

    return CostInfo(
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=input_cost + output_cost
    )


# Exception types for better error handling
class NoNError(Exception):
    """Base exception for NoN system errors."""
    pass


class OperatorError(NoNError):
    """Errors related to operator execution."""
    pass


class ValidationError(NoNError):
    """Errors related to type validation."""
    pass


class ConfigurationError(NoNError):
    """Errors related to configuration."""
    pass


class RateLimitError(NoNError):
    """Errors related to rate limiting."""
    pass


class NetworkError(NoNError):
    """Errors related to network execution."""
    pass