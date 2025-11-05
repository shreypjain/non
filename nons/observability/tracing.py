"""
Comprehensive tracing system for NoN execution with database-ready structured data.

Provides distributed tracing capabilities that track execution flows across
networks, layers, and nodes with detailed timing, metadata, and relationships.
"""

import time
import uuid
import threading
import asyncio
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from contextvars import ContextVar
from pydantic import BaseModel


class SpanKind(str, Enum):
    """Types of spans for different execution contexts."""

    NETWORK = "network"
    LAYER = "layer"
    NODE = "node"
    OPERATOR = "operator"
    LLM_CALL = "llm_call"
    RETRY = "retry"
    FALLBACK = "fallback"


class SpanStatus(str, Enum):
    """Span execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SpanContext:
    """Context information for span relationships."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """
    Represents a single execution span with comprehensive metadata.

    Database-ready structure for storing execution traces.
    """

    # Core identification (required fields)
    trace_id: str
    span_id: str
    operation_name: str
    kind: SpanKind

    # Optional identification
    parent_span_id: Optional[str] = None

    # Execution details
    status: SpanStatus = SpanStatus.PENDING

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None

    # Metadata
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)

    # NoN-specific data
    component_type: str = ""  # "network", "layer", "node", "operator"
    component_id: str = ""

    # Resource information
    model_config: Optional[Dict[str, Any]] = None
    token_usage: Optional[Dict[str, Any]] = None
    cost_info: Optional[Dict[str, Any]] = None

    # Error information
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_stack: Optional[str] = None

    # Network topology
    layer_index: Optional[int] = None
    node_index: Optional[int] = None

    def start(self) -> "Span":
        """Start the span and mark as running."""
        self.start_time = time.time()
        self.status = SpanStatus.RUNNING
        self.add_log("span_started", {"timestamp": self.start_time})
        return self

    def finish(self, status: Optional[SpanStatus] = None) -> "Span":
        """Finish the span and calculate duration."""
        self.end_time = time.time()
        if self.start_time:
            self.duration_ms = (self.end_time - self.start_time) * 1000

        if status:
            self.status = status
        elif self.status == SpanStatus.RUNNING:
            self.status = SpanStatus.SUCCESS

        self.add_log(
            "span_finished",
            {
                "timestamp": self.end_time,
                "duration_ms": self.duration_ms,
                "status": self.status.value,
            },
        )
        return self

    def add_tag(self, key: str, value: Any) -> "Span":
        """Add a tag to the span."""
        self.tags[key] = value
        return self

    def add_tags(self, tags: Dict[str, Any]) -> "Span":
        """Add multiple tags to the span."""
        self.tags.update(tags)
        return self

    def add_log(self, event: str, fields: Optional[Dict[str, Any]] = None) -> "Span":
        """Add a log entry to the span."""
        log_entry = {"timestamp": time.time(), "event": event, "fields": fields or {}}
        self.logs.append(log_entry)
        return self

    def set_error(self, error: Exception) -> "Span":
        """Record an error in the span."""
        self.status = SpanStatus.ERROR
        self.error_type = type(error).__name__
        self.error_message = str(error)

        import traceback

        self.error_stack = traceback.format_exc()

        self.add_log(
            "error",
            {"error_type": self.error_type, "error_message": self.error_message},
        )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for database storage."""
        return asdict(self)

    def to_json_serializable(self) -> Dict[str, Any]:
        """Convert span to JSON-serializable format."""
        data = self.to_dict()

        # Convert enums to strings
        data["kind"] = self.kind.value
        data["status"] = self.status.value

        # Convert timestamps to ISO format
        if self.start_time:
            data["start_time_iso"] = datetime.fromtimestamp(
                self.start_time, tz=timezone.utc
            ).isoformat()

        if self.end_time:
            data["end_time_iso"] = datetime.fromtimestamp(
                self.end_time, tz=timezone.utc
            ).isoformat()

        return data


class TraceManager:
    """
    Manages distributed tracing across NoN execution.

    Provides thread-safe and async-safe tracing with automatic
    parent-child relationships and context propagation.
    """

    def __init__(self, enable_tracing: bool = True):
        self.enable_tracing = enable_tracing
        self.active_spans: Dict[str, Span] = {}
        self.completed_spans: List[Span] = []
        self._lock = threading.RLock()

        # Context variables for automatic parent tracking
        self.current_span_context: ContextVar[Optional[SpanContext]] = ContextVar(
            "current_span_context", default=None
        )

    def start_span(
        self,
        operation_name: str,
        kind: SpanKind,
        parent_context: Optional[SpanContext] = None,
        **kwargs,
    ) -> Span:
        """Start a new span with automatic parent relationship."""
        if not self.enable_tracing:
            return self._create_noop_span()

        # Generate IDs
        span_id = str(uuid.uuid4())

        # Determine parent and trace ID
        if parent_context:
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
        else:
            # Try to get from context
            current_context = self.current_span_context.get()
            if current_context:
                trace_id = current_context.trace_id
                parent_span_id = current_context.span_id
            else:
                # Root span
                trace_id = str(uuid.uuid4())
                parent_span_id = None

        # Create span
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            kind=kind,
            **kwargs,
        ).start()

        # Store active span
        with self._lock:
            self.active_spans[span_id] = span

        # Set as current context
        new_context = SpanContext(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id
        )
        self.current_span_context.set(new_context)

        return span

    def finish_span(self, span: Span, status: Optional[SpanStatus] = None) -> None:
        """Finish a span and move to completed spans."""
        if not self.enable_tracing or not span:
            return

        span.finish(status)

        with self._lock:
            # Move from active to completed
            if span.span_id in self.active_spans:
                del self.active_spans[span.span_id]
            self.completed_spans.append(span)

    def get_current_context(self) -> Optional[SpanContext]:
        """Get the current span context."""
        return self.current_span_context.get()

    def get_active_spans(self) -> List[Span]:
        """Get all currently active spans."""
        with self._lock:
            return list(self.active_spans.values())

    def get_completed_spans(self) -> List[Span]:
        """Get all completed spans."""
        with self._lock:
            return self.completed_spans.copy()

    def get_trace_spans(self, trace_id: str) -> List[Span]:
        """Get all spans for a specific trace."""
        spans = []

        with self._lock:
            # Check active spans
            for span in self.active_spans.values():
                if span.trace_id == trace_id:
                    spans.append(span)

            # Check completed spans
            for span in self.completed_spans:
                if span.trace_id == trace_id:
                    spans.append(span)

        return sorted(spans, key=lambda s: s.start_time)

    def export_spans(
        self, trace_id: Optional[str] = None, since: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Export spans in database-ready format.

        Args:
            trace_id: Optional trace ID to filter spans
            since: Optional timestamp to get spans since

        Returns:
            List of span dictionaries ready for database insertion
        """
        spans = []

        if trace_id:
            span_list = self.get_trace_spans(trace_id)
        else:
            with self._lock:
                span_list = list(self.active_spans.values()) + self.completed_spans

        for span in span_list:
            if since and span.start_time < since:
                continue
            spans.append(span.to_json_serializable())

        return spans

    def clear_completed_spans(self) -> int:
        """Clear completed spans and return count cleared."""
        with self._lock:
            count = len(self.completed_spans)
            self.completed_spans.clear()
            return count

    def _create_noop_span(self) -> Span:
        """Create a no-op span when tracing is disabled."""
        return Span(
            trace_id="noop",
            span_id="noop",
            operation_name="noop",
            kind=SpanKind.OPERATOR,
        )


# Global trace manager
_trace_manager: Optional[TraceManager] = None


def get_tracer() -> TraceManager:
    """Get the global trace manager."""
    global _trace_manager
    if _trace_manager is None:
        _trace_manager = TraceManager()
    return _trace_manager


def configure_tracing(enable: bool = True) -> TraceManager:
    """Configure global tracing."""
    global _trace_manager
    _trace_manager = TraceManager(enable_tracing=enable)
    return _trace_manager


class TracedOperation:
    """Context manager for automatic span lifecycle management."""

    def __init__(
        self,
        operation_name: str,
        kind: SpanKind,
        tracer: Optional[TraceManager] = None,
        **span_kwargs,
    ):
        self.operation_name = operation_name
        self.kind = kind
        self.tracer = tracer or get_tracer()
        self.span_kwargs = span_kwargs
        self.span: Optional[Span] = None

    def __enter__(self) -> Span:
        self.span = self.tracer.start_span(
            self.operation_name, self.kind, **self.span_kwargs
        )
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type:
                self.span.set_error(exc_val)
            self.tracer.finish_span(self.span)


class AsyncTracedOperation:
    """Async context manager for automatic span lifecycle management."""

    def __init__(
        self,
        operation_name: str,
        kind: SpanKind,
        tracer: Optional[TraceManager] = None,
        **span_kwargs,
    ):
        self.operation_name = operation_name
        self.kind = kind
        self.tracer = tracer or get_tracer()
        self.span_kwargs = span_kwargs
        self.span: Optional[Span] = None

    async def __aenter__(self) -> Span:
        self.span = self.tracer.start_span(
            self.operation_name, self.kind, **self.span_kwargs
        )
        return self.span

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type:
                self.span.set_error(exc_val)
            self.tracer.finish_span(self.span)


def traced_operation(
    operation_name: str, kind: SpanKind, **span_kwargs
) -> TracedOperation:
    """Create a traced operation context manager."""
    return TracedOperation(operation_name, kind, **span_kwargs)


def async_traced_operation(
    operation_name: str, kind: SpanKind, **span_kwargs
) -> AsyncTracedOperation:
    """Create an async traced operation context manager."""
    return AsyncTracedOperation(operation_name, kind, **span_kwargs)
