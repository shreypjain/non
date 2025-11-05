"""
Integration utilities for combining tracing, logging, and metrics across NoN execution.

Provides unified observability that automatically correlates traces, logs, and metrics
with database-ready export capabilities.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

from .tracing import (
    TraceManager,
    Span,
    SpanKind,
    SpanStatus,
    get_tracer,
    async_traced_operation,
)
from .logging import LogManager, LogLevel, get_logger, get_log_manager
from .metrics import MetricsCollector, MetricType, get_metrics_collector


class ObservabilityManager:
    """
    Unified observability manager that coordinates tracing, logging, and metrics.

    Provides automatic correlation and database-ready export for all observability data.
    """

    def __init__(
        self,
        enable_tracing: bool = True,
        enable_logging: bool = True,
        enable_metrics: bool = True,
    ):
        self.tracer = TraceManager(enable_tracing=enable_tracing)
        self.log_manager = LogManager(enable_logging=enable_logging)
        self.metrics_collector = MetricsCollector(enable_metrics=enable_metrics)

        # Auto-correlation enabled
        self.auto_correlate = True

    def start_operation(
        self,
        operation_name: str,
        kind: SpanKind,
        component_type: str = "",
        component_id: str = "",
        **metadata,
    ) -> Span:
        """
        Start a new traced operation with automatic logging and metrics.

        Returns a span that automatically correlates logs and metrics.
        """
        # Start span
        span = self.tracer.start_span(
            operation_name=operation_name,
            kind=kind,
            component_type=component_type,
            component_id=component_id,
            **metadata,
        )

        # Set trace context for logging
        if self.auto_correlate:
            self.log_manager.set_trace_context(span.trace_id, span.span_id)

        # Log operation start
        logger = self.log_manager.get_logger(f"nons.{component_type}")
        logger.info(
            f"Starting {operation_name}",
            component_type=component_type,
            component_id=component_id,
            operation_name=operation_name,
        )

        # Record start metric
        self.metrics_collector.increment_counter(
            f"{operation_name}.started",
            trace_id=span.trace_id,
            span_id=span.span_id,
            component_type=component_type,
            component_id=component_id,
        )

        return span

    def finish_operation(
        self,
        span: Span,
        status: Optional[SpanStatus] = None,
        result: Optional[Any] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """
        Finish an operation with automatic logging and metrics.
        """
        operation_name = span.operation_name
        component_type = span.component_type
        component_id = span.component_id

        # Handle error if present
        if error:
            span.set_error(error)
            status = SpanStatus.ERROR

        # Finish span
        self.tracer.finish_span(span, status)

        # Log completion
        logger = self.log_manager.get_logger(f"nons.{component_type}")

        if error:
            logger.error(
                f"Failed {operation_name}",
                component_type=component_type,
                component_id=component_id,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        else:
            logger.info(
                f"Completed {operation_name}",
                component_type=component_type,
                component_id=component_id,
                duration_ms=span.duration_ms,
            )

        # Record completion metrics
        final_status = "error" if error else "success"
        self.metrics_collector.increment_counter(
            f"{operation_name}.{final_status}",
            trace_id=span.trace_id,
            span_id=span.span_id,
            component_type=component_type,
            component_id=component_id,
        )

        if span.duration_ms:
            self.metrics_collector.record_timing(
                f"{operation_name}.duration",
                span.duration_ms,
                trace_id=span.trace_id,
                span_id=span.span_id,
                component_type=component_type,
                component_id=component_id,
            )

        # Clear trace context
        if self.auto_correlate:
            self.log_manager.clear_trace_context()

    def record_cost_and_tokens(
        self,
        span: Span,
        token_count: int,
        cost_usd: float,
        provider: str = "",
        model: str = "",
    ) -> None:
        """Record cost and token metrics with trace correlation."""
        base_labels = {
            "provider": provider,
            "model": model,
            "component_type": span.component_type,
            "component_id": span.component_id,
        }

        # Record token metrics
        self.metrics_collector.record_tokens(
            "llm.tokens.total",
            token_count,
            trace_id=span.trace_id,
            span_id=span.span_id,
            labels=base_labels,
        )

        # Record cost metrics
        self.metrics_collector.record_cost(
            "llm.cost.total",
            cost_usd,
            trace_id=span.trace_id,
            span_id=span.span_id,
            labels=base_labels,
        )

        # Update span with cost/token info
        span.add_tags(
            {
                "tokens": token_count,
                "cost_usd": cost_usd,
                "provider": provider,
                "model": model,
            }
        )

        # Log cost information
        logger = self.log_manager.get_logger(f"nons.{span.component_type}")
        logger.info(
            "LLM call completed",
            component_type=span.component_type,
            component_id=span.component_id,
            tokens=token_count,
            cost_usd=cost_usd,
            provider=provider,
            model=model,
        )

    @asynccontextmanager
    async def trace_operation(
        self,
        operation_name: str,
        kind: SpanKind,
        component_type: str = "",
        component_id: str = "",
        **metadata,
    ):
        """Async context manager for automatic operation tracing."""
        span = self.start_operation(
            operation_name=operation_name,
            kind=kind,
            component_type=component_type,
            component_id=component_id,
            **metadata,
        )

        try:
            yield span
            self.finish_operation(span)
        except Exception as e:
            self.finish_operation(span, error=e)
            raise

    def export_all_data(
        self, since: Optional[float] = None, trace_id: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Export all observability data in database-ready format.

        Returns:
            Dictionary with 'spans', 'logs', and 'metrics' keys containing
            lists of database-ready dictionaries.
        """
        return {
            "spans": self.tracer.export_spans(trace_id=trace_id, since=since),
            "logs": self.log_manager.export_logs(
                trace_id=trace_id, since=since, format="dict"
            ),
            "metrics": self.metrics_collector.export_metrics(since=since),
        }

    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get comprehensive summary for a specific trace."""
        spans = self.tracer.get_trace_spans(trace_id)
        logs = self.log_manager.get_log_entries(trace_id=trace_id)
        metrics = self.metrics_collector.get_metric_points(trace_id=trace_id)

        if not spans:
            return {"error": "Trace not found"}

        # Calculate trace statistics
        start_time = min(span.start_time for span in spans if span.start_time)
        end_time = max(span.end_time for span in spans if span.end_time)
        total_duration = (
            (end_time - start_time) * 1000 if end_time and start_time else 0
        )

        # Count spans by status
        span_counts = {}
        for span in spans:
            status = span.status.value
            span_counts[status] = span_counts.get(status, 0) + 1

        # Aggregate costs and tokens
        total_cost = 0.0
        total_tokens = 0
        for metric in metrics:
            if metric.metric_type.value == "cost":
                total_cost += metric.value
            elif metric.metric_type.value == "token":
                total_tokens += metric.value

        return {
            "trace_id": trace_id,
            "total_spans": len(spans),
            "total_logs": len(logs),
            "total_metrics": len(metrics),
            "duration_ms": total_duration,
            "span_counts": span_counts,
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "start_time": start_time,
            "end_time": end_time,
            "success": span_counts.get("error", 0) == 0,
        }

    def clear_all_data(self) -> Dict[str, int]:
        """Clear all observability data and return counts."""
        return {
            "spans_cleared": self.tracer.clear_completed_spans(),
            "logs_cleared": self.log_manager.clear_logs(),
            "metrics_cleared": self.metrics_collector.clear_metrics(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive observability statistics."""
        return {
            "tracing": {
                "active_spans": len(self.tracer.get_active_spans()),
                "completed_spans": len(self.tracer.get_completed_spans()),
            },
            "logging": self.log_manager.get_stats(),
            "metrics": self.metrics_collector.get_stats(),
        }


# Global observability manager
_observability_manager: Optional[ObservabilityManager] = None


def get_observability() -> ObservabilityManager:
    """Get the global observability manager."""
    global _observability_manager
    if _observability_manager is None:
        _observability_manager = ObservabilityManager()
    return _observability_manager


def configure_observability(
    enable_tracing: bool = True,
    enable_logging: bool = True,
    enable_metrics: bool = True,
) -> ObservabilityManager:
    """Configure global observability."""
    global _observability_manager
    _observability_manager = ObservabilityManager(
        enable_tracing=enable_tracing,
        enable_logging=enable_logging,
        enable_metrics=enable_metrics,
    )
    return _observability_manager


# Convenience decorators and context managers
def trace_network_operation(operation_name: str, network_id: str = ""):
    """Decorator for network-level operations."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            obs = get_observability()
            async with obs.trace_operation(
                operation_name=operation_name,
                kind=SpanKind.NETWORK,
                component_type="network",
                component_id=network_id,
            ) as span:
                result = await func(*args, **kwargs)
                return result

        return wrapper

    return decorator


def trace_layer_operation(operation_name: str, layer_id: str = ""):
    """Decorator for layer-level operations."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            obs = get_observability()
            async with obs.trace_operation(
                operation_name=operation_name,
                kind=SpanKind.LAYER,
                component_type="layer",
                component_id=layer_id,
            ) as span:
                result = await func(*args, **kwargs)
                return result

        return wrapper

    return decorator


def trace_node_operation(operation_name: str, node_id: str = ""):
    """Decorator for node-level operations."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            obs = get_observability()
            async with obs.trace_operation(
                operation_name=operation_name,
                kind=SpanKind.NODE,
                component_type="node",
                component_id=node_id,
            ) as span:
                result = await func(*args, **kwargs)
                return result

        return wrapper

    return decorator
