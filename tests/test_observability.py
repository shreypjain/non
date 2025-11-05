"""
Tests for observability system (tracing, logging, metrics).
"""

import pytest
import time
from unittest.mock import patch, MagicMock
from nons.observability.tracing import TracingManager, Span, SpanKind, SpanStatus
from nons.observability.logging import LoggingManager, LogLevel
from nons.observability.metrics import MetricsManager, MetricType
from nons.observability.integration import (
    ObservabilityManager,
    get_observability,
    configure_observability,
)
from nons.core.types import TokenUsage, CostInfo, ModelProvider


class TestTracingManager:
    """Test the tracing system."""

    def test_tracing_manager_initialization(self):
        """Test tracing manager initialization."""
        manager = TracingManager()

        assert manager.enabled is True
        assert len(manager.spans) == 0
        assert len(manager.active_spans) == 0

    def test_start_span(self):
        """Test starting a new span."""
        manager = TracingManager()

        span = manager.start_span(
            operation_name="test_operation", kind=SpanKind.NODE, tags={"test": "value"}
        )

        assert isinstance(span, Span)
        assert span.operation_name == "test_operation"
        assert span.kind == SpanKind.NODE
        assert span.tags["test"] == "value"
        assert span.status == SpanStatus.ACTIVE
        assert span.span_id in manager.active_spans

    def test_finish_span(self):
        """Test finishing a span."""
        manager = TracingManager()

        span = manager.start_span("test_operation", SpanKind.NODE)
        span_id = span.span_id

        time.sleep(0.01)  # Small delay to ensure duration > 0
        manager.finish_span(span, SpanStatus.SUCCESS)

        assert span.status == SpanStatus.SUCCESS
        assert span.end_time > span.start_time
        assert span.duration_ms > 0
        assert span_id not in manager.active_spans
        assert span_id in [s.span_id for s in manager.spans]

    def test_span_hierarchy(self):
        """Test parent-child span relationships."""
        manager = TracingManager()

        parent_span = manager.start_span("parent", SpanKind.NETWORK)
        child_span = manager.start_span(
            "child", SpanKind.NODE, parent_span_id=parent_span.span_id
        )

        assert child_span.parent_span_id == parent_span.span_id
        assert child_span.trace_id == parent_span.trace_id

    def test_span_with_token_usage(self):
        """Test span with token usage information."""
        manager = TracingManager()

        token_usage = TokenUsage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )

        span = manager.start_span("test", SpanKind.LLM_CALL)
        span.token_usage = token_usage

        assert span.token_usage.prompt_tokens == 100
        assert span.token_usage.completion_tokens == 50
        assert span.token_usage.total_tokens == 150

    def test_span_with_cost_info(self):
        """Test span with cost information."""
        manager = TracingManager()

        cost_info = CostInfo(
            cost_usd=0.005,
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-haiku-20240307",
        )

        span = manager.start_span("test", SpanKind.LLM_CALL)
        span.cost_info = cost_info

        assert span.cost_info.cost_usd == 0.005
        assert span.cost_info.provider == ModelProvider.ANTHROPIC

    def test_get_trace_spans(self):
        """Test getting all spans for a trace."""
        manager = TracingManager()

        # Create spans with same trace ID
        parent = manager.start_span("parent", SpanKind.NETWORK)
        child1 = manager.start_span(
            "child1", SpanKind.NODE, parent_span_id=parent.span_id
        )
        child2 = manager.start_span(
            "child2", SpanKind.NODE, parent_span_id=parent.span_id
        )

        manager.finish_span(child1, SpanStatus.SUCCESS)
        manager.finish_span(child2, SpanStatus.SUCCESS)
        manager.finish_span(parent, SpanStatus.SUCCESS)

        trace_spans = manager.get_trace_spans(parent.trace_id)

        assert len(trace_spans) == 3
        span_ids = [s.span_id for s in trace_spans]
        assert parent.span_id in span_ids
        assert child1.span_id in span_ids
        assert child2.span_id in span_ids

    def test_export_spans(self):
        """Test exporting spans for database storage."""
        manager = TracingManager()

        span = manager.start_span("test", SpanKind.NODE, tags={"key": "value"})
        manager.finish_span(span, SpanStatus.SUCCESS)

        exported = manager.export_spans()

        assert len(exported) == 1
        exported_span = exported[0]

        # Check required fields for database storage
        assert "trace_id" in exported_span
        assert "span_id" in exported_span
        assert "operation_name" in exported_span
        assert "kind" in exported_span
        assert "status" in exported_span
        assert "start_time" in exported_span
        assert "end_time" in exported_span
        assert "duration_ms" in exported_span
        assert "tags" in exported_span

    def test_span_cleanup(self):
        """Test automatic span cleanup."""
        manager = TracingManager(max_spans=2)

        # Create 3 spans to trigger cleanup
        span1 = manager.start_span("span1", SpanKind.NODE)
        span2 = manager.start_span("span2", SpanKind.NODE)
        span3 = manager.start_span("span3", SpanKind.NODE)

        manager.finish_span(span1, SpanStatus.SUCCESS)
        manager.finish_span(span2, SpanStatus.SUCCESS)
        manager.finish_span(span3, SpanStatus.SUCCESS)

        # Should only keep the most recent spans
        assert len(manager.spans) <= 2


class TestLoggingManager:
    """Test the logging system."""

    def test_logging_manager_initialization(self):
        """Test logging manager initialization."""
        manager = LoggingManager()

        assert manager.enabled is True
        assert len(manager.logs) == 0

    def test_log_creation(self):
        """Test creating log entries."""
        manager = LoggingManager()

        manager.log(
            level=LogLevel.INFO,
            message="Test message",
            component_type="test",
            component_id="test-123",
            extra_data={"key": "value"},
        )

        assert len(manager.logs) == 1
        log_entry = manager.logs[0]

        assert log_entry["level"] == LogLevel.INFO
        assert log_entry["message"] == "Test message"
        assert log_entry["component_type"] == "test"
        assert log_entry["component_id"] == "test-123"
        assert log_entry["extra_data"]["key"] == "value"
        assert "timestamp" in log_entry

    def test_log_with_trace_context(self):
        """Test logging with trace context."""
        manager = LoggingManager()

        manager.set_trace_context("trace-123", "span-456")
        manager.log(LogLevel.DEBUG, "Debug message")

        log_entry = manager.logs[0]
        assert log_entry["trace_id"] == "trace-123"
        assert log_entry["span_id"] == "span-456"

    def test_log_filtering_by_level(self):
        """Test log filtering by level."""
        manager = LoggingManager(min_level=LogLevel.WARNING)

        # These should be ignored
        manager.log(LogLevel.DEBUG, "Debug message")
        manager.log(LogLevel.INFO, "Info message")

        # These should be recorded
        manager.log(LogLevel.WARNING, "Warning message")
        manager.log(LogLevel.ERROR, "Error message")

        assert len(manager.logs) == 2
        assert all(
            log["level"] in [LogLevel.WARNING, LogLevel.ERROR] for log in manager.logs
        )

    def test_export_logs(self):
        """Test exporting logs for database storage."""
        manager = LoggingManager()

        manager.log(LogLevel.INFO, "Test message", extra_data={"test": True})

        exported = manager.export_logs()

        assert len(exported) == 1
        exported_log = exported[0]

        # Check required fields for database storage
        assert "timestamp" in exported_log
        assert "level" in exported_log
        assert "message" in exported_log
        assert "component_type" in exported_log
        assert "component_id" in exported_log

    def test_log_cleanup(self):
        """Test automatic log cleanup."""
        manager = LoggingManager(max_entries=3)

        # Create 5 log entries to trigger cleanup
        for i in range(5):
            manager.log(LogLevel.INFO, f"Message {i}")

        # Should only keep the most recent entries
        assert len(manager.logs) <= 3


class TestMetricsManager:
    """Test the metrics system."""

    def test_metrics_manager_initialization(self):
        """Test metrics manager initialization."""
        manager = MetricsManager()

        assert manager.enabled is True
        assert len(manager.metrics) == 0

    def test_record_counter(self):
        """Test recording counter metrics."""
        manager = MetricsManager()

        manager.record_counter(
            name="test_counter", value=5, tags={"environment": "test"}
        )

        assert len(manager.metrics) == 1
        metric = manager.metrics[0]

        assert metric["name"] == "test_counter"
        assert metric["type"] == MetricType.COUNTER
        assert metric["value"] == 5
        assert metric["tags"]["environment"] == "test"
        assert "timestamp" in metric

    def test_record_gauge(self):
        """Test recording gauge metrics."""
        manager = MetricsManager()

        manager.record_gauge("test_gauge", 42.5)

        metric = manager.metrics[0]
        assert metric["type"] == MetricType.GAUGE
        assert metric["value"] == 42.5

    def test_record_histogram(self):
        """Test recording histogram metrics."""
        manager = MetricsManager()

        manager.record_histogram("test_histogram", 1.23)

        metric = manager.metrics[0]
        assert metric["type"] == MetricType.HISTOGRAM
        assert metric["value"] == 1.23

    def test_metric_aggregation(self):
        """Test metric aggregation capabilities."""
        manager = MetricsManager()

        # Record multiple values for the same metric
        manager.record_counter("requests", 1, {"endpoint": "api"})
        manager.record_counter("requests", 3, {"endpoint": "api"})
        manager.record_counter("requests", 2, {"endpoint": "web"})

        api_metrics = [m for m in manager.metrics if m["tags"].get("endpoint") == "api"]
        assert len(api_metrics) == 2

        total_api_requests = sum(m["value"] for m in api_metrics)
        assert total_api_requests == 4

    def test_export_metrics(self):
        """Test exporting metrics for database storage."""
        manager = MetricsManager()

        manager.record_gauge("cpu_usage", 75.5, {"host": "server1"})

        exported = manager.export_metrics()

        assert len(exported) == 1
        exported_metric = exported[0]

        # Check required fields for database storage
        assert "timestamp" in exported_metric
        assert "name" in exported_metric
        assert "type" in exported_metric
        assert "value" in exported_metric
        assert "tags" in exported_metric

    def test_metrics_cleanup(self):
        """Test automatic metrics cleanup."""
        manager = MetricsManager(max_points=3)

        # Create 5 metric points to trigger cleanup
        for i in range(5):
            manager.record_counter(f"counter_{i}", 1)

        # Should only keep the most recent points
        assert len(manager.metrics) <= 3


class TestObservabilityManager:
    """Test the unified observability manager."""

    def test_observability_manager_initialization(self):
        """Test observability manager initialization."""
        manager = ObservabilityManager()

        assert manager.tracing is not None
        assert manager.logging is not None
        assert manager.metrics is not None

    def test_start_operation(self):
        """Test starting an operation with automatic correlation."""
        manager = ObservabilityManager()

        span = manager.start_operation(
            operation_name="test_operation",
            kind=SpanKind.NODE,
            component_type="node",
            component_id="node-123",
        )

        assert isinstance(span, Span)
        assert span.operation_name == "test_operation"

        # Check that trace context was set for logging
        assert manager.logging.current_trace_id == span.trace_id
        assert manager.logging.current_span_id == span.span_id

    def test_finish_operation(self):
        """Test finishing an operation with automatic logging."""
        manager = ObservabilityManager()

        span = manager.start_operation("test_operation", SpanKind.NODE)

        # Add some execution data
        span.token_usage = TokenUsage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        span.cost_info = CostInfo(
            cost_usd=0.001, provider=ModelProvider.MOCK, model_name="test"
        )

        manager.finish_operation(span, result="success")

        # Check that span was finished
        assert span.status == SpanStatus.SUCCESS

        # Check that metrics were recorded
        assert len(manager.metrics.metrics) > 0

        # Check for operation completion log
        completion_logs = [
            log for log in manager.logging.logs if "completed" in log["message"]
        ]
        assert len(completion_logs) > 0

    def test_finish_operation_with_error(self):
        """Test finishing an operation with error."""
        manager = ObservabilityManager()

        span = manager.start_operation("test_operation", SpanKind.NODE)
        error = Exception("Test error")

        manager.finish_operation(span, error=error)

        # Check that span was marked as error
        assert span.status == SpanStatus.ERROR
        assert "error" in span.logs[-1]

        # Check for error log
        error_logs = [
            log for log in manager.logging.logs if log["level"] == LogLevel.ERROR
        ]
        assert len(error_logs) > 0

    def test_export_all_data(self):
        """Test exporting all observability data."""
        manager = ObservabilityManager()

        # Generate some data
        span = manager.start_operation("test", SpanKind.NODE)
        manager.finish_operation(span)

        exported = manager.export_all_data()

        assert "spans" in exported
        assert "logs" in exported
        assert "metrics" in exported

        assert len(exported["spans"]) > 0
        assert len(exported["logs"]) > 0
        assert len(exported["metrics"]) > 0

    def test_export_filtered_by_time(self):
        """Test exporting data filtered by timestamp."""
        manager = ObservabilityManager()

        # Record timestamp before operations
        start_time = time.time()

        # Generate some data
        span = manager.start_operation("test", SpanKind.NODE)
        manager.finish_operation(span)

        # Export only data since start_time
        exported = manager.export_all_data(since=start_time)

        # All exported data should be newer than start_time
        for span_data in exported["spans"]:
            assert span_data["start_time"] >= start_time

        for log_data in exported["logs"]:
            assert log_data["timestamp"] >= start_time

        for metric_data in exported["metrics"]:
            assert metric_data["timestamp"] >= start_time

    def test_export_filtered_by_trace(self):
        """Test exporting data filtered by trace ID."""
        manager = ObservabilityManager()

        # Create two separate traces
        span1 = manager.start_operation("operation1", SpanKind.NODE)
        manager.finish_operation(span1)

        span2 = manager.start_operation("operation2", SpanKind.NODE)
        manager.finish_operation(span2)

        # Export only first trace
        exported = manager.export_all_data(trace_id=span1.trace_id)

        # All exported spans should belong to the specified trace
        for span_data in exported["spans"]:
            assert span_data["trace_id"] == span1.trace_id

        # All exported logs should belong to the specified trace
        for log_data in exported["logs"]:
            if "trace_id" in log_data:
                assert log_data["trace_id"] == span1.trace_id

    def test_get_trace_summary(self):
        """Test getting trace summary with aggregated metrics."""
        manager = ObservabilityManager()

        # Create a trace with multiple operations
        parent_span = manager.start_operation("parent", SpanKind.NETWORK)
        child_span = manager.start_operation(
            "child", SpanKind.NODE, parent_span_id=parent_span.span_id
        )

        # Add execution data
        child_span.token_usage = TokenUsage(
            prompt_tokens=20, completion_tokens=10, total_tokens=30
        )
        child_span.cost_info = CostInfo(
            cost_usd=0.002, provider=ModelProvider.MOCK, model_name="test"
        )

        manager.finish_operation(child_span)
        manager.finish_operation(parent_span)

        summary = manager.get_trace_summary(parent_span.trace_id)

        assert "trace_id" in summary
        assert "total_duration_ms" in summary
        assert "total_cost_usd" in summary
        assert "total_tokens" in summary
        assert "success" in summary
        assert "spans" in summary

        assert summary["trace_id"] == parent_span.trace_id
        assert summary["total_cost_usd"] == 0.002
        assert summary["total_tokens"] == 30
        assert summary["success"] is True

    def test_get_stats(self):
        """Test getting comprehensive observability statistics."""
        manager = ObservabilityManager()

        # Generate some data
        span = manager.start_operation("test", SpanKind.NODE)
        manager.finish_operation(span)

        stats = manager.get_stats()

        assert "tracing" in stats
        assert "logging" in stats
        assert "metrics" in stats

        # Check tracing stats
        assert "total_spans" in stats["tracing"]
        assert "active_spans" in stats["tracing"]

        # Check logging stats
        assert "total_entries" in stats["logging"]

        # Check metrics stats
        assert "total_points" in stats["metrics"]


class TestObservabilityIntegration:
    """Test observability integration functions."""

    def test_get_observability_singleton(self):
        """Test that get_observability returns singleton instance."""
        obs1 = get_observability()
        obs2 = get_observability()

        assert obs1 is obs2

    def test_configure_observability(self):
        """Test observability configuration."""
        manager = configure_observability(
            enable_tracing=True, enable_logging=False, enable_metrics=True
        )

        assert manager.tracing.enabled is True
        assert manager.logging.enabled is False
        assert manager.metrics.enabled is True

    def test_observability_disabled(self):
        """Test behavior when observability is disabled."""
        manager = ObservabilityManager(
            enable_tracing=False, enable_logging=False, enable_metrics=False
        )

        # Operations should still work but not collect data
        span = manager.start_operation("test", SpanKind.NODE)
        manager.finish_operation(span)

        exported = manager.export_all_data()

        # No data should be collected when disabled
        assert len(exported["spans"]) == 0
        assert len(exported["logs"]) == 0
        assert len(exported["metrics"]) == 0


@pytest.mark.unit
class TestObservabilityPerformance:
    """Test observability system performance characteristics."""

    def test_high_volume_operations(self):
        """Test handling high volume of operations."""
        manager = ObservabilityManager()

        # Simulate high volume of operations
        for i in range(100):
            span = manager.start_operation(f"operation_{i}", SpanKind.NODE)
            manager.finish_operation(span)

        # Should handle without issues
        exported = manager.export_all_data()
        assert len(exported["spans"]) <= 100  # May be fewer due to cleanup

    def test_memory_management(self):
        """Test memory management with automatic cleanup."""
        # Create manager with small limits
        manager = ObservabilityManager()
        manager.tracing.max_spans = 10
        manager.logging.max_entries = 10
        manager.metrics.max_points = 10

        # Generate more data than limits
        for i in range(20):
            span = manager.start_operation(f"op_{i}", SpanKind.NODE)
            manager.finish_operation(span)

        # Should respect limits
        assert len(manager.tracing.spans) <= 10
        assert len(manager.logging.logs) <= 10
        assert len(manager.metrics.metrics) <= 10

    def test_concurrent_operations(self):
        """Test handling concurrent operations."""
        manager = ObservabilityManager()

        # Simulate concurrent operations with overlapping spans
        parent = manager.start_operation("parent", SpanKind.NETWORK)
        child1 = manager.start_operation(
            "child1", SpanKind.NODE, parent_span_id=parent.span_id
        )
        child2 = manager.start_operation(
            "child2", SpanKind.NODE, parent_span_id=parent.span_id
        )

        # Finish in different order
        manager.finish_operation(child1)
        manager.finish_operation(parent)
        manager.finish_operation(child2)

        # Should handle correctly
        trace_spans = manager.tracing.get_trace_spans(parent.trace_id)
        assert len(trace_spans) == 3
