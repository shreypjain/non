"""
Regression tests for four observability bug fixes.

Covers:
  - Issue #15: p95/p99 percentile off-by-one in MetricsCollector
  - Issue #12: Exported logs always empty due to wrong structlog logger factory
  - Issue #13: Finishing a child span corrupts parent span context in TraceManager
  - Issue #10: asyncio.CancelledError leaves orphaned spans; get_trace_summary
                crashes when all end_times are None
"""

import asyncio
import logging

import pytest

from nons.observability.metrics import MetricsCollector
from nons.observability.logging import LogManager
from nons.observability.tracing import TraceManager, SpanKind, SpanStatus
from nons.observability.integration import ObservabilityManager


# ---------------------------------------------------------------------------
# Issue #15 – p95/p99 percentile off-by-one
# ---------------------------------------------------------------------------


class TestPercentileCalculation:
    """Tests for the nearest-rank percentile fix in MetricsCollector."""

    def _collector_with_values(self, values):
        """Return a MetricsCollector pre-loaded with the given values."""
        collector = MetricsCollector()
        for v in values:
            collector.record_metric("test_metric", v)
        return collector

    def test_p95_does_not_overshoot_for_exact_multiple(self):
        """
        With n=20 values the old formula int(0.95*20)=19 returned the maximum
        (index 19), which is the 100th percentile.  The fixed formula returns
        index 18, i.e. the 19th-largest value.
        """
        # Values 1..20 sorted: [1,2,...,20]
        values = list(range(1, 21))
        collector = self._collector_with_values(values)
        summary = collector.get_metric_summary("test_metric")

        assert summary is not None
        # p95 of [1..20] via nearest-rank: ceil(0.95*20)-1 = 18 → value 19
        assert summary.p95_value == 19, (
            f"p95 should be 19 (index 18), got {summary.p95_value}"
        )
        # The old bug would have returned 20 (the maximum)
        assert summary.p95_value != 20, "p95 must not equal the maximum (off-by-one)"

    def test_p99_does_not_overshoot_for_exact_multiple(self):
        """
        With n=100 values the old formula int(0.99*100)=99 returned index 99
        (the maximum).  The fixed formula returns index 98.
        """
        values = list(range(1, 101))
        collector = self._collector_with_values(values)
        summary = collector.get_metric_summary("test_metric")

        assert summary is not None
        # nearest-rank: ceil(0.99*100)-1 = 98 → value 99
        assert summary.p99_value == 99, (
            f"p99 should be 99 (index 98), got {summary.p99_value}"
        )
        assert summary.p99_value != 100, "p99 must not equal the maximum (off-by-one)"

    def test_p95_within_bounds(self):
        """p95 index must always be within [0, n-1]."""
        # 20 is the smallest n that triggers percentile calculation
        values = list(range(1, 21))
        collector = self._collector_with_values(values)
        summary = collector.get_metric_summary("test_metric")

        assert summary is not None
        n = len(values)
        sorted_values = sorted(values)
        assert summary.p95_value in sorted_values, "p95 must be an actual data value"
        assert summary.p95_value <= sorted_values[-1]

    def test_p99_within_bounds(self):
        """p99 index must always be within [0, n-1]."""
        values = list(range(1, 101))
        collector = self._collector_with_values(values)
        summary = collector.get_metric_summary("test_metric")

        assert summary is not None
        sorted_values = sorted(values)
        assert summary.p99_value in sorted_values, "p99 must be an actual data value"
        assert summary.p99_value <= sorted_values[-1]


# ---------------------------------------------------------------------------
# Issue #12 – Exported logs always empty
# ---------------------------------------------------------------------------


class TestDatabaseLogHandlerFires:
    """
    Tests that DatabaseLogHandler receives records when structlog is backed
    by stdlib (LoggerFactory instead of PrintLoggerFactory).
    """

    def test_log_entries_stored_after_stdlib_logger_call(self):
        """
        Calling stdlib logging.getLogger(name).info() must produce a stored
        LogEntry when a DatabaseLogHandler is attached via LogManager.get_logger().
        """
        manager = LogManager(enable_logging=True)
        name = "test.logger.stdlib"

        # Attaches DatabaseLogHandler to the stdlib logger for `name`
        manager.get_logger(name)

        stdlib_logger = logging.getLogger(name)
        stdlib_logger.info("hello from stdlib")

        entries = manager.get_log_entries()
        assert len(entries) > 0, (
            "DatabaseLogHandler did not capture any log entry; "
            "structlog may still be using PrintLoggerFactory"
        )
        messages = [e.message for e in entries]
        assert "hello from stdlib" in messages

    def test_export_logs_non_empty_after_logging(self):
        """export_logs() must return at least one entry after a log call."""
        manager = LogManager(enable_logging=True)
        name = "test.logger.export"

        manager.get_logger(name)
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.warning("export test message")

        exported = manager.export_logs(format="dict")
        assert isinstance(exported, list)
        assert len(exported) > 0, "export_logs returned an empty list after logging"

    def test_log_structured_always_stores_entry(self):
        """
        log_structured() writes directly to the store and must work regardless
        of the factory change (sanity check for the direct path).
        """
        manager = LogManager(enable_logging=True)
        from nons.observability.logging import LogLevel

        manager.log_structured(LogLevel.INFO, "direct structured message")
        entries = manager.get_log_entries()
        assert any(e.message == "direct structured message" for e in entries)


# ---------------------------------------------------------------------------
# Issue #13 – Finishing child span corrupts parent context
# ---------------------------------------------------------------------------


class TestSpanContextRestoration:
    """
    Tests that finishing a child span restores the parent's SpanContext so
    subsequent start_span() calls correctly attach to the right parent.
    """

    def test_context_is_parent_after_child_finishes(self):
        """
        After finishing a child span the current context must point back to
        the parent span, not to the (now-finished) child.
        """
        tracer = TraceManager()

        parent = tracer.start_span("parent", SpanKind.NETWORK)
        parent_ctx = tracer.get_current_context()
        assert parent_ctx is not None
        assert parent_ctx.span_id == parent.span_id

        child = tracer.start_span("child", SpanKind.NODE)
        child_ctx = tracer.get_current_context()
        assert child_ctx is not None
        assert child_ctx.span_id == child.span_id

        tracer.finish_span(child)

        restored_ctx = tracer.get_current_context()
        assert restored_ctx is not None, "Context must not be None after finishing child"
        assert restored_ctx.span_id == parent.span_id, (
            f"Expected parent span_id {parent.span_id!r}, "
            f"got {restored_ctx.span_id!r}"
        )

    def test_sibling_span_gets_correct_parent(self):
        """
        A span started after a sibling is finished must use the shared parent,
        not the finished sibling.
        """
        tracer = TraceManager()

        parent = tracer.start_span("parent", SpanKind.NETWORK)

        child_a = tracer.start_span("child_a", SpanKind.NODE)
        tracer.finish_span(child_a)

        # child_b should see *parent* as parent, not child_a
        child_b = tracer.start_span("child_b", SpanKind.NODE)

        assert child_b.parent_span_id == parent.span_id, (
            f"child_b should have parent_span_id={parent.span_id!r}, "
            f"got {child_b.parent_span_id!r}"
        )

        tracer.finish_span(child_b)
        tracer.finish_span(parent)

    def test_context_is_none_after_root_span_finishes(self):
        """
        After the root span finishes the context must be restored to its
        initial state (None), not left pointing to the finished root.
        """
        tracer = TraceManager()

        root = tracer.start_span("root", SpanKind.NETWORK)
        tracer.finish_span(root)

        ctx = tracer.get_current_context()
        assert ctx is None, (
            f"Context should be None after root span finishes, got {ctx!r}"
        )


# ---------------------------------------------------------------------------
# Issue #10 – CancelledError leaves orphaned spans; get_trace_summary crashes
# ---------------------------------------------------------------------------


class TestCancelledErrorHandling:
    """
    Tests that asyncio.CancelledError is caught by trace_operation() so the
    span is properly closed, and that get_trace_summary() doesn't crash when
    spans have no end_time.
    """

    @pytest.mark.asyncio
    async def test_cancelled_span_is_moved_to_completed(self):
        """
        When an asyncio.CancelledError is raised inside trace_operation the
        span must not remain in active_spans (it must be finished/completed).
        """
        obs = ObservabilityManager()

        with pytest.raises(asyncio.CancelledError):
            async with obs.trace_operation("cancellable_op", SpanKind.NODE) as span:
                span_id = span.span_id
                raise asyncio.CancelledError()

        active_ids = {s.span_id for s in obs.tracer.get_active_spans()}
        assert span_id not in active_ids, (
            "Span must not remain active after CancelledError"
        )

    @pytest.mark.asyncio
    async def test_cancelled_span_recorded_as_error(self):
        """The span finished due to CancelledError must be present in completed spans."""
        obs = ObservabilityManager()

        span_id = None
        with pytest.raises(asyncio.CancelledError):
            async with obs.trace_operation("cancellable_op2", SpanKind.NODE) as span:
                span_id = span.span_id
                raise asyncio.CancelledError()

        completed_ids = {s.span_id for s in obs.tracer.get_completed_spans()}
        assert span_id in completed_ids, (
            "Span must appear in completed spans after CancelledError"
        )

    def test_get_trace_summary_no_crash_when_end_times_all_none(self):
        """
        get_trace_summary() must not raise ValueError when all spans in the
        trace are still running (end_time is None for every span).
        """
        obs = ObservabilityManager()

        # Start a span but do NOT finish it so end_time stays None
        span = obs.tracer.start_span("running_span", SpanKind.NODE)
        trace_id = span.trace_id

        # This must not raise
        summary = obs.get_trace_summary(trace_id)

        assert "trace_id" in summary
        assert summary["trace_id"] == trace_id
        # end_time is None when no spans have finished
        assert summary["end_time"] is None
        assert summary["duration_ms"] == 0

    def test_get_trace_summary_with_mixed_end_times(self):
        """
        When some spans have end_time and some do not, get_trace_summary()
        must succeed and return the max of the available end_times.
        """
        obs = ObservabilityManager()

        parent = obs.tracer.start_span("parent", SpanKind.NETWORK)
        child = obs.tracer.start_span("child", SpanKind.NODE)
        trace_id = parent.trace_id

        # Finish only the child; parent stays running
        obs.tracer.finish_span(child)

        summary = obs.get_trace_summary(trace_id)

        assert "error" not in summary
        assert summary["end_time"] is not None, (
            "end_time should reflect the finished child span"
        )
        assert summary["duration_ms"] >= 0
