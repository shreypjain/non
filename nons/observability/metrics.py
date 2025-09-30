"""
Metrics collection system for NoN with database-ready time series data.

Provides comprehensive metrics collection for performance monitoring,
cost tracking, and operational insights with database-ready structures.
"""

import time
import threading
import uuid
from typing import Any, Dict, List, Optional, Union, DefaultDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from collections import defaultdict, deque
import statistics


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    COST = "cost"
    TOKEN = "token"


@dataclass
class MetricPoint:
    """
    Single metric data point with comprehensive metadata.

    Database-ready structure for storing time series metrics.
    """
    # Core identification
    metric_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Metric details
    metric_name: str = ""
    metric_type: MetricType = MetricType.GAUGE
    value: Union[int, float] = 0.0
    unit: str = ""

    # Context
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    component_type: str = ""  # "network", "layer", "node", "operator"
    component_id: str = ""

    # Labels/tags for grouping
    labels: Dict[str, str] = field(default_factory=dict)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return asdict(self)

    def to_json_serializable(self) -> Dict[str, Any]:
        """Convert to JSON-serializable format."""
        data = self.to_dict()

        # Convert enums to strings
        data["metric_type"] = self.metric_type.value

        # Convert timestamp to ISO format
        data["timestamp_iso"] = datetime.fromtimestamp(
            self.timestamp, tz=timezone.utc
        ).isoformat()

        return data


@dataclass
class MetricSummary:
    """Summary statistics for a metric over a time period."""
    metric_name: str
    count: int = 0
    sum_value: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    avg_value: Optional[float] = None
    p50_value: Optional[float] = None
    p95_value: Optional[float] = None
    p99_value: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetricsCollector:
    """
    Collects and aggregates metrics with database-ready output.

    Provides thread-safe metrics collection with automatic aggregation
    and export capabilities for database storage and monitoring.
    """

    def __init__(self, enable_metrics: bool = True, max_points: int = 10000):
        self.enable_metrics = enable_metrics
        self.max_points = max_points
        self.metric_points: List[MetricPoint] = []
        self.metric_buffers: DefaultDict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.RLock()

        # Current values for gauges
        self.current_gauges: Dict[str, float] = {}

        # Counters
        self.counters: DefaultDict[str, float] = defaultdict(float)

    def record_metric(
        self,
        metric_name: str,
        value: Union[int, float],
        metric_type: MetricType = MetricType.GAUGE,
        unit: str = "",
        labels: Optional[Dict[str, str]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        component_type: str = "",
        component_id: str = "",
        **metadata
    ) -> None:
        """Record a metric point."""
        if not self.enable_metrics:
            return

        metric_point = MetricPoint(
            metric_name=metric_name,
            metric_type=metric_type,
            value=value,
            unit=unit,
            labels=labels or {},
            trace_id=trace_id,
            span_id=span_id,
            component_type=component_type,
            component_id=component_id,
            metadata=metadata
        )

        with self._lock:
            # Store the metric point
            self.metric_points.append(metric_point)

            # Update aggregations based on type
            if metric_type == MetricType.COUNTER:
                self.counters[metric_name] += value
            elif metric_type == MetricType.GAUGE:
                self.current_gauges[metric_name] = value

            # Add to buffer for summary calculations
            self.metric_buffers[metric_name].append(value)

            # Trim if we exceed max points
            if len(self.metric_points) > self.max_points:
                self.metric_points = self.metric_points[-self.max_points:]

    def increment_counter(
        self,
        metric_name: str,
        value: Union[int, float] = 1,
        **kwargs
    ) -> None:
        """Increment a counter metric."""
        self.record_metric(
            metric_name=metric_name,
            value=value,
            metric_type=MetricType.COUNTER,
            **kwargs
        )

    def set_gauge(
        self,
        metric_name: str,
        value: Union[int, float],
        **kwargs
    ) -> None:
        """Set a gauge metric."""
        self.record_metric(
            metric_name=metric_name,
            value=value,
            metric_type=MetricType.GAUGE,
            **kwargs
        )

    def record_timing(
        self,
        metric_name: str,
        duration_ms: float,
        **kwargs
    ) -> None:
        """Record a timing metric."""
        self.record_metric(
            metric_name=metric_name,
            value=duration_ms,
            metric_type=MetricType.TIMER,
            unit="milliseconds",
            **kwargs
        )

    def record_cost(
        self,
        metric_name: str,
        cost_usd: float,
        **kwargs
    ) -> None:
        """Record a cost metric."""
        self.record_metric(
            metric_name=metric_name,
            value=cost_usd,
            metric_type=MetricType.COST,
            unit="usd",
            **kwargs
        )

    def record_tokens(
        self,
        metric_name: str,
        token_count: int,
        **kwargs
    ) -> None:
        """Record a token usage metric."""
        self.record_metric(
            metric_name=metric_name,
            value=token_count,
            metric_type=MetricType.TOKEN,
            unit="tokens",
            **kwargs
        )

    def get_metric_summary(
        self,
        metric_name: str,
        since: Optional[float] = None
    ) -> Optional[MetricSummary]:
        """Get summary statistics for a metric."""
        with self._lock:
            # Get points for this metric
            points = [
                p for p in self.metric_points
                if p.metric_name == metric_name and (since is None or p.timestamp >= since)
            ]

            if not points:
                return None

            values = [p.value for p in points]
            timestamps = [p.timestamp for p in points]

            summary = MetricSummary(
                metric_name=metric_name,
                count=len(values),
                sum_value=sum(values),
                min_value=min(values),
                max_value=max(values),
                avg_value=statistics.mean(values),
                start_time=min(timestamps),
                end_time=max(timestamps)
            )

            # Calculate percentiles if we have enough data
            if len(values) >= 2:
                sorted_values = sorted(values)
                summary.p50_value = statistics.median(sorted_values)

            if len(values) >= 20:
                summary.p95_value = sorted_values[int(0.95 * len(sorted_values))]
                summary.p99_value = sorted_values[int(0.99 * len(sorted_values))]

            return summary

    def get_current_values(self) -> Dict[str, Union[int, float]]:
        """Get current values for all metrics."""
        with self._lock:
            current = {}
            current.update(self.current_gauges)
            current.update(self.counters)
            return current

    def get_metric_points(
        self,
        metric_name: Optional[str] = None,
        since: Optional[float] = None,
        trace_id: Optional[str] = None
    ) -> List[MetricPoint]:
        """Get metric points with optional filtering."""
        with self._lock:
            points = self.metric_points.copy()

        filtered = []
        for point in points:
            # Filter by metric name
            if metric_name and point.metric_name != metric_name:
                continue

            # Filter by timestamp
            if since and point.timestamp < since:
                continue

            # Filter by trace ID
            if trace_id and point.trace_id != trace_id:
                continue

            filtered.append(point)

        return sorted(filtered, key=lambda p: p.timestamp)

    def export_metrics(
        self,
        since: Optional[float] = None,
        metric_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Export metrics in database-ready format.

        Args:
            since: Optional timestamp to get metrics since
            metric_names: Optional list of metric names to export

        Returns:
            List of metric point dictionaries ready for database insertion
        """
        points = []

        with self._lock:
            for point in self.metric_points:
                # Filter by timestamp
                if since and point.timestamp < since:
                    continue

                # Filter by metric names
                if metric_names and point.metric_name not in metric_names:
                    continue

                points.append(point.to_json_serializable())

        return points

    def get_stats(self) -> Dict[str, Any]:
        """Get metrics collection statistics."""
        with self._lock:
            points = self.metric_points.copy()

        stats = {
            "total_points": len(points),
            "points_by_type": {},
            "points_by_component": {},
            "unique_metrics": set(),
            "oldest_point": None,
            "newest_point": None,
            "current_gauges": len(self.current_gauges),
            "current_counters": len(self.counters)
        }

        if points:
            # Count by type
            for point in points:
                metric_type = point.metric_type.value
                stats["points_by_type"][metric_type] = stats["points_by_type"].get(metric_type, 0) + 1

            # Count by component
            for point in points:
                component = point.component_type or "unknown"
                stats["points_by_component"][component] = stats["points_by_component"].get(component, 0) + 1

            # Unique metrics
            stats["unique_metrics"] = len(set(p.metric_name for p in points))

            # Time range
            timestamps = [point.timestamp for point in points]
            stats["oldest_point"] = min(timestamps)
            stats["newest_point"] = max(timestamps)

        return stats

    def clear_metrics(self) -> int:
        """Clear stored metric points and return count cleared."""
        with self._lock:
            count = len(self.metric_points)
            self.metric_points.clear()
            self.metric_buffers.clear()
            return count

    def reset_counters(self) -> None:
        """Reset all counters to zero."""
        with self._lock:
            self.counters.clear()


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def configure_metrics(enable: bool = True, max_points: int = 10000) -> MetricsCollector:
    """Configure global metrics collection."""
    global _metrics_collector
    _metrics_collector = MetricsCollector(enable_metrics=enable, max_points=max_points)
    return _metrics_collector


class TimedOperation:
    """Context manager for automatic timing metrics."""

    def __init__(
        self,
        metric_name: str,
        collector: Optional[MetricsCollector] = None,
        **metric_kwargs
    ):
        self.metric_name = metric_name
        self.collector = collector or get_metrics_collector()
        self.metric_kwargs = metric_kwargs
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.collector.record_timing(
                self.metric_name,
                duration_ms,
                **self.metric_kwargs
            )

            # Record success/failure
            status = "error" if exc_type else "success"
            self.collector.increment_counter(
                f"{self.metric_name}.{status}",
                **self.metric_kwargs
            )


def timed_operation(
    metric_name: str,
    **metric_kwargs
) -> TimedOperation:
    """Create a timed operation context manager."""
    return TimedOperation(metric_name, **metric_kwargs)