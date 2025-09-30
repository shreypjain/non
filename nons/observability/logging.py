"""
Structured logging system for NoN with database-ready log entries.

Provides structured, contextual logging that integrates with tracing
and produces JSON-formatted logs suitable for database storage.
"""

import logging
import json
import time
import threading
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
from enum import Enum
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict

import structlog


class LogLevel(str, Enum):
    """Standard log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """
    Structured log entry with comprehensive context.

    Database-ready structure for storing log data.
    """
    # Core identification
    log_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Message details
    level: LogLevel = LogLevel.INFO
    message: str = ""
    logger_name: str = ""

    # Context
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    component_type: str = ""  # "network", "layer", "node", "operator"
    component_id: str = ""

    # Structured data
    fields: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

    # Location information
    filename: Optional[str] = None
    function_name: Optional[str] = None
    line_number: Optional[int] = None

    # Error information (if applicable)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_stack: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return asdict(self)

    def to_json_serializable(self) -> Dict[str, Any]:
        """Convert to JSON-serializable format."""
        data = self.to_dict()

        # Convert enums to strings
        data["level"] = self.level.value

        # Convert timestamp to ISO format
        data["timestamp_iso"] = datetime.fromtimestamp(
            self.timestamp, tz=timezone.utc
        ).isoformat()

        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_json_serializable(), default=str)


class DatabaseLogHandler(logging.Handler):
    """
    Custom log handler that collects structured logs for database storage.
    """

    def __init__(self, log_manager: 'LogManager'):
        super().__init__()
        self.log_manager = log_manager

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the log manager."""
        try:
            # Get trace context if available
            trace_context = self.log_manager.get_current_trace_context()

            # Create structured log entry
            log_entry = LogEntry(
                timestamp=record.created,
                level=LogLevel(record.levelname.lower()),
                message=record.getMessage(),
                logger_name=record.name,
                trace_id=trace_context.get("trace_id") if trace_context else None,
                span_id=trace_context.get("span_id") if trace_context else None,
                filename=record.filename,
                function_name=record.funcName,
                line_number=record.lineno
            )

            # Add extra fields if present
            if hasattr(record, "fields"):
                log_entry.fields.update(record.fields)

            if hasattr(record, "tags"):
                log_entry.tags.update(record.tags)

            if hasattr(record, "component_type"):
                log_entry.component_type = record.component_type

            if hasattr(record, "component_id"):
                log_entry.component_id = record.component_id

            # Handle exceptions
            if record.exc_info:
                import traceback
                log_entry.error_type = record.exc_info[0].__name__
                log_entry.error_message = str(record.exc_info[1])
                log_entry.error_stack = "".join(traceback.format_exception(*record.exc_info))

            # Store the log entry
            self.log_manager.add_log_entry(log_entry)

        except Exception:
            # Don't let logging errors break the application
            self.handleError(record)


class LogManager:
    """
    Manages structured logging with database-ready output.

    Provides centralized log collection, formatting, and export
    capabilities for database storage.
    """

    def __init__(self, enable_logging: bool = True):
        self.enable_logging = enable_logging
        self.log_entries: List[LogEntry] = []
        self._lock = threading.RLock()

        # Context variable for trace correlation
        self.trace_context: ContextVar[Optional[Dict[str, str]]] = ContextVar(
            'trace_context', default=None
        )

        # Set up structured logging
        self._setup_structured_logging()

    def _setup_structured_logging(self) -> None:
        """Configure structlog for structured logging."""
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.dev.ConsoleRenderer() if self.enable_logging else structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def get_logger(self, name: str) -> structlog.BoundLogger:
        """Get a structured logger for the given name."""
        logger = structlog.get_logger(name)

        # Add database handler if logging is enabled
        if self.enable_logging:
            # Get the underlying stdlib logger and add our handler
            stdlib_logger = logging.getLogger(name)
            if not any(isinstance(h, DatabaseLogHandler) for h in stdlib_logger.handlers):
                db_handler = DatabaseLogHandler(self)
                stdlib_logger.addHandler(db_handler)
                stdlib_logger.setLevel(logging.DEBUG)

        return logger

    def set_trace_context(self, trace_id: str, span_id: str) -> None:
        """Set trace context for log correlation."""
        context = {
            "trace_id": trace_id,
            "span_id": span_id
        }
        self.trace_context.set(context)

        # Also set in structlog context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            trace_id=trace_id,
            span_id=span_id
        )

    def clear_trace_context(self) -> None:
        """Clear trace context."""
        self.trace_context.set(None)
        structlog.contextvars.clear_contextvars()

    def get_current_trace_context(self) -> Optional[Dict[str, str]]:
        """Get current trace context."""
        return self.trace_context.get()

    def add_log_entry(self, log_entry: LogEntry) -> None:
        """Add a log entry to the collection."""
        if not self.enable_logging:
            return

        with self._lock:
            self.log_entries.append(log_entry)

    def log_structured(
        self,
        level: LogLevel,
        message: str,
        component_type: str = "",
        component_id: str = "",
        **fields
    ) -> None:
        """Log a structured message directly."""
        if not self.enable_logging:
            return

        trace_context = self.get_current_trace_context()

        log_entry = LogEntry(
            level=level,
            message=message,
            component_type=component_type,
            component_id=component_id,
            trace_id=trace_context.get("trace_id") if trace_context else None,
            span_id=trace_context.get("span_id") if trace_context else None,
            fields=fields
        )

        self.add_log_entry(log_entry)

    def get_log_entries(
        self,
        since: Optional[float] = None,
        trace_id: Optional[str] = None,
        level: Optional[LogLevel] = None
    ) -> List[LogEntry]:
        """Get log entries with optional filtering."""
        with self._lock:
            entries = self.log_entries.copy()

        filtered = []
        for entry in entries:
            # Filter by timestamp
            if since and entry.timestamp < since:
                continue

            # Filter by trace ID
            if trace_id and entry.trace_id != trace_id:
                continue

            # Filter by level
            if level and entry.level != level:
                continue

            filtered.append(entry)

        return sorted(filtered, key=lambda e: e.timestamp)

    def export_logs(
        self,
        since: Optional[float] = None,
        trace_id: Optional[str] = None,
        format: str = "json"
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Export logs in database-ready format.

        Args:
            since: Optional timestamp to get logs since
            trace_id: Optional trace ID to filter logs
            format: Output format ("json" or "dict")

        Returns:
            List of log dictionaries or JSON string
        """
        entries = self.get_log_entries(since=since, trace_id=trace_id)

        if format == "dict":
            return [entry.to_json_serializable() for entry in entries]
        elif format == "json":
            return json.dumps([entry.to_json_serializable() for entry in entries], default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def clear_logs(self) -> int:
        """Clear stored log entries and return count cleared."""
        with self._lock:
            count = len(self.log_entries)
            self.log_entries.clear()
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        with self._lock:
            entries = self.log_entries.copy()

        stats = {
            "total_entries": len(entries),
            "entries_by_level": {},
            "entries_by_component": {},
            "oldest_entry": None,
            "newest_entry": None
        }

        if entries:
            # Count by level
            for entry in entries:
                level = entry.level.value
                stats["entries_by_level"][level] = stats["entries_by_level"].get(level, 0) + 1

            # Count by component
            for entry in entries:
                component = entry.component_type or "unknown"
                stats["entries_by_component"][component] = stats["entries_by_component"].get(component, 0) + 1

            # Time range
            timestamps = [entry.timestamp for entry in entries]
            stats["oldest_entry"] = min(timestamps)
            stats["newest_entry"] = max(timestamps)

        return stats


# Global log manager
_log_manager: Optional[LogManager] = None


def get_logger(name: str = "nons") -> structlog.BoundLogger:
    """Get a structured logger."""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager()
    return _log_manager.get_logger(name)


def get_log_manager() -> LogManager:
    """Get the global log manager."""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager()
    return _log_manager


def configure_logging(enable: bool = True) -> LogManager:
    """Configure global logging."""
    global _log_manager
    _log_manager = LogManager(enable_logging=enable)
    return _log_manager


class LoggedOperation:
    """Context manager for automatic log correlation with spans."""

    def __init__(
        self,
        logger: structlog.BoundLogger,
        operation_name: str,
        component_type: str = "",
        component_id: str = "",
        log_manager: Optional[LogManager] = None
    ):
        self.logger = logger
        self.operation_name = operation_name
        self.component_type = component_type
        self.component_id = component_id
        self.log_manager = log_manager or get_log_manager()

    def __enter__(self):
        self.logger.info(
            f"Starting {self.operation_name}",
            component_type=self.component_type,
            component_id=self.component_id
        )
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.error(
                f"Failed {self.operation_name}",
                component_type=self.component_type,
                component_id=self.component_id,
                error_type=exc_type.__name__,
                error_message=str(exc_val)
            )
        else:
            self.logger.info(
                f"Completed {self.operation_name}",
                component_type=self.component_type,
                component_id=self.component_id
            )


def logged_operation(
    operation_name: str,
    component_type: str = "",
    component_id: str = "",
    logger_name: str = "nons"
) -> LoggedOperation:
    """Create a logged operation context manager."""
    logger = get_logger(logger_name)
    return LoggedOperation(logger, operation_name, component_type, component_id)