"""
Request scheduler with rate limiting and queue management for NoN execution.

Provides intelligent request scheduling across multiple LLM providers with
configurable rate limits, request queuing, and automatic backoff strategies.
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import heapq
from datetime import datetime, timedelta

from .types import ModelProvider, ModelConfig
from ..observability.integration import get_observability
from ..observability.tracing import SpanKind


class QueueStrategy(str, Enum):
    """Request queue strategies."""

    FIFO = "fifo"  # First In, First Out
    PRIORITY = "priority"  # Priority-based scheduling
    ROUND_ROBIN = "round_robin"  # Round robin across providers
    LEAST_LOADED = "least_loaded"  # Route to least loaded provider


class BackoffStrategy(str, Enum):
    """Backoff strategies for rate limiting."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIXED = "fixed"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    requests_per_second: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    max_concurrent: int = 10
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_base_delay: float = 1.0
    backoff_max_delay: float = 60.0
    backoff_multiplier: float = 2.0


@dataclass
class RequestItem:
    """Individual request item in the queue."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher numbers = higher priority
    provider: ModelProvider = ModelProvider.OPENAI
    model_config: Optional[ModelConfig] = None

    # Request details
    operation: Optional[Callable[..., Awaitable[Any]]] = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # Response handling
    future: Optional[asyncio.Future] = field(default_factory=asyncio.Future)

    # Metadata
    estimated_tokens: int = 0
    component_type: str = ""
    component_id: str = ""

    def __lt__(self, other):
        """For priority queue ordering."""
        # Higher priority first, then earlier timestamp
        return (self.priority, -self.timestamp) > (other.priority, -other.timestamp)


@dataclass
class ProviderStats:
    """Statistics for a provider."""

    requests_in_queue: int = 0
    requests_in_flight: int = 0
    requests_completed: int = 0
    requests_failed: int = 0
    requests_rate_limited: int = 0

    # Timing
    total_request_time: float = 0.0
    avg_request_time: float = 0.0

    # Rate limiting
    last_request_time: float = 0.0
    request_timestamps: deque = field(default_factory=lambda: deque(maxlen=100))

    # Token usage
    total_tokens_used: int = 0
    tokens_per_minute_window: deque = field(default_factory=lambda: deque(maxlen=60))


class RequestScheduler:
    """
    Intelligent request scheduler with rate limiting and queue management.

    Manages request distribution across multiple LLM providers with
    configurable rate limits, queuing strategies, and automatic scaling.
    """

    def __init__(
        self,
        default_rate_limits: Optional[Dict[ModelProvider, RateLimitConfig]] = None,
        queue_strategy: QueueStrategy = QueueStrategy.PRIORITY,
        enable_observability: bool = True,
    ):
        self.queue_strategy = queue_strategy
        self.enable_observability = enable_observability

        # Default rate limits per provider
        self.rate_limits = default_rate_limits or self._get_default_rate_limits()

        # Request queues per provider
        self.request_queues: Dict[ModelProvider, List[RequestItem]] = {
            provider: [] for provider in ModelProvider
        }

        # Provider statistics
        self.provider_stats: Dict[ModelProvider, ProviderStats] = {
            provider: ProviderStats() for provider in ModelProvider
        }

        # Active requests tracking
        self.active_requests: Dict[str, RequestItem] = {}
        self.semaphores: Dict[ModelProvider, asyncio.Semaphore] = {}

        # Global state
        self.is_running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Round robin state
        self._round_robin_index = 0

        # Initialize semaphores
        for provider, config in self.rate_limits.items():
            self.semaphores[provider] = asyncio.Semaphore(config.max_concurrent)

    def _get_default_rate_limits(self) -> Dict[ModelProvider, RateLimitConfig]:
        """Get default rate limits for each provider."""
        return {
            ModelProvider.OPENAI: RateLimitConfig(
                requests_per_minute=500,
                requests_per_second=10,
                tokens_per_minute=200000,
                max_concurrent=20,
            ),
            ModelProvider.ANTHROPIC: RateLimitConfig(
                requests_per_minute=1000,
                requests_per_second=15,
                tokens_per_minute=400000,
                max_concurrent=25,
            ),
            ModelProvider.GOOGLE: RateLimitConfig(
                requests_per_minute=1500,
                requests_per_second=20,
                tokens_per_minute=1000000,
                max_concurrent=30,
            ),
            ModelProvider.MOCK: RateLimitConfig(
                requests_per_minute=10000, max_concurrent=100
            ),
        }

    async def start(self) -> None:
        """Start the request scheduler."""
        if self.is_running:
            return

        self.is_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())

        if self.enable_observability:
            obs = get_observability()
            async with obs.trace_operation(
                operation_name="scheduler_start",
                kind=SpanKind.OPERATOR,
                component_type="scheduler",
                component_id="global",
            ) as span:
                span.add_tags(
                    {
                        "queue_strategy": self.queue_strategy.value,
                        "providers": [p.value for p in self.rate_limits.keys()],
                    }
                )

    async def stop(self) -> None:
        """Stop the request scheduler."""
        self.is_running = False

        if self.scheduler_task and not self.scheduler_task.done():
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

        # Cancel any pending requests
        async with self._lock:
            for request_item in self.active_requests.values():
                if request_item.future and not request_item.future.done():
                    request_item.future.cancel()

    async def schedule_request(
        self,
        operation: Callable[..., Awaitable[Any]],
        provider: ModelProvider,
        model_config: Optional[ModelConfig] = None,
        priority: int = 0,
        estimated_tokens: int = 0,
        component_type: str = "",
        component_id: str = "",
        *args,
        **kwargs,
    ) -> Any:
        """
        Schedule a request for execution.

        Args:
            operation: The async operation to execute
            provider: The LLM provider to use
            model_config: Optional model configuration
            priority: Request priority (higher = more urgent)
            estimated_tokens: Estimated token usage
            component_type: Component type for observability
            component_id: Component ID for observability
            *args, **kwargs: Arguments to pass to the operation

        Returns:
            The result of the operation
        """
        request_item = RequestItem(
            provider=provider,
            model_config=model_config,
            operation=operation,
            args=args,
            kwargs=kwargs,
            priority=priority,
            estimated_tokens=estimated_tokens,
            component_type=component_type,
            component_id=component_id,
        )

        # Add to appropriate queue
        async with self._lock:
            if self.queue_strategy == QueueStrategy.PRIORITY:
                heapq.heappush(self.request_queues[provider], request_item)
            else:
                self.request_queues[provider].append(request_item)

            self.provider_stats[provider].requests_in_queue += 1

        if self.enable_observability:
            obs = get_observability()
            async with obs.trace_operation(
                operation_name="request_scheduled",
                kind=SpanKind.OPERATOR,
                component_type="scheduler",
                component_id=request_item.request_id,
            ) as span:
                span.add_tags(
                    {
                        "provider": provider.value,
                        "priority": priority,
                        "estimated_tokens": estimated_tokens,
                        "queue_length": len(self.request_queues[provider]),
                    }
                )

        # Wait for the request to complete
        return await request_item.future

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self.is_running:
            try:
                # Process requests based on strategy
                if self.queue_strategy == QueueStrategy.ROUND_ROBIN:
                    await self._process_round_robin()
                elif self.queue_strategy == QueueStrategy.LEAST_LOADED:
                    await self._process_least_loaded()
                else:
                    await self._process_all_queues()

                # Brief sleep to prevent busy waiting
                await asyncio.sleep(0.01)

            except Exception as e:
                if self.enable_observability:
                    obs = get_observability()
                    logger = obs.log_manager.get_logger("scheduler")
                    logger.error(
                        "Scheduler loop error",
                        component_type="scheduler",
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                await asyncio.sleep(1.0)  # Back off on errors

    async def _process_all_queues(self) -> None:
        """Process all provider queues."""
        for provider in ModelProvider:
            await self._process_provider_queue(provider)

    async def _process_round_robin(self) -> None:
        """Process queues in round-robin fashion."""
        providers = list(ModelProvider)
        if not providers:
            return

        provider = providers[self._round_robin_index % len(providers)]
        self._round_robin_index += 1

        await self._process_provider_queue(provider)

    async def _process_least_loaded(self) -> None:
        """Process the least loaded provider first."""
        # Find provider with smallest queue + in-flight ratio
        min_load = float("inf")
        least_loaded_provider = None

        for provider, stats in self.provider_stats.items():
            load = stats.requests_in_queue + stats.requests_in_flight
            if load < min_load and self.request_queues[provider]:
                min_load = load
                least_loaded_provider = provider

        if least_loaded_provider:
            await self._process_provider_queue(least_loaded_provider)

    async def _process_provider_queue(self, provider: ModelProvider) -> None:
        """Process requests for a specific provider."""
        queue = self.request_queues[provider]
        if not queue:
            return

        # Check rate limits
        if not await self._can_process_request(provider):
            return

        # Get next request
        async with self._lock:
            if self.queue_strategy == QueueStrategy.PRIORITY:
                request_item = heapq.heappop(queue)
            else:
                request_item = queue.pop(0)

            self.provider_stats[provider].requests_in_queue -= 1
            self.provider_stats[provider].requests_in_flight += 1
            self.active_requests[request_item.request_id] = request_item

        # Process the request
        asyncio.create_task(self._execute_request(request_item))

    async def _can_process_request(self, provider: ModelProvider) -> bool:
        """Check if we can process a request for the given provider."""
        config = self.rate_limits[provider]
        stats = self.provider_stats[provider]
        current_time = time.time()

        # Check concurrent limit
        if stats.requests_in_flight >= config.max_concurrent:
            return False

        # Check requests per second
        if config.requests_per_second:
            time_since_last = current_time - stats.last_request_time
            if time_since_last < (1.0 / config.requests_per_second):
                return False

        # Check requests per minute
        if config.requests_per_minute:
            # Clean old timestamps
            cutoff_time = current_time - 60.0
            while (
                stats.request_timestamps and stats.request_timestamps[0] < cutoff_time
            ):
                stats.request_timestamps.popleft()

            if len(stats.request_timestamps) >= config.requests_per_minute:
                return False

        # Check tokens per minute
        if config.tokens_per_minute:
            # Clean old token counts
            while (
                stats.tokens_per_minute_window
                and stats.tokens_per_minute_window[0][1] < current_time - 60.0
            ):
                stats.tokens_per_minute_window.popleft()

            total_tokens = sum(count for count, _ in stats.tokens_per_minute_window)
            if total_tokens >= config.tokens_per_minute:
                return False

        return True

    async def _execute_request(self, request_item: RequestItem) -> None:
        """Execute a single request."""
        provider = request_item.provider
        start_time = time.time()

        try:
            # Acquire semaphore
            async with self.semaphores[provider]:
                # Update rate limiting tracking
                self.provider_stats[provider].last_request_time = start_time
                self.provider_stats[provider].request_timestamps.append(start_time)

                if request_item.estimated_tokens:
                    self.provider_stats[provider].tokens_per_minute_window.append(
                        (request_item.estimated_tokens, start_time)
                    )

                # Execute the operation with observability
                if self.enable_observability:
                    obs = get_observability()
                    async with obs.trace_operation(
                        operation_name="request_execution",
                        kind=SpanKind.LLM_CALL,
                        component_type=request_item.component_type,
                        component_id=request_item.component_id,
                    ) as span:
                        span.add_tags(
                            {
                                "provider": provider.value,
                                "request_id": request_item.request_id,
                                "estimated_tokens": request_item.estimated_tokens,
                            }
                        )

                        result = await request_item.operation(
                            *request_item.args, **request_item.kwargs
                        )
                else:
                    result = await request_item.operation(
                        *request_item.args, **request_item.kwargs
                    )

                # Set the result
                if not request_item.future.done():
                    request_item.future.set_result(result)

                # Update stats
                execution_time = time.time() - start_time
                stats = self.provider_stats[provider]
                stats.requests_completed += 1
                stats.total_request_time += execution_time
                stats.avg_request_time = (
                    stats.total_request_time / stats.requests_completed
                )

        except Exception as e:
            # Handle errors
            self.provider_stats[provider].requests_failed += 1

            if not request_item.future.done():
                request_item.future.set_exception(e)

            if self.enable_observability:
                obs = get_observability()
                logger = obs.log_manager.get_logger("scheduler")
                logger.error(
                    "Request execution failed",
                    component_type=request_item.component_type,
                    component_id=request_item.component_id,
                    provider=provider.value,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )

        finally:
            # Clean up
            async with self._lock:
                self.provider_stats[provider].requests_in_flight -= 1
                if request_item.request_id in self.active_requests:
                    del self.active_requests[request_item.request_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive scheduler statistics."""
        stats = {
            "is_running": self.is_running,
            "queue_strategy": self.queue_strategy.value,
            "total_active_requests": len(self.active_requests),
            "providers": {},
        }

        for provider, provider_stats in self.provider_stats.items():
            queue_length = len(self.request_queues[provider])

            # Get rate limit config if available
            rate_config = self.rate_limits.get(provider)
            rate_limit_config = {
                "requests_per_minute": (
                    rate_config.requests_per_minute if rate_config else 0
                ),
                "max_concurrent": rate_config.max_concurrent if rate_config else 0,
            }

            stats["providers"][provider.value] = {
                "queue_length": queue_length,
                "requests_in_flight": provider_stats.requests_in_flight,
                "requests_completed": provider_stats.requests_completed,
                "requests_failed": provider_stats.requests_failed,
                "requests_rate_limited": provider_stats.requests_rate_limited,
                "avg_request_time": provider_stats.avg_request_time,
                "total_tokens_used": provider_stats.total_tokens_used,
                "rate_limit_config": rate_limit_config,
            }

        return stats

    def update_rate_limits(
        self, provider: ModelProvider, config: RateLimitConfig
    ) -> None:
        """Update rate limits for a provider."""
        self.rate_limits[provider] = config
        self.semaphores[provider] = asyncio.Semaphore(config.max_concurrent)

    async def wait_for_completion(self) -> None:
        """Wait for all queued and active requests to complete."""
        while True:
            total_pending = sum(len(queue) for queue in self.request_queues.values())
            total_active = len(self.active_requests)

            if total_pending == 0 and total_active == 0:
                break

            await asyncio.sleep(0.1)


# Global scheduler instance
_request_scheduler: Optional[RequestScheduler] = None


def get_scheduler() -> RequestScheduler:
    """Get the global request scheduler."""
    global _request_scheduler
    if _request_scheduler is None:
        _request_scheduler = RequestScheduler()
    return _request_scheduler


def configure_scheduler(
    rate_limits: Optional[Dict[ModelProvider, RateLimitConfig]] = None,
    queue_strategy: QueueStrategy = QueueStrategy.PRIORITY,
    enable_observability: bool = True,
) -> RequestScheduler:
    """Configure the global request scheduler."""
    global _request_scheduler
    _request_scheduler = RequestScheduler(
        default_rate_limits=rate_limits,
        queue_strategy=queue_strategy,
        enable_observability=enable_observability,
    )
    return _request_scheduler


async def start_scheduler() -> None:
    """Start the global request scheduler."""
    scheduler = get_scheduler()
    await scheduler.start()


async def stop_scheduler() -> None:
    """Stop the global request scheduler."""
    scheduler = get_scheduler()
    await scheduler.stop()
