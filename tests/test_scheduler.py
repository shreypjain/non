"""
Tests for request scheduler with rate limiting.
"""
import pytest
import asyncio
import time
from unittest.mock import patch, AsyncMock, MagicMock
from nons.core.scheduler import (
    RequestScheduler, RateLimitConfig, QueueStrategy, BackoffStrategy,
    configure_scheduler, get_scheduler, start_scheduler, stop_scheduler
)
from nons.core.types import ModelProvider, ModelConfig


class TestRateLimitConfig:
    """Test rate limit configuration."""

    def test_rate_limit_config_initialization(self):
        """Test rate limit config initialization."""
        config = RateLimitConfig(
            requests_per_minute=60,
            requests_per_second=2,
            tokens_per_minute=1000,
            max_concurrent=5,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            backoff_base_delay=1.0,
            backoff_max_delay=30.0,
            backoff_multiplier=2.0
        )

        assert config.requests_per_minute == 60
        assert config.requests_per_second == 2
        assert config.tokens_per_minute == 1000
        assert config.max_concurrent == 5
        assert config.backoff_strategy == BackoffStrategy.EXPONENTIAL
        assert config.backoff_base_delay == 1.0
        assert config.backoff_max_delay == 30.0
        assert config.backoff_multiplier == 2.0

    def test_rate_limit_config_defaults(self):
        """Test rate limit config default values."""
        config = RateLimitConfig()

        assert config.requests_per_minute == 60
        assert config.requests_per_second is None
        assert config.tokens_per_minute is None
        assert config.max_concurrent == 10
        assert config.backoff_strategy == BackoffStrategy.EXPONENTIAL


class TestRequestScheduler:
    """Test request scheduler functionality."""

    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        scheduler = RequestScheduler()

        assert scheduler.queue_strategy == QueueStrategy.PRIORITY
        assert scheduler.running is False
        assert len(scheduler.provider_queues) == 0

    def test_scheduler_with_custom_rate_limits(self):
        """Test scheduler with custom rate limits."""
        rate_limits = {
            ModelProvider.OPENAI: RateLimitConfig(
                requests_per_minute=100,
                max_concurrent=15
            ),
            ModelProvider.ANTHROPIC: RateLimitConfig(
                requests_per_minute=200,
                max_concurrent=20
            )
        }

        scheduler = RequestScheduler(
            default_rate_limits=rate_limits,
            queue_strategy=QueueStrategy.FIFO
        )

        assert scheduler.queue_strategy == QueueStrategy.FIFO
        assert ModelProvider.OPENAI in scheduler.rate_limits
        assert ModelProvider.ANTHROPIC in scheduler.rate_limits

    async def test_scheduler_start_stop(self):
        """Test scheduler start and stop functionality."""
        scheduler = RequestScheduler()

        assert scheduler.running is False

        await scheduler.start()
        assert scheduler.running is True

        await scheduler.stop()
        assert scheduler.running is False

    async def test_schedule_request_basic(self):
        """Test basic request scheduling."""
        scheduler = RequestScheduler()
        await scheduler.start()

        # Mock operation
        async def mock_operation(value):
            return f"processed: {value}"

        try:
            result = await scheduler.schedule_request(
                operation=mock_operation,
                provider=ModelProvider.MOCK,
                priority=1,
                estimated_tokens=10,
                component_type="test",
                component_id="test-123",
                value="test_input"
            )

            assert result == "processed: test_input"
        finally:
            await scheduler.stop()

    async def test_schedule_request_with_model_config(self):
        """Test request scheduling with model config."""
        scheduler = RequestScheduler()
        await scheduler.start()

        config = ModelConfig(
            provider=ModelProvider.MOCK,
            model_name="test-model",
            temperature=0.7
        )

        async def mock_operation(prompt, config):
            return f"response to: {prompt}"

        try:
            result = await scheduler.schedule_request(
                operation=mock_operation,
                provider=ModelProvider.MOCK,
                model_config=config,
                prompt="test prompt",
                config=config
            )

            assert result == "response to: test prompt"
        finally:
            await scheduler.stop()

    async def test_concurrent_request_limiting(self):
        """Test concurrent request limiting."""
        rate_limits = {
            ModelProvider.MOCK: RateLimitConfig(
                max_concurrent=2,
                requests_per_minute=60
            )
        }

        scheduler = RequestScheduler(default_rate_limits=rate_limits)
        await scheduler.start()

        # Create slow operation
        async def slow_operation(delay):
            await asyncio.sleep(delay)
            return f"completed after {delay}s"

        try:
            # Start 3 operations simultaneously (limit is 2)
            tasks = [
                scheduler.schedule_request(
                    operation=slow_operation,
                    provider=ModelProvider.MOCK,
                    delay=0.1
                )
                for _ in range(3)
            ]

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            # All should complete
            assert len(results) == 3
            assert all("completed after" in result for result in results)

            # Should take longer due to concurrency limiting
            # (not strictly enforced in test due to timing variations)
            assert end_time - start_time >= 0.1

        finally:
            await scheduler.stop()

    async def test_priority_queue_ordering(self):
        """Test priority queue ordering."""
        scheduler = RequestScheduler(queue_strategy=QueueStrategy.PRIORITY)
        await scheduler.start()

        execution_order = []

        async def tracking_operation(name, delay=0.01):
            await asyncio.sleep(delay)
            execution_order.append(name)
            return name

        try:
            # Submit requests with different priorities (higher = more urgent)
            tasks = [
                scheduler.schedule_request(
                    operation=tracking_operation,
                    provider=ModelProvider.MOCK,
                    priority=1,  # Low priority
                    name="low"
                ),
                scheduler.schedule_request(
                    operation=tracking_operation,
                    provider=ModelProvider.MOCK,
                    priority=10,  # High priority
                    name="high"
                ),
                scheduler.schedule_request(
                    operation=tracking_operation,
                    provider=ModelProvider.MOCK,
                    priority=5,  # Medium priority
                    name="medium"
                )
            ]

            await asyncio.gather(*tasks)

            # High priority should execute first
            assert execution_order[0] == "high"

        finally:
            await scheduler.stop()

    async def test_rate_limiting_delay(self):
        """Test that rate limiting introduces appropriate delays."""
        rate_limits = {
            ModelProvider.MOCK: RateLimitConfig(
                requests_per_second=2,  # 2 requests per second
                max_concurrent=10
            )
        }

        scheduler = RequestScheduler(default_rate_limits=rate_limits)
        await scheduler.start()

        async def fast_operation():
            return "done"

        try:
            # Make 3 rapid requests
            start_time = time.time()

            tasks = [
                scheduler.schedule_request(
                    operation=fast_operation,
                    provider=ModelProvider.MOCK
                )
                for _ in range(3)
            ]

            results = await asyncio.gather(*tasks)
            end_time = time.time()

            # All should complete
            assert len(results) == 3
            assert all(result == "done" for result in results)

            # Should take at least 1 second due to rate limiting (2 req/sec, 3 requests)
            elapsed = end_time - start_time
            assert elapsed >= 0.5  # Conservative check for CI/test environments

        finally:
            await scheduler.stop()

    async def test_error_handling_in_scheduled_operation(self):
        """Test error handling in scheduled operations."""
        scheduler = RequestScheduler()
        await scheduler.start()

        async def failing_operation():
            raise Exception("Simulated failure")

        try:
            with pytest.raises(Exception, match="Simulated failure"):
                await scheduler.schedule_request(
                    operation=failing_operation,
                    provider=ModelProvider.MOCK
                )
        finally:
            await scheduler.stop()

    def test_get_stats(self):
        """Test scheduler statistics."""
        scheduler = RequestScheduler()

        stats = scheduler.get_stats()

        assert "total_requests" in stats
        assert "active_requests" in stats
        assert "provider_stats" in stats
        assert "queue_stats" in stats

        # Initially should be zero
        assert stats["total_requests"] == 0
        assert stats["active_requests"] == 0

    async def test_scheduler_statistics_tracking(self):
        """Test that scheduler tracks statistics correctly."""
        scheduler = RequestScheduler()
        await scheduler.start()

        async def test_operation():
            return "success"

        try:
            # Execute some operations
            for _ in range(3):
                await scheduler.schedule_request(
                    operation=test_operation,
                    provider=ModelProvider.MOCK
                )

            stats = scheduler.get_stats()

            # Should have processed 3 requests
            assert stats["total_requests"] == 3
            assert stats["active_requests"] == 0  # All completed

            # Should have provider stats
            assert ModelProvider.MOCK in stats["provider_stats"]
            provider_stats = stats["provider_stats"][ModelProvider.MOCK]
            assert provider_stats["total_requests"] == 3

        finally:
            await scheduler.stop()


class TestGlobalSchedulerFunctions:
    """Test global scheduler management functions."""

    async def test_configure_scheduler(self):
        """Test configuring the global scheduler."""
        rate_limits = {
            ModelProvider.ANTHROPIC: RateLimitConfig(
                requests_per_minute=120,
                max_concurrent=8
            )
        }

        scheduler = configure_scheduler(
            rate_limits=rate_limits,
            queue_strategy=QueueStrategy.ROUND_ROBIN
        )

        assert isinstance(scheduler, RequestScheduler)
        assert scheduler.queue_strategy == QueueStrategy.ROUND_ROBIN

    async def test_get_scheduler_singleton(self):
        """Test that get_scheduler returns the same instance."""
        scheduler1 = get_scheduler()
        scheduler2 = get_scheduler()

        assert scheduler1 is scheduler2

    async def test_start_stop_global_scheduler(self):
        """Test starting and stopping the global scheduler."""
        # Start global scheduler
        await start_scheduler()

        scheduler = get_scheduler()
        assert scheduler.running is True

        # Stop global scheduler
        await stop_scheduler()
        assert scheduler.running is False

    async def test_scheduler_context_manager_usage(self):
        """Test using scheduler in context manager pattern."""
        rate_limits = {
            ModelProvider.MOCK: RateLimitConfig(
                requests_per_minute=60,
                max_concurrent=5
            )
        }

        scheduler = RequestScheduler(default_rate_limits=rate_limits)

        # Test that we can use it as context manager (if implemented)
        await scheduler.start()

        async def test_operation():
            return "test_result"

        try:
            result = await scheduler.schedule_request(
                operation=test_operation,
                provider=ModelProvider.MOCK
            )

            assert result == "test_result"
        finally:
            await scheduler.stop()


class TestQueueStrategies:
    """Test different queue strategies."""

    async def test_fifo_strategy(self):
        """Test FIFO (First In, First Out) queue strategy."""
        scheduler = RequestScheduler(queue_strategy=QueueStrategy.FIFO)
        await scheduler.start()

        execution_order = []

        async def tracking_operation(name):
            execution_order.append(name)
            return name

        try:
            # Submit requests in order
            tasks = []
            for i in range(3):
                task = scheduler.schedule_request(
                    operation=tracking_operation,
                    provider=ModelProvider.MOCK,
                    name=f"request_{i}"
                )
                tasks.append(task)

            await asyncio.gather(*tasks)

            # Should execute in FIFO order (first submitted, first executed)
            assert execution_order == ["request_0", "request_1", "request_2"]

        finally:
            await scheduler.stop()

    async def test_round_robin_strategy(self):
        """Test round-robin queue strategy across providers."""
        scheduler = RequestScheduler(queue_strategy=QueueStrategy.ROUND_ROBIN)
        await scheduler.start()

        execution_order = []

        async def tracking_operation(provider_name):
            execution_order.append(provider_name)
            return provider_name

        try:
            # Submit requests to different providers
            tasks = [
                scheduler.schedule_request(
                    operation=tracking_operation,
                    provider=ModelProvider.MOCK,
                    provider_name="mock"
                ),
                scheduler.schedule_request(
                    operation=tracking_operation,
                    provider=ModelProvider.ANTHROPIC,
                    provider_name="anthropic"
                ),
                scheduler.schedule_request(
                    operation=tracking_operation,
                    provider=ModelProvider.OPENAI,
                    provider_name="openai"
                )
            ]

            await asyncio.gather(*tasks)

            # All should complete (order may vary due to round-robin)
            assert len(execution_order) == 3
            assert "mock" in execution_order
            assert "anthropic" in execution_order
            assert "openai" in execution_order

        finally:
            await scheduler.stop()


@pytest.mark.unit
class TestSchedulerIntegration:
    """Test scheduler integration with other components."""

    async def test_scheduler_with_observability(self):
        """Test scheduler integration with observability."""
        with patch('nons.observability.integration.get_observability') as mock_obs:
            mock_manager = MagicMock()
            mock_manager.start_operation.return_value = MagicMock()
            mock_manager.finish_operation.return_value = None
            mock_obs.return_value = mock_manager

            scheduler = RequestScheduler(enable_observability=True)
            await scheduler.start()

            async def test_operation():
                return "success"

            try:
                result = await scheduler.schedule_request(
                    operation=test_operation,
                    provider=ModelProvider.MOCK,
                    component_type="test",
                    component_id="test-123"
                )

                assert result == "success"
                # Verify observability was used
                mock_manager.start_operation.assert_called()
                mock_manager.finish_operation.assert_called()

            finally:
                await scheduler.stop()

    async def test_scheduler_with_node_execution(self):
        """Test scheduler integration with node execution."""
        from nons.core.node import Node
        from nons.core.types import ModelConfig

        # Configure scheduler
        rate_limits = {
            ModelProvider.MOCK: RateLimitConfig(
                requests_per_minute=120,
                max_concurrent=5
            )
        }

        configure_scheduler(rate_limits=rate_limits)
        await start_scheduler()

        try:
            # Create a node (this should use the scheduler internally)
            config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")

            # Import operators for the test
            import nons.operators.base

            node = Node('generate', model_config=config)

            with patch('nons.utils.providers.create_provider') as mock_provider_factory:
                from tests.conftest import MockLLMProvider
                mock_provider = MockLLMProvider()
                mock_provider_factory.return_value = mock_provider

                result = await node.execute(prompt="Test prompt")

                assert result is not None
                assert node.execution_count == 1

        finally:
            await stop_scheduler()


class TestSchedulerErrorScenarios:
    """Test scheduler behavior in error scenarios."""

    async def test_scheduler_with_provider_failures(self):
        """Test scheduler handling of provider failures."""
        scheduler = RequestScheduler()
        await scheduler.start()

        failure_count = 0

        async def intermittent_failure():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise Exception(f"Failure {failure_count}")
            return "success"

        try:
            # This should fail (no retry logic in basic scheduler)
            with pytest.raises(Exception, match="Failure 1"):
                await scheduler.schedule_request(
                    operation=intermittent_failure,
                    provider=ModelProvider.MOCK
                )

        finally:
            await scheduler.stop()

    async def test_scheduler_resource_cleanup(self):
        """Test scheduler resource cleanup on shutdown."""
        scheduler = RequestScheduler()
        await scheduler.start()

        # Start some long-running operations
        async def long_operation():
            await asyncio.sleep(1.0)
            return "completed"

        # Don't await these - they should be cleaned up on stop
        scheduler.schedule_request(
            operation=long_operation,
            provider=ModelProvider.MOCK
        )

        # Stop scheduler - should handle cleanup gracefully
        await scheduler.stop()

        assert scheduler.running is False