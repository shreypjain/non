"""
Integration tests for full NoN workflows.
"""
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from nons.core.network import NoN
from nons.core.node import Node
from nons.core.layer import Layer
from nons.core.types import (
    ModelConfig, ModelProvider, ExecutionContext, LayerConfig, NetworkConfig,
    ErrorPolicy
)
from nons.core.scheduler import (
    configure_scheduler, start_scheduler, stop_scheduler,
    RateLimitConfig, QueueStrategy
)
from nons.observability.integration import get_observability, configure_observability
from tests.conftest import MockLLMProvider


class TestBasicWorkflows:
    """Test basic end-to-end workflows."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_simple_sequential_workflow(self):
        """Test a simple sequential workflow."""
        # Create a simple 3-step pipeline
        network = NoN.from_operators([
            'generate',    # Step 1: Generate content
            'transform',   # Step 2: Transform format
            'validate'     # Step 3: Validate result
        ])

        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            call_count = 0

            async def sequential_responses(prompt, config):
                nonlocal call_count
                call_count += 1

                responses = {
                    1: "Generated initial content",
                    2: "Transformed content format",
                    3: "Validated final result"
                }

                response = responses.get(call_count, "Default response")

                from nons.core.types import ExecutionMetrics, TokenUsage, CostInfo
                metrics = ExecutionMetrics(
                    execution_time=0.5,
                    token_usage=TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
                    cost_info=CostInfo(cost_usd=0.001, provider=config.provider, model_name=config.model_name),
                    provider=config.provider,
                    model_name=config.model_name,
                    success=True,
                    timestamp=1234567890.0
                )
                return response, metrics

            mock_provider = MagicMock()
            mock_provider.generate_completion = sequential_responses
            mock_provider_factory.return_value = mock_provider

            result = await network.forward("Initial input")

            assert result == "Validated final result"
            assert call_count == 3  # Should have called all three operators

    async def test_parallel_processing_workflow(self):
        """Test workflow with parallel processing."""
        # Create pipeline with parallel analysis layer
        network = NoN.from_operators([
            'transform',                           # Preprocessing
            ['classify', 'extract', 'condense'],   # Parallel analysis
            'synthesize'                           # Final synthesis
        ])

        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            responses = {
                "transform": "Preprocessed content",
                "classify": "Category: Technical",
                "extract": "Key info: API documentation",
                "condense": "Summary: Short version",
                "synthesize": "Final synthesized result"
            }

            async def contextual_responses(prompt, config):
                # Determine response based on the prompt context
                if "preprocessed" in prompt.lower() or "initial" in prompt.lower():
                    if "classify" in str(config.model_name):
                        return responses["classify"], self._create_metrics(config)
                    elif "extract" in str(config.model_name):
                        return responses["extract"], self._create_metrics(config)
                    elif "condense" in str(config.model_name):
                        return responses["condense"], self._create_metrics(config)
                    else:
                        return responses["synthesize"], self._create_metrics(config)
                else:
                    return responses["transform"], self._create_metrics(config)

            mock_provider = MagicMock()
            mock_provider.generate_completion = contextual_responses
            mock_provider_factory.return_value = mock_provider

            result = await network.forward("Initial technical document")

            assert result is not None

    def _create_metrics(self, config):
        """Helper to create execution metrics."""
        from nons.core.types import ExecutionMetrics, TokenUsage, CostInfo
        return ExecutionMetrics(
            execution_time=0.5,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
            cost_info=CostInfo(cost_usd=0.001, provider=config.provider, model_name=config.model_name),
            provider=config.provider,
            model_name=config.model_name,
            success=True,
            timestamp=1234567890.0
        )

    async def test_complex_multi_stage_workflow(self):
        """Test complex multi-stage workflow with various patterns."""
        # Create base nodes for multiplication
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        analysis_node = Node('generate', config)

        # Build complex network
        network = NoN.from_operators([
            'transform',                    # Input preprocessing
            analysis_node * 2,              # Parallel analysis (2 instances)
            ['classify', 'extract'],        # Parallel categorization
            'synthesize'                    # Final synthesis
        ])

        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = MockLLMProvider({
                "default": "Mock processing result"
            })
            mock_provider_factory.return_value = mock_provider

            result = await network.forward("Complex input data requiring multi-stage analysis")

            assert result == "Mock processing result"
            # Should have called provider 6 times (1 + 2 + 2 + 1)
            assert mock_provider.call_count == 6

    async def test_workflow_with_custom_configurations(self):
        """Test workflow with custom node and layer configurations."""
        # Create custom configurations
        fast_config = ModelConfig(
            provider=ModelProvider.MOCK,
            model_name="fast-model",
            temperature=0.1,
            max_tokens=50
        )

        slow_config = ModelConfig(
            provider=ModelProvider.MOCK,
            model_name="slow-model",
            temperature=0.9,
            max_tokens=200
        )

        resilient_layer_config = LayerConfig(
            error_policy=ErrorPolicy.SKIP_AND_CONTINUE,
            max_retries=2
        )

        # Create nodes with different configurations
        fast_node = Node('generate', fast_config)
        slow_node = Node('generate', slow_config)

        # Create layers with custom configurations
        preprocessing_layer = Layer([Node('transform', fast_config)])
        analysis_layer = Layer([fast_node, slow_node], resilient_layer_config)
        synthesis_layer = Layer([Node('synthesize', slow_config)])

        # Create network
        network = NoN([preprocessing_layer, analysis_layer, synthesis_layer])

        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = MockLLMProvider()
            mock_provider_factory.return_value = mock_provider

            result = await network.forward("Input requiring different processing speeds")

            assert result is not None


class TestErrorHandlingWorkflows:
    """Test workflows with various error scenarios."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_workflow_with_retry_recovery(self):
        """Test workflow that recovers from transient failures."""
        layer_config = LayerConfig(
            error_policy=ErrorPolicy.RETRY_WITH_BACKOFF,
            max_retries=2,
            retry_delay_seconds=0.1
        )

        network = NoN.from_operators(
            ['generate', 'validate'],
            layer_config=layer_config
        )

        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            attempt_count = 0

            async def failing_then_succeeding(prompt, config):
                nonlocal attempt_count
                attempt_count += 1

                # Fail on first attempt, succeed on retry
                if attempt_count == 1:
                    raise Exception("Transient failure")

                return f"Success on attempt {attempt_count}", self._create_metrics(config)

            mock_provider = MagicMock()
            mock_provider.generate_completion = failing_then_succeeding
            mock_provider_factory.return_value = mock_provider

            result = await network.forward("Test input")

            assert "Success on attempt 2" in result
            assert attempt_count == 4  # 2 attempts for each of 2 operators

    async def test_workflow_with_partial_failure_tolerance(self):
        """Test workflow that tolerates partial failures."""
        layer_config = LayerConfig(
            error_policy=ErrorPolicy.RETURN_PARTIAL,
            min_success_threshold=0.5  # Require at least 50% success
        )

        network = NoN.from_operators(
            [['generate', 'generate', 'generate'], 'synthesize'],
            layer_config=layer_config
        )

        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            call_count = 0

            async def partial_failure(prompt, config):
                nonlocal call_count
                call_count += 1

                # Fail every other call
                if call_count % 2 == 0:
                    raise Exception(f"Failure on call {call_count}")

                return f"Success on call {call_count}", self._create_metrics(config)

            mock_provider = MagicMock()
            mock_provider.generate_completion = partial_failure
            mock_provider_factory.return_value = mock_provider

            result = await network.forward("Test input")

            # Should complete despite partial failures
            assert result is not None

    def _create_metrics(self, config):
        """Helper to create execution metrics."""
        from nons.core.types import ExecutionMetrics, TokenUsage, CostInfo
        return ExecutionMetrics(
            execution_time=0.5,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
            cost_info=CostInfo(cost_usd=0.001, provider=config.provider, model_name=config.model_name),
            provider=config.provider,
            model_name=config.model_name,
            success=True,
            timestamp=1234567890.0
        )


class TestObservabilityIntegration:
    """Test integration with observability system."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_full_workflow_with_tracing(self):
        """Test full workflow with distributed tracing."""
        # Configure observability
        obs_manager = configure_observability(
            enable_tracing=True,
            enable_logging=True,
            enable_metrics=True
        )

        network = NoN.from_operators(['generate', 'validate'])

        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = MockLLMProvider()
            mock_provider_factory.return_value = mock_provider

            # Execute with execution context
            execution_context = ExecutionContext(
                trace_id="test-trace-123",
                user_id="test-user"
            )

            result = await network.forward(
                "Test input for tracing",
                execution_context=execution_context
            )

            assert result is not None

            # Verify observability data was collected
            all_data = obs_manager.export_all_data()

            assert len(all_data["spans"]) > 0
            assert len(all_data["logs"]) > 0
            assert len(all_data["metrics"]) > 0

            # Check trace correlation
            trace_spans = [span for span in all_data["spans"] if span["trace_id"] == "test-trace-123"]
            assert len(trace_spans) > 0

    async def test_cost_and_token_tracking_workflow(self):
        """Test end-to-end cost and token tracking."""
        network = NoN.from_operators(['generate', 'transform', 'validate'])

        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            call_count = 0

            async def cost_tracking_provider(prompt, config):
                nonlocal call_count
                call_count += 1

                # Different costs for different calls
                costs = [0.002, 0.003, 0.001]
                tokens = [30, 45, 25]

                from nons.core.types import ExecutionMetrics, TokenUsage, CostInfo
                metrics = ExecutionMetrics(
                    execution_time=0.5 + (call_count * 0.1),
                    token_usage=TokenUsage(
                        prompt_tokens=10 + call_count,
                        completion_tokens=tokens[call_count - 1],
                        total_tokens=10 + call_count + tokens[call_count - 1]
                    ),
                    cost_info=CostInfo(
                        cost_usd=costs[call_count - 1],
                        provider=config.provider,
                        model_name=config.model_name
                    ),
                    provider=config.provider,
                    model_name=config.model_name,
                    success=True,
                    timestamp=1234567890.0 + call_count
                )

                return f"Response {call_count}", metrics

            mock_provider = MagicMock()
            mock_provider.generate_completion = cost_tracking_provider
            mock_provider_factory.return_value = mock_provider

            result = await network.forward("Cost tracking test")

            assert result == "Response 3"

            # Verify cost tracking across all nodes
            total_cost = 0
            total_tokens = 0
            for layer in network.layers:
                for node in layer.nodes:
                    total_cost += node.total_cost
                    total_tokens += node.total_tokens

            assert total_cost == 0.006  # 0.002 + 0.003 + 0.001
            assert total_tokens == 100   # 41 + 50 + 38 (calculated from metrics)


class TestSchedulerIntegration:
    """Test integration with request scheduler."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_workflow_with_rate_limiting(self):
        """Test workflow with request scheduler and rate limiting."""
        # Configure scheduler with rate limits
        rate_limits = {
            ModelProvider.MOCK: RateLimitConfig(
                requests_per_second=2,  # Slow rate for testing
                max_concurrent=2
            )
        }

        configure_scheduler(
            rate_limits=rate_limits,
            queue_strategy=QueueStrategy.PRIORITY
        )

        await start_scheduler()

        try:
            network = NoN.from_operators(['generate', 'generate', 'generate'])

            with patch('nons.utils.providers.create_provider') as mock_provider_factory:
                mock_provider = MockLLMProvider()
                mock_provider_factory.return_value = mock_provider

                import time
                start_time = time.time()

                result = await network.forward("Rate limited test")

                end_time = time.time()
                elapsed = end_time - start_time

                assert result is not None
                # Should take some time due to rate limiting
                assert elapsed >= 0.5  # Conservative check

        finally:
            await stop_scheduler()

    async def test_workflow_with_priority_scheduling(self):
        """Test workflow with priority-based scheduling."""
        configure_scheduler(queue_strategy=QueueStrategy.PRIORITY)
        await start_scheduler()

        try:
            # Create nodes with different priorities via custom execution context
            network = NoN.from_operators(['generate', 'generate'])

            with patch('nons.utils.providers.create_provider') as mock_provider_factory:
                execution_order = []

                async def priority_tracking_provider(prompt, config):
                    execution_order.append(config.model_name)
                    return f"Response from {config.model_name}", self._create_metrics(config)

                mock_provider = MagicMock()
                mock_provider.generate_completion = priority_tracking_provider
                mock_provider_factory.return_value = mock_provider

                result = await network.forward("Priority test")

                assert result is not None
                assert len(execution_order) == 2

        finally:
            await stop_scheduler()

    def _create_metrics(self, config):
        """Helper to create execution metrics."""
        from nons.core.types import ExecutionMetrics, TokenUsage, CostInfo
        return ExecutionMetrics(
            execution_time=0.5,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
            cost_info=CostInfo(cost_usd=0.001, provider=config.provider, model_name=config.model_name),
            provider=config.provider,
            model_name=config.model_name,
            success=True,
            timestamp=1234567890.0
        )


class TestMultiProviderWorkflows:
    """Test workflows using multiple LLM providers."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_workflow_with_different_providers(self):
        """Test workflow using different providers for different tasks."""
        # Create nodes with different providers
        openai_config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.1
        )

        anthropic_config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-haiku-20240307",
            temperature=0.7
        )

        google_config = ModelConfig(
            provider=ModelProvider.GOOGLE,
            model_name="gemini-2.5-flash",
            temperature=0.5
        )

        # Create network with mixed providers
        network = NoN.from_operators([
            Node('generate', openai_config),
            Node('transform', anthropic_config),
            Node('validate', google_config)
        ])

        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            provider_calls = []

            def mock_provider_creation(provider, model_name, **kwargs):
                provider_calls.append((provider, model_name))
                return MockLLMProvider({
                    "default": f"Response from {provider}:{model_name}"
                })

            mock_provider_factory.side_effect = mock_provider_creation

            result = await network.forward("Multi-provider test")

            # Should have created providers for all three providers
            assert len(provider_calls) == 3
            providers_used = [call[0] for call in provider_calls]
            assert ModelProvider.OPENAI in providers_used
            assert ModelProvider.ANTHROPIC in providers_used
            assert ModelProvider.GOOGLE in providers_used

    async def test_workflow_with_provider_fallback(self):
        """Test workflow with provider fallback on failure."""
        # Create network with multiple nodes using same operation but different providers
        primary_config = ModelConfig(provider=ModelProvider.OPENAI, model_name="gpt-4")
        fallback_config = ModelConfig(provider=ModelProvider.ANTHROPIC, model_name="claude-3-haiku-20240307")

        # Parallel execution with potential fallback
        network = NoN.from_operators([
            [Node('generate', primary_config), Node('generate', fallback_config)],
            'validate'
        ])

        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            call_count = 0

            def provider_with_failure(provider, model_name, **kwargs):
                nonlocal call_count
                call_count += 1

                if provider == ModelProvider.OPENAI:
                    # OpenAI provider fails
                    failing_provider = MockLLMProvider()

                    async def failing_completion(prompt, config):
                        raise Exception("OpenAI provider unavailable")

                    failing_provider.generate_completion = failing_completion
                    return failing_provider
                else:
                    # Other providers work
                    return MockLLMProvider({
                        "default": f"Fallback response from {provider}"
                    })

            mock_provider_factory.side_effect = provider_with_failure

            # Configure layer to continue on partial failure
            for layer in network.layers:
                layer.layer_config.error_policy = ErrorPolicy.SKIP_AND_CONTINUE

            result = await network.forward("Fallback test")

            # Should complete using fallback provider
            assert "claude" in result.lower() or "anthropic" in result.lower()


@pytest.mark.integration
class TestRealWorldScenarios:
    """Test realistic use case scenarios."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_document_analysis_pipeline(self):
        """Test a realistic document analysis pipeline."""
        # Simulate a document analysis workflow
        network = NoN.from_operators([
            'transform',                              # Clean and preprocess
            ['classify', 'extract', 'condense'],      # Parallel analysis
            'synthesize'                              # Generate final report
        ])

        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            responses = {
                1: "Cleaned document content",
                2: "Classification: Technical Documentation",
                3: "Key entities: API, endpoints, authentication",
                4: "Summary: API documentation for REST service",
                5: "Final Report: Comprehensive API documentation analysis"
            }

            call_count = 0

            async def document_analysis_responses(prompt, config):
                nonlocal call_count
                call_count += 1
                response = responses.get(call_count, f"Response {call_count}")
                return response, self._create_metrics(config)

            mock_provider = MagicMock()
            mock_provider.generate_completion = document_analysis_responses
            mock_provider_factory.return_value = mock_provider

            document_content = """
            API Documentation

            This document describes the REST API endpoints for our service.
            Authentication is required via API keys.

            Endpoints:
            - GET /users
            - POST /users
            - PUT /users/{id}
            - DELETE /users/{id}
            """

            result = await network.forward(document_content)

            assert "Final Report" in result
            assert call_count == 5  # All processing steps completed

    async def test_content_generation_workflow(self):
        """Test a content generation and refinement workflow."""
        # Multi-stage content creation
        network = NoN.from_operators([
            'generate',      # Initial content generation
            'expand',        # Add detail and context
            'validate',      # Check quality and accuracy
            'transform'      # Final formatting
        ])

        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = MockLLMProvider({
                "default": "High-quality generated content"
            })
            mock_provider_factory.return_value = mock_provider

            content_request = "Write a technical blog post about microservices architecture"

            result = await network.forward(content_request)

            assert result == "High-quality generated content"
            assert mock_provider.call_count == 4

    async def test_data_processing_pipeline(self):
        """Test a data processing and analysis pipeline."""
        # Data analysis workflow
        base_analyzer = Node('generate', ModelConfig(provider=ModelProvider.MOCK, model_name="analyzer"))

        network = NoN.from_operators([
            'transform',              # Data preprocessing
            base_analyzer * 3,        # Parallel analysis (3 instances)
            'synthesize',             # Combine results
            'validate'                # Final validation
        ])

        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = MockLLMProvider()
            mock_provider_factory.return_value = mock_provider

            data_input = {
                "records": [
                    {"id": 1, "value": 100, "category": "A"},
                    {"id": 2, "value": 250, "category": "B"},
                    {"id": 3, "value": 175, "category": "A"}
                ]
            }

            result = await network.forward(str(data_input))

            assert result is not None
            # Should process through all stages: 1 + 3 + 1 + 1 = 6 calls
            assert mock_provider.call_count == 6

    def _create_metrics(self, config):
        """Helper to create execution metrics."""
        from nons.core.types import ExecutionMetrics, TokenUsage, CostInfo
        return ExecutionMetrics(
            execution_time=0.5,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
            cost_info=CostInfo(cost_usd=0.001, provider=config.provider, model_name=config.model_name),
            provider=config.provider,
            model_name=config.model_name,
            success=True,
            timestamp=1234567890.0
        )


class TestPerformanceCharacteristics:
    """Test performance characteristics of full workflows."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_high_concurrency_workflow(self):
        """Test workflow with high concurrency."""
        # Create network with many parallel operations
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="fast")
        base_node = Node('generate', config)

        network = NoN.from_operators([
            'transform',
            base_node * 10,  # 10 parallel operations
            'synthesize'
        ])

        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = MockLLMProvider()
            mock_provider_factory.return_value = mock_provider

            import time
            start_time = time.time()

            result = await network.forward("High concurrency test")

            end_time = time.time()
            elapsed = end_time - start_time

            assert result is not None
            assert mock_provider.call_count == 12  # 1 + 10 + 1
            # Should complete relatively quickly due to parallel execution
            assert elapsed < 5.0  # Should not take too long

    async def test_memory_efficient_workflow(self):
        """Test memory efficiency with large workflows."""
        # Create a large network to test memory usage
        network = NoN.from_operators([
            'transform',
            ['generate'] * 5,  # 5 parallel nodes
            ['classify'] * 3,   # 3 parallel nodes
            'synthesize'
        ])

        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = MockLLMProvider({
                "default": "Memory efficient response"
            })
            mock_provider_factory.return_value = mock_provider

            # Process multiple inputs sequentially
            for i in range(5):
                result = await network.forward(f"Input batch {i}")
                assert result == "Memory efficient response"

            # Should have processed all inputs
            assert mock_provider.call_count == 45  # 5 batches * 9 calls each