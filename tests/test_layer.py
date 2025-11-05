"""
Tests for Layer class parallel execution.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from nons.core.layer import Layer, create_parallel_layer
from nons.core.node import Node
from nons.core.types import (
    ModelConfig,
    ModelProvider,
    ExecutionContext,
    LayerConfig,
    ErrorPolicy,
    LayerResult,
    OperatorError,
)
from tests.conftest import MockLLMProvider, assert_execution_metrics


class TestLayerInitialization:
    """Test Layer initialization and configuration."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    def test_layer_basic_initialization(self):
        """Test basic layer initialization."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [
            Node("generate", config),
            Node("generate", config),
            Node("generate", config),
        ]

        layer = Layer(nodes)

        assert len(layer.nodes) == 3
        assert layer.layer_id is not None
        assert isinstance(layer.layer_id, str)
        assert layer.layer_config is not None
        assert len(layer) == 3

    def test_layer_initialization_with_custom_id(self):
        """Test layer initialization with custom ID."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [Node("generate", config)]
        custom_id = "custom-layer-123"

        layer = Layer(nodes, layer_id=custom_id)

        assert layer.layer_id == custom_id

    def test_layer_initialization_with_config(self):
        """Test layer initialization with custom configuration."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [Node("generate", config)]

        layer_config = LayerConfig(
            error_policy=ErrorPolicy.RETRY_WITH_BACKOFF,
            max_retries=5,
            timeout_seconds=30.0,
        )

        layer = Layer(nodes, layer_config=layer_config)

        assert layer.layer_config.error_policy == ErrorPolicy.RETRY_WITH_BACKOFF
        assert layer.layer_config.max_retries == 5
        assert layer.layer_config.timeout_seconds == 30.0

    def test_layer_initialization_empty_nodes(self):
        """Test layer initialization with empty nodes list."""
        with pytest.raises(ValueError, match="Layer must contain at least one node"):
            Layer([])

    def test_layer_initialization_invalid_node_type(self):
        """Test layer initialization with invalid node types."""
        with pytest.raises(
            TypeError, match="All items in nodes list must be Node instances"
        ):
            Layer(["not", "nodes"])


class TestLayerExecution:
    """Test Layer parallel execution."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_basic_parallel_execution(self):
        """Test basic parallel execution."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [
            Node("generate", config),
            Node("generate", config),
            Node("generate", config),
        ]
        layer = Layer(nodes)

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider({"default": "Mock response"})
            mock_provider_factory.return_value = mock_provider

            result = await layer.execute_parallel("Test input")

            assert isinstance(result, LayerResult)
            assert len(result.outputs) == 3
            assert all(output == "Mock response" for output in result.outputs)
            assert result.success_rate == 1.0
            assert result.execution_time > 0

    async def test_parallel_execution_with_different_responses(self):
        """Test parallel execution with different responses per node."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [Node("generate", config), Node("generate", config)]
        layer = Layer(nodes)

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            call_count = 0

            async def varying_completion(prompt, config):
                nonlocal call_count
                call_count += 1
                response = f"Response {call_count}"

                from nons.core.types import ExecutionMetrics, TokenUsage, CostInfo

                metrics = ExecutionMetrics(
                    execution_time=0.5,
                    token_usage=TokenUsage(
                        prompt_tokens=10, completion_tokens=15, total_tokens=25
                    ),
                    cost_info=CostInfo(
                        cost_usd=0.001,
                        provider=config.provider,
                        model_name=config.model_name,
                    ),
                    provider=config.provider,
                    model_name=config.model_name,
                    success=True,
                    timestamp=1234567890.0,
                )
                return response, metrics

            mock_provider = MagicMock()
            mock_provider.generate_completion = varying_completion
            mock_provider_factory.return_value = mock_provider

            result = await layer.execute_parallel("Test input")

            assert len(result.outputs) == 2
            assert "Response 1" in result.outputs
            assert "Response 2" in result.outputs
            assert result.success_rate == 1.0

    async def test_parallel_execution_with_list_inputs(self):
        """Test parallel execution with list of inputs."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [
            Node("generate", config),
            Node("generate", config),
            Node("generate", config),
        ]
        layer = Layer(nodes)

        inputs = ["Input 1", "Input 2", "Input 3"]

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider(
                {
                    "Input 1": "Response 1",
                    "Input 2": "Response 2",
                    "Input 3": "Response 3",
                    "default": "Default response",
                }
            )
            mock_provider_factory.return_value = mock_provider

            result = await layer.execute_parallel(inputs)

            assert len(result.outputs) == 3
            assert result.outputs == ["Response 1", "Response 2", "Response 3"]

    async def test_parallel_execution_with_mismatched_inputs(self):
        """Test parallel execution with mismatched input count."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [Node("generate", config), Node("generate", config)]
        layer = Layer(nodes)

        # More inputs than nodes - should use first input for all nodes
        inputs = ["Input 1", "Input 2", "Input 3"]

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider()
            mock_provider_factory.return_value = mock_provider

            result = await layer.execute_parallel(inputs)

            assert len(result.outputs) == 2  # Number of nodes

    async def test_parallel_execution_with_execution_context(self):
        """Test parallel execution with execution context."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [Node("generate", config), Node("generate", config)]
        layer = Layer(nodes)

        execution_context = ExecutionContext(trace_id="test-trace", user_id="test-user")

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider()
            mock_provider_factory.return_value = mock_provider

            result = await layer.execute_parallel(
                "Test input", execution_context=execution_context
            )

            assert isinstance(result, LayerResult)
            assert len(result.outputs) == 2


class TestLayerErrorHandling:
    """Test Layer error handling policies."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_fail_fast_policy(self):
        """Test FAIL_FAST error policy."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [Node("generate", config), Node("generate", config)]

        layer_config = LayerConfig(error_policy=ErrorPolicy.FAIL_FAST)
        layer = Layer(nodes, layer_config=layer_config)

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            # First call succeeds, second fails
            call_count = 0

            async def failing_completion(prompt, config):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise Exception("Simulated failure")

                from nons.core.types import ExecutionMetrics, TokenUsage, CostInfo

                metrics = ExecutionMetrics(
                    execution_time=0.5,
                    token_usage=TokenUsage(
                        prompt_tokens=10, completion_tokens=15, total_tokens=25
                    ),
                    cost_info=CostInfo(
                        cost_usd=0.001,
                        provider=config.provider,
                        model_name=config.model_name,
                    ),
                    provider=config.provider,
                    model_name=config.model_name,
                    success=True,
                    timestamp=1234567890.0,
                )
                return "Success", metrics

            mock_provider = MagicMock()
            mock_provider.generate_completion = failing_completion
            mock_provider_factory.return_value = mock_provider

            with pytest.raises(Exception, match="Simulated failure"):
                await layer.execute_parallel("Test input")

    async def test_skip_and_continue_policy(self):
        """Test SKIP_AND_CONTINUE error policy."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [
            Node("generate", config),
            Node("generate", config),
            Node("generate", config),
        ]

        layer_config = LayerConfig(error_policy=ErrorPolicy.SKIP_AND_CONTINUE)
        layer = Layer(nodes, layer_config=layer_config)

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            # Second call fails, others succeed
            call_count = 0

            async def partially_failing_completion(prompt, config):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise Exception("Simulated failure")

                from nons.core.types import ExecutionMetrics, TokenUsage, CostInfo

                metrics = ExecutionMetrics(
                    execution_time=0.5,
                    token_usage=TokenUsage(
                        prompt_tokens=10, completion_tokens=15, total_tokens=25
                    ),
                    cost_info=CostInfo(
                        cost_usd=0.001,
                        provider=config.provider,
                        model_name=config.model_name,
                    ),
                    provider=config.provider,
                    model_name=config.model_name,
                    success=True,
                    timestamp=1234567890.0,
                )
                return f"Success {call_count}", metrics

            mock_provider = MagicMock()
            mock_provider.generate_completion = partially_failing_completion
            mock_provider_factory.return_value = mock_provider

            result = await layer.execute_parallel("Test input")

            # Should have partial results
            assert len(result.outputs) == 2  # Only successful executions
            assert result.success_rate < 1.0  # Not all nodes succeeded
            assert "Success 1" in result.outputs
            assert "Success 3" in result.outputs

    async def test_retry_with_backoff_policy(self):
        """Test RETRY_WITH_BACKOFF error policy."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [Node("generate", config)]

        layer_config = LayerConfig(
            error_policy=ErrorPolicy.RETRY_WITH_BACKOFF,
            max_retries=2,
            retry_delay_seconds=0.1,  # Fast retry for testing
        )
        layer = Layer(nodes, layer_config=layer_config)

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            # Fail twice, then succeed
            attempt_count = 0

            async def retry_completion(prompt, config):
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count <= 2:
                    raise Exception(f"Attempt {attempt_count} failed")

                from nons.core.types import ExecutionMetrics, TokenUsage, CostInfo

                metrics = ExecutionMetrics(
                    execution_time=0.5,
                    token_usage=TokenUsage(
                        prompt_tokens=10, completion_tokens=15, total_tokens=25
                    ),
                    cost_info=CostInfo(
                        cost_usd=0.001,
                        provider=config.provider,
                        model_name=config.model_name,
                    ),
                    provider=config.provider,
                    model_name=config.model_name,
                    success=True,
                    timestamp=1234567890.0,
                )
                return "Success after retries", metrics

            mock_provider = MagicMock()
            mock_provider.generate_completion = retry_completion
            mock_provider_factory.return_value = mock_provider

            result = await layer.execute_parallel("Test input")

            assert len(result.outputs) == 1
            assert result.outputs[0] == "Success after retries"
            assert result.success_rate == 1.0
            assert attempt_count == 3  # 2 failures + 1 success

    async def test_return_partial_policy(self):
        """Test RETURN_PARTIAL error policy."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [Node("generate", config), Node("generate", config)]

        layer_config = LayerConfig(
            error_policy=ErrorPolicy.RETURN_PARTIAL,
            min_success_threshold=0.5,  # Require at least 50% success
        )
        layer = Layer(nodes, layer_config=layer_config)

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            # One succeeds, one fails
            call_count = 0

            async def mixed_completion(prompt, config):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise Exception("Second node failed")

                from nons.core.types import ExecutionMetrics, TokenUsage, CostInfo

                metrics = ExecutionMetrics(
                    execution_time=0.5,
                    token_usage=TokenUsage(
                        prompt_tokens=10, completion_tokens=15, total_tokens=25
                    ),
                    cost_info=CostInfo(
                        cost_usd=0.001,
                        provider=config.provider,
                        model_name=config.model_name,
                    ),
                    provider=config.provider,
                    model_name=config.model_name,
                    success=True,
                    timestamp=1234567890.0,
                )
                return "Success", metrics

            mock_provider = MagicMock()
            mock_provider.generate_completion = mixed_completion
            mock_provider_factory.return_value = mock_provider

            result = await layer.execute_parallel("Test input")

            assert len(result.outputs) == 1  # Only successful results
            assert result.outputs[0] == "Success"
            assert result.success_rate == 0.5  # 50% success rate


class TestLayerCreationHelpers:
    """Test layer creation helper functions."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    def test_create_parallel_layer(self):
        """Test create_parallel_layer helper function."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [
            Node("generate", config),
            Node("generate", config),
            Node("generate", config),
        ]

        layer = create_parallel_layer(nodes)

        assert isinstance(layer, Layer)
        assert len(layer.nodes) == 3
        assert layer.layer_id is not None

    def test_create_parallel_layer_empty_list(self):
        """Test create_parallel_layer with empty list."""
        with pytest.raises(
            ValueError, match="Cannot create layer from empty node list"
        ):
            create_parallel_layer([])

    def test_create_parallel_layer_invalid_types(self):
        """Test create_parallel_layer with invalid node types."""
        with pytest.raises(TypeError, match="All items must be Node instances"):
            create_parallel_layer(["not", "nodes"])

    def test_from_operators_class_method(self):
        """Test Layer.from_operators class method."""
        operator_names = ["generate", "generate", "transform"]

        layer = Layer.from_operators(operator_names)

        assert isinstance(layer, Layer)
        assert len(layer.nodes) == 3
        assert layer.nodes[0].operator_name == "generate"
        assert layer.nodes[1].operator_name == "generate"
        assert layer.nodes[2].operator_name == "transform"

    def test_from_operators_with_config(self):
        """Test Layer.from_operators with custom configuration."""
        operator_names = ["generate", "transform"]
        layer_config = LayerConfig(error_policy=ErrorPolicy.SKIP_AND_CONTINUE)
        node_config = ModelConfig(
            provider=ModelProvider.ANTHROPIC, model_name="claude-3-haiku-20240307"
        )

        layer = Layer.from_operators(
            operator_names, layer_config=layer_config, model_config=node_config
        )

        assert layer.layer_config.error_policy == ErrorPolicy.SKIP_AND_CONTINUE
        assert all(
            node.model_config.provider == ModelProvider.ANTHROPIC
            for node in layer.nodes
        )


class TestLayerStringRepresentation:
    """Test Layer string representation methods."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    def test_layer_str_representation(self):
        """Test layer string representation."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [
            Node("generate", config),
            Node("transform", config),
            Node("classify", config),
        ]
        layer = Layer(nodes)

        str_repr = str(layer)

        # Check that key information is included
        assert "Layer" in str_repr
        assert "generate" in str_repr
        assert "transform" in str_repr
        assert "classify" in str_repr
        assert str(len(nodes)) in str_repr

    def test_layer_repr_representation(self):
        """Test layer repr representation."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [Node("generate", config)]
        layer = Layer(nodes)

        repr_str = repr(layer)

        assert "Layer" in repr_str

    def test_layer_len(self):
        """Test layer length method."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [Node("generate", config) for _ in range(5)]
        layer = Layer(nodes)

        assert len(layer) == 5

    def test_layer_iter(self):
        """Test layer iteration."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [Node("generate", config) for _ in range(3)]
        layer = Layer(nodes)

        iterated_nodes = list(layer)

        assert len(iterated_nodes) == 3
        assert all(isinstance(node, Node) for node in iterated_nodes)


class TestLayerTimeout:
    """Test Layer timeout functionality."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_layer_timeout(self):
        """Test layer execution timeout."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [Node("generate", config)]

        layer_config = LayerConfig(timeout_seconds=0.1)  # Very short timeout
        layer = Layer(nodes, layer_config=layer_config)

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            # Create a slow provider
            async def slow_completion(prompt, config):
                await asyncio.sleep(0.2)  # Longer than timeout
                return "Response", MagicMock()

            mock_provider = MagicMock()
            mock_provider.generate_completion = slow_completion
            mock_provider_factory.return_value = mock_provider

            with pytest.raises(asyncio.TimeoutError):
                await layer.execute_parallel("Test input")


@pytest.mark.unit
class TestLayerIntegration:
    """Test Layer integration with other components."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_layer_with_multiplied_nodes(self):
        """Test layer execution with multiplied nodes."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        base_node = Node("generate", config)

        # Create layer from multiplied nodes
        multiplied_nodes = base_node * 4
        layer = create_parallel_layer(multiplied_nodes)

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider()
            mock_provider_factory.return_value = mock_provider

            result = await layer.execute_parallel("Test input")

            assert len(result.outputs) == 4
            assert all(output == "Mock LLM response" for output in result.outputs)

    async def test_layer_with_mixed_operators(self):
        """Test layer with different operator types."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        nodes = [
            Node("generate", config),
            Node("transform", config),
            Node("classify", config),
        ]
        layer = Layer(nodes)

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider({"default": "Mixed operator response"})
            mock_provider_factory.return_value = mock_provider

            result = await layer.execute_parallel("Test input")

            assert len(result.outputs) == 3
            assert all(output == "Mixed operator response" for output in result.outputs)
