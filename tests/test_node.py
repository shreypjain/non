"""
Tests for Node class with mocked LLM calls.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import uuid
from nons.core.node import Node
from nons.core.types import (
    ModelConfig,
    ModelProvider,
    ExecutionMetrics,
    TokenUsage,
    CostInfo,
    OperatorError,
    ValidationError,
    ExecutionContext,
)
from tests.conftest import MockLLMProvider, assert_execution_metrics, assert_node_state


class TestNodeInitialization:
    """Test Node initialization and configuration."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    def test_node_basic_initialization(self):
        """Test basic node initialization."""
        config = ModelConfig(
            provider=ModelProvider.MOCK, model_name="test-model", temperature=0.7
        )

        node = Node("generate", model_config=config)

        assert node.operator_name == "generate"
        assert node.model_config == config
        assert node.node_id is not None
        assert isinstance(node.node_id, str)
        assert len(node.node_id) > 0
        assert node._execution_count == 0
        assert node.get_total_cost() == 0.0
        assert node.get_total_tokens() == 0
        assert_node_state(node)

    def test_node_initialization_with_custom_id(self):
        """Test node initialization with custom ID."""
        custom_id = "custom-node-123"
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")

        node = Node("generate", model_config=config, node_id=custom_id)

        assert node.node_id == custom_id

    def test_node_initialization_with_context(self):
        """Test node initialization with additional context."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        context = "You are a helpful assistant specialized in testing."

        node = Node("generate", model_config=config, additional_prompt_context=context)

        assert node.additional_prompt_context == context

    def test_node_initialization_with_invalid_operator(self):
        """Test node initialization with invalid operator."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")

        with pytest.raises(ValidationError, match="Failed to create node"):
            Node("nonexistent", model_config=config)

    def test_node_auto_generated_id_uniqueness(self):
        """Test that auto-generated IDs are unique."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")

        node1 = Node("generate", model_config=config)
        node2 = Node("generate", model_config=config)

        assert node1.node_id != node2.node_id


class TestNodeExecution:
    """Test Node execution with mocked LLM calls."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_basic_node_execution(self):
        """Test basic node execution."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        node = Node("generate", model_config=config)

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider({"default": "Mock generated response"})
            mock_provider_factory.return_value = mock_provider

            result = await node.execute(prompt="Test prompt")

            assert result == "Mock generated response"
            assert node._execution_count == 1
            assert node.get_total_cost() > 0
            assert node.get_total_tokens() > 0
            assert len(node._execution_metrics) == 1

            # Check execution metrics
            metrics = node._execution_metrics[0]
            assert_execution_metrics(metrics)

    async def test_node_execution_with_context(self):
        """Test node execution with execution context."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        node = Node("generate", model_config=config)

        execution_context = ExecutionContext(trace_id="test-trace", user_id="test-user")

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider()
            mock_provider_factory.return_value = mock_provider

            result = await node.execute(
                prompt="Test prompt", execution_context=execution_context
            )

            assert result is not None
            assert node._execution_count == 1

    async def test_node_execution_multiple_calls(self):
        """Test multiple executions on the same node."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        node = Node("generate", model_config=config)

        responses = {
            "First prompt": "First response",
            "Second prompt": "Second response",
            "default": "Default response",
        }

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider(responses)
            mock_provider_factory.return_value = mock_provider

            result1 = await node.execute(prompt="First prompt")
            result2 = await node.execute(prompt="Second prompt")

            assert result1 == "First response"
            assert result2 == "Second response"
            assert node._execution_count == 2
            assert len(node._execution_metrics) == 2

            # Check that costs and tokens accumulated
            assert node.get_total_cost() > 0
            assert node.get_total_tokens() > 0

    async def test_node_execution_with_additional_prompt_context(self):
        """Test node execution with additional prompt context."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        context = "You are a test assistant."
        node = Node("generate", model_config=config, additional_prompt_context=context)

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider()
            mock_provider_factory.return_value = mock_provider

            await node.execute(prompt="Test prompt")

            # Check that the provider was called
            assert mock_provider.call_count == 1
            # The actual prompt should include the context (implementation dependent)
            assert mock_provider.last_prompt is not None

    async def test_node_execution_with_scheduler(self):
        """Test node execution with request scheduler."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        node = Node("generate", model_config=config)

        with patch("nons.core.scheduler.get_scheduler") as mock_get_scheduler:
            mock_scheduler = AsyncMock()
            mock_scheduler.schedule_request = AsyncMock(
                return_value="Scheduled response"
            )
            mock_get_scheduler.return_value = mock_scheduler

            result = await node.execute(prompt="Test prompt")

            assert result == "Scheduled response"
            mock_scheduler.schedule_request.assert_called_once()

    async def test_node_execution_error_handling(self):
        """Test node execution error handling."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        node = Node("generate", model_config=config)

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = AsyncMock()
            mock_provider.generate_completion.side_effect = Exception("Provider error")
            mock_provider_factory.return_value = mock_provider

            with pytest.raises(Exception, match="Provider error"):
                await node.execute(prompt="Test prompt")

            # Check that node state is consistent even after error
            assert node._execution_count == 0  # No successful executions
            assert len(node._execution_metrics) == 0


class TestNodeConfiguration:
    """Test Node configuration methods."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    def test_configure_model(self):
        """Test model configuration update."""
        initial_config = ModelConfig(
            provider=ModelProvider.MOCK, model_name="initial-model", temperature=0.5
        )

        node = Node("generate", model_config=initial_config)

        # Update configuration
        node.configure_model(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-haiku-20240307",
            temperature=0.8,
            max_tokens=200,
        )

        assert node.model_config.provider == ModelProvider.ANTHROPIC
        assert node.model_config.model_name == "claude-3-haiku-20240307"
        assert node.model_config.temperature == 0.8
        assert node.model_config.max_tokens == 200

    def test_configure_model_partial_update(self):
        """Test partial model configuration update."""
        initial_config = ModelConfig(
            provider=ModelProvider.MOCK,
            model_name="initial-model",
            temperature=0.5,
            max_tokens=100,
        )

        node = Node("generate", model_config=initial_config)

        # Partial update
        node.configure_model(temperature=0.9)

        assert node.model_config.provider == ModelProvider.MOCK  # Unchanged
        assert node.model_config.model_name == "initial-model"  # Unchanged
        assert node.model_config.temperature == 0.9  # Updated
        assert node.model_config.max_tokens == 100  # Unchanged


class TestNodeCloning:
    """Test Node cloning functionality."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    def test_clone_basic(self):
        """Test basic node cloning."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        original = Node(
            "generate",
            model_config=config,
            additional_prompt_context="Original context",
        )

        clone = original.clone()

        # Check that clone has different ID but same configuration
        assert clone.node_id != original.node_id
        assert clone.operator_name == original.operator_name
        assert clone.model_config == original.model_config
        assert clone.additional_prompt_context == original.additional_prompt_context

        # Check that execution state is reset
        assert clone._execution_count == 0
        assert clone.get_total_cost() == 0.0
        assert clone.get_total_tokens() == 0
        assert len(clone._execution_metrics) == 0

    def test_clone_with_custom_id(self):
        """Test node cloning with custom ID."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        original = Node("generate", model_config=config)

        custom_id = "custom-clone-id"
        clone = original.clone(new_node_id=custom_id)

        assert clone.node_id == custom_id

    async def test_clone_independence(self):
        """Test that cloned nodes are independent."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        original = Node("generate", model_config=config)

        clone = original.clone()

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider()
            mock_provider_factory.return_value = mock_provider

            # Execute only the original
            await original.execute(prompt="Test")

            # Check that clone is unaffected
            assert original._execution_count == 1
            assert clone._execution_count == 0
            assert original.get_total_cost() > 0
            assert clone.get_total_cost() == 0


class TestNodeMultiplication:
    """Test Node multiplication operator."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    def test_multiplication_operator(self):
        """Test node multiplication operator."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        node = Node("generate", model_config=config)

        # Test right multiplication: node * 3
        clones = node * 3

        assert len(clones) == 3
        assert all(isinstance(clone, Node) for clone in clones)
        assert all(clone.operator_name == node.operator_name for clone in clones)
        assert len(set(clone.node_id for clone in clones)) == 3  # All unique IDs

    def test_reverse_multiplication_operator(self):
        """Test node reverse multiplication operator."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        node = Node("generate", model_config=config)

        # Test left multiplication: 5 * node
        clones = 5 * node

        assert len(clones) == 5
        assert all(isinstance(clone, Node) for clone in clones)
        assert all(clone.operator_name == node.operator_name for clone in clones)
        assert len(set(clone.node_id for clone in clones)) == 5  # All unique IDs

    def test_multiplication_invalid_count(self):
        """Test multiplication with invalid count."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        node = Node("generate", model_config=config)

        with pytest.raises(
            ValueError, match="Multiplication count must be a positive integer"
        ):
            _ = node * 0

        with pytest.raises(
            ValueError, match="Multiplication count must be a positive integer"
        ):
            _ = node * -1

        with pytest.raises(TypeError):
            _ = node * "three"

    def test_multiplication_clone_independence(self):
        """Test that multiplied nodes are independent clones."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        original = Node("generate", model_config=config)

        clones = original * 3

        # Modify original
        original.configure_model(temperature=0.9)

        # Check that clones are unaffected
        for clone in clones:
            assert clone.model_config.temperature != 0.9


class TestNodeStringRepresentation:
    """Test Node string representation methods."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    def test_node_str_representation(self):
        """Test node string representation."""
        config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-haiku-20240307",
            temperature=0.7,
        )
        node = Node("generate", model_config=config)

        str_repr = str(node)

        # Check that key information is included
        assert "generate" in str_repr
        assert "claude-3-haiku-20240307" in str_repr
        assert "ANTHROPIC" in str_repr
        assert node.node_id[:8] in str_repr

    def test_node_repr_representation(self):
        """Test node repr representation."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        node = Node("generate", model_config=config)

        repr_str = repr(node)

        assert "Node" in repr_str
        assert "generate" in repr_str


class TestNodeMetrics:
    """Test Node metrics and cost tracking."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_cost_tracking(self):
        """Test that costs are tracked correctly."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        node = Node("generate", model_config=config)

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            # Create mock provider with specific cost
            mock_provider = MockLLMProvider()

            # Override to return specific metrics
            async def mock_completion(prompt, config):
                metrics = ExecutionMetrics(
                    execution_time=1.0,
                    token_usage=TokenUsage(
                        prompt_tokens=10, completion_tokens=20, total_tokens=30
                    ),
                    cost_info=CostInfo(
                        cost_usd=0.005,
                        provider=config.provider,
                        model_name=config.model_name,
                    ),
                    provider=config.provider,
                    model_name=config.model_name,
                    success=True,
                    timestamp=1234567890.0,
                )
                return "Response", metrics

            mock_provider.generate_completion = mock_completion
            mock_provider_factory.return_value = mock_provider

            # Execute twice
            await node.execute(prompt="First")
            await node.execute(prompt="Second")

            assert node._execution_count == 2
            assert node.get_total_cost() == 0.01  # 2 x 0.005
            assert node.get_total_tokens() == 60  # 2 x 30

    async def test_token_tracking(self):
        """Test that tokens are tracked correctly."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        node = Node("generate", model_config=config)

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider()

            async def mock_completion(prompt, config):
                # Variable token usage based on prompt length
                prompt_tokens = len(prompt.split())
                completion_tokens = 15
                total_tokens = prompt_tokens + completion_tokens

                metrics = ExecutionMetrics(
                    execution_time=1.0,
                    token_usage=TokenUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
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
                return "Response", metrics

            mock_provider.generate_completion = mock_completion
            mock_provider_factory.return_value = mock_provider

            # Execute with different prompt lengths
            await node.execute(prompt="Short")  # 1 + 15 = 16 tokens
            await node.execute(prompt="This is a longer prompt")  # 5 + 15 = 20 tokens

            assert node._execution_count == 2
            assert node.get_total_tokens() == 36  # 16 + 20


@pytest.mark.unit
class TestNodeIntegration:
    """Test Node integration with other components."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_node_with_observability(self):
        """Test node execution with observability."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        node = Node("generate", model_config=config)

        with patch("nons.observability.integration.get_observability") as mock_obs:
            mock_manager = MagicMock()
            mock_manager.start_operation.return_value = MagicMock()
            mock_manager.finish_operation.return_value = None
            mock_obs.return_value = mock_manager

            with patch("nons.utils.providers.create_provider") as mock_provider_factory:
                mock_provider = MockLLMProvider()
                mock_provider_factory.return_value = mock_provider

                result = await node.execute(prompt="Test")

                # Verify observability was used
                mock_manager.start_operation.assert_called()
                mock_manager.finish_operation.assert_called()

    async def test_node_execution_with_different_operators(self):
        """Test node execution with different operators."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")

        # Test different operators
        operators_to_test = ["generate", "transform", "classify"]

        for op_name in operators_to_test:
            node = Node(op_name, model_config=config)

            with patch("nons.utils.providers.create_provider") as mock_provider_factory:
                mock_provider = MockLLMProvider()
                mock_provider_factory.return_value = mock_provider

                # Execute with appropriate parameters for each operator
                if op_name == "generate":
                    result = await node.execute(prompt="Test prompt")
                elif op_name == "transform":
                    result = await node.execute(
                        text="Test text", target_format="summary"
                    )
                elif op_name == "classify":
                    result = await node.execute(
                        text="Test text", categories=["positive", "negative"]
                    )

                assert result is not None
                assert node._execution_count == 1
