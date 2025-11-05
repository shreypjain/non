"""
Tests for Network (NoN) class sequential execution.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from nons.core.network import NoN
from nons.core.layer import Layer
from nons.core.node import Node
from nons.core.types import (
    ModelConfig,
    ModelProvider,
    ExecutionContext,
    NetworkConfig,
    LayerConfig,
    ErrorPolicy,
    NetworkResult,
    OperatorError,
)
from tests.conftest import MockLLMProvider


class TestNetworkInitialization:
    """Test Network initialization and configuration."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    def test_network_basic_initialization(self):
        """Test basic network initialization."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        layer1 = Layer([Node("generate", config)])
        layer2 = Layer([Node("transform", config)])

        network = NoN([layer1, layer2])

        assert len(network.layers) == 2
        assert network.network_id is not None
        assert isinstance(network.network_id, str)
        assert network.network_config is not None

    def test_network_initialization_with_custom_id(self):
        """Test network initialization with custom ID."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        layer = Layer([Node("generate", config)])
        custom_id = "custom-network-123"

        network = NoN([layer], network_id=custom_id)

        assert network.network_id == custom_id

    def test_network_initialization_with_config(self):
        """Test network initialization with custom configuration."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        layer = Layer([Node("generate", config)])

        network_config = NetworkConfig(
            max_concurrent_layers=2,
            layer_timeout_seconds=120.0,
            enable_layer_caching=True,
        )

        network = NoN([layer], network_config=network_config)

        assert network.network_config.max_concurrent_layers == 2
        assert network.network_config.layer_timeout_seconds == 120.0
        assert network.network_config.enable_layer_caching is True

    def test_network_initialization_empty_layers(self):
        """Test network initialization with empty layers list."""
        with pytest.raises(ValueError, match="Network must contain at least one layer"):
            NoN([])

    def test_network_initialization_invalid_layer_type(self):
        """Test network initialization with invalid layer types."""
        with pytest.raises(
            TypeError, match="All items in layers list must be Layer instances"
        ):
            NoN(["not", "layers"])


class TestNetworkFromOperators:
    """Test Network creation from operator specifications."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    def test_from_operators_simple_list(self):
        """Test creating network from simple operator list."""
        network = NoN.from_operators(["generate", "transform", "validate"])

        assert len(network.layers) == 3
        assert len(network.layers[0]) == 1  # Each layer has one node
        assert len(network.layers[1]) == 1
        assert len(network.layers[2]) == 1

        assert network.layers[0].nodes[0].operator_name == "generate"
        assert network.layers[1].nodes[0].operator_name == "transform"
        assert network.layers[2].nodes[0].operator_name == "validate"

    def test_from_operators_with_parallel_layers(self):
        """Test creating network with parallel layers."""
        network = NoN.from_operators(
            [
                "generate",  # Single node layer
                ["classify", "extract"],  # Parallel nodes layer
                "transform",  # Single node layer
            ]
        )

        assert len(network.layers) == 3
        assert len(network.layers[0]) == 1  # Single node
        assert len(network.layers[1]) == 2  # Parallel nodes
        assert len(network.layers[2]) == 1  # Single node

        # Check parallel layer operators
        parallel_operators = [node.operator_name for node in network.layers[1].nodes]
        assert "classify" in parallel_operators
        assert "extract" in parallel_operators

    def test_from_operators_with_node_objects(self):
        """Test creating network with Node objects."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        custom_node = Node(
            "generate", config, additional_prompt_context="Custom context"
        )

        network = NoN.from_operators(
            [
                "transform",
                [custom_node, "classify"],  # Mix of Node and string
                "validate",
            ]
        )

        assert len(network.layers) == 3
        assert len(network.layers[1]) == 2

        # Check that custom node was used
        nodes_in_parallel = network.layers[1].nodes
        custom_nodes = [
            n
            for n in nodes_in_parallel
            if n.additional_prompt_context == "Custom context"
        ]
        assert len(custom_nodes) == 1

    def test_from_operators_with_multiplied_nodes(self):
        """Test creating network with multiplied nodes."""
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        base_node = Node("generate", config)
        multiplied_nodes = base_node * 3

        network = NoN.from_operators(
            ["transform", multiplied_nodes, "validate"]  # 3 parallel nodes
        )

        assert len(network.layers) == 3
        assert len(network.layers[1]) == 3  # 3 multiplied nodes

    def test_from_operators_with_configs(self):
        """Test creating network with custom configurations."""
        network_config = NetworkConfig(max_concurrent_layers=2)
        layer_config = LayerConfig(error_policy=ErrorPolicy.SKIP_AND_CONTINUE)
        node_config = ModelConfig(
            provider=ModelProvider.ANTHROPIC, model_name="claude-3-haiku-20240307"
        )

        network = NoN.from_operators(
            ["generate", "transform"],
            network_config=network_config,
            layer_config=layer_config,
            model_config=node_config,
        )

        assert network.network_config.max_concurrent_layers == 2

        # Check that all nodes have the specified config
        for layer in network.layers:
            for node in layer.nodes:
                assert node.model_config.provider == ModelProvider.ANTHROPIC

    def test_from_operators_empty_list(self):
        """Test creating network from empty operator list."""
        with pytest.raises(ValueError, match="operator_specs cannot be empty"):
            NoN.from_operators([])

    def test_from_operators_invalid_spec(self):
        """Test creating network with invalid operator specification."""
        with pytest.raises(ValueError, match="Invalid operator specification"):
            NoN.from_operators([123])  # Invalid type


class TestNetworkExecution:
    """Test Network sequential execution."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_basic_sequential_execution(self):
        """Test basic sequential execution."""
        network = NoN.from_operators(["generate", "transform", "validate"])

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            # Create different responses for each layer
            call_count = 0

            async def sequential_completion(prompt, config):
                nonlocal call_count
                call_count += 1
                response = f"Response from step {call_count}"

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
            mock_provider.generate_completion = sequential_completion
            mock_provider_factory.return_value = mock_provider

            result = await network.forward("Initial input")

            # Final result should be from last layer
            assert result == "Response from step 3"
            assert call_count == 3  # One call per layer

    async def test_sequential_execution_with_parallel_layers(self):
        """Test sequential execution with parallel layers."""
        network = NoN.from_operators(
            [
                "generate",  # Layer 1: Single
                ["classify", "extract"],  # Layer 2: Parallel
                "synthesize",  # Layer 3: Single
            ]
        )

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider({"default": "Mock response"})
            mock_provider_factory.return_value = mock_provider

            result = await network.forward("Initial input")

            assert result == "Mock response"
            # Should have called provider 4 times (1 + 2 + 1)
            assert mock_provider.call_count == 4

    async def test_execution_with_execution_context(self):
        """Test network execution with execution context."""
        network = NoN.from_operators(["generate", "transform"])

        execution_context = ExecutionContext(trace_id="test-trace", user_id="test-user")

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider()
            mock_provider_factory.return_value = mock_provider

            result = await network.forward(
                "Initial input", execution_context=execution_context
            )

            assert result is not None

    async def test_execution_with_data_flow(self):
        """Test that data flows correctly between layers."""
        network = NoN.from_operators(["generate", "transform"])

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            received_prompts = []

            async def tracking_completion(prompt, config):
                received_prompts.append(prompt)
                response = f"Processed: {prompt}"

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
            mock_provider.generate_completion = tracking_completion
            mock_provider_factory.return_value = mock_provider

            result = await network.forward("Initial input")

            assert len(received_prompts) == 2
            # Second layer should receive output from first layer
            assert "Processed: Initial input" in received_prompts[1]

    async def test_execution_with_list_outputs_from_parallel_layer(self):
        """Test handling of list outputs from parallel layers."""
        network = NoN.from_operators(
            [
                ["generate", "generate"],  # Parallel layer producing list
                "synthesize",  # Single layer consuming list
            ]
        )

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            call_count = 0

            async def list_producing_completion(prompt, config):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    response = f"Parallel output {call_count}"
                else:
                    # Synthesize layer should receive list as input
                    response = "Synthesized result"

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
            mock_provider.generate_completion = list_producing_completion
            mock_provider_factory.return_value = mock_provider

            result = await network.forward("Initial input")

            assert result == "Synthesized result"
            assert call_count == 3


class TestNetworkErrorHandling:
    """Test Network error handling."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_network_layer_failure(self):
        """Test network behavior when a layer fails."""
        # Create layer with FAIL_FAST policy
        layer_config = LayerConfig(error_policy=ErrorPolicy.FAIL_FAST)
        network = NoN.from_operators(
            ["generate", "transform"], layer_config=layer_config
        )

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            call_count = 0

            async def failing_completion(prompt, config):
                nonlocal call_count
                call_count += 1
                if call_count == 2:  # Second call (second layer) fails
                    raise Exception("Layer 2 failed")

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

            with pytest.raises(Exception, match="Layer 2 failed"):
                await network.forward("Initial input")

    async def test_network_with_resilient_layers(self):
        """Test network with resilient layer configurations."""
        # Create layer with SKIP_AND_CONTINUE policy
        layer_config = LayerConfig(error_policy=ErrorPolicy.SKIP_AND_CONTINUE)
        network = NoN.from_operators(
            [
                ["generate", "generate"],
                "transform",
            ],  # Parallel first layer, single second
            layer_config=layer_config,
        )

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            call_count = 0

            async def partially_failing_completion(prompt, config):
                nonlocal call_count
                call_count += 1
                if call_count == 2:  # Second parallel node fails
                    raise Exception("Node 2 failed")

                response = f"Success {call_count}"
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
            mock_provider.generate_completion = partially_failing_completion
            mock_provider_factory.return_value = mock_provider

            result = await network.forward("Initial input")

            # Should complete despite partial failure in first layer
            assert result is not None


class TestNetworkStringRepresentation:
    """Test Network string representation methods."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    def test_network_str_representation(self):
        """Test network string representation."""
        network = NoN.from_operators(
            ["generate", ["classify", "extract"], "synthesize"]
        )

        str_repr = str(network)

        # Check that key information is included
        assert "Network" in str_repr
        assert "3 layers" in str_repr
        assert network.network_id[:8] in str_repr

    def test_network_repr_representation(self):
        """Test network repr representation."""
        network = NoN.from_operators(["generate"])

        repr_str = repr(network)

        assert "NoN" in repr_str

    def test_network_len(self):
        """Test network length method."""
        network = NoN.from_operators(["generate", "transform", "validate"])

        assert len(network) == 3


class TestNetworkConfiguration:
    """Test Network configuration methods."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    def test_network_layer_access(self):
        """Test accessing network layers."""
        network = NoN.from_operators(["generate", "transform", "validate"])

        # Test indexing
        first_layer = network.layers[0]
        assert len(first_layer) == 1
        assert first_layer.nodes[0].operator_name == "generate"

        # Test iteration
        layer_count = 0
        for layer in network.layers:
            layer_count += 1
            assert isinstance(layer, Layer)

        assert layer_count == 3

    def test_network_total_nodes(self):
        """Test counting total nodes in network."""
        network = NoN.from_operators(
            [
                "generate",  # 1 node
                ["classify", "extract"],  # 2 nodes
                "synthesize",  # 1 node
            ]
        )

        total_nodes = sum(len(layer) for layer in network.layers)
        assert total_nodes == 4


class TestNetworkObservability:
    """Test Network observability integration."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_network_with_observability(self):
        """Test network execution with observability."""
        network = NoN.from_operators(["generate", "transform"])

        with patch("nons.observability.integration.get_observability") as mock_obs:
            mock_manager = MagicMock()
            mock_manager.start_operation.return_value = MagicMock()
            mock_manager.finish_operation.return_value = None
            mock_obs.return_value = mock_manager

            with patch("nons.utils.providers.create_provider") as mock_provider_factory:
                mock_provider = MockLLMProvider()
                mock_provider_factory.return_value = mock_provider

                result = await network.forward("Test input")

                # Verify observability was used
                assert mock_manager.start_operation.call_count > 0
                assert mock_manager.finish_operation.call_count > 0

    async def test_network_result_tracking(self):
        """Test network result tracking."""
        network = NoN.from_operators(["generate"])

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider()
            mock_provider_factory.return_value = mock_provider

            result = await network.forward("Test input")

            # Check that last result is tracked (if implemented)
            assert hasattr(network, "_last_result") or result is not None


class TestNetworkPerformance:
    """Test Network performance characteristics."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_concurrent_layer_limitation(self):
        """Test that concurrent layer limitation is respected."""
        network_config = NetworkConfig(max_concurrent_layers=1)
        network = NoN.from_operators(
            ["generate", "transform"], network_config=network_config
        )

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider()
            mock_provider_factory.return_value = mock_provider

            result = await network.forward("Test input")

            # Should complete successfully with sequential execution
            assert result is not None

    async def test_network_timeout(self):
        """Test network-level timeout behavior."""
        network_config = NetworkConfig(layer_timeout_seconds=0.1)
        network = NoN.from_operators(["generate"], network_config=network_config)

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            # Create a slow provider
            async def slow_completion(prompt, config):
                await asyncio.sleep(0.2)  # Longer than timeout
                return "Response", MagicMock()

            mock_provider = MagicMock()
            mock_provider.generate_completion = slow_completion
            mock_provider_factory.return_value = mock_provider

            # Should timeout at layer level, not network level
            with pytest.raises((asyncio.TimeoutError, Exception)):
                await network.forward("Test input")


@pytest.mark.unit
class TestNetworkIntegration:
    """Test Network integration with other components."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_network_with_custom_nodes(self):
        """Test network with custom configured nodes."""
        # Create nodes with different configurations
        fast_config = ModelConfig(
            provider=ModelProvider.MOCK, model_name="fast-model", temperature=0.1
        )
        slow_config = ModelConfig(
            provider=ModelProvider.MOCK, model_name="slow-model", temperature=0.9
        )

        fast_node = Node("generate", fast_config)
        slow_node = Node("generate", slow_config)

        network = NoN.from_operators([fast_node, slow_node])

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider()
            mock_provider_factory.return_value = mock_provider

            result = await network.forward("Test input")

            assert result is not None
            # Should have called provider twice
            assert mock_provider.call_count == 2

    async def test_network_with_scheduler_integration(self):
        """Test network execution with request scheduler integration."""
        network = NoN.from_operators(["generate", "transform"])

        with patch("nons.core.scheduler.get_scheduler") as mock_get_scheduler:
            mock_scheduler = AsyncMock()

            async def scheduled_operation(operation, *args, **kwargs):
                return await operation(*args, **kwargs)

            mock_scheduler.schedule_request = scheduled_operation
            mock_get_scheduler.return_value = mock_scheduler

            with patch("nons.utils.providers.create_provider") as mock_provider_factory:
                mock_provider = MockLLMProvider()
                mock_provider_factory.return_value = mock_provider

                result = await network.forward("Test input")

                assert result is not None

    async def test_complex_network_execution(self):
        """Test execution of a complex network structure."""
        # Create a complex network with multiple patterns
        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        base_node = Node("generate", config)
        multiplied_nodes = base_node * 2

        network = NoN.from_operators(
            [
                "transform",  # Preprocessing
                multiplied_nodes,  # Parallel processing
                ["classify", "extract"],  # Parallel analysis
                "synthesize",  # Final synthesis
            ]
        )

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            mock_provider = MockLLMProvider()
            mock_provider_factory.return_value = mock_provider

            result = await network.forward("Complex input data")

            assert result is not None
            # Should have called provider 6 times (1 + 2 + 2 + 1)
            assert mock_provider.call_count == 6

    async def test_network_error_recovery(self):
        """Test network error recovery across layers."""
        # Create network with different error policies per layer
        resilient_config = LayerConfig(error_policy=ErrorPolicy.SKIP_AND_CONTINUE)
        strict_config = LayerConfig(error_policy=ErrorPolicy.FAIL_FAST)

        config = ModelConfig(provider=ModelProvider.MOCK, model_name="test")
        layer1 = Layer(
            [Node("generate", config), Node("generate", config)], resilient_config
        )
        layer2 = Layer([Node("validate", config)], strict_config)

        network = NoN([layer1, layer2])

        with patch("nons.utils.providers.create_provider") as mock_provider_factory:
            call_count = 0

            async def mixed_completion(prompt, config):
                nonlocal call_count
                call_count += 1
                if call_count == 2:  # Second node in first layer fails
                    raise Exception("Node failure")

                response = f"Success {call_count}"
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
            mock_provider.generate_completion = mixed_completion
            mock_provider_factory.return_value = mock_provider

            result = await network.forward("Test input")

            # Should complete despite failure in first layer
            assert result is not None
            assert call_count == 3  # 2 attempts in layer 1, 1 success in layer 2
