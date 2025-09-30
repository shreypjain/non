"""
Pytest configuration and shared fixtures.
"""
import asyncio
import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, Optional

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import NoN components
from nons.core.types import ModelConfig, ModelProvider, ExecutionMetrics, TokenUsage, CostInfo
from nons.core.node import Node
from nons.core.layer import Layer
from nons.core.network import NoN
from nons.operators.registry import OperatorRegistry
from nons.utils.providers import create_provider


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_model_config():
    """Create a mock model configuration."""
    return ModelConfig(
        provider=ModelProvider.MOCK,
        model_name="test-model",
        temperature=0.7,
        max_tokens=100
    )


@pytest.fixture
def mock_execution_metrics():
    """Create mock execution metrics."""
    return ExecutionMetrics(
        execution_time=1.23,
        token_usage=TokenUsage(
            prompt_tokens=50,
            completion_tokens=30,
            total_tokens=80
        ),
        cost_info=CostInfo(
            cost_usd=0.001234,
            provider=ModelProvider.MOCK,
            model_name="test-model"
        ),
        provider=ModelProvider.MOCK,
        model_name="test-model",
        success=True,
        timestamp=1234567890.0
    )


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return {
        "content": "Mock response from LLM",
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 30,
            "total_tokens": 80
        },
        "model": "test-model"
    }


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {"default": "Mock LLM response"}
        self.call_count = 0
        self.last_prompt = None
        self.last_config = None

    async def generate_completion(self, prompt: str, config: ModelConfig) -> tuple[str, ExecutionMetrics]:
        """Mock completion generation."""
        self.call_count += 1
        self.last_prompt = prompt
        self.last_config = config

        # Get response based on prompt or use default
        response = self.responses.get(prompt, self.responses.get("default", "Mock response"))

        # Create mock metrics
        metrics = ExecutionMetrics(
            execution_time=0.5,
            token_usage=TokenUsage(
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(response.split()),
                total_tokens=len(prompt.split()) + len(response.split())
            ),
            cost_info=CostInfo(
                cost_usd=0.001,
                provider=config.provider,
                model_name=config.model_name
            ),
            provider=config.provider,
            model_name=config.model_name,
            success=True,
            timestamp=1234567890.0
        )

        return response, metrics


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def mock_failing_llm_provider():
    """Create a mock LLM provider that fails."""
    provider = MockLLMProvider()

    async def failing_completion(prompt: str, config: ModelConfig):
        raise Exception("Mock LLM failure")

    provider.generate_completion = failing_completion
    return provider


@pytest.fixture
def sample_node(mock_model_config):
    """Create a sample node for testing."""
    # Ensure operators are imported
    import nons.operators.base

    return Node(
        operator_name='generate',
        model_config=mock_model_config,
        additional_prompt_context="Test context"
    )


@pytest.fixture
def sample_layer(mock_model_config):
    """Create a sample layer for testing."""
    # Ensure operators are imported
    import nons.operators.base

    nodes = [
        Node('generate', mock_model_config),
        Node('generate', mock_model_config),
        Node('generate', mock_model_config)
    ]
    return Layer(nodes)


@pytest.fixture
def sample_network(mock_model_config):
    """Create a sample network for testing."""
    # Ensure operators are imported
    import nons.operators.base

    return NoN.from_operators([
        'generate',
        ['generate', 'generate'],
        'generate'
    ])


@pytest.fixture
def clean_operator_registry():
    """Provide a clean operator registry for testing."""
    registry = OperatorRegistry()
    # Register a test operator

    async def test_operator(text: str) -> str:
        return f"Processed: {text}"

    registry.register(
        name="test_operator",
        function=test_operator,
        schema={
            "input": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            },
            "output": {"type": "string"},
            "description": "Test operator"
        }
    )

    return registry


@pytest.fixture
def mock_provider_factory():
    """Mock the provider factory to return mock providers."""
    with patch('nons.utils.providers.create_provider') as mock_factory:
        mock_factory.return_value = MockLLMProvider()
        yield mock_factory


@pytest.fixture
def mock_observability():
    """Mock the observability system."""
    with patch('nons.observability.integration.get_observability') as mock_obs:
        mock_manager = MagicMock()
        mock_manager.start_operation.return_value = MagicMock()
        mock_manager.finish_operation.return_value = None
        mock_obs.return_value = mock_manager
        yield mock_manager


@pytest.fixture
def mock_scheduler():
    """Mock the request scheduler."""
    with patch('nons.core.scheduler.get_scheduler') as mock_scheduler:
        scheduler = AsyncMock()
        scheduler.schedule_request = AsyncMock(side_effect=lambda op, *args, **kwargs: op(*args, **kwargs))
        mock_scheduler.return_value = scheduler
        yield scheduler


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment with mocked providers."""
    # Patch the provider creation to use mock providers
    with patch('nons.utils.providers.create_provider') as mock_create:
        mock_create.return_value = MockLLMProvider()
        yield


class AsyncTestCase:
    """Base class for async test cases."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up method for each test."""
        # Ensure operators are imported
        import nons.operators.base


# Utility functions for tests
def assert_execution_metrics(metrics: ExecutionMetrics):
    """Assert that execution metrics are valid."""
    assert isinstance(metrics, ExecutionMetrics)
    assert metrics.execution_time >= 0
    assert metrics.token_usage.total_tokens >= 0
    assert metrics.cost_info.cost_usd >= 0
    assert metrics.provider is not None
    assert metrics.model_name is not None
    assert isinstance(metrics.success, bool)
    assert metrics.timestamp > 0


def assert_node_state(node: Node):
    """Assert that a node is in a valid state."""
    assert node.node_id is not None
    assert node.operator_name is not None
    assert node.model_config is not None
    assert isinstance(node._execution_count, int)
    assert isinstance(node.get_total_cost(), float)
    assert isinstance(node.get_total_tokens(), int)


def create_test_input_data():
    """Create test input data for various operators."""
    return {
        "simple_text": "This is a test input",
        "complex_text": "This is a more complex test input with multiple sentences. It contains various types of content.",
        "structured_data": {
            "text": "Sample text",
            "metadata": {"source": "test", "type": "sample"}
        },
        "list_data": ["item1", "item2", "item3"]
    }