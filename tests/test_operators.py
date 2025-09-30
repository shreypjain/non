"""
Tests for operator registry and base operators.
"""
import pytest
from unittest.mock import patch, AsyncMock
from nons.operators.registry import OperatorRegistry, operator, get_operator, list_operators, get_operator_info
from nons.core.types import OperatorError, ValidationError


class TestOperatorRegistry:
    """Test the operator registry functionality."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = OperatorRegistry()
        assert registry._operators == {}

    def test_register_operator(self):
        """Test registering an operator."""
        registry = OperatorRegistry()

        async def test_func(text: str) -> str:
            return f"processed: {text}"

        schema = {
            "input": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            },
            "output": {"type": "string"},
            "description": "Test function"
        }

        from nons.core.types import OperatorMetadata
        metadata = OperatorMetadata(name="test_op", description="Test function", examples=[], tags=[])
        registry.register(test_func, name="test_op", input_schema=schema["input"], output_schema=schema["output"], metadata=metadata)

        assert "test_op" in registry._operators
        registered = registry._operators["test_op"]
        assert registered.name == "test_op"
        assert registered.function == test_func
        assert registered.metadata.description == "Test function"

    def test_get_operator(self):
        """Test getting a registered operator."""
        registry = OperatorRegistry()

        async def test_func(text: str) -> str:
            return f"processed: {text}"

        schema = {
            "input": {"type": "object"},
            "output": {"type": "string"},
            "description": "Test function"
        }

        from nons.core.types import OperatorMetadata
        metadata = OperatorMetadata(name="test_op", description="Test function", examples=[], tags=[])
        registry.register(test_func, name="test_op", input_schema=schema["input"], output_schema=schema["output"], metadata=metadata)
        retrieved = registry.get("test_op")

        assert retrieved.name == "test_op"
        assert retrieved.function == test_func

    def test_get_nonexistent_operator(self):
        """Test getting a non-existent operator raises error."""
        registry = OperatorRegistry()

        with pytest.raises(OperatorError, match="Operator 'nonexistent' not found"):
            registry.get("nonexistent")

    def test_list_operators(self):
        """Test listing registered operators."""
        registry = OperatorRegistry()

        async def func1(text: str) -> str:
            return text

        async def func2(text: str) -> str:
            return text

        schema = {"input": {}, "output": {}, "description": ""}

        from nons.core.types import OperatorMetadata
        metadata = OperatorMetadata(name="op1", description="", examples=[], tags=[])
        registry.register(func1, name="op1", input_schema=schema["input"], output_schema=schema["output"], metadata=metadata)
        registry.register(func2, name="op2", input_schema=schema["input"], output_schema=schema["output"], metadata=metadata)

        operators = registry.list()
        assert set(operators) == {"op1", "op2"}

    def test_get_operator_info(self):
        """Test getting operator information."""
        registry = OperatorRegistry()

        async def test_func(text: str) -> str:
            return text

        schema = {
            "input": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            },
            "output": {"type": "string"},
            "description": "Test function description"
        }

        from nons.core.types import OperatorMetadata
        metadata = OperatorMetadata(name="test_op", description="Test function description", examples=[], tags=[])
        registry.register(test_func, name="test_op", input_schema=schema["input"], output_schema=schema["output"], metadata=metadata)
        from nons.operators.registry import get_operator_info
        info = get_operator_info("test_op")

        assert info["name"] == "test_op"
        assert info["description"] == "Test function description"
        assert "input_schema" in info
        assert "output_schema" in info


class TestOperatorDecorator:
    """Test the @operator decorator."""

    def test_operator_decorator_registration(self):
        """Test that @operator decorator registers the function."""
        # Clear any existing operators for clean test
        with patch('nons.operators.registry._registry') as mock_registry:
            mock_registry.register = AsyncMock()

            from nons.core.types import InputSchema, OutputSchema, OperatorMetadata
            @operator(
                input_schema=InputSchema(required_params=["text"], optional_params=[], param_types={"text": "str"}),
                output_schema=OutputSchema(return_type="str", description="Test output"),
                metadata=OperatorMetadata(name="test_op", description="Test operator", examples=[], tags=[])
            )
            async def test_op(text: str) -> str:
                return f"processed: {text}"

            # Check that register was called
            mock_registry.register.assert_called_once()
            args = mock_registry.register.call_args[0]
            assert args[0] == "test_op"  # function name
            assert args[1] == test_op    # function itself

    def test_operator_decorator_with_custom_name(self):
        """Test @operator decorator with custom name."""
        with patch('nons.operators.registry._registry') as mock_registry:
            mock_registry.register = AsyncMock()

            from nons.core.types import InputSchema, OutputSchema, OperatorMetadata
            @operator(
                name="custom_name",
                input_schema=InputSchema(required_params=["text"], optional_params=[], param_types={"text": "str"}),
                output_schema=OutputSchema(return_type="str", description="Test output"),
                metadata=OperatorMetadata(name="custom_name", description="Custom named operator", examples=[], tags=[])
            )
            async def test_function(text: str) -> str:
                return text

            args = mock_registry.register.call_args[0]
            assert args[0] == "custom_name"

    async def test_operator_function_execution(self):
        """Test that decorated operator function executes correctly."""
        from nons.core.types import InputSchema, OutputSchema, OperatorMetadata
        @operator(
            input_schema=InputSchema(required_params=["text"], optional_params=[], param_types={"text": "str"}),
            output_schema=OutputSchema(return_type="str", description="Test output"),
            metadata=OperatorMetadata(name="test_op", description="Test operator", examples=[], tags=[])
        )
        async def test_op(text: str) -> str:
            return f"processed: {text}"

        result = await test_op("hello")
        assert result == "processed: hello"


class TestBaseOperators:
    """Test the base operators."""

    def setup_method(self):
        """Set up test method."""
        # Import operators to ensure they're registered
        import nons.operators.base

    async def test_transform_operator(self):
        """Test the transform operator."""
        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = AsyncMock()
            mock_provider.generate_completion.return_value = (
                "Transformed text",
                AsyncMock()  # Mock metrics
            )
            mock_provider_factory.return_value = mock_provider

            from nons.operators.registry import get_operator
            transform_op = get_operator('transform')

            result = await transform_op.function(
                text="Hello world",
                target_format="uppercase"
            )

            assert result is not None
            mock_provider.generate_completion.assert_called_once()

    async def test_generate_operator(self):
        """Test the generate operator."""
        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = AsyncMock()
            mock_provider.generate_completion.return_value = (
                "Generated content",
                AsyncMock()  # Mock metrics
            )
            mock_provider_factory.return_value = mock_provider

            from nons.operators.registry import get_operator
            generate_op = get_operator('generate')

            result = await generate_op.function(prompt="Write a story")

            assert result is not None
            mock_provider.generate_completion.assert_called_once()

    async def test_classify_operator(self):
        """Test the classify operator."""
        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = AsyncMock()
            mock_provider.generate_completion.return_value = (
                "positive",
                AsyncMock()  # Mock metrics
            )
            mock_provider_factory.return_value = mock_provider

            from nons.operators.registry import get_operator
            classify_op = get_operator('classify')

            result = await classify_op.function(
                text="I love this!",
                categories=["positive", "negative", "neutral"]
            )

            assert result is not None
            mock_provider.generate_completion.assert_called_once()

    async def test_extract_operator(self):
        """Test the extract operator."""
        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = AsyncMock()
            mock_provider.generate_completion.return_value = (
                "John Doe, jane@example.com",
                AsyncMock()  # Mock metrics
            )
            mock_provider_factory.return_value = mock_provider

            from nons.operators.registry import get_operator
            extract_op = get_operator('extract')

            result = await extract_op.function(
                text="Contact John Doe at jane@example.com",
                target_info="names and emails"
            )

            assert result is not None
            mock_provider.generate_completion.assert_called_once()

    async def test_condense_operator(self):
        """Test the condense operator."""
        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = AsyncMock()
            mock_provider.generate_completion.return_value = (
                "Brief summary",
                AsyncMock()  # Mock metrics
            )
            mock_provider_factory.return_value = mock_provider

            from nons.operators.registry import get_operator
            condense_op = get_operator('condense')

            result = await condense_op.function(
                text="Long text to summarize...",
                target_length="brief"
            )

            assert result is not None
            mock_provider.generate_completion.assert_called_once()

    async def test_expand_operator(self):
        """Test the expand operator."""
        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = AsyncMock()
            mock_provider.generate_completion.return_value = (
                "Expanded detailed text",
                AsyncMock()  # Mock metrics
            )
            mock_provider_factory.return_value = mock_provider

            from nons.operators.registry import get_operator
            expand_op = get_operator('expand')

            result = await expand_op.function(
                text="Brief text",
                expansion_type="detailed"
            )

            assert result is not None
            mock_provider.generate_completion.assert_called_once()

    async def test_compare_operator(self):
        """Test the compare operator."""
        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = AsyncMock()
            mock_provider.generate_completion.return_value = (
                "Similar concepts with different approaches",
                AsyncMock()  # Mock metrics
            )
            mock_provider_factory.return_value = mock_provider

            from nons.operators.registry import get_operator
            compare_op = get_operator('compare')

            result = await compare_op.function(
                text1="First text",
                text2="Second text",
                comparison_type="similarities"
            )

            assert result is not None
            mock_provider.generate_completion.assert_called_once()

    async def test_validate_operator(self):
        """Test the validate operator."""
        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = AsyncMock()
            mock_provider.generate_completion.return_value = (
                "Valid: meets all criteria",
                AsyncMock()  # Mock metrics
            )
            mock_provider_factory.return_value = mock_provider

            from nons.operators.registry import get_operator
            validate_op = get_operator('validate')

            result = await validate_op.function(
                text="Text to validate",
                criteria="grammar and clarity"
            )

            assert result is not None
            mock_provider.generate_completion.assert_called_once()

    async def test_route_operator(self):
        """Test the route operator."""
        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = AsyncMock()
            mock_provider.generate_completion.return_value = (
                "technical",
                AsyncMock()  # Mock metrics
            )
            mock_provider_factory.return_value = mock_provider

            from nons.operators.registry import get_operator
            route_op = get_operator('route')

            result = await route_op.function(
                text="Technical documentation about APIs",
                routes=["technical", "creative", "business"]
            )

            assert result is not None
            mock_provider.generate_completion.assert_called_once()

    async def test_synthesize_operator(self):
        """Test the synthesize operator."""
        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = AsyncMock()
            mock_provider.generate_completion.return_value = (
                "Synthesized content from multiple sources",
                AsyncMock()  # Mock metrics
            )
            mock_provider_factory.return_value = mock_provider

            from nons.operators.registry import get_operator
            synthesize_op = get_operator('synthesize')

            result = await synthesize_op.function(
                inputs=["Input 1", "Input 2", "Input 3"],
                synthesis_type="comprehensive"
            )

            assert result is not None
            mock_provider.generate_completion.assert_called_once()

    def test_all_operators_registered(self):
        """Test that all expected operators are registered."""
        from nons.operators.registry import list_operators

        operators = list_operators()
        expected_operators = {
            'transform', 'generate', 'classify', 'extract',
            'condense', 'expand', 'compare', 'validate',
            'route', 'synthesize'
        }

        # Check that all expected operators are present
        for op in expected_operators:
            assert op in operators, f"Operator '{op}' not found in registered operators"

    def test_operator_schemas(self):
        """Test that all operators have valid schemas."""
        from nons.operators.registry import get_operator_info, list_operators

        operators = list_operators()

        for op_name in operators:
            info = get_operator_info(op_name)

            # Check that info has required fields
            assert 'name' in info
            assert 'input_schema' in info
            assert 'output_schema' in info
            assert 'description' in info

            # Check that schemas are dictionaries
            assert isinstance(info['input_schema'], dict)
            assert isinstance(info['output_schema'], dict)
            assert isinstance(info['description'], str)

            # Check that description is not empty
            assert len(info['description']) > 0


class TestGlobalOperatorFunctions:
    """Test global operator functions."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    def test_get_operator_function(self):
        """Test the global get_operator function."""
        from nons.operators.registry import get_operator

        op = get_operator('generate')
        assert op.name == 'generate'
        assert callable(op.function)

    def test_list_operators_function(self):
        """Test the global list_operators function."""
        from nons.operators.registry import list_operators

        operators = list_operators()
        assert isinstance(operators, list)
        assert 'generate' in operators

    def test_get_operator_info_function(self):
        """Test the global get_operator_info function."""
        from nons.operators.registry import get_operator_info

        info = get_operator_info('generate')
        assert isinstance(info, dict)
        assert 'name' in info
        assert info['name'] == 'generate'

    def test_get_nonexistent_operator_info(self):
        """Test getting info for non-existent operator."""
        from nons.operators.registry import get_operator_info

        with pytest.raises(OperatorError, match="Operator 'nonexistent' not found"):
            get_operator_info('nonexistent')


@pytest.mark.unit
class TestOperatorValidation:
    """Test operator input validation."""

    def setup_method(self):
        """Set up test method."""
        import nons.operators.base

    async def test_operator_with_invalid_input(self):
        """Test that operators handle invalid input appropriately."""
        # This test depends on how validation is implemented in the operators
        # For now, we'll test that the operator functions can be called
        from nons.operators.registry import get_operator

        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = AsyncMock()
            mock_provider.generate_completion.return_value = (
                "Response",
                AsyncMock()
            )
            mock_provider_factory.return_value = mock_provider

            generate_op = get_operator('generate')

            # Test with minimal valid input
            result = await generate_op.function(prompt="test")
            assert result is not None

    async def test_operator_error_handling(self):
        """Test operator error handling."""
        from nons.operators.registry import get_operator

        with patch('nons.utils.providers.create_provider') as mock_provider_factory:
            mock_provider = AsyncMock()
            mock_provider.generate_completion.side_effect = Exception("Provider error")
            mock_provider_factory.return_value = mock_provider

            generate_op = get_operator('generate')

            with pytest.raises(Exception, match="Provider error"):
                await generate_op.function(prompt="test")