"""
Operator registry system with decorator pattern for function registration.

This module provides the core registry functionality for operators, including
the @operator decorator and registry management for compile-time validation.
"""

from typing import Dict, Callable, Any, List, Optional, Type
from functools import wraps, partial
import asyncio
import inspect
from ..core.types import (
    InputSchema,
    OutputSchema,
    OperatorMetadata,
    Content,
    ValidationError,
    OperatorError,
)


class OperatorRegistry:
    """
    Central registry for all operators in the NoN system.

    Maintains function references, I/O validation schemas, metadata,
    and compilation hooks for optimization.
    """

    def __init__(self) -> None:
        self._operators: Dict[str, "RegisteredOperator"] = {}
        self._compilation_hooks: List[Callable] = []

    def register(
        self,
        func: Callable,
        name: Optional[str] = None,
        input_schema: Optional[InputSchema] = None,
        output_schema: Optional[OutputSchema] = None,
        metadata: Optional[OperatorMetadata] = None,
    ) -> "RegisteredOperator":
        """
        Register an operator function with the registry.

        Args:
            func: The operator function to register
            name: Optional name override (defaults to function name)
            input_schema: Schema defining required/optional inputs
            output_schema: Schema defining expected output types
            metadata: Operator metadata for documentation

        Returns:
            RegisteredOperator instance
        """
        operator_name = name or func.__name__

        if operator_name in self._operators:
            raise ValidationError(f"Operator '{operator_name}' already registered")

        # Auto-generate schemas if not provided
        if input_schema is None:
            input_schema = self._infer_input_schema(func)

        if output_schema is None:
            output_schema = self._infer_output_schema(func)

        if metadata is None:
            metadata = OperatorMetadata(
                name=operator_name,
                description=func.__doc__ or f"Operator: {operator_name}",
                examples=[],
                tags=[],
            )

        registered_op = RegisteredOperator(
            name=operator_name,
            function=func,
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata,
        )

        self._operators[operator_name] = registered_op
        return registered_op

    def get(self, name: str) -> "RegisteredOperator":
        """Get a registered operator by name."""
        if name not in self._operators:
            raise ValidationError(f"Operator '{name}' not found in registry")
        return self._operators[name]

    def list_operators(self) -> List[str]:
        """Get list of all registered operator names."""
        return list(self._operators.keys())

    def add_compilation_hook(self, hook: Callable) -> None:
        """Add a compilation optimization hook."""
        self._compilation_hooks.append(hook)

    def _infer_input_schema(self, func: Callable) -> InputSchema:
        """Infer input schema from function signature."""
        sig = inspect.signature(func)
        required_params = []
        optional_params = []
        param_types = {}

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Convert type annotations to strings for pydantic compatibility
            if param.annotation != inspect.Parameter.empty:
                param_types[param_name] = str(param.annotation)
            else:
                param_types[param_name] = "Any"

            if param.default == inspect.Parameter.empty:
                required_params.append(param_name)
            else:
                optional_params.append(param_name)

        return InputSchema(
            required_params=required_params,
            optional_params=optional_params,
            param_types=param_types,
        )

    def _infer_output_schema(self, func: Callable) -> OutputSchema:
        """Infer output schema from function return annotation."""
        sig = inspect.signature(func)
        return_type = (
            str(sig.return_annotation)
            if sig.return_annotation != inspect.Signature.empty
            else "Any"
        )

        return OutputSchema(
            return_type=return_type, description=f"Return type for {func.__name__}"
        )


class RegisteredOperator:
    """
    Container for a registered operator with all its metadata and schemas.
    """

    def __init__(
        self,
        name: str,
        function: Callable,
        input_schema: InputSchema,
        output_schema: OutputSchema,
        metadata: OperatorMetadata,
    ):
        self.name = name
        self.function = function
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.metadata = metadata

        # Ensure function is async
        if not asyncio.iscoroutinefunction(function):
            self.function = self._make_async(function)

    def _make_async(self, func: Callable) -> Callable:
        """Convert sync function to async."""

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            # Use partial to properly handle kwargs with run_in_executor
            func_with_args = partial(func, *args, **kwargs)
            return await loop.run_in_executor(None, func_with_args)

        return async_wrapper

    def validate_inputs(self, *args, **kwargs) -> None:
        """Validate inputs against the input schema."""
        # Check required parameters are provided
        provided_kwargs = set(kwargs.keys())
        required_params = set(self.input_schema.required_params)

        # Account for positional arguments
        sig = inspect.signature(self.function)
        param_names = list(sig.parameters.keys())

        # Map positional args to parameter names
        for i, arg in enumerate(args):
            if i < len(param_names):
                provided_kwargs.add(param_names[i])

        missing_params = required_params - provided_kwargs
        if missing_params:
            raise ValidationError(f"Missing required parameters: {missing_params}")

    def __mul__(self, n: int) -> "ParallelOperator":
        """Support multiplication operator for parallel execution."""
        if not isinstance(n, int) or n < 1:
            raise ValidationError("Multiplication factor must be a positive integer")
        return ParallelOperator(self, n)


class ParallelOperator:
    """
    Represents n parallel instances of an operator for batch execution.
    Created via the multiplication operator (operator * n).
    """

    def __init__(self, operator: RegisteredOperator, count: int):
        self.operator = operator
        self.count = count
        self.name = f"{operator.name}_x{count}"

    def __repr__(self) -> str:
        return f"ParallelOperator({self.operator.name} x {self.count})"


# Global registry instance
_global_registry = OperatorRegistry()


def operator(
    name: Optional[str] = None,
    input_schema: Optional[InputSchema] = None,
    output_schema: Optional[OutputSchema] = None,
    metadata: Optional[OperatorMetadata] = None,
) -> Callable:
    """
    Decorator for registering operator functions.

    Usage:
        @operator
        async def transform(content: Content, transform_spec: str) -> Content:
            # Implementation here
            pass

    Args:
        name: Optional name override for the operator
        input_schema: Optional input validation schema
        output_schema: Optional output type schema
        metadata: Optional metadata for documentation

    Returns:
        Decorated function registered in the global registry
    """

    def decorator(func: Callable) -> Callable:
        # Register the function
        registered_op = _global_registry.register(
            func=func,
            name=name,
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata,
        )

        # Return a wrapper that preserves the original function interface
        # but adds registry metadata
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Validate inputs before execution
            registered_op.validate_inputs(*args, **kwargs)

            try:
                result = await registered_op.function(*args, **kwargs)
                return result
            except Exception as e:
                raise OperatorError(
                    f"Error in operator '{registered_op.name}': {str(e)}"
                ) from e

        # Attach registry information to the wrapper
        wrapper._registered_operator = registered_op
        wrapper.__mul__ = registered_op.__mul__  # Support multiplication

        return wrapper

    return decorator


def get_registry() -> OperatorRegistry:
    """Get the global operator registry."""
    return _global_registry


def get_operator(name: str) -> RegisteredOperator:
    """Get a registered operator by name from the global registry."""
    return _global_registry.get(name)


def list_operators() -> List[str]:
    """List all registered operators."""
    return _global_registry.list_operators()


def get_operator_info(name: str) -> Dict[str, Any]:
    """Get detailed information about a registered operator."""
    operator = _global_registry.get(name)
    return {
        "name": operator.name,
        "input_schema": {
            "required_params": operator.input_schema.required_params,
            "optional_params": operator.input_schema.optional_params,
            "param_types": operator.input_schema.param_types,
        },
        "output_schema": {
            "return_type": operator.output_schema.return_type,
            "description": operator.output_schema.description,
        },
        "description": operator.metadata.description,
    }


def clear_registry() -> None:
    """Clear the global registry (mainly for testing)."""
    global _global_registry
    _global_registry = OperatorRegistry()
