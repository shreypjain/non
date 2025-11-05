from typing import Any, Awaitable, Callable, Dict, Optional, Type
from pydantic import BaseModel, ValidationError

AsyncCallable = Callable[..., Awaitable[Any]]


class ToolRegistry:
    """
    Registry for tools with Pydantic validation and proper error handling.
    Inspired by browser-use's Controller but simplified for general agent use.
    """

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def _register(
        self,
        name: str,
        description: str,
        fn: AsyncCallable,
        param_model: Optional[Type[BaseModel]] = None,
        is_stop_tool: bool = False,
    ):
        """Register a tool with validation schema"""
        self._tools[name] = {
            "fn": fn,
            "description": description,
            "param_model": param_model,
            "is_stop_tool": is_stop_tool,
        }

    def tool(
        self,
        name: str,
        description: str,
        param_model: Optional[Type[BaseModel]] = None,
        is_stop_tool: bool = False,
    ):
        """Decorator for registering tools"""

        def decorator(fn: AsyncCallable):
            self._register(name, description, fn, param_model, is_stop_tool)
            return fn

        return decorator

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Return all tool descriptions for LLM context"""
        return {name: tool["description"] for name, tool in self._tools.items()}

    def is_stop_tool(self, name: str) -> bool:
        """Check if a tool is a termination tool"""
        return self._tools.get(name, {}).get("is_stop_tool", False)

    async def execute(self, name: str, params: Any, **context) -> Dict[str, Any]:
        """
        Execute tool with validation and error handling.
        Returns standardized result format.
        """
        if name not in self._tools:
            return {
                "success": False,
                "error": f"Tool '{name}' does not exist in registry",
                "available_tools": list(self._tools.keys()),
            }

        tool = self._tools[name]

        try:
            # Validate params if model exists
            if tool["param_model"]:
                validated_params = tool["param_model"].model_validate(params)
            else:
                validated_params = params

            # Execute tool
            result = await tool["fn"](validated_params, **context)

            # Standardize output
            return {
                "success": True,
                "tool_name": name,
                "output": result,
                "is_stop_tool": tool["is_stop_tool"],
            }

        except ValidationError as e:
            return {
                "success": False,
                "error": f"Parameter validation failed: {str(e)}",
                "tool_name": name,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}",
                "tool_name": name,
            }
