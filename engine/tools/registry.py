from __future__ import annotations

from typing import Any, Callable, Awaitable

from engine.config.models import ToolParameter


ToolCallable = Callable[..., Awaitable[str]]

_REGISTRY: dict[str, ToolCallable] = {}
_SCHEMAS: dict[str, dict[str, ToolParameter]] = {}


def tool(
    tool_id: str,
    description: str,
    parameters: dict[str, ToolParameter] | None = None,
):
    def decorator(fn: ToolCallable) -> ToolCallable:
        _REGISTRY[tool_id] = fn
        _SCHEMAS[tool_id] = parameters or {}
        fn._tool_id = tool_id  # type: ignore[attr-defined]
        fn._tool_description = description  # type: ignore[attr-defined]
        return fn
    return decorator


def get_tool(tool_id: str) -> ToolCallable:
    if tool_id not in _REGISTRY:
        raise KeyError(f"Tool '{tool_id}' is not registered")
    return _REGISTRY[tool_id]


def get_tool_schema(tool_id: str) -> dict[str, ToolParameter]:
    return _SCHEMAS.get(tool_id, {})


def list_registered_tools() -> list[str]:
    return list(_REGISTRY.keys())


def build_openai_tool_spec(tool_id: str, definition: Any) -> dict:
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in definition.parameters.items():
        properties[param_name] = {
            "type": param.type,
            "description": param.description,
        }
        if param.required:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": tool_id,
            "description": definition.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
