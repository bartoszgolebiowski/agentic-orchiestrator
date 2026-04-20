from __future__ import annotations

from engine.config.models import ToolParameter
from engine.tools.registry import tool


@tool(
    tool_id="add",
    description="Adds two numbers and returns the sum as a string.",
    parameters={
        "a": ToolParameter(type="number", description="First addend."),
        "b": ToolParameter(type="number", description="Second addend."),
    },
)
async def add(a: float, b: float) -> str:
    return str(a + b)


@tool(
    tool_id="multiply",
    description="Multiplies two numbers and returns the product as a string.",
    parameters={
        "a": ToolParameter(type="number", description="First factor."),
        "b": ToolParameter(type="number", description="Second factor."),
    },
)
async def multiply(a: float, b: float) -> str:
    return str(a * b)


@tool(
    tool_id="subtract",
    description="Subtracts the second number from the first and returns the difference as a string.",
    parameters={
        "a": ToolParameter(type="number", description="Minuend."),
        "b": ToolParameter(type="number", description="Subtrahend."),
    },
)
async def subtract(a: float, b: float) -> str:
    return str(a - b)
