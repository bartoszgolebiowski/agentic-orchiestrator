"""Compatibility re-export for math tools.

Prefer importing implementations from engine.tools.implementations.math_tools.
"""

from engine.tools.implementations.math_tools import add, multiply, subtract

__all__ = ["add", "multiply", "subtract"]