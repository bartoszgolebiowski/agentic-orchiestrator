"""Compatibility re-export for markdown tools.

Prefer importing implementations from engine.tools.implementations.markdown_tools.
"""

from engine.tools.implementations.markdown_tools import read_markdown_structure

__all__ = ["read_markdown_structure"]
