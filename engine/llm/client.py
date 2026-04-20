from __future__ import annotations

"""Compatibility facade for LLM completion and usage helpers.

This module intentionally re-exports public functions from smaller modules:
- completion concerns live in engine.llm.completion
- usage accounting lives in engine.llm.usage
"""

from engine.llm.completion import (
    chat_completion,
    get_base_url,
    get_model,
    get_raw_client,
    get_reasoning_effort,
    structured_completion,
)
from engine.llm.usage import (
    estimate_io_usage,
    estimate_tokens,
    get_last_usage_details,
    merge_usage_summaries,
    summarize_usage,
)

__all__ = [
    "chat_completion",
    "estimate_io_usage",
    "estimate_tokens",
    "get_base_url",
    "get_last_usage_details",
    "get_model",
    "get_raw_client",
    "get_reasoning_effort",
    "merge_usage_summaries",
    "structured_completion",
    "summarize_usage",
]
