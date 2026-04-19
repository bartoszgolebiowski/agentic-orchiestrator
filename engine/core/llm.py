"""Backward-compat shim — use engine.llm.client and engine.llm.tracing directly."""
from engine.llm.client import (
    chat_completion,
    get_base_url,
    get_model,
    get_raw_client,
    structured_completion,
)
from engine.llm.tracing import (
    flush,
    observe,
)

__all__ = [
    "chat_completion",
    "flush",
    "get_base_url",
    "get_model",
    "get_raw_client",
    "observe",
    "structured_completion",
]
