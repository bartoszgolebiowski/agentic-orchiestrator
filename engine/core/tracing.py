"""Backward-compat shim — use engine.llm.tracing directly."""
from engine.llm.tracing import (
    flush,
    get_langfuse,
    get_langfuse_base_url,
    get_langfuse_import_error,
    is_langfuse_enabled,
    log_langfuse_connection_status,
    observe,
)

__all__ = [
    "flush",
    "get_langfuse",
    "get_langfuse_base_url",
    "get_langfuse_import_error",
    "is_langfuse_enabled",
    "log_langfuse_connection_status",
    "observe",
]
