"""Backward-compat shim — use engine.config.contracts directly."""
from engine.config.contracts import (
    describe_model_contract,
    format_model_spec,
    model_schema_json,
    normalize_handoff_payload,
    resolve_model,
    resolve_model_or_none,
    validate_model_payload,
)

__all__ = [
    "describe_model_contract",
    "format_model_spec",
    "model_schema_json",
    "normalize_handoff_payload",
    "resolve_model",
    "resolve_model_or_none",
    "validate_model_payload",
]
