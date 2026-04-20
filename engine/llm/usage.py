from __future__ import annotations

from contextvars import ContextVar
import json
from math import ceil
from typing import Any


_LAST_USAGE_DETAILS: ContextVar[dict[str, Any] | None] = ContextVar(
    "_last_usage_details",
    default=None,
)


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _coerce_usage_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _extract_nested_value(mapping: dict[str, Any], path: str) -> Any:
    current: Any = mapping
    for segment in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
        if current is None:
            return None
    return current


def _first_usage_int(mapping: dict[str, Any], *paths: str) -> int | None:
    for path in paths:
        value = _extract_nested_value(mapping, path)
        parsed = _coerce_usage_int(value)
        if parsed is not None:
            return parsed
    return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except TypeError:
            return _json_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    return str(value)


def normalize_usage_details(raw_usage: Any) -> dict[str, Any] | None:
    usage_dump = _model_dump(raw_usage)
    if usage_dump is None:
        return None
    if not isinstance(usage_dump, dict):
        return None

    usage_payload = _json_safe(usage_dump)
    if not isinstance(usage_payload, dict):
        return None

    normalized = dict(usage_payload)
    input_tokens = _first_usage_int(normalized, "input_tokens", "prompt_tokens")
    output_tokens = _first_usage_int(normalized, "output_tokens", "completion_tokens")
    total_tokens = _first_usage_int(normalized, "total_tokens")
    reasoning_tokens = _first_usage_int(
        normalized,
        "reasoning_tokens",
        "completion_tokens_details.reasoning_tokens",
        "output_tokens_details.reasoning_tokens",
    )
    cached_input_tokens = _first_usage_int(
        normalized,
        "cached_input_tokens",
        "cache_read_input_tokens",
        "prompt_tokens_details.cached_tokens",
        "input_token_details.cached_tokens",
    )

    if input_tokens is not None:
        normalized["input_tokens"] = input_tokens
    if output_tokens is not None:
        normalized["output_tokens"] = output_tokens
    if total_tokens is not None:
        normalized["total_tokens"] = total_tokens
    elif input_tokens is not None and output_tokens is not None:
        normalized["total_tokens"] = input_tokens + output_tokens
    if reasoning_tokens is not None:
        normalized["reasoning_tokens"] = reasoning_tokens
    if cached_input_tokens is not None:
        normalized["cached_input_tokens"] = cached_input_tokens

    return normalized


def extract_usage_details(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is not None:
        return normalize_usage_details(usage)

    nested_response = getattr(response, "response", None)
    nested_usage = getattr(nested_response, "usage", None) if nested_response is not None else None
    return normalize_usage_details(nested_usage)


def set_last_usage_details(usage: dict[str, Any] | None) -> None:
    _LAST_USAGE_DETAILS.set(None if usage is None else dict(usage))


def get_last_usage_details() -> dict[str, Any] | None:
    usage = _LAST_USAGE_DETAILS.get()
    if usage is None:
        return None
    return dict(usage)


def summarize_usage(usage_details: dict[str, Any] | None) -> dict[str, int] | None:
    if usage_details is None:
        return None

    summary: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens", "cached_input_tokens"):
        parsed = _coerce_usage_int(usage_details.get(key))
        if parsed is not None:
            summary[key] = parsed

    return summary or None


def merge_usage_summaries(*summaries: dict[str, int] | None) -> dict[str, int] | None:
    merged: dict[str, int] = {}
    for summary in summaries:
        if summary is None:
            continue
        for key, value in summary.items():
            merged[key] = merged.get(key, 0) + value
    return merged or None


def estimate_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        payload = value
    else:
        payload = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    if not payload:
        return 0
    return max(1, ceil(len(payload) / 4))


def estimate_io_usage(input_value: Any, output_value: Any) -> dict[str, int]:
    return {
        "input_tokens_estimate": estimate_tokens(input_value),
        "output_tokens_estimate": estimate_tokens(output_value),
    }


__all__ = [
    "estimate_io_usage",
    "estimate_tokens",
    "extract_usage_details",
    "get_last_usage_details",
    "merge_usage_summaries",
    "normalize_usage_details",
    "set_last_usage_details",
    "summarize_usage",
]
