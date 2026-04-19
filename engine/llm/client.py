from __future__ import annotations

from contextvars import ContextVar
import inspect
import json
import logging
from math import ceil
import os
from enum import Enum
from textwrap import indent
from typing import Any

from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel

from engine.llm.tracing import observe


logger = logging.getLogger(__name__)
_LAST_USAGE_DETAILS: ContextVar[dict[str, Any] | None] = ContextVar(
    "_last_usage_details",
    default=None,
)


def get_base_url() -> str | None:
    return (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENROUTER_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
    )


def _client_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"api_key": os.environ["OPENAI_API_KEY"]}
    base_url = get_base_url()
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    return kwargs


def get_raw_client() -> AsyncOpenAI:
    """Raw OpenAI client for native tool-calling (Subagent → Tool layer)."""
    return AsyncOpenAI(**_client_kwargs())


def get_model() -> str:
    return os.environ.get("LLM_MODEL", "gpt-4o")


def get_reasoning_effort(default: str | None = None) -> str | None:
    value = (
        os.environ.get("LLM_REASONING_EFFORT")
        or os.environ.get("OPENAI_REASONING_EFFORT")
        or os.environ.get("REASONING_EFFORT")
    )
    if value is None:
        return default
    effort = value.strip().lower()
    return effort or default


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
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


def _format_trace_value(value: Any) -> str:
    if value is None:
        return "(none)"
    if isinstance(value, str):
        return value
    return json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True)


def _tool_name(tool_spec: dict[str, Any]) -> str:
    function = tool_spec.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            return name
    return str(tool_spec)


def _structured_completion_endpoint(client: AsyncOpenAI) -> tuple[Any, str]:
    responses = getattr(client, "responses", None)
    if responses is not None:
        return responses, "responses"

    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None) if chat is not None else None
    if completions is not None:
        return completions, "chat"

    raise AttributeError("OpenAI client does not expose responses or chat.completions")


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


def _normalize_usage_details(raw_usage: Any) -> dict[str, Any] | None:
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


def _extract_usage_details(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is not None:
        return _normalize_usage_details(usage)

    nested_response = getattr(response, "response", None)
    nested_usage = getattr(nested_response, "usage", None) if nested_response is not None else None
    return _normalize_usage_details(nested_usage)


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


def _format_message_thread(
    messages: list[dict[str, Any]],
    *,
    model_name: str,
    trace_kind: str,
    tools: list[dict[str, Any]] | None = None,
    response_model: str | None = None,
) -> str:
    lines: list[str] = [
        f"Trace kind: {trace_kind}",
        f"Model: {model_name}",
        f"Message count: {len(messages)}",
    ]

    if response_model is not None:
        lines.append(f"Response model: {response_model}")

    if tools:
        lines.append("Available tools:")
        for tool_spec in tools:
            lines.append(f"- {_tool_name(tool_spec)}")

    lines.append("Conversation:")

    for index, message in enumerate(messages, start=1):
        role = str(message.get("role", "message")).upper()
        name = message.get("name")
        header = f"{index}. {role}"
        if isinstance(name, str) and name.strip():
            header += f" ({name})"
        lines.append(header)

        content = message.get("content")
        if content not in (None, ""):
            lines.append(indent(_format_trace_value(content), "  "))

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            lines.append("  Tool calls:")
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    function = tool_call.get("function") or {}
                    tool_name = (
                        function.get("name", "unknown")
                        if isinstance(function, dict)
                        else "unknown"
                    )
                    arguments = (
                        function.get("arguments", "")
                        if isinstance(function, dict)
                        else ""
                    )
                    lines.append(f"    - {tool_name}({arguments})")
                else:
                    lines.append(f"    - {_format_trace_value(tool_call)}")

        tool_call_id = message.get("tool_call_id")
        if tool_call_id:
            lines.append(f"  Tool call id: {tool_call_id}")

    return "\n".join(lines)


async def chat_completion(
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
    trace_name: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
    temperature: float = 0.0,
) -> Any:
    """Raw chat completion — used only by SubagentExecutor (native tool calls)."""
    _LAST_USAGE_DETAILS.set(None)
    model_name = model or get_model()
    trace_input = _format_message_thread(
        messages,
        model_name=model_name,
        trace_kind="chat_completion",
        tools=tools,
    )
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
    }
    supports_reasoning = False
    reasoning_effort = get_reasoning_effort()
    if reasoning_effort:
        try:
            supports_reasoning = "reasoning_effort" in inspect.signature(
                client.chat.completions.create
            ).parameters
        except (ValueError, TypeError, AttributeError):
            supports_reasoning = False
    if supports_reasoning and reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    with observe(
        name=trace_name or "llm.chat_completion",
        as_type="generation",
        input=trace_input,
        model=model_name,
        metadata=trace_metadata,
    ) as generation:
        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            if generation is not None:
                generation.update(output=f"ERROR: {type(exc).__name__}: {exc}")
            raise

        message = response.choices[0].message
        usage_details = _extract_usage_details(response)
        _LAST_USAGE_DETAILS.set(usage_details)
        if generation is not None:
            update_kwargs: dict[str, Any] = {
                "output": _format_trace_value(_model_dump(message))
            }
            if usage_details is not None:
                update_kwargs["usage_details"] = usage_details
            generation.update(**update_kwargs)

        return message


async def structured_completion(
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    model: str | None = None,
    trace_name: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
    temperature: float = 0.0,
) -> Any:
    """Native structured-output completion — returns a validated Pydantic model."""
    _LAST_USAGE_DETAILS.set(None)
    model_name = model or get_model()
    completions, api_kind = _structured_completion_endpoint(client)
    trace_input = _format_message_thread(
        messages,
        model_name=model_name,
        trace_kind="structured_completion",
        response_model=response_model.__name__,
    )
    with observe(
        name=trace_name or "llm.structured_completion",
        as_type="generation",
        input=trace_input,
        model=model_name,
        metadata=trace_metadata,
    ) as generation:
        reasoning_effort = get_reasoning_effort(default="medium")
        supports_reasoning_parse = False
        supports_reasoning_create = False
        try:
            supports_reasoning_parse = 'reasoning_effort' in inspect.signature(
                completions.parse
            ).parameters
        except (ValueError, TypeError, AttributeError):
            supports_reasoning_parse = False
        try:
            supports_reasoning_create = (
                hasattr(completions, "create")
                and 'reasoning_effort' in inspect.signature(completions.create).parameters
            )
        except (ValueError, TypeError, AttributeError):
            supports_reasoning_create = False

        parse_kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
        }
        if supports_reasoning_parse and reasoning_effort:
            parse_kwargs["reasoning_effort"] = reasoning_effort

        if api_kind == "responses":
            parse_kwargs["input"] = messages
            parse_kwargs["text_format"] = response_model
        else:
            parse_kwargs["messages"] = messages
            parse_kwargs["response_format"] = response_model

        try:
            response = await completions.parse(**parse_kwargs)
        except TypeError as exc:
            if "reasoning_effort" in str(exc) and "reasoning_effort" in parse_kwargs:
                parse_kwargs.pop("reasoning_effort", None)
                try:
                    response = await completions.parse(**parse_kwargs)
                except Exception as exc2:
                    if generation is not None:
                        generation.update(output=f"ERROR: {type(exc2).__name__}: {exc2}")
                    raise
            else:
                if generation is not None:
                    generation.update(output=f"ERROR: {type(exc).__name__}: {exc}")
                raise
        except BadRequestError:
            logger.warning("Structured output parse was rejected by the provider; retrying with JSON mode")
            create_kwargs: dict[str, Any] = {
                "model": model_name,
                "temperature": temperature,
            }
            if supports_reasoning_create and reasoning_effort:
                create_kwargs["reasoning_effort"] = reasoning_effort
            if api_kind == "responses":
                create_kwargs["input"] = messages
                create_kwargs["text"] = {"format": {"type": "json_object"}}
            else:
                create_kwargs["messages"] = messages
                create_kwargs["response_format"] = {"type": "json_object"}

            try:
                response = await completions.create(**create_kwargs)
            except Exception as exc:
                if generation is not None:
                    generation.update(output=f"ERROR: {type(exc).__name__}: {exc}")
                raise
        except Exception as exc:
            if generation is not None:
                generation.update(output=f"ERROR: {type(exc).__name__}: {exc}")
            raise

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            message = (
                response.choices[0].message
                if getattr(response, "choices", None)
                else None
            )
            parsed = getattr(message, "parsed", None) if message is not None else None
        if parsed is None:
            content = getattr(response, "output_text", None)
            if content is None:
                message = (
                    response.choices[0].message
                    if getattr(response, "choices", None)
                    else None
                )
                content = (
                    getattr(message, "content", None) if message is not None else None
                )
            if content is None:
                raise ValueError("Structured completion returned no parsable output")
            parsed = response_model.model_validate_json(content)

        usage_details = _extract_usage_details(response)
        _LAST_USAGE_DETAILS.set(usage_details)

        if generation is not None:
            update_kwargs: dict[str, Any] = {
                "output": _format_trace_value(_model_dump(parsed))
            }
            if usage_details is not None:
                update_kwargs["usage_details"] = usage_details
            generation.update(**update_kwargs)

        return parsed
