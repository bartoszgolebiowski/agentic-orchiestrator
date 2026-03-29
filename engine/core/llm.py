from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel

from engine.core.tracing import observe


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


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


async def chat_completion(
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
    trace_name: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
) -> Any:
    """Raw chat completion — used only by SubagentExecutor (native tool calls)."""
    model_name = model or get_model()
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    with observe(
        name=trace_name or "llm.chat_completion",
        as_type="generation",
        input={"messages": messages, "tools": tools, "model": model_name},
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
        if generation is not None:
            update_kwargs: dict[str, Any] = {"output": _model_dump(message)}
            usage = getattr(response, "usage", None)
            usage_dump = _model_dump(usage)
            if usage_dump is not None:
                update_kwargs["usage_details"] = usage_dump
            generation.update(**update_kwargs)

        return message


async def structured_completion(
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    model: str | None = None,
    trace_name: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
) -> Any:
    """Native structured-output completion — returns a validated Pydantic model."""
    model_name = model or get_model()
    with observe(
        name=trace_name or "llm.structured_completion",
        as_type="generation",
        input={"messages": messages, "response_model": response_model.__name__, "model": model_name},
        model=model_name,
        metadata=trace_metadata,
    ) as generation:
        try:
            response = await client.responses.parse(
                model=model_name,
                input=messages,
                text_format=response_model,
            )
        except BadRequestError as exc:
            error_text = str(exc).lower()
            if "response_format" not in error_text or "invalid" not in error_text:
                if generation is not None:
                    generation.update(output=f"ERROR: {type(exc).__name__}: {exc}")
                raise

            try:
                response = await client.responses.create(
                    model=model_name,
                    input=messages,
                    text={"format": {"type": "json_object"}},
                )
            except Exception as fallback_exc:
                if generation is not None:
                    generation.update(output=f"ERROR: {type(fallback_exc).__name__}: {fallback_exc}")
                raise
        except Exception as exc:
            if generation is not None:
                generation.update(output=f"ERROR: {type(exc).__name__}: {exc}")
            raise

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            message = response.choices[0].message if getattr(response, "choices", None) else None
            parsed = getattr(message, "parsed", None) if message is not None else None
        if parsed is None:
            content = getattr(response, "output_text", None)
            if content is None:
                message = response.choices[0].message if getattr(response, "choices", None) else None
                content = getattr(message, "content", None) if message is not None else None
            if content is None:
                raise ValueError("Structured completion returned no parsable output")
            parsed = response_model.model_validate_json(content)

        if generation is not None:
            generation.update(output=_model_dump(parsed))

        return parsed
