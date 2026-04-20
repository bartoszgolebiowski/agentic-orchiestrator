from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from engine.llm.completion import ChatCompletionService, StructuredCompletionService


@dataclass
class LLMClientEntity:
    """Domain entity that composes completion services over an AsyncOpenAI client."""

    client: AsyncOpenAI
    default_model: str | None = None
    chat_service: ChatCompletionService = field(default_factory=ChatCompletionService)
    structured_service: StructuredCompletionService = field(default_factory=StructuredCompletionService)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        trace_name: str | None = None,
        trace_metadata: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> Any:
        return await self.chat_service.complete(
            client=self.client,
            messages=messages,
            tools=tools,
            model=model or self.default_model,
            trace_name=trace_name,
            trace_metadata=trace_metadata,
            temperature=temperature,
        )

    async def structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        model: str | None = None,
        trace_name: str | None = None,
        trace_metadata: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> Any:
        return await self.structured_service.complete(
            client=self.client,
            messages=messages,
            response_model=response_model,
            model=model or self.default_model,
            trace_name=trace_name,
            trace_metadata=trace_metadata,
            temperature=temperature,
        )
