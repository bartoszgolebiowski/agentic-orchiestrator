from __future__ import annotations

from pydantic import BaseModel, Field


class CompletionOptions(BaseModel):
    model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    trace_name: str | None = None
    trace_metadata: dict[str, object] | None = None


class UsageDetails(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
