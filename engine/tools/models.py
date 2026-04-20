from __future__ import annotations

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    id: str
    description: str
    parameters: dict[str, dict[str, object]] = Field(default_factory=dict)


class ToolParameterModel(BaseModel):
    type: str
    description: str
    required: bool = True
