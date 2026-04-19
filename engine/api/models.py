from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    config_dir: str = "configs"


class HitlResponseRequest(BaseModel):
    approved: bool
    modified_arguments: dict | None = None
    rejection_reason: str | None = None


class SessionSummary(BaseModel):
    id: str
    status: str
    query: str
    created_at: float
    updated_at: float
