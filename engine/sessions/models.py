"""Session domain models."""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    RUNNING = "running"
    PAUSED_FOR_HITL = "paused_for_hitl"
    COMPLETED = "completed"
    FAILED = "failed"


class PendingToolCall(BaseModel):
    """Captures a tool invocation that is awaiting human confirmation."""
    tool_name: str
    arguments: dict[str, Any]
    tool_call_id: str
    subagent_id: str | None = None
    source: str = "local"  # "local" or "mcp"


class HitlResponse(BaseModel):
    """Human response to a pending tool call."""
    approved: bool
    modified_arguments: dict[str, Any] | None = None
    rejection_reason: str | None = None


class ConversationTurn(BaseModel):
    """A single user ↔ assistant exchange persisted for multi-message context."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = Field(default_factory=time.time)


class SessionData(BaseModel):
    """Full session state persisted to the repository."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    status: SessionStatus = SessionStatus.RUNNING
    query: str
    config_dir: str = "configs"
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    pending_tool_call: PendingToolCall | None = None
    result: str | None = None
    error: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
