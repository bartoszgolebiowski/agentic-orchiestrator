"""Typed runtime event schema and async event bus for streaming UI."""
from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class EventType(str, Enum):
    # lifecycle
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_ERROR = "run_error"

    # orchestrator
    ROUTING_STARTED = "routing_started"
    ROUTING_DECISION = "routing_decision"

    # enrichment
    ENRICHMENT_STARTED = "enrichment_started"
    ENRICHMENT_RESULT = "enrichment_result"

    # agent
    AGENT_STARTED = "agent_started"
    AGENT_STEP = "agent_step"
    AGENT_DELEGATION = "agent_delegation"
    AGENT_FINISHED = "agent_finished"

    # subagent
    SUBAGENT_STARTED = "subagent_started"
    SUBAGENT_STEP = "subagent_step"
    SUBAGENT_FINISHED = "subagent_finished"

    # tool
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_FINISHED = "tool_call_finished"

    # human-in-the-loop
    HITL_REQUIRED = "hitl_required"
    HITL_RESPONSE = "hitl_response"

    # LLM streaming tokens
    TOKEN_DELTA = "token_delta"

    # warnings / info
    WARNING = "warning"


@dataclass
class StreamEvent:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d


# ── Async event bus ──────────────────────────────────────────────────────

_event_queue: ContextVar[asyncio.Queue[StreamEvent | None] | None] = ContextVar(
    "_event_queue", default=None,
)


def get_event_queue() -> asyncio.Queue[StreamEvent | None] | None:
    return _event_queue.get(None)


def set_event_queue(q: asyncio.Queue[StreamEvent | None] | None) -> None:
    _event_queue.set(q)


def emit(event: StreamEvent) -> None:
    """Non-blocking publish.  No-op when no queue is active (CLI mode)."""
    q = _event_queue.get(None)
    if q is not None:
        q.put_nowait(event)


def emit_event(event_type: EventType, **data: Any) -> None:
    emit(StreamEvent(type=event_type, data=data))
