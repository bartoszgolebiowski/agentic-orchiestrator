"""Typed runtime event schema for streaming UI."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
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
        payload = asdict(self)
        payload["type"] = self.type.value
        return payload
