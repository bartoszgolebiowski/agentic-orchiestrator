from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


_FAILURE_PREFIXES = (
    "error:",
    "i'm unable",
    "i am unable",
    "i cannot",
    "could not",
    "failed to",
)


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StageResult:
    stage_id: str
    status: StageStatus
    output: str | None = None
    error: str | None = None


def is_failure_observation(observation: str) -> bool:
    stripped = observation.strip()
    if not stripped:
        return True

    try:
        parsed = json.loads(stripped)
    except Exception:
        parsed = None

    if isinstance(parsed, dict) and parsed.get("status") == "error":
        return True

    lower = stripped.lower()
    return any(lower.startswith(prefix) for prefix in _FAILURE_PREFIXES)
