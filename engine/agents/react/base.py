from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel


class MaxStepsExceededError(Exception):
    pass


@dataclass
class Action:
    name: str
    arguments: dict[str, Any]


@dataclass
class StepResult:
    thought: str | None
    action: Action | None
    observation: str | None
    is_final: bool = False
    final_answer: str | None = None
    llm_usage: dict[str, int] | None = None
    llm_usage_details: dict[str, Any] | None = None


ActionHandler = Callable[[str, dict[str, Any]], Awaitable[str]]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: to_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def structured_payload_to_mapping(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return {key: to_jsonable(item) for key, item in payload.items()}
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    if hasattr(payload, "__dict__"):
        return {
            key: to_jsonable(item)
            for key, item in vars(payload).items()
            if not key.startswith("_")
        }
    raise TypeError(f"Unsupported structured payload type: {type(payload)!r}")


def init_user_message(task: str, input_json: Any | None = None) -> str:
    if input_json is None:
        return task
    return json.dumps(
        {"task": task, "input_json": input_json},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
