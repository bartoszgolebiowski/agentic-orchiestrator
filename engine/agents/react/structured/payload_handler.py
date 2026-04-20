from __future__ import annotations

from typing import Any

from engine.config.contracts import normalize_handoff_payload
from engine.pipeline import is_failure_observation


class PayloadHandler:
    def __init__(self, initial_payload: Any | None = None) -> None:
        self._handoff_payload = initial_payload

    @property
    def handoff_payload(self) -> Any | None:
        return self._handoff_payload

    def action_arguments(self, *, task: str, explicit_input_json: dict[str, Any] | None) -> dict[str, Any]:
        action_args: dict[str, Any] = {"task": task}
        if explicit_input_json is not None:
            action_args["input_json"] = explicit_input_json
        elif self._handoff_payload is not None:
            action_args["input_json"] = self._handoff_payload
        return action_args

    def observe(self, observation: str) -> None:
        if observation.strip() and not is_failure_observation(observation):
            self._handoff_payload = normalize_handoff_payload(observation)
