from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


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


@dataclass
class PipelineState:
    """First-class pipeline tracker that replaces ad-hoc list-based completion tracking."""

    stages: list[str]
    _results: dict[str, StageResult] = field(default_factory=dict, init=False, repr=False)

    @property
    def is_empty(self) -> bool:
        return not self.stages

    @property
    def is_complete(self) -> bool:
        if self.is_empty:
            return True
        return all(
            self._results.get(s, StageResult(s, StageStatus.PENDING)).status == StageStatus.COMPLETED
            for s in self.stages
        )

    @property
    def next_stage(self) -> str | None:
        for s in self.stages:
            result = self._results.get(s)
            if result is None or result.status in (StageStatus.PENDING, StageStatus.FAILED):
                return s
        return None

    @property
    def missing_stages(self) -> list[str]:
        return [
            s for s in self.stages
            if s not in self._results or self._results[s].status != StageStatus.COMPLETED
        ]

    @property
    def completed_stages(self) -> list[str]:
        return [
            s for s in self.stages
            if s in self._results and self._results[s].status == StageStatus.COMPLETED
        ]

    def get_result(self, stage_id: str) -> StageResult | None:
        return self._results.get(stage_id)

    def is_stage_completed(self, stage_id: str) -> bool:
        result = self._results.get(stage_id)
        return result is not None and result.status == StageStatus.COMPLETED

    def mark_running(self, stage_id: str) -> None:
        self._results[stage_id] = StageResult(stage_id, StageStatus.RUNNING)

    def mark_completed(self, stage_id: str, output: str) -> None:
        self._results[stage_id] = StageResult(stage_id, StageStatus.COMPLETED, output=output)

    def mark_failed(self, stage_id: str, error: str) -> None:
        self._results[stage_id] = StageResult(stage_id, StageStatus.FAILED, error=error)

    def observe_result(self, stage_id: str, observation: str) -> None:
        """Observe an action result and update the pipeline stage status accordingly."""
        if stage_id not in set(self.stages):
            return
        if self.is_stage_completed(stage_id):
            return

        if is_failure_observation(observation):
            self.mark_failed(stage_id, observation)
            logger.warning("Pipeline stage '%s' reported a failure; not marking as complete.", stage_id)
        else:
            self.mark_completed(stage_id, observation)
            remaining = self.missing_stages
            logger.info(
                "Pipeline progress: completed '%s'. Remaining: %s",
                stage_id,
                remaining if remaining else "none (pipeline complete)",
            )

    def summary(self) -> dict[str, Any]:
        return {
            "stages": self.stages,
            "completed": self.completed_stages,
            "missing": self.missing_stages,
            "is_complete": self.is_complete,
            "results": {
                stage_id: {
                    "status": result.status.value,
                    "has_output": result.output is not None,
                    "has_error": result.error is not None,
                }
                for stage_id, result in self._results.items()
            },
        }


def is_failure_observation(observation: str) -> bool:
    lower = observation.lower()
    return any(lower.startswith(prefix) for prefix in _FAILURE_PREFIXES)
