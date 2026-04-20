from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from engine.pipeline.models import StageResult, StageStatus, is_failure_observation

logger = logging.getLogger(__name__)


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
        for stage_id in self.stages:
            result = self._results.get(stage_id)
            if result is None or result.status in (StageStatus.PENDING, StageStatus.FAILED):
                return stage_id
        return None

    @property
    def missing_stages(self) -> list[str]:
        return [
            stage_id for stage_id in self.stages
            if stage_id not in self._results or self._results[stage_id].status != StageStatus.COMPLETED
        ]

    @property
    def completed_stages(self) -> list[str]:
        return [
            stage_id for stage_id in self.stages
            if stage_id in self._results and self._results[stage_id].status == StageStatus.COMPLETED
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
        """Observe an action result and update pipeline stage status."""
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


@dataclass
class PipelineEntity:
    """Domain entity wrapping an identified pipeline state."""

    pipeline_id: str
    state: PipelineState

    @property
    def is_complete(self) -> bool:
        return self.state.is_complete

    def observe_result(self, stage_id: str, observation: str) -> None:
        self.state.observe_result(stage_id, observation)

    def summary(self) -> dict[str, Any]:
        summary = self.state.summary()
        summary["pipeline_id"] = self.pipeline_id
        return summary
