from __future__ import annotations

import logging

from engine.agents.react.base import StepResult
from engine.pipeline import PipelineState

logger = logging.getLogger(__name__)


class PipelineEnforcer:
    def __init__(self, pipeline: PipelineState) -> None:
        self._pipeline = pipeline

    @property
    def pipeline(self) -> PipelineState:
        return self._pipeline

    def enforce_before_final(
        self,
        *,
        thought: str,
        messages: list[dict[str, str]],
        steps: list[StepResult],
    ) -> bool:
        if self._pipeline.is_empty or self._pipeline.is_complete:
            return False

        missing = self._pipeline.missing_stages
        next_stage = self._pipeline.next_stage
        if not missing or not next_stage:
            return False

        logger.warning(
            "  Pipeline enforcement: LLM tried to finish but %s required stages remain: %s. "
            "Forcing delegation to '%s'.",
            len(missing),
            missing,
            next_stage,
        )

        messages.append({"role": "assistant", "content": f"Thought: {thought}"})
        messages.append({
            "role": "user",
            "content": (
                f"PIPELINE VIOLATION: You cannot produce a final answer yet. "
                f"The following mandatory stages have NOT been executed: {missing}. "
                f"You MUST delegate to '{next_stage}' now. "
                f"Continue the pipeline from where you left off."
            ),
        })

        steps.append(StepResult(
            thought=thought,
            action=None,
            observation=f"Pipeline enforcement: redirected to {next_stage}",
        ))
        return True
