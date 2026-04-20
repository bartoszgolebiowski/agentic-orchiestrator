from __future__ import annotations

from engine.pipeline.entities import PipelineEntity, PipelineState


class PipelineFactory:
    @staticmethod
    def create(stages: list[str], pipeline_id: str = "pipeline") -> PipelineEntity:
        return PipelineEntity(pipeline_id=pipeline_id, state=PipelineState(list(stages)))

    @staticmethod
    def from_state(state: PipelineState, pipeline_id: str = "pipeline") -> PipelineEntity:
        return PipelineEntity(pipeline_id=pipeline_id, state=state)
