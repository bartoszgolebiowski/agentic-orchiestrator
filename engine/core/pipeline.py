"""Backward-compat shim — use engine.pipeline directly."""
from engine.pipeline import (
    PipelineState,
    StageResult,
    StageStatus,
    is_failure_observation,
)

__all__ = ["PipelineState", "StageResult", "StageStatus", "is_failure_observation"]
