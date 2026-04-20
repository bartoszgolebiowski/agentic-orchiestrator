from __future__ import annotations

from dataclasses import dataclass

from engine.tools.projection.models import ProjectionSpec
from engine.tools.projection.projector import (
    project_tool_result,
    project_tool_result_strict,
    validate_projection_spec,
)


@dataclass
class Projector:
    """Projection domain entity with safe and strict projection modes."""

    def validate(self, spec: ProjectionSpec) -> None:
        validate_projection_spec(spec)

    def project(self, raw: str, projection: ProjectionSpec) -> str:
        return project_tool_result(raw, projection)

    def project_strict(self, raw: str, projection: ProjectionSpec) -> str:
        return project_tool_result_strict(raw, projection)
