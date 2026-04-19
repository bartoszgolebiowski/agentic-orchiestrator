"""Backward-compat shim — use engine.security directly."""
from engine.security import (
    BoundaryViolationError,
    enforce_agent_boundary,
    enforce_no_direct_tool_access,
    enforce_no_subagent_calling_subagent,
)

__all__ = [
    "BoundaryViolationError",
    "enforce_agent_boundary",
    "enforce_no_direct_tool_access",
    "enforce_no_subagent_calling_subagent",
]
