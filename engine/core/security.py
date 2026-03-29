from __future__ import annotations

from engine.core.models import RoleType


class BoundaryViolationError(Exception):
    pass


def enforce_agent_boundary(caller_role: RoleType, callee_id: str, allowed_ids: list[str]) -> None:
    if caller_role == RoleType.AGENT:
        if callee_id not in allowed_ids:
            raise BoundaryViolationError(
                f"Agent attempted to call '{callee_id}' which is not in its allowed "
                f"subagents: {allowed_ids}"
            )

    elif caller_role == RoleType.SUBAGENT:
        if callee_id not in allowed_ids:
            raise BoundaryViolationError(
                f"Subagent attempted to call '{callee_id}' which is not in its allowed "
                f"tools: {allowed_ids}"
            )


def enforce_no_direct_tool_access(caller_role: RoleType, tool_id: str) -> None:
    if caller_role == RoleType.AGENT:
        raise BoundaryViolationError(
            f"Agent is forbidden from directly calling tool '{tool_id}'. "
            f"Agents may only delegate to subagents."
        )


def enforce_no_subagent_calling_subagent(caller_role: RoleType, target_id: str) -> None:
    if caller_role == RoleType.SUBAGENT:
        raise BoundaryViolationError(
            f"Subagent is forbidden from calling another subagent '{target_id}'. "
            f"Subagents may only call tools."
        )
