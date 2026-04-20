from __future__ import annotations

from typing import Any

from engine.events import EventType, emit_event
from engine.sessions.hitl import HitlCallback
from engine.sessions.models import PendingToolCall


class HitlHandler:
    def __init__(self, callback: HitlCallback | None) -> None:
        self._callback = callback

    @property
    def enabled(self) -> bool:
        return self._callback is not None

    async def authorize(
        self,
        *,
        subagent_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        source: str,
        skip_tools: set[str] | None = None,
    ) -> tuple[bool, dict[str, Any], str | None]:
        if self._callback is None:
            return True, arguments, None

        if skip_tools is not None and tool_name in skip_tools:
            return True, arguments, None

        pending = PendingToolCall(
            tool_name=tool_name,
            arguments=arguments,
            tool_call_id=f"{subagent_id}:{tool_name}:{id(arguments)}",
            subagent_id=subagent_id,
            source=source,
        )

        emit_event(
            EventType.HITL_REQUIRED,
            subagent_id=subagent_id,
            tool=tool_name,
            arguments=arguments,
            source=source,
            tool_call_id=pending.tool_call_id,
        )

        response = await self._callback(pending)

        emit_event(
            EventType.HITL_RESPONSE,
            tool=tool_name,
            approved=response.approved,
            approval_scope=response.approval_scope,
            rejection_reason=response.rejection_reason,
        )

        if not response.approved:
            reason = response.rejection_reason or "rejected by human"
            return False, arguments, reason

        if response.modified_arguments is not None:
            return True, response.modified_arguments, None

        return True, arguments, None
