"""Human-in-the-loop manager for tool confirmation via async queues."""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from engine.sessions.models import HitlApprovalScope, HitlResponse, PendingToolCall

logger = logging.getLogger(__name__)

# Type alias: async callback invoked when a tool needs human confirmation.
HitlCallback = Callable[[PendingToolCall], Awaitable[HitlResponse]]


class HitlManager:
    """Manages per-session asyncio queues for human-in-the-loop confirmations."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[HitlResponse]] = {}
        self._approval_scopes: dict[str, HitlApprovalScope] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}

    def register_session(
        self,
        session_id: str,
        approval_scope: HitlApprovalScope = HitlApprovalScope.ONCE,
    ) -> None:
        if session_id not in self._queues:
            self._queues[session_id] = asyncio.Queue(maxsize=1)
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        self._approval_scopes[session_id] = approval_scope

    def unregister_session(self, session_id: str) -> None:
        self._queues.pop(session_id, None)
        self._approval_scopes.pop(session_id, None)
        self._session_locks.pop(session_id, None)

    def is_session_auto_approved(self, session_id: str) -> bool:
        return self._approval_scopes.get(session_id) == HitlApprovalScope.SESSION

    def set_approval_scope(self, session_id: str, approval_scope: HitlApprovalScope) -> None:
        if session_id in self._queues:
            self._approval_scopes[session_id] = approval_scope

    async def wait_for_human(self, session_id: str) -> HitlResponse:
        """Called inside the engine task to block until the human responds."""
        queue = self._queues.get(session_id)
        if queue is None:
            raise ValueError(f"No HITL queue registered for session {session_id}")
        logger.info(f"HITL: waiting for human response on session {session_id}")
        return await queue.get()

    async def submit_response(self, session_id: str, response: HitlResponse) -> None:
        """Called by the REST endpoint when the human submits a decision."""
        queue = self._queues.get(session_id)
        if queue is None:
            raise ValueError(f"No HITL queue registered for session {session_id}")
        if response.approved and response.approval_scope == HitlApprovalScope.SESSION:
            self._approval_scopes[session_id] = HitlApprovalScope.SESSION
        if queue.full():
            raise ValueError(f"HITL response already queued for session {session_id}")
        try:
            queue.put_nowait(response)
        except asyncio.QueueFull as exc:
            raise ValueError(f"HITL response already queued for session {session_id}") from exc
        logger.info(f"HITL: human response submitted for session {session_id}: approved={response.approved}")

    def build_callback(
        self,
        session_id: str,
        on_pause: Callable[[PendingToolCall], Awaitable[None]] | None = None,
    ) -> HitlCallback:
        """Create an HitlCallback bound to a specific session."""

        async def _callback(pending: PendingToolCall) -> HitlResponse:
            lock = self._session_locks.get(session_id)
            if lock is None:
                raise ValueError(f"No HITL lock registered for session {session_id}")

            # Ensure only one HITL action can be pending on the same session at a time.
            async with lock:
                if self.is_session_auto_approved(session_id):
                    return HitlResponse(
                        approved=True,
                        approval_scope=HitlApprovalScope.SESSION,
                    )
                if on_pause is not None:
                    await on_pause(pending)
                response = await self.wait_for_human(session_id)
                if response.approved and response.approval_scope == HitlApprovalScope.SESSION:
                    self._approval_scopes[session_id] = HitlApprovalScope.SESSION
                return response

        return _callback
