"""Human-in-the-loop manager for tool confirmation via async queues."""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from engine.sessions.models import HitlResponse, PendingToolCall

logger = logging.getLogger(__name__)

# Type alias: async callback invoked when a tool needs human confirmation.
HitlCallback = Callable[[PendingToolCall], Awaitable[HitlResponse]]


class HitlManager:
    """Manages per-session asyncio queues for human-in-the-loop confirmations."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[HitlResponse]] = {}

    def register_session(self, session_id: str) -> None:
        if session_id not in self._queues:
            self._queues[session_id] = asyncio.Queue(maxsize=1)

    def unregister_session(self, session_id: str) -> None:
        self._queues.pop(session_id, None)

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
        await queue.put(response)
        logger.info(f"HITL: human response submitted for session {session_id}: approved={response.approved}")

    def build_callback(
        self,
        session_id: str,
        on_pause: Callable[[PendingToolCall], Awaitable[None]] | None = None,
    ) -> HitlCallback:
        """Create an HitlCallback bound to a specific session."""

        async def _callback(pending: PendingToolCall) -> HitlResponse:
            if on_pause is not None:
                await on_pause(pending)
            return await self.wait_for_human(session_id)

        return _callback
