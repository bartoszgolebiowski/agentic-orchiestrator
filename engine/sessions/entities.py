from __future__ import annotations

from dataclasses import dataclass

from engine.sessions.models import SessionData, SessionStatus


@dataclass
class SessionEntity:
    """Behavior-focused session entity."""

    data: SessionData

    def run(self) -> None:
        self.data.status = SessionStatus.RUNNING

    def pause(self) -> None:
        self.data.status = SessionStatus.PAUSED_FOR_HITL

    def complete(self, result: str) -> None:
        self.data.status = SessionStatus.COMPLETED
        self.data.result = result

    def fail(self, error: str) -> None:
        self.data.status = SessionStatus.FAILED
        self.data.error = error

    def get_state(self) -> SessionData:
        return self.data
