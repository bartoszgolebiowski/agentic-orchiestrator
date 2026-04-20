from __future__ import annotations

from engine.sessions.entities import SessionEntity
from engine.sessions.models import SessionData


class SessionFactory:
    @staticmethod
    def create(query: str, config_dir: str = "configs") -> SessionEntity:
        return SessionEntity(data=SessionData(query=query, config_dir=config_dir))

    @staticmethod
    def from_data(data: SessionData) -> SessionEntity:
        return SessionEntity(data=data)
