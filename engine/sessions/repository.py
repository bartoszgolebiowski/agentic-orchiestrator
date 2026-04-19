"""Session repository interface and SQLite implementation."""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from abc import ABC, abstractmethod

from engine.sessions.models import (
    ConversationTurn,
    HitlResponse,
    PendingToolCall,
    SessionData,
    SessionStatus,
)


class SessionRepository(ABC):
    """Abstract base for session persistence. Swap SQLite for Postgres, Redis, etc."""

    @abstractmethod
    async def create(self, session: SessionData) -> SessionData: ...

    @abstractmethod
    async def get(self, session_id: str) -> SessionData | None: ...

    @abstractmethod
    async def update(self, session: SessionData) -> SessionData: ...

    @abstractmethod
    async def list_sessions(self, limit: int = 50, offset: int = 0) -> list[SessionData]: ...

    @abstractmethod
    async def delete(self, session_id: str) -> bool: ...


class SQLiteSessionRepository(SessionRepository):
    """SQLite-backed session repository using the built-in sqlite3 module."""

    def __init__(self, db_path: str = "data/sessions.db") -> None:
        self._db_path = db_path
        self._ensure_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_tables(self) -> None:
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    query TEXT NOT NULL,
                    config_dir TEXT NOT NULL DEFAULT 'configs',
                    conversation_history TEXT NOT NULL DEFAULT '[]',
                    events TEXT NOT NULL DEFAULT '[]',
                    pending_tool_call TEXT,
                    result TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _row_to_session(self, row: sqlite3.Row) -> SessionData:
        return SessionData(
            id=row["id"],
            status=SessionStatus(row["status"]),
            query=row["query"],
            config_dir=row["config_dir"],
            conversation_history=[
                ConversationTurn(**t) for t in json.loads(row["conversation_history"])
            ],
            events=json.loads(row["events"]),
            pending_tool_call=(
                PendingToolCall(**json.loads(row["pending_tool_call"]))
                if row["pending_tool_call"]
                else None
            ),
            result=row["result"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _sync_create(self, session: SessionData) -> SessionData:
        conn = self._get_connection()
        try:
            conn.execute(
                """INSERT INTO sessions
                   (id, status, query, config_dir, conversation_history, events,
                    pending_tool_call, result, error, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session.id,
                    session.status.value,
                    session.query,
                    session.config_dir,
                    json.dumps([t.model_dump() for t in session.conversation_history]),
                    json.dumps(session.events),
                    (
                        json.dumps(session.pending_tool_call.model_dump())
                        if session.pending_tool_call
                        else None
                    ),
                    session.result,
                    session.error,
                    session.created_at,
                    session.updated_at,
                ),
            )
            conn.commit()
            return session
        finally:
            conn.close()

    def _sync_get(self, session_id: str) -> SessionData | None:
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            return self._row_to_session(row) if row else None
        finally:
            conn.close()

    def _sync_update(self, session: SessionData) -> SessionData:
        session.updated_at = time.time()
        conn = self._get_connection()
        try:
            conn.execute(
                """UPDATE sessions SET
                       status = ?, query = ?, config_dir = ?,
                       conversation_history = ?, events = ?,
                       pending_tool_call = ?, result = ?, error = ?,
                       updated_at = ?
                   WHERE id = ?""",
                (
                    session.status.value,
                    session.query,
                    session.config_dir,
                    json.dumps([t.model_dump() for t in session.conversation_history]),
                    json.dumps(session.events),
                    (
                        json.dumps(session.pending_tool_call.model_dump())
                        if session.pending_tool_call
                        else None
                    ),
                    session.result,
                    session.error,
                    session.updated_at,
                    session.id,
                ),
            )
            conn.commit()
            return session
        finally:
            conn.close()

    def _sync_list(self, limit: int, offset: int) -> list[SessionData]:
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [self._row_to_session(r) for r in rows]
        finally:
            conn.close()

    def _sync_delete(self, session_id: str) -> bool:
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    async def create(self, session: SessionData) -> SessionData:
        return await asyncio.to_thread(self._sync_create, session)

    async def get(self, session_id: str) -> SessionData | None:
        return await asyncio.to_thread(self._sync_get, session_id)

    async def update(self, session: SessionData) -> SessionData:
        return await asyncio.to_thread(self._sync_update, session)

    async def list_sessions(self, limit: int = 50, offset: int = 0) -> list[SessionData]:
        return await asyncio.to_thread(self._sync_list, limit, offset)

    async def delete(self, session_id: str) -> bool:
        return await asyncio.to_thread(self._sync_delete, session_id)
