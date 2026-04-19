"""Business logic for the API layer — session lifecycle, SSE streaming, HITL."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from engine.config.graph import validate_config_graph
from engine.config.loader import load_engine_config
from engine.events import EventType, StreamEvent, emit_event, set_event_queue
from engine.main import Engine, build_mcp_manager
from engine.sessions.hitl import HitlManager
from engine.sessions.models import (
    ConversationTurn,
    HitlApprovalScope,
    HitlResponse,
    PendingToolCall,
    SessionData,
    SessionStatus,
)
from engine.sessions.repository import SessionRepository, SQLiteSessionRepository
from engine.api.models import HitlResponseRequest, SessionSummary

logger = logging.getLogger(__name__)

CONFIG_DIR = Path("configs")

# ── Singletons ────────────────────────────────────────────────────────────

repository: SessionRepository = SQLiteSessionRepository()
hitl_manager = HitlManager()


# ── App lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    engine_config = load_engine_config(CONFIG_DIR)

    issues = validate_config_graph(engine_config)
    for issue in issues:
        if issue.level == "error":
            logger.error("Config graph error: %s", issue.message)
        else:
            logger.warning("Config graph warning: %s", issue.message)

    errors = [issue for issue in issues if issue.level == "error"]
    if errors:
        raise ValueError(
            f"Config graph has {len(errors)} error(s); fix them before running the orchestrator"
        )

    mcp_manager = build_mcp_manager(engine_config)
    if mcp_manager is not None:
        await mcp_manager.warmup()

    app.state.engine_config = engine_config
    app.state.mcp_manager = mcp_manager

    try:
        yield
    finally:
        if mcp_manager is not None:
            await mcp_manager.aclose()
        app.state.engine_config = None
        app.state.mcp_manager = None


# ── SSE helpers ───────────────────────────────────────────────────────────

async def _event_generator(
    queue: asyncio.Queue[StreamEvent | None],
) -> AsyncGenerator[str, None]:
    while True:
        event = await queue.get()
        if event is None:
            break
        payload = json.dumps(event.to_dict(), ensure_ascii=False)
        yield f"data: {payload}\n\n"


async def _run_engine_with_session(
    session: SessionData,
    queue: asyncio.Queue[StreamEvent | None],
    app_state: object,
) -> None:
    try:
        set_event_queue(queue)

        async def _on_pause(pending: PendingToolCall) -> None:
            session.status = SessionStatus.PAUSED_FOR_HITL
            session.pending_tool_call = pending
            await repository.update(session)

        hitl_callback = hitl_manager.build_callback(session.id, on_pause=_on_pause)

        engine = Engine(
            config_dir=session.config_dir,
            engine_config=getattr(app_state, "engine_config", None),
            mcp_manager=getattr(app_state, "mcp_manager", None),
            owns_mcp_manager=False,
        )
        result = await engine.run(query=session.query, hitl_callback=hitl_callback)

        session.result = result
        session.status = SessionStatus.COMPLETED
        if hitl_manager.is_session_auto_approved(session.id):
            session.hitl_approval_scope = HitlApprovalScope.SESSION
        session.conversation_history.append(
            ConversationTurn(role="assistant", content=result)
        )
        session.pending_tool_call = None
        await repository.update(session)

    except Exception as exc:
        emit_event(EventType.RUN_ERROR, error=f"{type(exc).__name__}: {exc}")
        session.status = SessionStatus.FAILED
        session.error = f"{type(exc).__name__}: {exc}"
        if hitl_manager.is_session_auto_approved(session.id):
            session.hitl_approval_scope = HitlApprovalScope.SESSION
        session.pending_tool_call = None
        await repository.update(session)
    finally:
        queue.put_nowait(None)
        set_event_queue(None)
        hitl_manager.unregister_session(session.id)


# ── Service functions ─────────────────────────────────────────────────────

async def start_session(query: str, config_dir: str, app_state: object) -> StreamingResponse:
    if config_dir != str(CONFIG_DIR):
        raise HTTPException(
            status_code=400,
            detail=f"config_dir is fixed to {CONFIG_DIR.as_posix()} in server mode",
        )
    if getattr(app_state, "engine_config", None) is None:
        raise HTTPException(status_code=503, detail="Orchestrator runtime is not ready")

    session = SessionData(
        query=query,
        config_dir=config_dir,
        conversation_history=[ConversationTurn(role="user", content=query)],
    )
    await repository.create(session)
    hitl_manager.register_session(session.id, session.hitl_approval_scope)

    queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
    asyncio.create_task(_run_engine_with_session(session, queue, app_state))

    return StreamingResponse(
        _event_generator(queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session.id,
        },
    )


async def get_session(session_id: str) -> SessionData:
    session = await repository.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


async def list_sessions(limit: int, offset: int) -> list[SessionSummary]:
    sessions = await repository.list_sessions(limit=limit, offset=offset)
    return [
        SessionSummary(
            id=s.id,
            status=s.status.value,
            query=s.query,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in sessions
    ]


async def submit_hitl_response(session_id: str, request: HitlResponseRequest) -> dict:
    session = await repository.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != SessionStatus.PAUSED_FOR_HITL:
        raise HTTPException(
            status_code=409,
            detail=f"Session is not paused for HITL (status: {session.status.value})",
        )

    response = HitlResponse(
        approved=request.approved,
        approval_scope=request.approval_scope,
        modified_arguments=request.modified_arguments,
        rejection_reason=request.rejection_reason,
    )
    if response.approved and response.approval_scope == HitlApprovalScope.SESSION:
        session.hitl_approval_scope = HitlApprovalScope.SESSION
    session.status = SessionStatus.RUNNING
    session.pending_tool_call = None
    await repository.update(session)
    try:
        await hitl_manager.submit_response(session_id, response)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {"status": "ok", "session_id": session_id}


async def delete_session(session_id: str) -> dict:
    deleted = await repository.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}
