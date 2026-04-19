"""FastAPI server that exposes the orchestrator as an SSE streaming endpoint with session management and HITL."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import engine.tools  # noqa: F401 — trigger tool registration

from engine.core.events import EventType, StreamEvent, emit_event, set_event_queue
from engine.core.hitl import HitlManager
from engine.core.storage import (
    ConversationTurn,
    HitlResponse,
    PendingToolCall,
    SessionData,
    SessionRepository,
    SessionStatus,
    SQLiteSessionRepository,
)
from engine.core.graph import validate_config_graph
from engine.core.loader import load_engine_config
from engine.main import build_mcp_manager, main as engine_main

logger = logging.getLogger(__name__)

load_dotenv()

CONFIG_DIR = Path("configs")


@asynccontextmanager
async def lifespan(app: FastAPI):
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


app = FastAPI(title="Agentic Orchestrator UI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_DIR = Path(__file__).resolve().parent / "ui"

# ── Singletons ───────────────────────────────────────────────────────────

repository: SessionRepository = SQLiteSessionRepository()
hitl_manager = HitlManager()


# ── Request / Response Models ────────────────────────────────────────────


class ChatRequest(BaseModel):
    query: str
    config_dir: str = "configs"


class HitlResponseRequest(BaseModel):
    approved: bool
    modified_arguments: dict | None = None
    rejection_reason: str | None = None


class SessionSummary(BaseModel):
    id: str
    status: str
    query: str
    created_at: float
    updated_at: float


# ── SSE helpers ──────────────────────────────────────────────────────────


async def _event_generator(
    queue: asyncio.Queue[StreamEvent | None],
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted lines from the event queue until a sentinel arrives."""
    while True:
        event = await queue.get()
        if event is None:
            break
        payload = json.dumps(event.to_dict(), ensure_ascii=False)
        yield f"data: {payload}\n\n"


# ── Engine runner ────────────────────────────────────────────────────────


async def _run_engine_with_session(
    session: SessionData,
    queue: asyncio.Queue[StreamEvent | None],
    app_state: object,
) -> None:
    """Run the orchestrator engine for a session, publishing events and persisting state."""
    try:
        set_event_queue(queue)

        # Build HITL callback that pauses session and emits events
        async def _on_pause(pending: PendingToolCall) -> None:
            session.status = SessionStatus.PAUSED_FOR_HITL
            session.pending_tool_call = pending
            await repository.update(session)

        hitl_callback = hitl_manager.build_callback(session.id, on_pause=_on_pause)

        result = await engine_main(
            session.query,
            session.config_dir,
            hitl_callback=hitl_callback,
            engine_config=getattr(app_state, "engine_config", None),
            mcp_manager=getattr(app_state, "mcp_manager", None),
            owns_mcp_manager=False,
        )

        session.result = result
        session.status = SessionStatus.COMPLETED
        session.conversation_history.append(
            ConversationTurn(role="assistant", content=result)
        )
        session.pending_tool_call = None
        await repository.update(session)

    except Exception as exc:
        emit_event(EventType.RUN_ERROR, error=f"{type(exc).__name__}: {exc}")
        session.status = SessionStatus.FAILED
        session.error = f"{type(exc).__name__}: {exc}"
        session.pending_tool_call = None
        await repository.update(session)
    finally:
        queue.put_nowait(None)  # sentinel
        set_event_queue(None)
        hitl_manager.unregister_session(session.id)


# ── API Endpoints ────────────────────────────────────────────────────────


@app.post("/api/session")
async def create_session(request: ChatRequest, app_request: Request) -> StreamingResponse:
    """Start a new session. Returns SSE stream of events."""
    if request.config_dir != str(CONFIG_DIR):
        raise HTTPException(status_code=400, detail=f"config_dir is fixed to {CONFIG_DIR.as_posix()} in server mode")

    app_state = getattr(app_request.app.state, "engine_config", None)
    if app_state is None:
        raise HTTPException(status_code=503, detail="Orchestrator runtime is not ready")

    session = SessionData(
        query=request.query,
        config_dir=request.config_dir,
        conversation_history=[ConversationTurn(role="user", content=request.query)],
    )
    await repository.create(session)

    hitl_manager.register_session(session.id)

    queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()

    asyncio.create_task(
        _run_engine_with_session(
            session,
            queue,
            app_request.app.state,
        )
    )

    return StreamingResponse(
        _event_generator(queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session.id,
        },
    )


@app.get("/api/session/{session_id}")
async def get_session(session_id: str) -> SessionData:
    """Get the current state of a session."""
    session = await repository.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.get("/api/sessions")
async def list_sessions(limit: int = 50, offset: int = 0) -> list[SessionSummary]:
    """List all sessions (most recent first)."""
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


@app.post("/api/session/{session_id}/respond")
async def submit_hitl_response(session_id: str, request: HitlResponseRequest) -> dict:
    """Submit a human-in-the-loop response for a paused session."""
    session = await repository.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != SessionStatus.PAUSED_FOR_HITL:
        raise HTTPException(status_code=409, detail=f"Session is not paused for HITL (status: {session.status.value})")

    response = HitlResponse(
        approved=request.approved,
        modified_arguments=request.modified_arguments,
        rejection_reason=request.rejection_reason,
    )

    session.status = SessionStatus.RUNNING
    session.pending_tool_call = None
    await repository.update(session)

    await hitl_manager.submit_response(session_id, response)

    return {"status": "ok", "session_id": session_id}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Delete a session from the repository."""
    deleted = await repository.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


# ── Legacy endpoint (backward compatible) ────────────────────────────────


@app.post("/api/chat")
async def chat(request: ChatRequest, app_request: Request) -> StreamingResponse:
    """Legacy endpoint: starts a session and returns SSE stream."""
    return await create_session(request, app_request)


# ── Static UI ────────────────────────────────────────────────────────────

if UI_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/")
async def index() -> HTMLResponse:
    index_path = UI_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not found — run from project root</h1>", status_code=404)
