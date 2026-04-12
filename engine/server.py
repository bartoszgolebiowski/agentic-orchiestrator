"""FastAPI server that exposes the orchestrator as an SSE streaming endpoint."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import engine.tools  # noqa: F401 — trigger tool registration

from engine.core.events import EventType, StreamEvent, emit_event, set_event_queue
from engine.main import main as engine_main

logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Agentic Orchestrator UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_DIR = Path(__file__).resolve().parent / "ui"


class ChatRequest(BaseModel):
    query: str
    config_dir: str = "configs"


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


async def _run_engine(query: str, config_dir: str, queue: asyncio.Queue[StreamEvent | None]) -> None:
    """Run the orchestrator engine, publishing events, then send the sentinel."""
    try:
        set_event_queue(queue)
        await engine_main(query, config_dir)
    except Exception as exc:
        emit_event(EventType.RUN_ERROR, error=f"{type(exc).__name__}: {exc}")
    finally:
        queue.put_nowait(None)  # sentinel
        set_event_queue(None)


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()

    # Launch engine in background; events stream to the client as SSE.
    asyncio.create_task(_run_engine(request.query, request.config_dir, queue))

    return StreamingResponse(
        _event_generator(queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# Serve static UI files
if UI_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/")
async def index() -> HTMLResponse:
    index_path = UI_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not found — run from project root</h1>", status_code=404)
