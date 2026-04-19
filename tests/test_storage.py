"""Tests for session storage repository pattern."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time

import pytest

from engine.core.storage import (
    ConversationTurn,
    HitlApprovalScope,
    HitlResponse,
    PendingToolCall,
    SessionData,
    SessionStatus,
    SQLiteSessionRepository,
)


@pytest.fixture()
def repo(tmp_path):
    db_path = str(tmp_path / "test_sessions.db")
    return SQLiteSessionRepository(db_path=db_path)


@pytest.mark.asyncio
async def test_create_and_get_session(repo):
    session = SessionData(
        query="What is 2+2?",
        config_dir="configs",
        conversation_history=[ConversationTurn(role="user", content="What is 2+2?")],
    )
    created = await repo.create(session)
    assert created.id == session.id

    loaded = await repo.get(session.id)
    assert loaded is not None
    assert loaded.id == session.id
    assert loaded.query == "What is 2+2?"
    assert loaded.status == SessionStatus.RUNNING
    assert loaded.hitl_approval_scope == HitlApprovalScope.ONCE
    assert len(loaded.conversation_history) == 1
    assert loaded.conversation_history[0].role == "user"
    assert loaded.conversation_history[0].content == "What is 2+2?"


@pytest.mark.asyncio
async def test_get_nonexistent_returns_none(repo):
    result = await repo.get("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_update_session(repo):
    session = SessionData(query="test query")
    await repo.create(session)

    session.status = SessionStatus.COMPLETED
    session.result = "The answer is 4"
    session.conversation_history.append(
        ConversationTurn(role="assistant", content="The answer is 4")
    )
    await repo.update(session)

    loaded = await repo.get(session.id)
    assert loaded is not None
    assert loaded.status == SessionStatus.COMPLETED
    assert loaded.result == "The answer is 4"
    assert len(loaded.conversation_history) == 1
    assert loaded.conversation_history[0].content == "The answer is 4"


@pytest.mark.asyncio
async def test_update_with_pending_tool_call(repo):
    session = SessionData(query="run a tool")
    await repo.create(session)

    pending = PendingToolCall(
        tool_name="trello_create_card",
        arguments={"title": "New card", "list_id": "abc123"},
        tool_call_id="tc-001",
        subagent_id="trello_publisher",
        source="mcp",
    )
    session.status = SessionStatus.PAUSED_FOR_HITL
    session.pending_tool_call = pending
    await repo.update(session)

    loaded = await repo.get(session.id)
    assert loaded is not None
    assert loaded.status == SessionStatus.PAUSED_FOR_HITL
    assert loaded.pending_tool_call is not None
    assert loaded.pending_tool_call.tool_name == "trello_create_card"
    assert loaded.pending_tool_call.arguments == {"title": "New card", "list_id": "abc123"}
    assert loaded.pending_tool_call.tool_call_id == "tc-001"
    assert loaded.pending_tool_call.subagent_id == "trello_publisher"
    assert loaded.pending_tool_call.source == "mcp"


@pytest.mark.asyncio
async def test_clear_pending_tool_call(repo):
    session = SessionData(query="run a tool")
    session.pending_tool_call = PendingToolCall(
        tool_name="some_tool",
        arguments={},
        tool_call_id="tc-x",
    )
    await repo.create(session)

    session.pending_tool_call = None
    session.status = SessionStatus.RUNNING
    await repo.update(session)

    loaded = await repo.get(session.id)
    assert loaded is not None
    assert loaded.pending_tool_call is None


@pytest.mark.asyncio
async def test_list_sessions_ordered_by_creation(repo):
    for i in range(5):
        s = SessionData(query=f"query {i}")
        s.created_at = time.time() + i
        await repo.create(s)

    sessions = await repo.list_sessions(limit=3)
    assert len(sessions) == 3
    # Most recent first
    assert sessions[0].query == "query 4"
    assert sessions[1].query == "query 3"
    assert sessions[2].query == "query 2"


@pytest.mark.asyncio
async def test_list_sessions_with_offset(repo):
    for i in range(5):
        s = SessionData(query=f"query {i}")
        s.created_at = time.time() + i
        await repo.create(s)

    sessions = await repo.list_sessions(limit=2, offset=3)
    assert len(sessions) == 2
    assert sessions[0].query == "query 1"
    assert sessions[1].query == "query 0"


@pytest.mark.asyncio
async def test_delete_session(repo):
    session = SessionData(query="to be deleted")
    await repo.create(session)

    deleted = await repo.delete(session.id)
    assert deleted is True

    loaded = await repo.get(session.id)
    assert loaded is None


@pytest.mark.asyncio
async def test_delete_nonexistent_returns_false(repo):
    deleted = await repo.delete("does-not-exist")
    assert deleted is False


@pytest.mark.asyncio
async def test_events_serialization(repo):
    session = SessionData(query="test events")
    session.events = [
        {"type": "run_started", "data": {"query": "test"}, "timestamp": 1234567890.0},
        {"type": "routing_decision", "data": {"agent_id": "math_agent"}, "timestamp": 1234567891.0},
    ]
    await repo.create(session)

    loaded = await repo.get(session.id)
    assert loaded is not None
    assert len(loaded.events) == 2
    assert loaded.events[0]["type"] == "run_started"
    assert loaded.events[1]["data"]["agent_id"] == "math_agent"


@pytest.mark.asyncio
async def test_complex_round_trip(repo):
    """Verify that a session with nested tool calls and conversation history
    round-trips through SQLite without data loss."""
    session = SessionData(
        query="Create a Trello card for task X",
        config_dir="configs",
        conversation_history=[
            ConversationTurn(role="user", content="Create a Trello card for task X"),
        ],
        events=[
            {"type": "run_started", "data": {"query": "Create a Trello card"}},
            {"type": "tool_call_started", "data": {"tool": "trello_create", "arguments": {"name": "Task X"}}},
        ],
        pending_tool_call=PendingToolCall(
            tool_name="trello_create",
            arguments={"name": "Task X", "description": "Do the thing", "labels": ["urgent"]},
            tool_call_id="tc-complex",
            subagent_id="trello_publisher",
            source="mcp",
        ),
        status=SessionStatus.PAUSED_FOR_HITL,
    )
    await repo.create(session)

    loaded = await repo.get(session.id)
    assert loaded is not None

    # Verify complete structure
    assert loaded.query == session.query
    assert loaded.status == SessionStatus.PAUSED_FOR_HITL
    assert len(loaded.conversation_history) == 1
    assert loaded.conversation_history[0].role == "user"
    assert len(loaded.events) == 2
    assert loaded.pending_tool_call is not None
    assert loaded.pending_tool_call.arguments["labels"] == ["urgent"]
    assert loaded.pending_tool_call.source == "mcp"


@pytest.mark.asyncio
async def test_hitl_approval_scope_round_trip(repo):
    session = SessionData(query="approve for session")
    session.hitl_approval_scope = HitlApprovalScope.SESSION
    await repo.create(session)

    loaded = await repo.get(session.id)
    assert loaded is not None
    assert loaded.hitl_approval_scope == HitlApprovalScope.SESSION
