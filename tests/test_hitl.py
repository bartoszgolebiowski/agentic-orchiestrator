"""Tests for the Human-in-the-Loop manager."""
from __future__ import annotations

import asyncio

import pytest

from engine.core.hitl import HitlManager
from engine.core.storage import HitlResponse, PendingToolCall


@pytest.fixture()
def manager():
    return HitlManager()


@pytest.mark.asyncio
async def test_register_and_submit_response(manager):
    session_id = "test-session-1"
    manager.register_session(session_id)

    pending = PendingToolCall(
        tool_name="trello_create",
        arguments={"title": "Test"},
        tool_call_id="tc-1",
        source="mcp",
    )

    response = HitlResponse(approved=True)

    # Submit response from a concurrent task
    async def _submit_later():
        await asyncio.sleep(0.05)
        await manager.submit_response(session_id, response)

    asyncio.create_task(_submit_later())

    result = await manager.wait_for_human(session_id)
    assert result.approved is True


@pytest.mark.asyncio
async def test_submit_rejected_response(manager):
    session_id = "test-session-2"
    manager.register_session(session_id)

    response = HitlResponse(approved=False, rejection_reason="Not now")

    async def _submit_later():
        await asyncio.sleep(0.05)
        await manager.submit_response(session_id, response)

    asyncio.create_task(_submit_later())

    result = await manager.wait_for_human(session_id)
    assert result.approved is False
    assert result.rejection_reason == "Not now"


@pytest.mark.asyncio
async def test_build_callback(manager):
    session_id = "test-session-3"
    manager.register_session(session_id)

    pause_calls = []

    async def _on_pause(pending: PendingToolCall):
        pause_calls.append(pending)

    callback = manager.build_callback(session_id, on_pause=_on_pause)

    pending = PendingToolCall(
        tool_name="some_tool",
        arguments={"a": 1},
        tool_call_id="tc-3",
    )

    async def _submit_later():
        await asyncio.sleep(0.05)
        await manager.submit_response(session_id, HitlResponse(approved=True))

    asyncio.create_task(_submit_later())

    result = await callback(pending)
    assert result.approved is True
    assert len(pause_calls) == 1
    assert pause_calls[0].tool_name == "some_tool"


@pytest.mark.asyncio
async def test_unregister_session(manager):
    session_id = "test-session-4"
    manager.register_session(session_id)
    manager.unregister_session(session_id)

    with pytest.raises(ValueError, match="No HITL queue"):
        await manager.wait_for_human(session_id)


@pytest.mark.asyncio
async def test_submit_to_unregistered_raises(manager):
    with pytest.raises(ValueError, match="No HITL queue"):
        await manager.submit_response("nonexistent", HitlResponse(approved=True))


@pytest.mark.asyncio
async def test_modified_arguments_in_response(manager):
    session_id = "test-session-5"
    manager.register_session(session_id)

    modified_args = {"title": "Modified Title", "priority": "high"}
    response = HitlResponse(approved=True, modified_arguments=modified_args)

    async def _submit_later():
        await asyncio.sleep(0.05)
        await manager.submit_response(session_id, response)

    asyncio.create_task(_submit_later())

    result = await manager.wait_for_human(session_id)
    assert result.approved is True
    assert result.modified_arguments == modified_args
