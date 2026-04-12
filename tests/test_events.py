"""Tests for the streaming event bus and stream contract."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from engine.core.events import (
    EventType,
    StreamEvent,
    emit,
    emit_event,
    get_event_queue,
    set_event_queue,
)


class TestStreamEvent:
    def test_to_dict_serializes_type(self):
        e = StreamEvent(type=EventType.RUN_STARTED, data={"query": "hello"})
        d = e.to_dict()
        assert d["type"] == "run_started"
        assert d["data"] == {"query": "hello"}
        assert isinstance(d["timestamp"], float)

    def test_to_dict_roundtrips_through_json(self):
        e = StreamEvent(type=EventType.TOOL_CALL_FINISHED, data={"tool": "add", "result": "5"})
        raw = json.dumps(e.to_dict())
        loaded = json.loads(raw)
        assert loaded["type"] == "tool_call_finished"
        assert loaded["data"]["tool"] == "add"


class TestEventQueue:
    def test_default_queue_is_none(self):
        assert get_event_queue() is None

    def test_set_and_get_queue(self):
        q: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        set_event_queue(q)
        assert get_event_queue() is q
        set_event_queue(None)
        assert get_event_queue() is None

    def test_emit_noop_without_queue(self):
        # Should not raise
        set_event_queue(None)
        emit(StreamEvent(type=EventType.RUN_STARTED))

    def test_emit_puts_event_on_queue(self):
        q: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        set_event_queue(q)
        try:
            emit_event(EventType.AGENT_STARTED, agent_id="math_agent")
            assert not q.empty()
            event = q.get_nowait()
            assert event.type == EventType.AGENT_STARTED
            assert event.data["agent_id"] == "math_agent"
        finally:
            set_event_queue(None)


class TestEventContract:
    """Verify that every EventType is a valid SSE-safe string."""

    def test_all_event_types_are_strings(self):
        for et in EventType:
            assert isinstance(et.value, str)
            assert et.value == et.value.strip()
            assert "\n" not in et.value

    def test_event_types_are_unique(self):
        values = [et.value for et in EventType]
        assert len(values) == len(set(values))


class TestEmitEventHelper:
    def test_emit_event_creates_stream_event(self):
        q: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        set_event_queue(q)
        try:
            emit_event(EventType.TOOL_CALL_STARTED, tool="add", arguments={"a": 1, "b": 2})
            event = q.get_nowait()
            assert event.type == EventType.TOOL_CALL_STARTED
            assert event.data["tool"] == "add"
            assert event.data["arguments"] == {"a": 1, "b": 2}
        finally:
            set_event_queue(None)
