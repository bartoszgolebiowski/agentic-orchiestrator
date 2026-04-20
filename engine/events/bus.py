"""Async runtime event bus for streaming UI."""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Any

from engine.events.models import EventType, StreamEvent


_event_queue: ContextVar[asyncio.Queue[StreamEvent | None] | None] = ContextVar(
    "_event_queue",
    default=None,
)


class EventBus:
    """Small event bus wrapper around a context-local asyncio queue."""

    def get_event_queue(self) -> asyncio.Queue[StreamEvent | None] | None:
        return _event_queue.get(None)

    def set_event_queue(self, queue: asyncio.Queue[StreamEvent | None] | None) -> None:
        _event_queue.set(queue)

    def emit(self, event: StreamEvent) -> None:
        """Publish an event without blocking; no-op when queue is missing."""
        queue = _event_queue.get(None)
        if queue is not None:
            queue.put_nowait(event)

    def emit_event(self, event_type: EventType, **data: Any) -> None:
        self.emit(StreamEvent(type=event_type, data=data))


_default_bus = EventBus()


def get_event_queue() -> asyncio.Queue[StreamEvent | None] | None:
    return _default_bus.get_event_queue()


def set_event_queue(queue: asyncio.Queue[StreamEvent | None] | None) -> None:
    _default_bus.set_event_queue(queue)


def emit(event: StreamEvent) -> None:
    _default_bus.emit(event)


def emit_event(event_type: EventType, **data: Any) -> None:
    _default_bus.emit_event(event_type, **data)
