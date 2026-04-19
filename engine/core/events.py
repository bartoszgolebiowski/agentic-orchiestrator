"""Backward-compat shim — use engine.events directly."""
from engine.events import (
    EventType,
    StreamEvent,
    emit,
    emit_event,
    get_event_queue,
    set_event_queue,
)

__all__ = [
    "EventType",
    "StreamEvent",
    "emit",
    "emit_event",
    "get_event_queue",
    "set_event_queue",
]
