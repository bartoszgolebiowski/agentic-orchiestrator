from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from engine.tools.models import ToolSpec


ToolCallable = Callable[..., Awaitable[str]]


@dataclass(frozen=True)
class ToolEntity:
    id: str
    description: str
    call_fn: ToolCallable
    spec: ToolSpec
