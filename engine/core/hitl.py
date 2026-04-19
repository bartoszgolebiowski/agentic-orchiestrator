"""Backward-compat shim — use engine.sessions.hitl directly."""
from engine.sessions.hitl import HitlCallback, HitlManager

__all__ = ["HitlCallback", "HitlManager"]
