"""Backward-compat shim — use engine.config.context directly."""
from engine.config.context import (
    clear_engine_config,
    get_engine_config,
    set_engine_config,
)

__all__ = ["clear_engine_config", "get_engine_config", "set_engine_config"]
