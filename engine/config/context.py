from __future__ import annotations

import contextvars

from engine.config.models import EngineConfig


_ENGINE_CONFIG_VAR: contextvars.ContextVar[EngineConfig | None] = contextvars.ContextVar(
    "engine_config", default=None
)


def set_engine_config(config: EngineConfig) -> None:
    _ENGINE_CONFIG_VAR.set(config)


def clear_engine_config() -> None:
    _ENGINE_CONFIG_VAR.set(None)


def get_engine_config() -> EngineConfig:
    config = _ENGINE_CONFIG_VAR.get()
    if config is None:
        raise RuntimeError("Engine config has not been initialized")
    return config
