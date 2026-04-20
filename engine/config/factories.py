from __future__ import annotations

from pathlib import Path

from engine.config.entities import EngineConfigEntity
from engine.config.loader import load_engine_config
from engine.config.models import EngineConfig


class EngineConfigFactory:
    @staticmethod
    def from_model(model: EngineConfig) -> EngineConfigEntity:
        return EngineConfigEntity(model=model)

    @staticmethod
    def from_yaml(config_dir: Path) -> EngineConfigEntity:
        model = load_engine_config(config_dir)
        return EngineConfigEntity(model=model)

    @staticmethod
    def to_model(entity: EngineConfigEntity) -> EngineConfig:
        return entity.to_model()
