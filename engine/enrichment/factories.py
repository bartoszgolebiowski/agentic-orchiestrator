from __future__ import annotations

from typing import Callable

from engine.config.models import EnricherConfig
from engine.enrichment.entities import EnricherEntity
from engine.enrichment.models import EnrichmentResult


class EnricherFactory:
    @staticmethod
    def create(
        config: EnricherConfig,
        execute_fn: Callable[[str], EnrichmentResult] | None = None,
    ) -> EnricherEntity:
        return EnricherEntity(config=config, execute_fn=execute_fn)
