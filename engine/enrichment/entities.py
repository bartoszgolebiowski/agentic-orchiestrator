from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from engine.config.models import EnricherConfig
from engine.enrichment.models import EnrichmentResult


@dataclass
class EnricherEntity:
    """Domain enricher with selection metadata and execution behavior."""

    config: EnricherConfig
    execute_fn: Callable[[str], EnrichmentResult] | None = None

    @property
    def enricher_id(self) -> str:
        return self.config.id

    def supports(self, query: str) -> bool:
        return bool(query.strip())

    def enrich(self, query: str) -> EnrichmentResult:
        if self.execute_fn is None:
            return EnrichmentResult(enricher_id=self.config.id, payloads=[])
        return self.execute_fn(query)
