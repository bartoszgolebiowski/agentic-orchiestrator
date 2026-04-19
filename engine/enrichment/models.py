from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnrichmentResult:
    """Outcome of the enrichment stage."""
    enricher_id: str | None
    payloads: list[dict[str, Any]]
