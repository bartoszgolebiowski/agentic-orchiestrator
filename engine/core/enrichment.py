"""Backward-compat shim — use engine.enrichment directly."""
from engine.enrichment.models import EnrichmentResult
from engine.enrichment.executor import apply_enrichment

__all__ = ["EnrichmentResult", "apply_enrichment"]
