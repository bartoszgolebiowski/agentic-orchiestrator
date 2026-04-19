"""Generic LLM-based enrichment layer."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from engine.llm.client import structured_completion
from engine.config.models import EnricherConfig, EnrichmentDecision
from engine.llm.tracing import observe
from engine.enrichment.document import (
    discover_markdown_sources,
    load_document_workflow_config,
)
from engine.enrichment.models import EnrichmentResult

logger = logging.getLogger(__name__)


# ─── Executor registry ───

_EXECUTORS: dict[str, Any] = {}


def _register_builtin_executors() -> None:
    _EXECUTORS["glob_file_discovery"] = _execute_glob_file_discovery


async def _execute_glob_file_discovery(
    config: dict[str, Any],
    config_dir: Path,
) -> list[dict[str, Any]]:
    """Built-in executor: discovers files via glob and returns one payload per file."""
    workflow_config_file = config.get("workflow_config_file")
    if workflow_config_file is None:
        raise ValueError("glob_file_discovery requires 'workflow_config_file' in executor_config")

    workflow_config = load_document_workflow_config(
        config_dir,
        str(workflow_config_file),
    )
    if workflow_config is None:
        workflow_config_path = (config_dir / Path(str(workflow_config_file))).resolve()
        logger.warning("glob_file_discovery: workflow config not found at %s", workflow_config_path)
        return []

    sources = discover_markdown_sources(workflow_config.source_dir, workflow_config.scan_pattern)
    if not sources:
        return []

    input_key = config.get("input_key", "source_path")
    return [{input_key: str(path)} for path in sources]


# ─── LLM selector ───


async def _select_enricher(
    client: AsyncOpenAI,
    query: str,
    enrichers: list[EnricherConfig],
    model: str | None,
) -> EnrichmentDecision:
    """Ask the LLM which enricher (if any) should be applied."""
    enricher_list = "\n".join(
        f"- {e.id}: {e.description}" for e in enrichers
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a pre-routing selector. Given the user query, decide whether "
                "any of the available enrichers should run to prepare input data before "
                "the main agent handles the query.\n\n"
                "Available enrichers:\n"
                f"{enricher_list}\n\n"
                "Return a JSON object with:\n"
                "- enricher_id: the ID of the best matching enricher, or null if none "
                "is needed.\n"
                "- reason: a brief explanation.\n\n"
                "Only select an enricher when the query clearly benefits from the "
                "additional data that enricher would provide. When in doubt, return "
                "enricher_id as null."
            ),
        },
        {"role": "user", "content": query},
    ]

    selector_model = model
    for enricher in enrichers:
        if enricher.model is not None:
            selector_model = enricher.model
            break

    return await structured_completion(
        client=client,
        messages=messages,
        response_model=EnrichmentDecision,
        model=selector_model,
        trace_name="enrichment-selector",
        trace_metadata={"component": "enrichment"},
    )


# ─── Public API ───


async def apply_enrichment(
    client: AsyncOpenAI,
    query: str,
    enrichers: dict[str, EnricherConfig],
    config_dir: Path,
    model: str | None = None,
) -> EnrichmentResult:
    """Run the LLM enrichment selector and execute the chosen enricher."""
    _register_builtin_executors()

    active = sorted(
        (e for e in enrichers.values() if e.enabled),
        key=lambda e: e.priority,
        reverse=True,
    )
    if not active:
        return EnrichmentResult(enricher_id=None, payloads=[])

    with observe(
        name="enrichment",
        as_type="span",
        input={"query": query, "enricher_count": len(active)},
        metadata={"component": "enrichment"},
    ) as enrich_span:
        try:
            decision = await _select_enricher(client, query, active, model)
        except Exception:
            logger.warning("Enrichment selector failed; skipping enrichment", exc_info=True)
            if enrich_span is not None:
                enrich_span.update(output={"status": "selector_failed"})
            return EnrichmentResult(enricher_id=None, payloads=[])

        selected_id = decision.enricher_id
        if isinstance(selected_id, str):
            selected_id = selected_id.strip() or None

        if selected_id is None or selected_id not in enrichers:
            logger.info("Enrichment selector chose no enricher (reason: %s)", decision.reason)
            if enrich_span is not None:
                enrich_span.update(output={"status": "none_selected", "reason": decision.reason})
            return EnrichmentResult(enricher_id=None, payloads=[])

        enricher = enrichers[selected_id]
        if not enricher.enabled:
            logger.warning("Enrichment selector chose disabled enricher '%s'; skipping", selected_id)
            if enrich_span is not None:
                enrich_span.update(output={"status": "disabled", "enricher_id": selected_id})
            return EnrichmentResult(enricher_id=None, payloads=[])

        executor_fn = _EXECUTORS.get(enricher.executor)
        if executor_fn is None:
            logger.error("Unknown executor '%s' for enricher '%s'; skipping", enricher.executor, selected_id)
            if enrich_span is not None:
                enrich_span.update(output={"status": "unknown_executor", "enricher_id": selected_id})
            return EnrichmentResult(enricher_id=None, payloads=[])

        logger.info("Enrichment: running executor '%s' for enricher '%s'", enricher.executor, selected_id)
        try:
            payloads = await executor_fn(enricher.executor_config, config_dir)
        except Exception:
            logger.warning("Enricher '%s' executor failed; skipping enrichment", selected_id, exc_info=True)
            if enrich_span is not None:
                enrich_span.update(output={"status": "executor_failed", "enricher_id": selected_id})
            return EnrichmentResult(enricher_id=None, payloads=[])

        if enrich_span is not None:
            enrich_span.update(output={
                "status": "enriched",
                "enricher_id": selected_id,
                "payload_count": len(payloads),
            })

        return EnrichmentResult(enricher_id=selected_id, payloads=payloads)
