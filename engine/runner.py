"""Core engine execution logic: config loading, enrichment, orchestration."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from engine.agents.orchestrator import Orchestrator
from engine.config.context import clear_engine_config, set_engine_config
from engine.config.graph import validate_config_graph
from engine.config.loader import load_engine_config
from engine.config.models import EngineConfig
from engine.enrichment.executor import apply_enrichment
from engine.events import EventType, emit_event
from engine.llm.client import get_raw_client
from engine.llm.tracing import flush, log_langfuse_connection_status, observe
from engine.mcp.runtime import McpManager
from engine.sessions.hitl import HitlCallback

logger = logging.getLogger(__name__)


def build_mcp_manager(engine_config: EngineConfig) -> McpManager | None:
    referenced_subagents = {
        dep
        for agent in engine_config.agents.values()
        for dep in agent.dependencies
    }
    has_mcp = any(
        engine_config.subagents[sid].mcp_dependencies
        for sid in referenced_subagents
        if sid in engine_config.subagents
    )
    return McpManager(engine_config.mcps) if has_mcp else None


def _load_and_validate_config(config_dir: Path) -> EngineConfig:
    engine_config = load_engine_config(config_dir)
    issues = validate_config_graph(engine_config)
    for issue in issues:
        if issue.level == "error":
            logger.error("Config graph error: %s", issue.message)
        else:
            logger.warning("Config graph warning: %s", issue.message)
    errors = [i for i in issues if i.level == "error"]
    if errors:
        raise ValueError(
            f"Config graph has {len(errors)} error(s); fix them before running the orchestrator"
        )
    return engine_config


async def _run_query(
    orchestrator: Orchestrator,
    query: str,
    config_dir: str,
    input_json: dict[str, Any] | None = None,
) -> str:
    with observe(
        name="agent-run",
        as_type="span",
        input={"query": query, "config_dir": config_dir, "input_json": input_json},
        metadata={"entrypoint": "main"},
    ) as span:
        result = await orchestrator.handle(query, input_json=input_json)
        if span is not None:
            span.update(output=result)
        return result


async def run_engine(
    query: str,
    config_dir: str = "configs",
    input_json: dict[str, Any] | None = None,
    hitl_callback: HitlCallback | None = None,
    engine_config: EngineConfig | None = None,
    mcp_manager: McpManager | None = None,
    owns_mcp_manager: bool | None = None,
) -> str:
    load_dotenv()
    log_langfuse_connection_status()

    emit_event(EventType.RUN_STARTED, query=query, config_dir=config_dir)

    config_path = Path(config_dir)
    if engine_config is None:
        engine_config = _load_and_validate_config(config_path)

    set_engine_config(engine_config)

    owns = owns_mcp_manager if owns_mcp_manager is not None else (mcp_manager is None)
    if mcp_manager is None:
        mcp_manager = build_mcp_manager(engine_config)
        owns = mcp_manager is not None

    logger.info(
        "Loaded %d agents, %d subagents, %d tools, %d MCP servers, %d enrichers",
        len(engine_config.agents),
        len(engine_config.subagents),
        len(engine_config.tools),
        len(engine_config.mcps),
        len(engine_config.enrichers),
    )

    client = get_raw_client()
    try:
        if owns and mcp_manager is not None:
            await mcp_manager.warmup()

        orchestrator = Orchestrator(
            engine_config=engine_config,
            client=client,
            mcp_manager=mcp_manager,
            hitl_callback=hitl_callback,
        )

        run_id = uuid4().hex
        with observe(
            name="agent-request",
            as_type="span",
            input={"query": query, "config_dir": config_dir, "input_json": input_json},
            metadata={"run_id": run_id, "entrypoint": "main"},
        ) as root_span:
            if input_json is None and engine_config.enrichers:
                emit_event(EventType.ENRICHMENT_STARTED)
                enrichment = await apply_enrichment(
                    client=client,
                    query=query,
                    enrichers=engine_config.enrichers,
                    config_dir=config_path,
                    model=engine_config.orchestrator.model,
                )

                if enrichment.payloads:
                    emit_event(
                        EventType.ENRICHMENT_RESULT,
                        enricher_id=enrichment.enricher_id,
                        payload_count=len(enrichment.payloads),
                    )
                    if len(enrichment.payloads) == 1:
                        result = await _run_query(orchestrator, query, config_dir, enrichment.payloads[0])
                        if root_span is not None:
                            root_span.update(output=result)
                        emit_event(EventType.TOKEN_DELTA, content=result)
                        emit_event(EventType.RUN_FINISHED, result=result[:500])
                        return result

                    results: list[str] = []
                    for payload in enrichment.payloads:
                        label = next(iter(payload.values()), "item") if payload else "item"
                        r = await _run_query(orchestrator, query, config_dir, payload)
                        results.append(f"## {label}\n{r}")

                    final_result = "\n\n".join(results)
                    if root_span is not None:
                        root_span.update(output=final_result)
                    emit_event(EventType.TOKEN_DELTA, content=final_result)
                    emit_event(EventType.RUN_FINISHED, result=final_result[:500])
                    return final_result

            result = await _run_query(orchestrator, query, config_dir, input_json)
            if root_span is not None:
                root_span.update(output=result)
            emit_event(EventType.TOKEN_DELTA, content=result)
            emit_event(EventType.RUN_FINISHED, result=result[:500])
            return result
    finally:
        clear_engine_config()
        if owns and mcp_manager is not None:
            await mcp_manager.aclose()
        flush()
