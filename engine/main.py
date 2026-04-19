from __future__ import annotations

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from uuid import uuid4
from typing import Any

from dotenv import load_dotenv

import engine.tools  # noqa: F401 — trigger tool registration

from engine.core.llm import get_raw_client
from engine.core.loader import load_engine_config
from engine.core.graph import validate_config_graph, build_dependency_tree
from engine.core.runtime_state import clear_engine_config, set_engine_config
from engine.core.tracing import observe, flush, log_langfuse_connection_status
from engine.core.enrichment import apply_enrichment
from engine.core.events import EventType, emit_event
from engine.core.models import EngineConfig
from engine.core.hitl import HitlCallback
from engine.mcp.runtime import McpManager
from engine.roles.orchestrator import Orchestrator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_mcp_manager(engine_config: EngineConfig) -> McpManager | None:
    referenced_subagents = {
        dependency
        for agent in engine_config.agents.values()
        for dependency in agent.dependencies
    }
    active_mcp_dependencies = any(
        engine_config.subagents[sub_id].mcp_dependencies
        for sub_id in referenced_subagents
        if sub_id in engine_config.subagents
    )
    return McpManager(engine_config.mcps) if active_mcp_dependencies else None


async def _run_orchestrator_query(
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
    ) as root_span:
        result = await orchestrator.handle(query, input_json=input_json)
        if root_span is not None:
            root_span.update(output=result)
        return result


async def main(
    query: str,
    config_dir: str = "configs",
    input_json: dict[str, Any] | None = None,
    hitl_callback: HitlCallback | None = None,
    *,
    engine_config: EngineConfig | None = None,
    mcp_manager: McpManager | None = None,
    owns_mcp_manager: bool | None = None,
) -> str:
    load_dotenv()
    log_langfuse_connection_status()

    emit_event(EventType.RUN_STARTED, query=query, config_dir=config_dir)

    config_path = Path(config_dir)
    if engine_config is None:
        engine_config = load_engine_config(config_path)

        # Config graph validation at startup
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

    set_engine_config(engine_config)

    if owns_mcp_manager is None:
        owns_mcp_manager = mcp_manager is None

    if mcp_manager is None:
        mcp_manager = build_mcp_manager(engine_config)
        if mcp_manager is not None:
            owns_mcp_manager = True

    logger.info(
        f"Loaded {len(engine_config.agents)} agents, "
        f"{len(engine_config.subagents)} subagents, "
        f"{len(engine_config.tools)} tools, "
        f"{len(engine_config.mcps)} MCP servers, "
        f"{len(engine_config.enrichers)} enrichers"
    )

    client = get_raw_client()
    try:
        if owns_mcp_manager and mcp_manager is not None:
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
            # ── Generic enrichment: LLM-selected, config-driven ──
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
                        result = await _run_orchestrator_query(
                            orchestrator=orchestrator,
                            query=query,
                            config_dir=config_dir,
                            input_json=enrichment.payloads[0],
                        )
                        if root_span is not None:
                            root_span.update(output=result)
                        emit_event(EventType.TOKEN_DELTA, content=result)
                        emit_event(EventType.RUN_FINISHED, result=result[:500])
                        return result

                    results: list[str] = []
                    for payload in enrichment.payloads:
                        label = next(iter(payload.values()), "item") if payload else "item"
                        result = await _run_orchestrator_query(
                            orchestrator=orchestrator,
                            query=query,
                            config_dir=config_dir,
                            input_json=payload,
                        )
                        results.append(f"## {label}\n{result}")

                    final_result = "\n\n".join(results)
                    if root_span is not None:
                        root_span.update(output=final_result)
                    emit_event(EventType.TOKEN_DELTA, content=final_result)
                    emit_event(EventType.RUN_FINISHED, result=final_result[:500])
                    return final_result

            result = await _run_orchestrator_query(
                orchestrator=orchestrator,
                query=query,
                config_dir=config_dir,
                input_json=input_json,
            )
            if root_span is not None:
                root_span.update(output=result)

            emit_event(EventType.TOKEN_DELTA, content=result)
            emit_event(EventType.RUN_FINISHED, result=result[:500])
            return result
    finally:
        clear_engine_config()
        if owns_mcp_manager and mcp_manager is not None:
            await mcp_manager.aclose()
        flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the agentic orchestrator.")
    parser.add_argument("query", nargs="?", help="User query to send to the orchestrator.")
    parser.add_argument("config_dir", nargs="?", default="configs", help="Path to the config directory.")
    parser.add_argument("--graph", action="store_true", help="Print the dependency graph and exit.")
    parser.add_argument("--validate", action="store_true", help="Validate the config graph and exit.")

    args = parser.parse_args()

    if args.graph or args.validate:
        load_dotenv()
        config_path = Path(args.config_dir)
        engine_config = load_engine_config(config_path)
        issues = validate_config_graph(engine_config)
        for issue in issues:
            prefix = "ERROR" if issue.level == "error" else "WARNING"
            print(f"[{prefix}] {issue.message}")
        if args.graph:
            print()
            print(build_dependency_tree(engine_config))
        errors = [i for i in issues if i.level == "error"]
        if errors:
            print(f"\n{len(errors)} error(s) found.")
            sys.exit(1)
        if not args.graph:
            print("Config graph OK.")
        sys.exit(0)

    if not args.query:
        parser.print_usage()
        sys.exit(1)

    result = asyncio.run(main(args.query, args.config_dir))

    print(f"\n{'='*60}")
    print(f"RESULT: {result}")
    print(f"{'='*60}")
