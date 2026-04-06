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
from engine.mcp.runtime import McpManager
from engine.roles.orchestrator import Orchestrator
from engine.workflows.document import (
    build_document_input,
    discover_markdown_sources,
    load_document_workflow_config,
    should_run_document_workflow,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def _run_orchestrator_query(
    orchestrator: Orchestrator,
    query: str,
    config_dir: str,
    input_json: dict[str, Any] | None = None,
) -> str:
    run_id = uuid4().hex
    with observe(
        name="agent-request",
        as_type="span",
        input={"query": query, "config_dir": config_dir, "input_json": input_json},
        metadata={"run_id": run_id, "entrypoint": "main"},
    ) as root_span:
        result = await orchestrator.handle(query, input_json=input_json)
        if root_span is not None:
            root_span.update(output=result)
        return result


async def main(query: str, config_dir: str = "configs", input_json: dict[str, Any] | None = None) -> str:
    load_dotenv()
    log_langfuse_connection_status()

    config_path = Path(config_dir)
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

    logger.info(
        f"Loaded {len(engine_config.agents)} agents, "
        f"{len(engine_config.subagents)} subagents, "
        f"{len(engine_config.tools)} tools, "
        f"{len(engine_config.mcps)} MCP servers"
    )

    client = get_raw_client()
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
    mcp_manager = McpManager(engine_config.mcps) if active_mcp_dependencies else None

    try:
        if mcp_manager is not None:
            await mcp_manager.warmup()

        orchestrator = Orchestrator(
            engine_config=engine_config,
            client=client,
            mcp_manager=mcp_manager,
        )

        workflow_config = load_document_workflow_config(config_path)
        if input_json is None and workflow_config is not None and should_run_document_workflow(query):
            source_paths = discover_markdown_sources(workflow_config.source_dir, workflow_config.scan_pattern)
            if not source_paths:
                raise FileNotFoundError(
                    f"No markdown files found in '{workflow_config.source_dir}' using pattern '{workflow_config.scan_pattern}'"
                )

            if len(source_paths) == 1:
                return await _run_orchestrator_query(
                    orchestrator=orchestrator,
                    query=query,
                    config_dir=config_dir,
                    input_json=build_document_input(source_paths[0]),
                )

            results: list[str] = []
            for source_path in source_paths:
                result = await _run_orchestrator_query(
                    orchestrator=orchestrator,
                    query=query,
                    config_dir=config_dir,
                    input_json=build_document_input(source_path),
                )
                results.append(f"## {source_path.name}\n{result}")

            return "\n\n".join(results)

        return await _run_orchestrator_query(
            orchestrator=orchestrator,
            query=query,
            config_dir=config_dir,
            input_json=input_json,
        )
    finally:
        clear_engine_config()
        if mcp_manager is not None:
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
