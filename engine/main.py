from __future__ import annotations

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

import engine.tools  # noqa: F401 — trigger tool registration

from engine.core.llm import get_raw_client
from engine.core.loader import load_engine_config
from engine.core.tracing import observe, flush, log_langfuse_connection_status
from engine.mcp.runtime import McpManager
from engine.roles.orchestrator import Orchestrator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main(query: str, config_dir: str = "configs") -> str:
    load_dotenv()
    log_langfuse_connection_status()

    config_path = Path(config_dir)
    engine_config = load_engine_config(config_path)

    logger.info(
        f"Loaded {len(engine_config.agents)} agents, "
        f"{len(engine_config.subagents)} subagents, "
        f"{len(engine_config.tools)} tools, "
        f"{len(engine_config.mcps)} MCP servers"
    )

    client = get_raw_client()
    mcp_manager = McpManager(engine_config.mcps)

    try:
        if engine_config.mcps:
            await mcp_manager.warmup()

        orchestrator = Orchestrator(
            engine_config=engine_config,
            client=client,
            mcp_manager=mcp_manager,
        )

        run_id = uuid4().hex
        with observe(
            name="agent-request",
            as_type="span",
            input={"query": query, "config_dir": config_dir},
            metadata={"run_id": run_id, "entrypoint": "main"},
        ) as root_span:
            result = await orchestrator.handle(query)
            if root_span is not None:
                root_span.update(output=result)
            return result
    finally:
        await mcp_manager.aclose()
        flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the agentic orchestrator.")
    parser.add_argument("query", nargs="?", help="User query to send to the orchestrator.")
    parser.add_argument("config_dir", nargs="?", default="configs", help="Path to the config directory.")

    args = parser.parse_args()

    if not args.query:
        parser.print_usage()
        sys.exit(1)

    result = asyncio.run(main(args.query, args.config_dir))

    print(f"\n{'='*60}")
    print(f"RESULT: {result}")
    print(f"{'='*60}")
