from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

import engine.tools  # noqa: F401 — trigger tool registration

from engine.core.llm import get_raw_client
from engine.core.loader import load_engine_config
from engine.core.tracing import observe, flush, log_langfuse_connection_status
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
        f"{len(engine_config.tools)} tools"
    )

    client = get_raw_client()
    orchestrator = Orchestrator(
        engine_config=engine_config,
        client=client,
    )

    run_id = uuid4().hex
    try:
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
        flush()


async def analyze_transcript(
    transcript_path: str,
    config_dir: str = "configs",
    output_base: str = "output",
) -> str:
    """Convenience entry point for transcript analysis with auto-generated output dir."""
    load_dotenv()
    log_langfuse_connection_status()

    transcript_file = Path(transcript_path)
    if not transcript_file.exists():
        return f"Error: Transcript file not found: {transcript_path}"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(output_base) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "action-points").mkdir(exist_ok=True)

    query = (
        f"Analyze the transcript at '{transcript_file}'. "
        f"Write all output files to '{output_dir}'. "
        f"The output directory for action points is '{output_dir}/action-points/'."
    )

    logger.info(f"Starting transcript analysis: {transcript_path} -> {output_dir}")

    return await main(query, config_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m engine.main <query> [config_dir]")
        print("  python -m engine.main --transcript <file> [config_dir] [output_dir]")
        sys.exit(1)

    if sys.argv[1] == "--transcript":
        if len(sys.argv) < 3:
            print("Usage: python -m engine.main --transcript <file> [config_dir] [output_dir]")
            sys.exit(1)
        transcript_path = sys.argv[2]
        config_dir = sys.argv[3] if len(sys.argv) > 3 else "configs"
        output_base = sys.argv[4] if len(sys.argv) > 4 else "output"
        result = asyncio.run(analyze_transcript(transcript_path, config_dir, output_base))
    else:
        query = sys.argv[1]
        config_dir = sys.argv[2] if len(sys.argv) > 2 else "configs"
        result = asyncio.run(main(query, config_dir))

    print(f"\n{'='*60}")
    print(f"RESULT: {result}")
    print(f"{'='*60}")
