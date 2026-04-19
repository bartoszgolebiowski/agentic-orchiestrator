"""Backward-compat shim — use engine.config.loader directly."""
from engine.config.loader import (
    load_engine_config,
    load_enricher_definitions,
    load_mcp_definitions,
    load_nodes_from_dir,
    load_orchestrator_config,
    load_tool_definitions,
)

__all__ = [
    "load_engine_config",
    "load_enricher_definitions",
    "load_mcp_definitions",
    "load_nodes_from_dir",
    "load_orchestrator_config",
    "load_tool_definitions",
]
