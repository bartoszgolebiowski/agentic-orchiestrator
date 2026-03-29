from __future__ import annotations

from pathlib import Path

import yaml

from engine.core.models import (
    EngineConfig,
    NodeConfig,
    OrchestratorConfig,
    ToolDefinition,
)


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_nodes_from_dir(directory: Path) -> dict[str, NodeConfig]:
    nodes: dict[str, NodeConfig] = {}
    if not directory.exists():
        return nodes
    for file in sorted(directory.glob("*.yaml")):
        raw = _load_yaml(file)
        config = NodeConfig(**raw)
        if config.id in nodes:
            raise ValueError(
                f"Duplicate node id '{config.id}' found in {file}"
            )
        nodes[config.id] = config
    return nodes


def load_tool_definitions(directory: Path) -> dict[str, ToolDefinition]:
    tools: dict[str, ToolDefinition] = {}
    if not directory.exists():
        return tools
    for file in sorted(directory.glob("*.yaml")):
        raw = _load_yaml(file)
        defn = ToolDefinition(**raw)
        if defn.id in tools:
            raise ValueError(
                f"Duplicate tool id '{defn.id}' found in {file}"
            )
        tools[defn.id] = defn
    return tools


def load_orchestrator_config(config_dir: Path) -> OrchestratorConfig:
    path = config_dir / "orchestrator.yaml"
    if path.exists():
        raw = _load_yaml(path)
        return OrchestratorConfig(**raw)
    return OrchestratorConfig()


def load_engine_config(config_dir: Path) -> EngineConfig:
    orchestrator = load_orchestrator_config(config_dir)
    agents = load_nodes_from_dir(config_dir / "agents")
    subagents = load_nodes_from_dir(config_dir / "subagents")
    tools = load_tool_definitions(config_dir / "tools")

    return EngineConfig(
        orchestrator=orchestrator,
        agents=agents,
        subagents=subagents,
        tools=tools,
    )
