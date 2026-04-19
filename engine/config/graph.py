from __future__ import annotations

import logging
from dataclasses import dataclass

from engine.config.models import EngineConfig

logger = logging.getLogger(__name__)


@dataclass
class ConfigIssue:
    level: str  # "error" or "warning"
    message: str


def validate_config_graph(config: EngineConfig) -> list[ConfigIssue]:
    """Validate the full config graph: cross-namespace collisions, orphans, dependency integrity."""
    issues: list[ConfigIssue] = []

    # 1. Cross-namespace ID collisions
    namespace_map: dict[str, list[str]] = {}
    for agent_id in config.agents:
        namespace_map.setdefault(agent_id, []).append("agents")
    for sub_id in config.subagents:
        namespace_map.setdefault(sub_id, []).append("subagents")
    for tool_id in config.tools:
        namespace_map.setdefault(tool_id, []).append("tools")
    for mcp_id in config.mcps:
        namespace_map.setdefault(mcp_id, []).append("mcps")

    for id_, namespaces in namespace_map.items():
        if len(namespaces) > 1:
            issues.append(ConfigIssue(
                "error",
                f"ID '{id_}' collides across namespaces: {', '.join(namespaces)}",
            ))

    # 1b. Enricher ID collisions with other namespaces
    for enricher_id in config.enrichers:
        collisions = []
        if enricher_id in config.agents:
            collisions.append("agents")
        if enricher_id in config.subagents:
            collisions.append("subagents")
        if enricher_id in config.tools:
            collisions.append("tools")
        if enricher_id in config.mcps:
            collisions.append("mcps")
        if collisions:
            issues.append(ConfigIssue(
                "error",
                f"Enricher ID '{enricher_id}' collides with: {', '.join(collisions)}",
            ))

    # 1c. Disabled enrichers
    for enricher_id, enricher in config.enrichers.items():
        if not enricher.enabled:
            issues.append(ConfigIssue(
                "warning",
                f"Enricher '{enricher_id}' is disabled",
            ))

    # 2. Orphaned subagents (not referenced by any agent)
    referenced_subagents: set[str] = set()
    for agent in config.agents.values():
        referenced_subagents.update(agent.dependencies)
    for sub_id in config.subagents:
        if sub_id not in referenced_subagents:
            issues.append(ConfigIssue(
                "warning",
                f"Subagent '{sub_id}' is not referenced by any agent",
            ))

    # 3. Orphaned tools (not referenced by any subagent)
    referenced_tools: set[str] = set()
    for sub in config.subagents.values():
        referenced_tools.update(sub.dependencies)
    for tool_id in config.tools:
        if tool_id not in referenced_tools:
            issues.append(ConfigIssue(
                "warning",
                f"Tool '{tool_id}' is not referenced by any subagent",
            ))

    # 4. Orphaned MCP servers (not referenced by any subagent)
    referenced_mcps: set[str] = set()
    for sub in config.subagents.values():
        referenced_mcps.update(sub.mcp_dependencies)
    for mcp_id in config.mcps:
        if mcp_id not in referenced_mcps:
            issues.append(ConfigIssue(
                "warning",
                f"MCP server '{mcp_id}' is not referenced by any subagent",
            ))

    # 5. Pipeline stages exist and are valid subagent dependencies
    for agent_id, agent in config.agents.items():
        for step in agent.required_pipeline:
            if step not in config.subagents:
                issues.append(ConfigIssue(
                    "error",
                    f"Agent '{agent_id}' required_pipeline references "
                    f"unknown subagent '{step}'",
                ))

    # 6. Agent dependencies reference known subagents
    for agent_id, agent in config.agents.items():
        for dep in agent.dependencies:
            if dep not in config.subagents:
                issues.append(ConfigIssue(
                    "error",
                    f"Agent '{agent_id}' dependencies references unknown subagent '{dep}'",
                ))

    # 7. Subagent mcp_dependencies reference known MCP servers
    for sub_id, sub in config.subagents.items():
        for mcp_dep in sub.mcp_dependencies:
            if mcp_dep not in config.mcps:
                issues.append(ConfigIssue(
                    "error",
                    f"Subagent '{sub_id}' mcp_dependencies references unknown MCP server '{mcp_dep}'",
                ))

    return issues


def build_dependency_tree(config: EngineConfig) -> str:
    """Render the full config graph as a human-readable tree."""
    lines: list[str] = ["Orchestrator"]
    agents = sorted(config.agents.items())

    for a_idx, (agent_id, agent) in enumerate(agents):
        is_last_agent = a_idx == len(agents) - 1
        a_branch = "└── " if is_last_agent else "├── "
        a_prefix = "    " if is_last_agent else "│   "

        pipeline_tag = ""
        if agent.required_pipeline:
            pipeline_tag = f"  [pipeline: {' -> '.join(agent.required_pipeline)}]"
        lines.append(f"{a_branch}Agent: {agent_id}{pipeline_tag}")

        deps = list(agent.dependencies)
        for s_idx, sub_id in enumerate(deps):
            is_last_sub = s_idx == len(deps) - 1
            s_branch = "└── " if is_last_sub else "├── "
            s_prefix = "    " if is_last_sub else "│   "
            sub = config.subagents.get(sub_id)
            lines.append(f"{a_prefix}{s_branch}Subagent: {sub_id}")
            if sub is None:
                continue

            all_deps: list[tuple[str, str]] = [
                ("Tool", dep) for dep in sub.dependencies
            ] + [
                ("MCP", dep) for dep in sub.mcp_dependencies
            ]
            for d_idx, (dep_kind, dep_id) in enumerate(all_deps):
                is_last_dep = d_idx == len(all_deps) - 1
                d_branch = "└── " if is_last_dep else "├── "
                suffix = ""
                if dep_kind == "MCP" and dep_id in sub.mcp_include_tools:
                    count = len(sub.mcp_include_tools[dep_id])
                    suffix = f" (filtered: {count} tool{'s' if count != 1 else ''})"
                if dep_kind == "MCP" and dep_id in sub.mcp_skip_hitl_tools:
                    skip_count = len(sub.mcp_skip_hitl_tools[dep_id])
                    suffix += f" (hitl-skip: {skip_count} tool{'s' if skip_count != 1 else ''})"
                lines.append(f"{a_prefix}{s_prefix}{d_branch}{dep_kind}: {dep_id}{suffix}")

    return "\n".join(lines)


def build_dependency_dict(config: EngineConfig) -> dict:
    """Render the full config graph as a machine-readable dict."""
    graph: dict = {"orchestrator": {}}
    for agent_id, agent in config.agents.items():
        subagents: dict = {}
        for sub_id in agent.dependencies:
            sub = config.subagents.get(sub_id)
            tools = list(sub.dependencies) if sub else []
            mcps = list(sub.mcp_dependencies) if sub else []
            subagents[sub_id] = {"tools": tools, "mcps": mcps}
        graph["orchestrator"][agent_id] = {
            "subagents": subagents,
            "required_pipeline": list(agent.required_pipeline),
        }
    return graph
