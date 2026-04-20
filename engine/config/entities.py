from __future__ import annotations

from dataclasses import dataclass

from engine.config.models import EngineConfig, NodeConfig
from engine.mcp.models import McpServerConfig


@dataclass(frozen=True)
class EngineConfigEntity:
    """Domain entity facade over validated EngineConfig model."""

    model: EngineConfig

    def get_agent(self, agent_id: str) -> NodeConfig:
        return self.model.agents[agent_id]

    def get_subagent(self, subagent_id: str) -> NodeConfig:
        return self.model.subagents[subagent_id]

    def get_tool_ids(self) -> list[str]:
        return sorted(self.model.tools)

    def get_mcp(self, server_id: str) -> McpServerConfig:
        return self.model.mcps[server_id]

    def subagents_for_agent(self, agent_id: str) -> list[NodeConfig]:
        agent = self.get_agent(agent_id)
        return [self.get_subagent(sub_id) for sub_id in agent.dependencies]

    def to_model(self) -> EngineConfig:
        return self.model
