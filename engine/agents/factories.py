from __future__ import annotations

from openai import AsyncOpenAI

from engine.agents.agent_planner import AgentPlanner
from engine.agents.execution.executor import SubagentExecutor
from engine.config.models import EngineConfig, NodeConfig
from engine.mcp.runtime import McpManager
from engine.sessions.hitl import HitlCallback


class AgentPlannerFactory:
    @staticmethod
    def create(
        *,
        config: NodeConfig,
        engine_config: EngineConfig,
        client: AsyncOpenAI,
        mcp_manager: McpManager | None = None,
        model: str | None = None,
        hitl_callback: HitlCallback | None = None,
    ) -> AgentPlanner:
        return AgentPlanner(
            config=config,
            engine_config=engine_config,
            client=client,
            mcp_manager=mcp_manager,
            model=model,
            hitl_callback=hitl_callback,
        )


class SubagentFactory:
    @staticmethod
    def create(
        *,
        config: NodeConfig,
        engine_config: EngineConfig,
        client: AsyncOpenAI,
        mcp_manager: McpManager | None = None,
        model: str | None = None,
        hitl_callback: HitlCallback | None = None,
    ) -> SubagentExecutor:
        return SubagentExecutor(
            config=config,
            engine_config=engine_config,
            client=client,
            mcp_manager=mcp_manager,
            model=model,
            hitl_callback=hitl_callback,
        )
