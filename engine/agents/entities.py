from __future__ import annotations

from dataclasses import dataclass

from engine.agents.agent_planner import AgentPlanner
from engine.agents.orchestrator import Orchestrator


@dataclass(frozen=True)
class AgentPlannerEntity:
    planner: AgentPlanner


@dataclass(frozen=True)
class OrchestratorEntity:
    orchestrator: Orchestrator
