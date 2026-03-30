from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI
from pydantic import Field, create_model

from engine.core.llm import structured_completion
from engine.core.models import EngineConfig, RoutingDecision
from engine.core.tracing import observe
from engine.roles.agent import AgentPlanner

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        engine_config: EngineConfig,
        client: AsyncOpenAI,
        model: str | None = None,
    ) -> None:
        self.engine_config = engine_config
        self.client = client
        self.model = model

        self._agent_planners: dict[str, AgentPlanner] = {}
        for agent_id, agent_config in engine_config.agents.items():
            self._agent_planners[agent_id] = AgentPlanner(
                config=agent_config,
                engine_config=engine_config,
                client=client,
                model=model,
            )

        self._routing_model = self._build_routing_model()

    def _build_routing_model(self) -> type[RoutingDecision]:
        """Dynamically create a RoutingDecision subclass with an enum-constrained agent_id."""
        agent_ids = tuple(self.engine_config.agents.keys())
        from enum import Enum
        AgentEnum = Enum("AgentEnum", {aid: aid for aid in agent_ids})

        agents_desc = []
        for agent_id, agent_config in self.engine_config.agents.items():
            agents_desc.append(f"- {agent_id}: {agent_config.description}")
        agents_list = "\n".join(agents_desc)

        return create_model(
            "DynamicRoutingDecision",
            __base__=RoutingDecision,
            agent_id=(
                AgentEnum,
                Field(description=f"The ID of the selected agent. Available:\n{agents_list}"),
            ),
        )

    async def handle(self, user_query: str) -> str:
        logger.info(f"Orchestrator received query: {user_query[:200]}")

        messages = [
            {"role": "system", "content": self.engine_config.orchestrator.system_prompt},
            {"role": "user", "content": user_query},
        ]

        with observe(
            name="orchestrator.routing",
            as_type="span",
            input={"user_query": user_query},
            metadata={"component": "orchestrator"},
        ) as routing_span:
            decision: RoutingDecision = await structured_completion(
                client=self.client,
                messages=messages,
                response_model=self._routing_model,
                model=self.engine_config.orchestrator.model or self.model,
                trace_name="orchestrator-routing",
                trace_metadata={"component": "orchestrator"},
            )

            if routing_span is not None:
                routing_span.update(output=decision.model_dump())

        agent_id = decision.agent_id
        if isinstance(agent_id, str):
            pass
        else:
            agent_id = agent_id.value

        task = decision.task

        if agent_id not in self._agent_planners:
            return f"Error: Unknown agent '{agent_id}'"

        logger.info(f"Orchestrator routed to agent '{agent_id}' with task: {task[:200]}")

        planner = self._agent_planners[agent_id]
        with observe(
            name=f"agent-run:{agent_id}",
            as_type="span",
            input={"task": task, "agent_id": agent_id},
            metadata={"component": "orchestrator", "agent_id": agent_id},
        ) as agent_span:
            result = await planner.plan_and_execute(task)
            if agent_span is not None:
                agent_span.update(output=result)

        return result
