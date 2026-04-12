from __future__ import annotations

import logging
from typing import Any
from enum import Enum

from openai import AsyncOpenAI
from pydantic import Field, create_model

from engine.core.events import EventType, emit_event
from engine.core.llm import structured_completion
from engine.core.models import EngineConfig, RoutingDecision, RoutingDecisionOutput
from engine.core.tracing import observe
from engine.mcp.runtime import McpManager
from engine.roles.agent import AgentPlanner

logger = logging.getLogger(__name__)


def _structured_payload_to_mapping(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    if hasattr(payload, "__dict__"):
        return {key: value for key, value in vars(payload).items() if not key.startswith("_")}
    raise TypeError(f"Unsupported structured payload type: {type(payload)!r}")


class Orchestrator:
    def __init__(
        self,
        engine_config: EngineConfig,
        client: AsyncOpenAI,
        mcp_manager: McpManager | None = None,
        model: str | None = None,
    ) -> None:
        self.engine_config = engine_config
        self.client = client
        self.mcp_manager = mcp_manager
        self.model = model

        self._agent_planners: dict[str, AgentPlanner] = {}
        for agent_id, agent_config in engine_config.agents.items():
            self._agent_planners[agent_id] = AgentPlanner(
                config=agent_config,
                engine_config=engine_config,
                client=client,
                mcp_manager=mcp_manager,
                model=model,
            )

        self._routing_model = self._build_routing_model()

    def _build_routing_model(self) -> type[RoutingDecisionOutput]:
        """Dynamically create a RoutingDecision subclass with an enum-constrained agent_id."""

        class _AgentChoice(str):
            def __new__(cls, value: str, title: str, description: str):
                obj = str.__new__(cls, value)
                obj._value_ = value
                obj.title = title
                obj.description = description
                return obj

        agent_members: dict[str, tuple[str, str, str]] = {}
        agent_descriptions: list[str] = []
        for agent_id, agent_config in self.engine_config.agents.items():
            title = agent_id.replace("_", " ").title()
            description = agent_config.description.strip()
            agent_members[agent_id] = (agent_id, title, description)
            agent_descriptions.append(f"- {title} ({agent_id}): {description}")

        AgentEnum = Enum("AgentEnum", agent_members, type=_AgentChoice)

        return create_model(
            "DynamicRoutingDecision",
            __base__=RoutingDecisionOutput,
            agent_id=(
                AgentEnum,
                Field(
                    description="The ID of the selected agent.",
                    validation_alias=RoutingDecisionOutput.model_fields["agent_id"].validation_alias,
                    serialization_alias="agent_id",
                    json_schema_extra={
                        "x-enumTitles": [member.title for member in AgentEnum],
                        "x-enumDescriptions": [member.description for member in AgentEnum],
                        "x-agent-list": agent_descriptions,
                    },
                ),
            ),
        )

    async def handle(self, user_query: str, input_json: dict[str, Any] | None = None) -> str:
        logger.info(f"Orchestrator received query: {user_query[:200]}")

        emit_event(EventType.ROUTING_STARTED, query=user_query)

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
            decision_output = await structured_completion(
                client=self.client,
                messages=messages,
                response_model=self._routing_model,
                model=self.engine_config.orchestrator.model or self.model,
                trace_name="orchestrator-routing",
                trace_metadata={"component": "orchestrator"},
            )
            decision: RoutingDecision = RoutingDecision.model_validate(
                _structured_payload_to_mapping(decision_output)
            )

            if routing_span is not None:
                routing_span.update(output=decision.model_dump())

        agent_id = decision.agent_id
        if hasattr(agent_id, "value"):
            agent_id = agent_id.value

        task = decision.task

        effective_input_json: dict[str, Any] | None = decision.input_json
        if input_json is not None:
            if isinstance(effective_input_json, dict):
                effective_input_json = {**effective_input_json, **input_json}
            else:
                effective_input_json = input_json

        emit_event(
            EventType.ROUTING_DECISION,
            agent_id=agent_id,
            task=task,
            input_json=effective_input_json,
        )

        if agent_id not in self._agent_planners:
            return f"Error: Unknown agent '{agent_id}'"

        logger.info(f"Orchestrator routed to agent '{agent_id}' with task: {task[:200]}")

        emit_event(EventType.AGENT_STARTED, agent_id=agent_id, task=task)

        planner = self._agent_planners[agent_id]
        with observe(
            name=f"agent-run:{agent_id}",
            as_type="span",
            input={"task": task, "agent_id": agent_id, "input_json": effective_input_json},
            metadata={"component": "orchestrator", "agent_id": agent_id},
        ) as agent_span:
            result = await planner.plan_and_execute(task, input_json=effective_input_json)
            if agent_span is not None:
                agent_span.update(output=result)

        emit_event(EventType.AGENT_FINISHED, agent_id=agent_id, result=result[:500])
        return result
