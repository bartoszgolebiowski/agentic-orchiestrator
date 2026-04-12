from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from engine.core.events import EventType, emit_event
from engine.core.models import NodeConfig, RoleType, EngineConfig
from engine.core.react import StructuredReActLoop
from engine.core.security import enforce_agent_boundary, enforce_no_direct_tool_access
from engine.core.tracing import observe
from engine.mcp.runtime import McpManager
from engine.roles.subagent import SubagentExecutor

logger = logging.getLogger(__name__)


class AgentPlanner:
    def __init__(
        self,
        config: NodeConfig,
        engine_config: EngineConfig,
        client: AsyncOpenAI,
        mcp_manager: McpManager | None = None,
        model: str | None = None,
    ) -> None:
        self.config = config
        self.engine_config = engine_config
        self.client = client
        self.mcp_manager = mcp_manager
        self.model = model

        self._executors: dict[str, SubagentExecutor] = {}

        for sub_id in self.config.dependencies:
            sub_config = self.engine_config.subagents[sub_id]
            self._executors[sub_id] = SubagentExecutor(
                config=sub_config,
                engine_config=engine_config,
                client=client,
                mcp_manager=mcp_manager,
                model=sub_config.model or model,
            )

    def _build_available_actions_description(self) -> str:
        lines: list[str] = []
        for sub_id in self.config.dependencies:
            sub_config = self.engine_config.subagents[sub_id]
            lines.append(f"- {sub_id}: {sub_config.description}")
        return "\n".join(lines)

    async def _handle_action(self, name: str, arguments: dict[str, Any]) -> str:
        if name in self.engine_config.tools:
            enforce_no_direct_tool_access(RoleType.AGENT, name)

        enforce_agent_boundary(
            caller_role=RoleType.AGENT,
            callee_id=name,
            allowed_ids=self.config.dependencies,
        )

        executor = self._executors[name]
        task = arguments.get("task", "")
        input_json = arguments.get("input_json")

        logger.info(f"AgentPlanner[{self.config.id}] delegating to subagent '{name}': {task[:100]}")

        emit_event(
            EventType.AGENT_DELEGATION,
            agent_id=self.config.id,
            subagent_id=name,
            task=task,
        )

        with observe(
            name=f"subagent-run:{name}",
            as_type="span",
            input={"task": task, "subagent_id": name, "input_json": input_json},
            metadata={"component": "agent", "agent_id": self.config.id, "subagent_id": name},
        ) as subagent_span:
            result = await executor.execute(task, input_json=input_json)
            if subagent_span is not None:
                subagent_span.update(output=result)
            return result

    async def plan_and_execute(self, task: str, input_json: Any | None = None) -> str:
        logger.info(f"AgentPlanner[{self.config.id}] starting task: {task[:100]}")

        system_prompt = self.config.system_prompt

        with observe(
            name=f"agent-plan:{self.config.id}",
            as_type="span",
            input={"task": task, "agent_id": self.config.id, "input_json": input_json},
            metadata={"component": "agent", "agent_id": self.config.id},
        ) as agent_span:
            loop = StructuredReActLoop(
                client=self.client,
                system_prompt=system_prompt,
                action_handler=self._handle_action,
                available_actions=self._build_available_actions_description(),
                max_steps=self.config.max_steps,
                model=self.config.model or self.model,
                required_pipeline=self.config.required_pipeline,
            )

            result = await loop.run(task, input_json=input_json)
            if agent_span is not None:
                agent_span.update(output=result)
            return result
