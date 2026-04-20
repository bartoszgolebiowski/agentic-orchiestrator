from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from engine.agents.execution.hitl_handler import HitlHandler
from engine.agents.execution.local_tool_handler import LocalToolHandler
from engine.agents.execution.mcp_tool_handler import McpToolHandler
from engine.agents.react import ReActLoop
from engine.config.models import EngineConfig, NodeConfig, RoleType
from engine.events import EventType, emit_event
from engine.mcp.runtime import McpManager
from engine.security import enforce_agent_boundary
from engine.sessions.hitl import HitlCallback
from engine.llm.tracing import observe

logger = logging.getLogger(__name__)


class SubagentExecutor:
    def __init__(
        self,
        config: NodeConfig,
        engine_config: EngineConfig,
        client: AsyncOpenAI,
        mcp_manager: McpManager | None = None,
        model: str | None = None,
        hitl_callback: HitlCallback | None = None,
    ) -> None:
        self.config = config
        self.engine_config = engine_config
        self.client = client
        self.mcp_manager = mcp_manager
        self.model = model
        self.hitl_callback = hitl_callback

        self._hitl_handler = HitlHandler(hitl_callback)
        self._local_handler = LocalToolHandler(config=config, engine_config=engine_config)
        self._mcp_handler = McpToolHandler(
            config=config,
            mcp_manager=mcp_manager,
            hitl_handler=self._hitl_handler,
        )

    # Compatibility helper for existing callers/tests that inspect local tool context.
    def _build_local_tool_specs(self) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        return self._local_handler.build_specs()

    # Compatibility helper for existing callers/tests that inspect MCP tool context.
    async def _build_mcp_tool_context(self) -> tuple[list[dict[str, Any]], list[str], list[str], set[str]]:
        return await self._mcp_handler.build_context()

    def _build_available_actions_description(
        self,
        local_descriptions: list[str],
        mcp_descriptions: list[str],
    ) -> str:
        sections: list[str] = []
        if local_descriptions:
            sections.append("Local tools:\n" + "\n".join(local_descriptions))
        if mcp_descriptions:
            sections.append("MCP tools:\n" + "\n".join(mcp_descriptions))
        return "\n\n".join(sections)

    async def execute(self, task: str, input_json: Any | None = None) -> str:
        logger.info(f"SubagentExecutor[{self.config.id}] starting task: {task[:100]}")

        emit_event(EventType.SUBAGENT_STARTED, subagent_id=self.config.id, task=task[:200])

        system_prompt = self.config.system_prompt

        local_tool_specs, local_descriptions, local_allowed_ids = self._local_handler.build_specs()
        mcp_tool_specs, mcp_descriptions, mcp_allowed_ids, mcp_skip_hitl = await self._mcp_handler.build_context()

        collisions = set(local_allowed_ids) & set(mcp_allowed_ids)
        if collisions:
            raise ValueError(
                f"Subagent '{self.config.id}' has overlapping local and MCP tool names: {sorted(collisions)}"
            )

        unknown_projection_keys = set(self.config.tool_result_projection) - set([*local_allowed_ids, *mcp_allowed_ids])
        if unknown_projection_keys:
            raise ValueError(
                f"Subagent '{self.config.id}' tool_result_projection references unknown tools: {sorted(unknown_projection_keys)}"
            )

        self._build_available_actions_description(local_descriptions, mcp_descriptions)
        tool_specs = [*local_tool_specs, *mcp_tool_specs]
        allowed_ids = [*local_allowed_ids, *mcp_allowed_ids]
        projections = self.config.tool_result_projection

        if self.config.mcp_include_tools:
            missing_projection_ids = [tool_name for tool_name in mcp_allowed_ids if tool_name not in projections]
            if missing_projection_ids:
                raise ValueError(
                    f"Subagent '{self.config.id}' must define tool_result_projection for MCP tools: {sorted(missing_projection_ids)}"
                )

        async def handle_action(name: str, arguments: dict[str, Any]) -> str:
            enforce_agent_boundary(
                caller_role=RoleType.SUBAGENT,
                callee_id=name,
                allowed_ids=allowed_ids,
            )

            if name in self.engine_config.tools:
                return await self._local_handler.execute(
                    subagent_id=self.config.id,
                    tool_name=name,
                    arguments=arguments,
                    projections=projections,
                )

            if name in mcp_allowed_ids:
                return await self._mcp_handler.execute(
                    subagent_id=self.config.id,
                    tool_name=name,
                    arguments=arguments,
                    projections=projections,
                    skip_hitl=mcp_skip_hitl,
                )

            raise KeyError(f"Subagent '{self.config.id}' cannot resolve action '{name}'")

        with observe(
            name=f"subagent-plan:{self.config.id}",
            as_type="span",
            input={"task": task, "subagent_id": self.config.id, "input_json": input_json},
            metadata={"component": "subagent", "subagent_id": self.config.id},
        ) as subagent_span:
            loop = ReActLoop(
                client=self.client,
                system_prompt=system_prompt,
                tools_spec=tool_specs,
                action_handler=handle_action,
                max_steps=self.config.max_steps,
                model=self.config.model or self.model,
            )

            result = await loop.run(task, input_json=input_json)
            if subagent_span is not None:
                subagent_span.update(output=result)

            emit_event(
                EventType.SUBAGENT_FINISHED,
                subagent_id=self.config.id,
                result=result[:500],
            )
            return result
