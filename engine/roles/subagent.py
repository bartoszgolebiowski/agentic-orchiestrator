from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from engine.core.events import EventType, emit_event
from engine.core.models import NodeConfig, RoleType, EngineConfig
from engine.core.react import ReActLoop
from engine.core.security import enforce_agent_boundary
from engine.core.tracing import observe
from engine.tools.registry import get_tool, build_openai_tool_spec
from engine.mcp.runtime import McpManager, build_openai_mcp_tool_spec

logger = logging.getLogger(__name__)


class SubagentExecutor:
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

    def _build_local_tool_specs(self) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        specs: list[dict[str, Any]] = []
        descriptions: list[str] = []
        allowed_ids: list[str] = []

        for tool_id in self.config.dependencies:
            defn = self.engine_config.tools[tool_id]
            specs.append(build_openai_tool_spec(tool_id, defn))
            descriptions.append(f"- {tool_id}: {defn.description}")
            allowed_ids.append(tool_id)

        return specs, descriptions, allowed_ids

    async def _build_mcp_tool_context(self) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        if not self.config.mcp_dependencies:
            return [], [], []

        if self.mcp_manager is None:
            raise ValueError(
                f"Subagent '{self.config.id}' declares mcp_dependencies but no MCP manager was provided"
            )

        descriptors = await self.mcp_manager.describe_tools(self.config.mcp_dependencies)

        if self.config.mcp_include_tools:
            filtered = []
            for descriptor in descriptors:
                server_filter = self.config.mcp_include_tools.get(descriptor.server_id)
                if server_filter is None:
                    filtered.append(descriptor)
                elif descriptor.remote_name in set(server_filter):
                    filtered.append(descriptor)
            for server_id, allowed_names in self.config.mcp_include_tools.items():
                discovered = {d.remote_name for d in descriptors if d.server_id == server_id}
                missing = set(allowed_names) - discovered
                if missing:
                    raise ValueError(
                        f"Subagent '{self.config.id}' mcp_include_tools references "
                        f"unknown tools from server '{server_id}': {sorted(missing)}"
                    )
            descriptors = filtered

        specs: list[dict[str, Any]] = []
        descriptions: list[str] = []
        allowed_ids: list[str] = []

        for descriptor in descriptors:
            specs.append(build_openai_mcp_tool_spec(descriptor))
            descriptions.append(
                f"- {descriptor.exposed_name}: {descriptor.description} (MCP server: {descriptor.server_id})"
            )
            allowed_ids.append(descriptor.exposed_name)

        return specs, descriptions, allowed_ids

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

        local_tool_specs, local_descriptions, local_allowed_ids = self._build_local_tool_specs()
        mcp_tool_specs, mcp_descriptions, mcp_allowed_ids = await self._build_mcp_tool_context()

        collisions = set(local_allowed_ids) & set(mcp_allowed_ids)
        if collisions:
            raise ValueError(
                f"Subagent '{self.config.id}' has overlapping local and MCP tool names: {sorted(collisions)}"
            )

        available_actions = self._build_available_actions_description(local_descriptions, mcp_descriptions)
        tool_specs = [*local_tool_specs, *mcp_tool_specs]
        allowed_ids = [*local_allowed_ids, *mcp_allowed_ids]

        async def handle_action(name: str, arguments: dict[str, Any]) -> str:
            enforce_agent_boundary(
                caller_role=RoleType.SUBAGENT,
                callee_id=name,
                allowed_ids=allowed_ids,
            )

            if name in self.engine_config.tools:
                tool_fn = get_tool(name)
                emit_event(
                    EventType.TOOL_CALL_STARTED,
                    subagent_id=self.config.id,
                    tool=name,
                    arguments=arguments,
                )
                with observe(
                    name=f"tool:{name}",
                    as_type="span",
                    input={"arguments": arguments, "tool": name},
                    metadata={
                        "component": "subagent",
                        "subagent_id": self.config.id,
                        "tool": name,
                        "tool_source": "local",
                    },
                ) as tool_span:
                    result = await tool_fn(**arguments)
                    result_text = str(result)
                    if tool_span is not None:
                        tool_span.update(output=result_text)
                    emit_event(
                        EventType.TOOL_CALL_FINISHED,
                        subagent_id=self.config.id,
                        tool=name,
                        result=result_text[:500],
                    )
                    return result_text

            if name in mcp_allowed_ids:
                if self.mcp_manager is None:
                    raise ValueError(
                        f"Subagent '{self.config.id}' cannot call MCP tool '{name}' without an MCP manager"
                    )

                emit_event(
                    EventType.TOOL_CALL_STARTED,
                    subagent_id=self.config.id,
                    tool=name,
                    arguments=arguments,
                    source="mcp",
                )

                with observe(
                    name=f"tool:{name}",
                    as_type="span",
                    input={"arguments": arguments, "tool": name},
                    metadata={
                        "component": "subagent",
                        "subagent_id": self.config.id,
                        "tool": name,
                        "tool_source": "mcp",
                    },
                ) as tool_span:
                    result_text = await self.mcp_manager.call_tool(name, arguments)
                    if tool_span is not None:
                        tool_span.update(output=result_text)
                    emit_event(
                        EventType.TOOL_CALL_FINISHED,
                        subagent_id=self.config.id,
                        tool=name,
                        result=result_text[:500],
                        source="mcp",
                    )
                    return result_text

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
