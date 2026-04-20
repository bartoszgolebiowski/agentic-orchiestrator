from __future__ import annotations

from typing import Any

from engine.config.models import NodeConfig
from engine.events import EventType, emit_event
from engine.llm.client import estimate_io_usage
from engine.llm.tracing import observe
from engine.mcp.runtime import McpManager, build_openai_mcp_tool_spec
from engine.tools.projection import project_tool_result_strict
from engine.agents.execution.hitl_handler import HitlHandler


class McpToolHandler:
    def __init__(
        self,
        *,
        config: NodeConfig,
        mcp_manager: McpManager | None,
        hitl_handler: HitlHandler,
    ) -> None:
        self._config = config
        self._mcp_manager = mcp_manager
        self._hitl_handler = hitl_handler

    async def build_context(self) -> tuple[list[dict[str, Any]], list[str], list[str], set[str]]:
        if not self._config.mcp_dependencies:
            return [], [], [], set()

        if self._mcp_manager is None:
            raise ValueError(
                f"Subagent '{self._config.id}' declares mcp_dependencies but no MCP manager was provided"
            )

        descriptors = await self._mcp_manager.describe_tools(self._config.mcp_dependencies)

        if self._config.mcp_include_tools:
            filtered = []
            for descriptor in descriptors:
                server_filter = self._config.mcp_include_tools.get(descriptor.server_id)
                if server_filter is None:
                    filtered.append(descriptor)
                elif descriptor.remote_name in set(server_filter):
                    filtered.append(descriptor)

            for server_id, allowed_names in self._config.mcp_include_tools.items():
                discovered = {d.remote_name for d in descriptors if d.server_id == server_id}
                missing = set(allowed_names) - discovered
                if missing:
                    raise ValueError(
                        f"Subagent '{self._config.id}' mcp_include_tools references "
                        f"unknown tools from server '{server_id}': {sorted(missing)}"
                    )
            descriptors = filtered

        skip_hitl_ids: set[str] = set()
        if self._config.mcp_skip_hitl_tools:
            for server_id, skip_names in self._config.mcp_skip_hitl_tools.items():
                discovered = {d.remote_name for d in descriptors if d.server_id == server_id}
                missing = set(skip_names) - discovered
                if missing:
                    raise ValueError(
                        f"Subagent '{self._config.id}' mcp_skip_hitl_tools references "
                        f"unknown tools from server '{server_id}': {sorted(missing)}"
                    )
            for descriptor in descriptors:
                server_skip = self._config.mcp_skip_hitl_tools.get(descriptor.server_id)
                if server_skip is not None and descriptor.remote_name in set(server_skip):
                    skip_hitl_ids.add(descriptor.exposed_name)

        specs: list[dict[str, Any]] = []
        descriptions: list[str] = []
        allowed_ids: list[str] = []

        for descriptor in descriptors:
            specs.append(build_openai_mcp_tool_spec(descriptor))
            descriptions.append(
                f"- {descriptor.exposed_name}: {descriptor.description} (MCP server: {descriptor.server_id})"
            )
            allowed_ids.append(descriptor.exposed_name)

        return specs, descriptions, allowed_ids, skip_hitl_ids

    async def execute(
        self,
        *,
        subagent_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        projections: dict[str, Any],
        skip_hitl: set[str],
    ) -> str:
        if self._mcp_manager is None:
            raise ValueError(
                f"Subagent '{self._config.id}' cannot call MCP tool '{tool_name}' without an MCP manager"
            )

        approved, maybe_modified_arguments, rejection_reason = await self._hitl_handler.authorize(
            subagent_id=subagent_id,
            tool_name=tool_name,
            arguments=arguments,
            source="mcp",
            skip_tools=skip_hitl,
        )
        if not approved:
            return f"TOOL REJECTED: {rejection_reason or 'rejected by human'}"

        arguments = maybe_modified_arguments

        emit_event(
            EventType.TOOL_CALL_STARTED,
            subagent_id=subagent_id,
            tool=tool_name,
            arguments=arguments,
            source="mcp",
        )

        with observe(
            name=f"tool:{tool_name}",
            as_type="span",
            input={"arguments": arguments, "tool": tool_name},
            metadata={
                "component": "subagent",
                "subagent_id": subagent_id,
                "tool": tool_name,
                "tool_source": "mcp",
            },
        ) as tool_span:
            result_text = await self._mcp_manager.call_tool(tool_name, arguments)

            projection = projections.get(tool_name)
            if projection is not None:
                result_text = project_tool_result_strict(result_text, projection)

            io_usage = estimate_io_usage(arguments, result_text)
            if tool_span is not None:
                tool_span.update(output=result_text, metadata={"io_usage_estimate": io_usage})

            emit_event(
                EventType.TOOL_CALL_FINISHED,
                subagent_id=subagent_id,
                tool=tool_name,
                result=result_text[:500],
                source="mcp",
                io_usage_estimate=io_usage,
            )

            return result_text
