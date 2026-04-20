from __future__ import annotations

from typing import Any

from engine.config.models import EngineConfig, NodeConfig
from engine.events import EventType, emit_event
from engine.llm.client import estimate_io_usage
from engine.llm.tracing import observe
from engine.tools.projection import project_tool_result_strict
from engine.tools.registry import build_openai_tool_spec, get_tool


class LocalToolHandler:
    def __init__(self, *, config: NodeConfig, engine_config: EngineConfig) -> None:
        self._config = config
        self._engine_config = engine_config

    def build_specs(self) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        specs: list[dict[str, Any]] = []
        descriptions: list[str] = []
        allowed_ids: list[str] = []

        for tool_id in self._config.dependencies:
            definition = self._engine_config.tools[tool_id]
            specs.append(build_openai_tool_spec(tool_id, definition))
            descriptions.append(f"- {tool_id}: {definition.description}")
            allowed_ids.append(tool_id)

        return specs, descriptions, allowed_ids

    async def execute(
        self,
        *,
        subagent_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        projections: dict[str, Any],
    ) -> str:
        tool_fn = get_tool(tool_name)

        emit_event(
            EventType.TOOL_CALL_STARTED,
            subagent_id=subagent_id,
            tool=tool_name,
            arguments=arguments,
        )

        with observe(
            name=f"tool:{tool_name}",
            as_type="span",
            input={"arguments": arguments, "tool": tool_name},
            metadata={
                "component": "subagent",
                "subagent_id": subagent_id,
                "tool": tool_name,
                "tool_source": "local",
            },
        ) as tool_span:
            result = await tool_fn(**arguments)
            result_text = str(result)

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
                io_usage_estimate=io_usage,
            )

            return result_text
