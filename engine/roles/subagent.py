from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from engine.core.models import NodeConfig, RoleType, EngineConfig
from engine.core.react import ReActLoop
from engine.core.security import enforce_agent_boundary
from engine.core.tracing import observe
from engine.tools.registry import get_tool, build_openai_tool_spec

logger = logging.getLogger(__name__)


class SubagentExecutor:
    def __init__(
        self,
        config: NodeConfig,
        engine_config: EngineConfig,
        client: AsyncOpenAI,
        model: str | None = None,
    ) -> None:
        self.config = config
        self.engine_config = engine_config
        self.client = client
        self.model = model

        self._tool_specs = self._build_tool_specs()

    def _build_tool_specs(self) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for tool_id in self.config.dependencies:
            defn = self.engine_config.tools[tool_id]
            specs.append(build_openai_tool_spec(tool_id, defn))
        return specs

    async def _handle_action(self, name: str, arguments: dict[str, Any]) -> str:
        enforce_agent_boundary(
            caller_role=RoleType.SUBAGENT,
            callee_id=name,
            allowed_ids=self.config.dependencies,
        )

        tool_fn = get_tool(name)
        with observe(
            name=f"tool:{name}",
            as_type="span",
            input={"arguments": arguments, "tool": name},
            metadata={"component": "subagent", "subagent_id": self.config.id, "tool": name},
        ) as tool_span:
            result = await tool_fn(**arguments)
            result_text = str(result)
            if tool_span is not None:
                tool_span.update(output=result_text)
            return result_text

    async def execute(self, task: str) -> str:
        logger.info(f"SubagentExecutor[{self.config.id}] starting task: {task[:100]}")

        with observe(
            name=f"subagent-plan:{self.config.id}",
            as_type="span",
            input={"task": task, "subagent_id": self.config.id},
            metadata={"component": "subagent", "subagent_id": self.config.id},
        ) as subagent_span:
            loop = ReActLoop(
                client=self.client,
                system_prompt=self.config.system_prompt,
                tools_spec=self._tool_specs,
                action_handler=self._handle_action,
                max_steps=self.config.max_steps,
                model=self.config.model or self.model,
            )

            result = await loop.run(task)
            if subagent_span is not None:
                subagent_span.update(output=result)
            return result
