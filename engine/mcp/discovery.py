from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.types import PaginatedRequestParams

from engine.mcp.models import McpServerConfig


@dataclass(frozen=True)
class ResolvedMcpTool:
    server_id: str
    remote_name: str
    exposed_name: str
    description: str
    input_schema: dict[str, Any]


class McpToolDiscovery:
    def __init__(self, config: McpServerConfig) -> None:
        self.config = config

    async def discover(self, session: ClientSession) -> list[ResolvedMcpTool]:
        discovered: list[ResolvedMcpTool] = []
        discovered_remote_names: set[str] = set()
        discovered_exposed_names: set[str] = set()
        requested_tools = set(self.config.include_tools)
        excluded_tools = set(self.config.exclude_tools)
        cursor: str | None = None

        while True:
            params = PaginatedRequestParams(cursor=cursor) if cursor else None
            result = await session.list_tools(params=params)

            for tool in result.tools:
                if requested_tools and tool.name not in requested_tools:
                    continue
                if tool.name in excluded_tools:
                    continue

                exposed_name = self.config.exposed_tool_name(tool.name)
                if exposed_name in discovered_exposed_names:
                    raise ValueError(
                        f"MCP server '{self.config.id}' produced a duplicate exposed tool name '{exposed_name}'"
                    )

                discovered.append(
                    ResolvedMcpTool(
                        server_id=self.config.id,
                        remote_name=tool.name,
                        exposed_name=exposed_name,
                        description=tool.description or f"MCP tool '{tool.name}' from server '{self.config.id}'",
                        input_schema=tool.inputSchema,
                    )
                )
                discovered_remote_names.add(tool.name)
                discovered_exposed_names.add(exposed_name)

            cursor = result.nextCursor
            if not cursor:
                break

        if requested_tools:
            missing = sorted(requested_tools - discovered_remote_names)
            if missing:
                raise ValueError(
                    f"MCP server '{self.config.id}' did not expose requested tools: {missing}"
                )

        if not discovered:
            raise ValueError(f"MCP server '{self.config.id}' did not expose any usable tools")

        return discovered
