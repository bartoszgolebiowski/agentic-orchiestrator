from __future__ import annotations

from dataclasses import dataclass

from engine.mcp.discovery import ResolvedMcpTool
from engine.mcp.models import McpServerConfig


@dataclass(frozen=True)
class McpServerEntity:
    config: McpServerConfig


@dataclass(frozen=True)
class McpToolEntity:
    descriptor: ResolvedMcpTool

    @property
    def exposed_name(self) -> str:
        return self.descriptor.exposed_name
