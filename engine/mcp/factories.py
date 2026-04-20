from __future__ import annotations

from engine.mcp.discovery import ResolvedMcpTool
from engine.mcp.entities import McpServerEntity, McpToolEntity
from engine.mcp.models import McpServerConfig


class McpServerFactory:
    @staticmethod
    def create(config: McpServerConfig) -> McpServerEntity:
        return McpServerEntity(config=config)


class McpToolFactory:
    @staticmethod
    def create(descriptor: ResolvedMcpTool) -> McpToolEntity:
        return McpToolEntity(descriptor=descriptor)
