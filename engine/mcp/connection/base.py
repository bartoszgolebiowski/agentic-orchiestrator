from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AsyncExitStack

from mcp import ClientSession


class McpConnector(ABC):
    @abstractmethod
    async def connect(self) -> tuple[AsyncExitStack, ClientSession]:
        """Create and initialize an MCP client session."""
