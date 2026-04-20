from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from datetime import timedelta

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from engine.mcp.connection.base import McpConnector
from engine.mcp.models import McpServerConfig, McpStdioConnectionConfig


class StdioConnector(McpConnector):
    def __init__(self, config: McpServerConfig) -> None:
        if not isinstance(config.connection, McpStdioConnectionConfig):
            raise TypeError("StdioConnector requires McpStdioConnectionConfig")
        self._config = config
        self._connection = config.connection

    async def connect(self) -> tuple[AsyncExitStack, ClientSession]:
        stack = AsyncExitStack()
        try:
            server_parameters = StdioServerParameters(
                command=self._connection.command,
                args=self._connection.args,
                env=self._connection.env or None,
                cwd=self._connection.cwd,
                encoding=self._connection.encoding,
                encoding_error_handler=self._connection.encoding_error_handler,
            )
            read_stream, write_stream = await stack.enter_async_context(stdio_client(server_parameters))
            read_timeout = timedelta(seconds=self._connection.request_timeout_seconds)
            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=read_timeout,
                )
            )
            await asyncio.wait_for(session.initialize(), timeout=self._connection.startup_timeout_seconds)
            return stack, session
        except Exception:
            await stack.aclose()
            raise
