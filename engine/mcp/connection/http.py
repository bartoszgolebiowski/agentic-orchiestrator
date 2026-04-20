from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from datetime import timedelta

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from engine.mcp.connection.base import McpConnector
from engine.mcp.models import McpHttpConnectionConfig, McpServerConfig


class HttpConnector(McpConnector):
    def __init__(self, config: McpServerConfig) -> None:
        if not isinstance(config.connection, McpHttpConnectionConfig):
            raise TypeError("HttpConnector requires McpHttpConnectionConfig")
        self._config = config
        self._connection = config.connection

    async def connect(self) -> tuple[AsyncExitStack, ClientSession]:
        stack = AsyncExitStack()
        try:
            timeout = httpx.Timeout(self._connection.request_timeout_seconds)
            http_client = create_mcp_http_client(headers=self._connection.headers or None, timeout=timeout)
            await stack.enter_async_context(http_client)
            read_stream, write_stream, _ = await stack.enter_async_context(
                streamable_http_client(
                    self._connection.url,
                    http_client=http_client,
                    terminate_on_close=self._connection.terminate_on_close,
                )
            )
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
