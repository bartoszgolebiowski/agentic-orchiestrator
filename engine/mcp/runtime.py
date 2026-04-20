from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from typing import Any, TypeVar

import anyio
from mcp import ClientSession

from engine.llm.tracing import observe
from engine.mcp.caller import (
    McpToolCaller,
    format_error_hint as _format_error_hint,
    serialize_call_tool_result,
    serialize_error_response as _serialize_error_response,
)
from engine.mcp.connection import HttpConnector, StdioConnector
from engine.mcp.discovery import McpToolDiscovery, ResolvedMcpTool
from engine.mcp.models import McpHttpConnectionConfig, McpServerConfig, McpStdioConnectionConfig

logger = logging.getLogger(__name__)
T = TypeVar("T")


def build_openai_mcp_tool_spec(tool: ResolvedMcpTool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.exposed_name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


class McpServerRuntime:
    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self._session_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._session_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._tools: list[ResolvedMcpTool] | None = None

        self._discovery = McpToolDiscovery(config)
        self._caller = McpToolCaller()

    def _session_is_open(self) -> bool:
        session = self._session
        if session is None or self._session_stack is None:
            return False

        try:
            read_stats = session._read_stream.statistics()  # type: ignore[attr-defined]
            write_stats = session._write_stream.statistics()  # type: ignore[attr-defined]
        except Exception:
            return False

        return read_stats.open_receive_streams > 0 and write_stats.open_send_streams > 0

    async def _close_session(self) -> None:
        stack = self._session_stack
        self._session_stack = None
        self._session = None
        self._tools = None

        if stack is None:
            return

        try:
            await stack.aclose()
        except Exception:
            logger.debug(
                "Ignoring error while closing MCP session for '%s'",
                self.config.id,
                exc_info=True,
            )

    async def _run_with_reconnect(
        self,
        operation: Callable[[ClientSession], Awaitable[T]],
    ) -> T:
        session = await self._ensure_session()
        try:
            return await operation(session)
        except (anyio.ClosedResourceError, anyio.BrokenResourceError, BrokenPipeError, ConnectionResetError):
            logger.warning("MCP session for '%s' closed unexpectedly; reconnecting", self.config.id)
            await self._close_session()
            session = await self._ensure_session()
            return await operation(session)

    async def _connect_stdio(self, connection: McpStdioConnectionConfig) -> tuple[AsyncExitStack, ClientSession]:
        return await StdioConnector(self.config).connect()

    async def _connect_http(self, connection: McpHttpConnectionConfig) -> tuple[AsyncExitStack, ClientSession]:
        return await HttpConnector(self.config).connect()

    async def _connect(self) -> tuple[AsyncExitStack, ClientSession]:
        connection = self.config.connection
        if isinstance(connection, McpStdioConnectionConfig):
            return await self._connect_stdio(connection)
        return await self._connect_http(connection)

    async def _ensure_session(self) -> ClientSession:
        if self._session is not None and self._session_stack is not None and self._session_is_open():
            return self._session

        async with self._session_lock:
            if self._session is not None and self._session_stack is not None and self._session_is_open():
                return self._session

            if self._session is not None or self._session_stack is not None:
                await self._close_session()

            logger.info("Connecting to MCP server '%s'", self.config.id)
            stack, session = await self._connect()
            self._session_stack = stack
            self._session = session
            return session

    async def list_tools(self) -> list[ResolvedMcpTool]:
        async with self._request_lock:
            await self._ensure_session()

            if self._tools is not None:
                return self._tools

            with observe(
                name=f"mcp.list_tools:{self.config.id}",
                as_type="span",
                input={"server_id": self.config.id},
                metadata={"component": "mcp", "server_id": self.config.id},
            ) as span:
                async def discover(active_session: ClientSession) -> list[ResolvedMcpTool]:
                    return await self._discovery.discover(active_session)

                discovered = await self._run_with_reconnect(discover)

                if span is not None:
                    span.update(output={"tool_count": len(discovered)})

            self._tools = discovered
            return discovered

    async def call_tool(self, exposed_name: str, arguments: dict[str, Any]) -> str:
        tools = await self.list_tools()
        tool = next((item for item in tools if item.exposed_name == exposed_name), None)
        if tool is None:
            raise KeyError(f"MCP tool '{exposed_name}' is not available on server '{self.config.id}'")

        async with self._request_lock:
            with observe(
                name=f"mcp.call_tool:{exposed_name}",
                as_type="span",
                input={"server_id": self.config.id, "tool": exposed_name, "arguments": arguments},
                metadata={"component": "mcp", "server_id": self.config.id, "tool": exposed_name},
            ) as span:
                async def invoke(active_session: ClientSession) -> str:
                    return await self._caller.call(active_session, tool.remote_name, arguments)

                result_text = await self._run_with_reconnect(invoke)
                if span is not None:
                    span.update(output=result_text)
                return result_text

    async def aclose(self) -> None:
        await self._close_session()


class McpManager:
    def __init__(self, configs: dict[str, McpServerConfig]) -> None:
        self._runtimes = {server_id: McpServerRuntime(config) for server_id, config in configs.items()}
        self._tool_index: dict[str, ResolvedMcpTool] = {}

    def _get_runtime(self, server_id: str) -> McpServerRuntime:
        if server_id not in self._runtimes:
            raise KeyError(f"MCP server '{server_id}' is not registered")
        return self._runtimes[server_id]

    async def describe_tools(self, server_ids: Sequence[str]) -> list[ResolvedMcpTool]:
        seen_servers = list(dict.fromkeys(server_ids))

        discovery_results: list[list[ResolvedMcpTool]] = []
        for server_id in seen_servers:
            discovery_results.append(await self._get_runtime(server_id).list_tools())

        resolved: list[ResolvedMcpTool] = []
        for tools in discovery_results:
            for tool in tools:
                existing = self._tool_index.get(tool.exposed_name)
                if existing is not None and existing.server_id != tool.server_id:
                    raise ValueError(
                        f"MCP tool name collision for '{tool.exposed_name}' between '{existing.server_id}' and '{tool.server_id}'"
                    )
                self._tool_index[tool.exposed_name] = tool
                resolved.append(tool)

        return resolved

    async def call_tool(self, exposed_name: str, arguments: dict[str, Any]) -> str:
        try:
            tool = self._tool_index.get(exposed_name)
            if tool is None:
                await self.warmup()
                tool = self._tool_index.get(exposed_name)

            if tool is None:
                raise KeyError(f"MCP tool '{exposed_name}' is not registered")

            runtime = self._get_runtime(tool.server_id)
            return await runtime.call_tool(exposed_name, arguments)
        except Exception as exc:
            logger.warning("MCP tool call failed for '%s'", exposed_name, exc_info=True)
            return _serialize_error_response(_format_error_hint(exc))

    async def warmup(self) -> None:
        if not self._runtimes:
            return
        await self.describe_tools(self._runtimes.keys())

    async def health(self) -> dict[str, dict[str, Any]]:
        status: dict[str, dict[str, Any]] = {}
        for server_id, runtime in self._runtimes.items():
            try:
                tools = await runtime.list_tools()
                status[server_id] = {"status": "ok", "tool_count": len(tools)}
            except Exception as exc:
                status[server_id] = {"status": "error", "error": str(exc)}
        return status

    async def aclose(self) -> None:
        for runtime in self._runtimes.values():
            await runtime.aclose()


__all__ = [
    "McpManager",
    "McpServerRuntime",
    "ResolvedMcpTool",
    "build_openai_mcp_tool_spec",
    "serialize_call_tool_result",
]
