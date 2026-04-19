from __future__ import annotations

import asyncio
import json
import logging
import anyio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from collections.abc import Awaitable, Callable
from typing import Any, Sequence, TypeVar

import httpx
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from mcp.types import CallToolResult, PaginatedRequestParams

from engine.core.tracing import observe
from engine.mcp.models import (
    McpHttpConnectionConfig,
    McpServerConfig,
    McpStdioConnectionConfig,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class ResolvedMcpTool:
    server_id: str
    remote_name: str
    exposed_name: str
    description: str
    input_schema: dict[str, Any]


def build_openai_mcp_tool_spec(tool: ResolvedMcpTool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.exposed_name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _format_content_block(content_block: Any) -> str:
    if getattr(content_block, "type", None) == "text":
        return getattr(content_block, "text", "")

    if hasattr(content_block, "model_dump"):
        try:
            payload = content_block.model_dump(mode="json")
        except TypeError:
            payload = content_block.model_dump()
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    return str(content_block)


def serialize_call_tool_result(result: CallToolResult) -> str:
    parts: list[str] = []

    if result.structuredContent is not None:
        parts.append(json.dumps(result.structuredContent, ensure_ascii=False, indent=2, sort_keys=True))

    for content_block in result.content:
        content_text = _format_content_block(content_block).strip()
        if content_text:
            parts.append(content_text)

    result_text = "\n".join(parts).strip()
    if result.isError:
        if result_text:
            return f"ERROR: {result_text}"
        return "ERROR: MCP tool call failed"
    return result_text


class McpServerRuntime:
    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self._session_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._session_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._tools: list[ResolvedMcpTool] | None = None

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

    async def _connect_stdio(self, connection: McpStdioConnectionConfig, stack: AsyncExitStack) -> ClientSession:
        server_parameters = StdioServerParameters(
            command=connection.command,
            args=connection.args,
            env=connection.env or None,
            cwd=connection.cwd,
            encoding=connection.encoding,
            encoding_error_handler=connection.encoding_error_handler,
        )
        read_stream, write_stream = await stack.enter_async_context(stdio_client(server_parameters))
        read_timeout = timedelta(seconds=connection.request_timeout_seconds)
        return await stack.enter_async_context(
            ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=read_timeout,
            )
        )

    async def _connect_http(self, connection: McpHttpConnectionConfig, stack: AsyncExitStack) -> ClientSession:
        http_timeout = httpx.Timeout(connection.request_timeout_seconds)
        http_client = create_mcp_http_client(headers=connection.headers or None, timeout=http_timeout)
        await stack.enter_async_context(http_client)
        read_stream, write_stream, _ = await stack.enter_async_context(
            streamable_http_client(
                connection.url,
                http_client=http_client,
                terminate_on_close=connection.terminate_on_close,
            )
        )
        read_timeout = timedelta(seconds=connection.request_timeout_seconds)
        return await stack.enter_async_context(
            ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=read_timeout,
            )
        )

    async def _connect(self) -> tuple[AsyncExitStack, ClientSession]:
        stack = AsyncExitStack()
        try:
            connection = self.config.connection
            if isinstance(connection, McpStdioConnectionConfig):
                session = await self._connect_stdio(connection, stack)
            else:
                session = await self._connect_http(connection, stack)

            await asyncio.wait_for(session.initialize(), timeout=connection.startup_timeout_seconds)
            return stack, session
        except Exception:
            await stack.aclose()
            raise

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
                    discovered: list[ResolvedMcpTool] = []
                    discovered_remote_names: set[str] = set()
                    discovered_exposed_names: set[str] = set()
                    requested_tools = set(self.config.include_tools)
                    excluded_tools = set(self.config.exclude_tools)
                    cursor: str | None = None

                    while True:
                        params = PaginatedRequestParams(cursor=cursor) if cursor else None
                        result = await active_session.list_tools(params=params)

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
                    result = await active_session.call_tool(tool.remote_name, arguments=arguments)
                    return serialize_call_tool_result(result)

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

        # Keep MCP discovery serialized so reconnects and tool registration stay
        # deterministic per server.
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
        tool = self._tool_index.get(exposed_name)
        if tool is None:
            await self.warmup()
            tool = self._tool_index.get(exposed_name)

        if tool is None:
            raise KeyError(f"MCP tool '{exposed_name}' is not registered")

        runtime = self._get_runtime(tool.server_id)
        return await runtime.call_tool(exposed_name, arguments)

    async def warmup(self) -> None:
        if not self._runtimes:
            return
        await self.describe_tools(self._runtimes.keys())

    async def health(self) -> dict[str, dict[str, Any]]:
        """Return per-server health status with tool counts or error details."""
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
