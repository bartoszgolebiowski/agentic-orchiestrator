from __future__ import annotations

import asyncio
from types import SimpleNamespace

import anyio

import pytest

from engine.mcp.runtime import McpManager, McpServerRuntime, ResolvedMcpTool, build_openai_mcp_tool_spec
from engine.mcp.models import McpHttpConnectionConfig, McpServerConfig
from engine.core.models import NodeConfig, RoleType
from engine.roles.subagent import SubagentExecutor


class _ConcurrentGuardRuntime:
    def __init__(self, server_id: str) -> None:
        self.server_id = server_id
        self.active_calls = 0
        self.max_active_calls = 0
        self.closed = False

    async def list_tools(self) -> list[ResolvedMcpTool]:
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        await asyncio.sleep(0)
        self.active_calls -= 1
        return [
            ResolvedMcpTool(
                server_id=self.server_id,
                remote_name="list_items",
                exposed_name=f"{self.server_id}__list_items",
                description=f"tool from {self.server_id}",
                input_schema={"type": "object", "properties": {}},
            )
        ]

    async def aclose(self) -> None:
        self.closed = True


class _StreamStats:
    def __init__(self, *, open_send_streams: int, open_receive_streams: int) -> None:
        self.open_send_streams = open_send_streams
        self.open_receive_streams = open_receive_streams


class _FakeStream:
    def __init__(self, *, open_send_streams: int = 1, open_receive_streams: int = 1) -> None:
        self._stats = _StreamStats(
            open_send_streams=open_send_streams,
            open_receive_streams=open_receive_streams,
        )

    def statistics(self) -> _StreamStats:
        return self._stats


class _FakeExitStack:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _ReconnectableSession:
    def __init__(
        self,
        *,
        server_id: str,
        tools: list[tuple[str, str]] | None = None,
        call_result: str = "ok",
        fail_first_call: bool = False,
        open_send_streams: int = 1,
        open_receive_streams: int = 1,
    ) -> None:
        self.server_id = server_id
        self._read_stream = _FakeStream(
            open_send_streams=1,
            open_receive_streams=open_receive_streams,
        )
        self._write_stream = _FakeStream(
            open_send_streams=open_send_streams,
            open_receive_streams=1,
        )
        self._tools = tools or [("echo", "echo desc")]
        self._call_result = call_result
        self._fail_first_call = fail_first_call
        self.list_tools_calls = 0
        self.call_tool_calls = 0

    async def list_tools(self, params=None) -> SimpleNamespace:
        self.list_tools_calls += 1
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name=name,
                    description=description,
                    inputSchema={"type": "object", "properties": {}},
                )
                for name, description in self._tools
            ],
            nextCursor=None,
        )

    async def call_tool(self, name, arguments) -> SimpleNamespace:
        self.call_tool_calls += 1
        if self._fail_first_call and self.call_tool_calls == 1:
            raise anyio.BrokenResourceError

        return SimpleNamespace(
            structuredContent=None,
            content=[SimpleNamespace(type="text", text=self._call_result)],
            isError=False,
        )


def _make_runtime(server_id: str = "srv") -> tuple[McpServerRuntime, McpServerConfig]:
    config = McpServerConfig(
        id=server_id,
        description="test mcp server",
        connection=McpHttpConnectionConfig(url="https://example.com/mcp"),
    )
    return McpServerRuntime(config), config


@pytest.mark.asyncio
async def test_mcp_warmup_runs_server_discovery_without_overlap() -> None:
    manager = McpManager({})
    runtimes = {
        "alpha": _ConcurrentGuardRuntime("alpha"),
        "beta": _ConcurrentGuardRuntime("beta"),
    }
    manager._runtimes = runtimes  # type: ignore[attr-defined]

    await manager.warmup()

    assert runtimes["alpha"].max_active_calls == 1
    assert runtimes["beta"].max_active_calls == 1
    assert set(manager._tool_index) == {"alpha__list_items", "beta__list_items"}

    await manager.aclose()
    assert runtimes["alpha"].closed is True
    assert runtimes["beta"].closed is True


@pytest.mark.asyncio
async def test_mcp_ensure_session_reconnects_closed_session() -> None:
    runtime, _ = _make_runtime()

    old_stack = _FakeExitStack()
    closed_session = _ReconnectableSession(
        server_id="srv",
        open_send_streams=0,
        open_receive_streams=0,
    )
    new_session = _ReconnectableSession(server_id="srv")
    connect_calls = 0

    async def fake_connect():
        nonlocal connect_calls
        connect_calls += 1
        return _FakeExitStack(), new_session

    runtime._session = closed_session  # type: ignore[attr-defined]
    runtime._session_stack = old_stack  # type: ignore[attr-defined]
    runtime._connect = fake_connect  # type: ignore[method-assign]

    session = await runtime._ensure_session()  # type: ignore[attr-defined]

    assert session is new_session
    assert old_stack.closed is True
    assert connect_calls == 1


@pytest.mark.asyncio
async def test_mcp_call_tool_retries_after_broken_resource_error() -> None:
    runtime, _ = _make_runtime()

    tool = ResolvedMcpTool(
        server_id="srv",
        remote_name="echo",
        exposed_name="srv__echo",
        description="echo tool",
        input_schema={"type": "object", "properties": {}},
    )
    runtime._tools = [tool]  # type: ignore[attr-defined]

    first_stack = _FakeExitStack()
    first_session = _ReconnectableSession(
        server_id="srv",
        fail_first_call=True,
    )
    second_session = _ReconnectableSession(
        server_id="srv",
        call_result="recovered",
    )
    connect_calls = 0

    async def fake_connect():
        nonlocal connect_calls
        connect_calls += 1
        return _FakeExitStack(), second_session

    runtime._session = first_session  # type: ignore[attr-defined]
    runtime._session_stack = first_stack  # type: ignore[attr-defined]
    runtime._connect = fake_connect  # type: ignore[method-assign]

    result = await runtime.call_tool("srv__echo", {"value": 1})

    assert result == "recovered"
    assert first_stack.closed is True
    assert first_session.call_tool_calls == 1
    assert second_session.call_tool_calls == 1
    assert connect_calls == 1


# ─── mcp_include_tools filtering ───


class _MultiToolRuntime:
    """Fake runtime returning multiple tools per server."""

    def __init__(self, server_id: str, tools: list[tuple[str, str]]) -> None:
        self.server_id = server_id
        self._tools = [
            ResolvedMcpTool(
                server_id=server_id,
                remote_name=remote,
                exposed_name=exposed,
                description=f"{remote} desc",
                input_schema={"type": "object", "properties": {}},
            )
            for remote, exposed in tools
        ]
        self.closed = False

    async def list_tools(self) -> list[ResolvedMcpTool]:
        return list(self._tools)

    async def aclose(self) -> None:
        self.closed = True


def _make_mcp_manager(server_id: str, tools: list[tuple[str, str]]) -> McpManager:
    manager = McpManager({})
    manager._runtimes = {server_id: _MultiToolRuntime(server_id, tools)}  # type: ignore[dict-item]
    return manager


def _make_engine_config(**overrides):
    from engine.core.models import EngineConfig, OrchestratorConfig
    defaults = {
        "orchestrator": OrchestratorConfig(),
        "agents": {},
        "subagents": {},
        "tools": {},
        "mcps": {},
        "enrichers": {},
    }
    defaults.update(overrides)
    return EngineConfig(**defaults)


def _make_subagent_config(
    sub_id: str = "test_sub",
    mcp_deps: list[str] | None = None,
    mcp_include: dict[str, list[str]] | None = None,
) -> NodeConfig:
    return NodeConfig(
        id=sub_id,
        role_type=RoleType.SUBAGENT,
        description="test",
        system_prompt="test",
        dependencies=[],
        mcp_dependencies=mcp_deps or [],
        mcp_include_tools=mcp_include or {},
    )


@pytest.mark.asyncio
async def test_mcp_include_tools_filters_to_allowlist() -> None:
    """Only tools listed in mcp_include_tools should appear in the tool context."""
    manager = _make_mcp_manager("srv", [
        ("get_card", "srv__get_card"),
        ("add_card", "srv__add_card"),
        ("delete_card", "srv__delete_card"),
    ])

    sub_config = _make_subagent_config(
        mcp_deps=["srv"],
        mcp_include={"srv": ["get_card", "add_card"]},
    )
    engine_config = _make_engine_config()

    executor = SubagentExecutor(
        config=sub_config,
        engine_config=engine_config,
        client=None,  # type: ignore[arg-type]
        mcp_manager=manager,
    )
    specs, descriptions, allowed_ids = await executor._build_mcp_tool_context()

    assert set(allowed_ids) == {"srv__get_card", "srv__add_card"}
    assert len(specs) == 2
    assert "srv__delete_card" not in allowed_ids
    await manager.aclose()


@pytest.mark.asyncio
async def test_mcp_include_tools_empty_filter_blocks_all() -> None:
    """An empty list for a server means zero tools from that server."""
    manager = _make_mcp_manager("srv", [
        ("get_card", "srv__get_card"),
        ("add_card", "srv__add_card"),
    ])

    sub_config = _make_subagent_config(
        mcp_deps=["srv"],
        mcp_include={"srv": []},
    )
    engine_config = _make_engine_config()

    executor = SubagentExecutor(
        config=sub_config,
        engine_config=engine_config,
        client=None,  # type: ignore[arg-type]
        mcp_manager=manager,
    )
    specs, descriptions, allowed_ids = await executor._build_mcp_tool_context()

    assert allowed_ids == []
    assert specs == []
    await manager.aclose()


@pytest.mark.asyncio
async def test_mcp_include_tools_no_filter_returns_all() -> None:
    """Without mcp_include_tools, all tools pass through (backward compat)."""
    manager = _make_mcp_manager("srv", [
        ("get_card", "srv__get_card"),
        ("add_card", "srv__add_card"),
    ])

    sub_config = _make_subagent_config(mcp_deps=["srv"])
    engine_config = _make_engine_config()

    executor = SubagentExecutor(
        config=sub_config,
        engine_config=engine_config,
        client=None,  # type: ignore[arg-type]
        mcp_manager=manager,
    )
    specs, descriptions, allowed_ids = await executor._build_mcp_tool_context()

    assert set(allowed_ids) == {"srv__get_card", "srv__add_card"}
    await manager.aclose()


@pytest.mark.asyncio
async def test_mcp_include_tools_unknown_tool_raises() -> None:
    """Referencing a tool name not exposed by the server should raise ValueError."""
    manager = _make_mcp_manager("srv", [
        ("get_card", "srv__get_card"),
    ])

    sub_config = _make_subagent_config(
        mcp_deps=["srv"],
        mcp_include={"srv": ["get_card", "nonexistent_tool"]},
    )
    engine_config = _make_engine_config()

    executor = SubagentExecutor(
        config=sub_config,
        engine_config=engine_config,
        client=None,  # type: ignore[arg-type]
        mcp_manager=manager,
    )

    with pytest.raises(ValueError, match="nonexistent_tool"):
        await executor._build_mcp_tool_context()
    await manager.aclose()


@pytest.mark.asyncio
async def test_mcp_include_tools_unfiltered_server_keeps_all() -> None:
    """If mcp_include_tools omits a server, all its tools pass through."""
    mgr = McpManager({})
    mgr._runtimes = {  # type: ignore[dict-item]
        "srv_a": _MultiToolRuntime("srv_a", [("toolA1", "a__toolA1"), ("toolA2", "a__toolA2")]),
        "srv_b": _MultiToolRuntime("srv_b", [("toolB1", "b__toolB1")]),
    }

    sub_config = _make_subagent_config(
        mcp_deps=["srv_a", "srv_b"],
        mcp_include={"srv_a": ["toolA1"]},  # filter only srv_a
    )
    engine_config = _make_engine_config()

    executor = SubagentExecutor(
        config=sub_config,
        engine_config=engine_config,
        client=None,  # type: ignore[arg-type]
        mcp_manager=mgr,
    )
    _, _, allowed_ids = await executor._build_mcp_tool_context()

    assert "a__toolA1" in allowed_ids
    assert "a__toolA2" not in allowed_ids
    assert "b__toolB1" in allowed_ids
    await mgr.aclose()