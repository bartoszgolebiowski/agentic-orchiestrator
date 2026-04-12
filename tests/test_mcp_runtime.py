from __future__ import annotations

import asyncio

import pytest

from engine.mcp.runtime import McpManager, ResolvedMcpTool, build_openai_mcp_tool_spec
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