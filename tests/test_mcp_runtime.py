from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import anyio

import pytest

from engine.mcp.runtime import McpManager, McpServerRuntime, ResolvedMcpTool, build_openai_mcp_tool_spec, serialize_call_tool_result
from engine.mcp.models import McpHttpConnectionConfig, McpServerConfig
from engine.events import EventType
from engine.config.models import NodeConfig, RoleType
from engine.agents.subagent import SubagentExecutor


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


class _FailingRuntime:
    def __init__(self, server_id: str, *, error_message: str) -> None:
        self.server_id = server_id
        self.error_message = error_message

    async def list_tools(self) -> list[ResolvedMcpTool]:
        return [
            ResolvedMcpTool(
                server_id=self.server_id,
                remote_name="echo",
                exposed_name=f"{self.server_id}__echo",
                description="echo tool",
                input_schema={"type": "object", "properties": {}},
            )
        ]

    async def call_tool(self, exposed_name: str, arguments: dict[str, object]) -> str:
        raise ValueError(self.error_message)

    async def aclose(self) -> None:
        return None


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


def test_serialize_call_tool_result_wraps_error_payload() -> None:
    result = SimpleNamespace(
        structuredContent=None,
        content=[SimpleNamespace(type="text", text="validation failed")],
        isError=True,
    )

    payload = json.loads(serialize_call_tool_result(result))

    assert payload == {"hint": "validation failed", "status": "error"}


@pytest.mark.asyncio
async def test_mcp_manager_call_tool_returns_error_json_on_runtime_failure() -> None:
    manager = McpManager({})
    runtime = _FailingRuntime("srv", error_message="board not selected")
    manager._runtimes = {"srv": runtime}  # type: ignore[dict-item]
    manager._tool_index = {
        "srv__echo": ResolvedMcpTool(
            server_id="srv",
            remote_name="echo",
            exposed_name="srv__echo",
            description="echo tool",
            input_schema={"type": "object", "properties": {}},
        )
    }  # type: ignore[attr-defined]

    payload = json.loads(await manager.call_tool("srv__echo", {"value": 1}))

    assert payload == {"hint": "board not selected", "status": "error"}


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
    from engine.config.models import EngineConfig, OrchestratorConfig
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
    specs, descriptions, allowed_ids, _ = await executor._build_mcp_tool_context()

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
    specs, descriptions, allowed_ids, _ = await executor._build_mcp_tool_context()

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
    specs, descriptions, allowed_ids, _ = await executor._build_mcp_tool_context()

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
    _, _, allowed_ids, _ = await executor._build_mcp_tool_context()

    assert "a__toolA1" in allowed_ids
    assert "a__toolA2" not in allowed_ids
    assert "b__toolB1" in allowed_ids
    await mgr.aclose()


class _DummyAssistantMessage:
    def __init__(self, *, content: str, tool_calls: list[SimpleNamespace] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "role": "assistant",
            "content": self.content,
        }
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in self.tool_calls
            ]
        return payload


class _DummyMcpManager:
    def __init__(self, *, call_results: list[str]) -> None:
        self._call_results = list(call_results)
        self.call_history: list[tuple[str, dict[str, object]]] = []

    async def describe_tools(self, server_ids: list[str]) -> list[ResolvedMcpTool]:
        assert server_ids == ["trello"]
        return [
            ResolvedMcpTool(
                server_id="trello",
                remote_name="add_card_to_list",
                exposed_name="trello__add_card_to_list",
                description="add a card",
                input_schema={"type": "object", "properties": {}},
            )
        ]

    async def call_tool(self, exposed_name: str, arguments: dict[str, object]) -> str:
        self.call_history.append((exposed_name, arguments))
        return self._call_results.pop(0)


def _tool_call(*, call_id: str, name: str, arguments: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _snapshot_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    return json.loads(json.dumps(messages))


@pytest.mark.asyncio
async def test_subagent_execute_projects_mcp_output_before_next_react_turn(monkeypatch) -> None:
    verbose_result = json.dumps(
        {
            "id": "card-1",
            "name": "Implement login",
            "desc": "x" * 500,
            "due": "2026-04-12T12:00:00.000Z",
            "dueComplete": False,
            "idList": "list-1",
            "labels": [
                {"id": "lbl-1", "name": "backend", "color": "blue"},
                {"id": "lbl-2", "name": "urgent", "color": "red"},
            ],
            "url": "https://trello.example/card/1",
            "shortUrl": "https://trello.example/c/1",
            "memberships": [{"idMember": "m1", "memberType": "normal"}],
            "badges": {"votes": 99, "attachments": 12},
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    mcp_manager = _DummyMcpManager(call_results=[verbose_result])

    sub_config = NodeConfig(
        id="trello_publisher_test",
        role_type=RoleType.SUBAGENT,
        description="test",
        system_prompt="test",
        mcp_dependencies=["trello"],
        mcp_include_tools={"trello": ["add_card_to_list"]},
        tool_result_projection={
            "trello__add_card_to_list": {
                "id": "$.id",
                "name": "$.name",
                "desc": "$.desc",
                "due": "$.due",
                "dueComplete": "$.dueComplete",
                "idList": "$.idList",
                "labels": "$.labels[*].name",
                "url": "$.url",
                "shortUrl": "$.shortUrl",
            }
        },
        max_steps=3,
    )
    executor = SubagentExecutor(
        config=sub_config,
        engine_config=_make_engine_config(),
        client=SimpleNamespace(),
        mcp_manager=mcp_manager,
    )

    captured_threads: list[list[dict[str, object]]] = []
    responses = [
        _DummyAssistantMessage(
            content="Need to create a card",
            tool_calls=[
                _tool_call(
                    call_id="call_1",
                    name="trello__add_card_to_list",
                    arguments={"listId": "list-1", "name": "Implement login"},
                )
            ],
        ),
        _DummyAssistantMessage(content='{"status":"ok"}'),
    ]
    captured_events: list[tuple[EventType, dict[str, object]]] = []

    async def fake_chat_completion(*, client, messages, tools, model, trace_name, trace_metadata, temperature=0.0):
        captured_threads.append(_snapshot_messages(messages))
        return responses.pop(0)

    def fake_emit_event(event_type: EventType, **data: object) -> None:
        captured_events.append((event_type, data))

    monkeypatch.setattr("engine.agents.react.native.chat_completion", fake_chat_completion)
    monkeypatch.setattr("engine.agents.execution.executor.emit_event", fake_emit_event)
    monkeypatch.setattr("engine.agents.execution.local_tool_handler.emit_event", fake_emit_event)
    monkeypatch.setattr("engine.agents.execution.mcp_tool_handler.emit_event", fake_emit_event)
    monkeypatch.setattr("engine.agents.execution.hitl_handler.emit_event", fake_emit_event)

    result = await executor.execute("publish", input_json={"plan": "demo"})

    assert result == '{"status":"ok"}'
    assert mcp_manager.call_history == [
        (
            "trello__add_card_to_list",
            {"listId": "list-1", "name": "Implement login"},
        )
    ]
    assert len(captured_threads) == 2

    first_turn_tool_messages = [m for m in captured_threads[0] if m.get("role") == "tool"]
    assert first_turn_tool_messages == []

    second_turn_tool_messages = [m for m in captured_threads[1] if m.get("role") == "tool"]
    assert len(second_turn_tool_messages) == 1
    projected_observation = str(second_turn_tool_messages[0]["content"])
    projected_payload = json.loads(projected_observation)

    assert projected_payload == {
        "desc": "x" * 500,
        "due": "2026-04-12T12:00:00.000Z",
        "dueComplete": False,
        "id": "card-1",
        "idList": "list-1",
        "labels": ["backend", "urgent"],
        "name": "Implement login",
        "shortUrl": "https://trello.example/c/1",
        "url": "https://trello.example/card/1",
    }
    assert "memberships" not in projected_observation
    assert "badges" not in projected_observation
    assert len(projected_observation) < len(verbose_result)

    tool_finished_events = [
        event_data
        for event_type, event_data in captured_events
        if event_type == EventType.TOOL_CALL_FINISHED and event_data.get("tool") == "trello__add_card_to_list"
    ]
    assert len(tool_finished_events) == 1
    assert "io_usage_estimate" in tool_finished_events[0]
    io_usage = tool_finished_events[0]["io_usage_estimate"]
    assert io_usage["input_tokens_estimate"] > 0
    assert io_usage["output_tokens_estimate"] > 0


@pytest.mark.asyncio
async def test_subagent_execute_projects_every_mcp_observation_across_react_turns(monkeypatch) -> None:
    verbose_results = [
        json.dumps(
            {
                "id": "card-1",
                "name": "Task A",
                "desc": "first",
                "due": None,
                "dueComplete": False,
                "idList": "list-1",
                "labels": [{"name": "backend", "color": "blue"}],
                "url": "https://trello.example/card/1",
                "shortUrl": "https://trello.example/c/1",
                "attachments": [{"id": "a1", "name": "spec.pdf"}],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        json.dumps(
            {
                "id": "card-2",
                "name": "Task B",
                "desc": "second",
                "due": "2026-04-13T10:00:00.000Z",
                "dueComplete": True,
                "idList": "list-1",
                "labels": [{"name": "frontend", "color": "green"}],
                "url": "https://trello.example/card/2",
                "shortUrl": "https://trello.example/c/2",
                "attachments": [{"id": "a2", "name": "wireframe.png"}],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    ]
    mcp_manager = _DummyMcpManager(call_results=verbose_results)

    sub_config = NodeConfig(
        id="trello_publisher_test",
        role_type=RoleType.SUBAGENT,
        description="test",
        system_prompt="test",
        mcp_dependencies=["trello"],
        mcp_include_tools={"trello": ["add_card_to_list"]},
        tool_result_projection={
            "trello__add_card_to_list": {
                "id": "$.id",
                "name": "$.name",
                "desc": "$.desc",
                "due": "$.due",
                "dueComplete": "$.dueComplete",
                "idList": "$.idList",
                "labels": "$.labels[*].name",
                "url": "$.url",
                "shortUrl": "$.shortUrl",
            }
        },
        max_steps=4,
    )
    executor = SubagentExecutor(
        config=sub_config,
        engine_config=_make_engine_config(),
        client=SimpleNamespace(),
        mcp_manager=mcp_manager,
    )

    captured_threads: list[list[dict[str, object]]] = []
    responses = [
        _DummyAssistantMessage(
            content="Create first card",
            tool_calls=[
                _tool_call(
                    call_id="call_1",
                    name="trello__add_card_to_list",
                    arguments={"listId": "list-1", "name": "Task A"},
                )
            ],
        ),
        _DummyAssistantMessage(
            content="Create second card",
            tool_calls=[
                _tool_call(
                    call_id="call_2",
                    name="trello__add_card_to_list",
                    arguments={"listId": "list-1", "name": "Task B"},
                )
            ],
        ),
        _DummyAssistantMessage(content='{"status":"published"}'),
    ]

    async def fake_chat_completion(*, client, messages, tools, model, trace_name, trace_metadata, temperature=0.0):
        captured_threads.append(_snapshot_messages(messages))
        return responses.pop(0)

    monkeypatch.setattr("engine.agents.react.native.chat_completion", fake_chat_completion)

    result = await executor.execute("publish", input_json={"plan": "demo"})

    assert result == '{"status":"published"}'
    assert len(captured_threads) == 3

    third_turn_tool_messages = [m for m in captured_threads[2] if m.get("role") == "tool"]
    assert len(third_turn_tool_messages) == 2

    first_observation = str(third_turn_tool_messages[0]["content"])
    second_observation = str(third_turn_tool_messages[1]["content"])
    first_payload = json.loads(first_observation)
    second_payload = json.loads(second_observation)

    assert first_payload["id"] == "card-1"
    assert first_payload["labels"] == ["backend"]
    assert second_payload["id"] == "card-2"
    assert second_payload["labels"] == ["frontend"]
    assert "attachments" not in first_observation
    assert "attachments" not in second_observation


@pytest.mark.asyncio
async def test_subagent_execute_fails_closed_when_projection_cannot_apply(monkeypatch) -> None:
    verbose_raw_result = json.dumps(
        {
            "unexpected": "SENSITIVE-" + ("X" * 1000),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    mcp_manager = _DummyMcpManager(call_results=[verbose_raw_result])

    sub_config = NodeConfig(
        id="trello_publisher_test",
        role_type=RoleType.SUBAGENT,
        description="test",
        system_prompt="test",
        mcp_dependencies=["trello"],
        mcp_include_tools={"trello": ["add_card_to_list"]},
        tool_result_projection={
            "trello__add_card_to_list": {
                "id": "$.id",
                "name": "$.name",
                "desc": "$.desc",
                "due": "$.due",
                "dueComplete": "$.dueComplete",
                "idList": "$.idList",
                "labels": "$.labels[*].name",
                "url": "$.url",
                "shortUrl": "$.shortUrl",
            }
        },
        max_steps=3,
    )
    executor = SubagentExecutor(
        config=sub_config,
        engine_config=_make_engine_config(),
        client=SimpleNamespace(),
        mcp_manager=mcp_manager,
    )

    captured_threads: list[list[dict[str, object]]] = []
    responses = [
        _DummyAssistantMessage(
            content="Need to create a card",
            tool_calls=[
                _tool_call(
                    call_id="call_1",
                    name="trello__add_card_to_list",
                    arguments={"listId": "list-1", "name": "Implement login"},
                )
            ],
        ),
        _DummyAssistantMessage(content='{"status":"ok"}'),
    ]

    async def fake_chat_completion(*, client, messages, tools, model, trace_name, trace_metadata, temperature=0.0):
        captured_threads.append(_snapshot_messages(messages))
        return responses.pop(0)

    monkeypatch.setattr("engine.agents.react.native.chat_completion", fake_chat_completion)

    result = await executor.execute("publish", input_json={"plan": "demo"})

    assert result == '{"status":"ok"}'
    assert len(captured_threads) == 2

    second_turn_tool_messages = [m for m in captured_threads[1] if m.get("role") == "tool"]
    assert len(second_turn_tool_messages) == 1
    observation = str(second_turn_tool_messages[0]["content"])

    assert observation.startswith("ERROR:")
    assert "SENSITIVE-" not in observation
    assert "unexpected" not in observation
    assert verbose_raw_result not in json.dumps(captured_threads[1], ensure_ascii=False)


@pytest.mark.asyncio
async def test_subagent_execute_preserves_mcp_error_json(monkeypatch) -> None:
    error_result = json.dumps(
        {"status": "error", "hint": "board not selected"},
        ensure_ascii=False,
        sort_keys=True,
    )
    mcp_manager = _DummyMcpManager(call_results=[error_result])

    sub_config = NodeConfig(
        id="trello_publisher_test",
        role_type=RoleType.SUBAGENT,
        description="test",
        system_prompt="test",
        mcp_dependencies=["trello"],
        mcp_include_tools={"trello": ["add_card_to_list"]},
        tool_result_projection={
            "trello__add_card_to_list": {
                "id": "$.id",
                "name": "$.name",
                "desc": "$.desc",
                "due": "$.due",
                "dueComplete": "$.dueComplete",
                "idList": "$.idList",
                "labels": "$.labels[*].name",
                "url": "$.url",
                "shortUrl": "$.shortUrl",
            }
        },
        max_steps=3,
    )
    executor = SubagentExecutor(
        config=sub_config,
        engine_config=_make_engine_config(),
        client=SimpleNamespace(),
        mcp_manager=mcp_manager,
    )

    captured_threads: list[list[dict[str, object]]] = []
    responses = [
        _DummyAssistantMessage(
            content="Need to create a card",
            tool_calls=[
                _tool_call(
                    call_id="call_1",
                    name="trello__add_card_to_list",
                    arguments={"listId": "list-1", "name": "Implement login"},
                )
            ],
        ),
        _DummyAssistantMessage(content='{"status":"ok"}'),
    ]

    async def fake_chat_completion(*, client, messages, tools, model, trace_name, trace_metadata, temperature=0.0):
        captured_threads.append(_snapshot_messages(messages))
        return responses.pop(0)

    monkeypatch.setattr("engine.agents.react.native.chat_completion", fake_chat_completion)

    result = await executor.execute("publish", input_json={"plan": "demo"})

    assert result == '{"status":"ok"}'
    assert len(captured_threads) == 2

    second_turn_tool_messages = [m for m in captured_threads[1] if m.get("role") == "tool"]
    assert len(second_turn_tool_messages) == 1
    observation = json.loads(str(second_turn_tool_messages[0]["content"]))

    assert observation == {"hint": "board not selected", "status": "error"}