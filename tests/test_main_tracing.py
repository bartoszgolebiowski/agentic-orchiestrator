from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.core.models import EngineConfig, EnricherConfig, OrchestratorConfig
import engine.main as main_module


class _DummySpan:
    def __init__(self, state: dict[str, list[tuple[str, object]]], name: str) -> None:
        self._state = state
        self._name = name

    def __enter__(self) -> "_DummySpan":
        self._state["stack"].append(self._name)
        self._state["events"].append(("enter", self._name))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._state["events"].append(("exit", self._name))
        self._state["stack"].pop()
        return False

    def update(self, **kwargs) -> None:
        self._state["events"].append(("update", self._name, kwargs))


@pytest.mark.asyncio
async def test_main_keeps_enrichment_and_orchestrator_under_one_root_trace(monkeypatch):
    state: dict[str, list[tuple[str, object]]] = {
        "events": [],
        "stack": [],
    }

    def fake_observe(**kwargs):
        state["events"].append(("observe", kwargs["name"], kwargs["as_type"]))
        return _DummySpan(state, kwargs["name"])

    async def fake_apply_enrichment(*, client, query, enrichers, config_dir, model):
        assert state["stack"] == ["agent-request"]
        state["events"].append(("enrichment", query, tuple(enrichers)))
        return SimpleNamespace(
            enricher_id="document_discovery",
            payloads=[{"source_path": "documents/example.md"}],
        )

    class DummyOrchestrator:
        def __init__(self, *args, **kwargs) -> None:
            state["events"].append(("orchestrator-init",))

        async def handle(self, query, input_json=None):
            assert state["stack"] == ["agent-request", "agent-run"]
            state["events"].append(("handle", query, input_json))
            return "final result"

    config = EngineConfig(
        orchestrator=OrchestratorConfig(),
        agents={},
        subagents={},
        tools={},
        enrichers={
            "document_discovery": EnricherConfig(
                id="document_discovery",
                description="Discover markdown files before routing.",
                executor="glob_file_discovery",
            )
        },
    )

    monkeypatch.setattr(main_module, "observe", fake_observe)
    monkeypatch.setattr(main_module, "load_engine_config", lambda path: config)
    monkeypatch.setattr(main_module, "validate_config_graph", lambda config: [])
    monkeypatch.setattr(main_module, "set_engine_config", lambda config: None)
    monkeypatch.setattr(main_module, "clear_engine_config", lambda: None)
    monkeypatch.setattr(main_module, "flush", lambda: None)
    monkeypatch.setattr(main_module, "log_langfuse_connection_status", lambda: True)
    monkeypatch.setattr(main_module, "get_raw_client", lambda: object())
    monkeypatch.setattr(main_module, "apply_enrichment", fake_apply_enrichment)
    monkeypatch.setattr(main_module, "Orchestrator", DummyOrchestrator)

    result = await main_module.main("Route this request", config_dir="configs")

    assert result == "final result"
    root_index = state["events"].index(("observe", "agent-request", "span"))
    child_index = state["events"].index(("observe", "agent-run", "span"))
    enrichment_index = state["events"].index(("enrichment", "Route this request", ("document_discovery",)))
    handle_index = state["events"].index(("handle", "Route this request", {"source_path": "documents/example.md"}))

    assert root_index < enrichment_index < handle_index
    assert root_index < child_index < handle_index
    assert ("enrichment", "Route this request", ("document_discovery",)) in state["events"]
    assert ("handle", "Route this request", {"source_path": "documents/example.md"}) in state["events"]
    assert ("update", "agent-request", {"output": "final result"}) in state["events"]
    assert state["stack"] == []


@pytest.mark.asyncio
async def test_main_leaves_externally_owned_mcp_manager_open(monkeypatch):
    state: dict[str, list[tuple[str, object]]] = {
        "events": [],
        "stack": [],
    }

    def fake_observe(**kwargs):
        state["events"].append(("observe", kwargs["name"], kwargs["as_type"]))
        return _DummySpan(state, kwargs["name"])

    class DummyMcpManager:
        async def warmup(self):
            state["events"].append(("warmup",))

        async def aclose(self):
            state["events"].append(("aclose",))

    class DummyOrchestrator:
        def __init__(self, *args, **kwargs) -> None:
            state["events"].append(("orchestrator-init",))

        async def handle(self, query, input_json=None):
            assert state["stack"] == ["agent-request", "agent-run"]
            state["events"].append(("handle", query, input_json))
            return "final result"

    config = EngineConfig(
        orchestrator=OrchestratorConfig(),
        agents={},
        subagents={},
        tools={},
        enrichers={},
    )
    manager = DummyMcpManager()

    monkeypatch.setattr(main_module, "observe", fake_observe)
    monkeypatch.setattr(main_module, "set_engine_config", lambda config: None)
    monkeypatch.setattr(main_module, "clear_engine_config", lambda: None)
    monkeypatch.setattr(main_module, "flush", lambda: None)
    monkeypatch.setattr(main_module, "log_langfuse_connection_status", lambda: True)
    monkeypatch.setattr(main_module, "get_raw_client", lambda: object())
    monkeypatch.setattr(main_module, "Orchestrator", DummyOrchestrator)

    result = await main_module.main(
        "Route this request",
        config_dir="configs",
        engine_config=config,
        mcp_manager=manager,
        owns_mcp_manager=False,
    )

    assert result == "final result"
    assert ("warmup",) not in state["events"]
    assert ("aclose",) not in state["events"]
    assert state["stack"] == []
