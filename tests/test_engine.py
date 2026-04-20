import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from engine.config.loader import load_engine_config
import engine.llm.client as llm
import engine.llm.tracing as tracing
from engine.config.models import (
    AgentReActStep,
    AgentReActStepOutput,
    DelegationAction,
    DelegationActionOutput,
    NodeConfig,
    RoleType,
    RoutingDecision,
)
from engine.security import (
    BoundaryViolationError,
    enforce_agent_boundary,
    enforce_no_direct_tool_access,
    enforce_no_subagent_calling_subagent,
)
from engine.tools.registry import get_tool, list_registered_tools
import engine.tools  # noqa: F401 — trigger registration


CONFIGS_DIR = Path(__file__).parent.parent / "configs"


# ─── YAML Loading & Validation ───


class TestYAMLLoading:
    def test_load_engine_config_succeeds(self):
        config = load_engine_config(CONFIGS_DIR)
        assert set(config.agents) == {"math_agent", "document_agent", "trello_update_agent"}
        assert set(config.subagents) == {
            "calculator_subagent",
            "markdown_extractor",
            "agile_mapper",
            "trello_publisher",
            "trello_intake_parser",
            "trello_task_matcher",
            "trello_task_operator",
        }
        assert set(config.tools) == {
            "add",
            "multiply",
            "subtract",
            "read_markdown_structure",
        }
        assert set(config.mcps) == {"trello"}
        assert set(config.enrichers) == {"document_discovery"}
        assert config.enrichers["document_discovery"].executor == "glob_file_discovery"
        assert config.enrichers["document_discovery"].enabled is True
        assert config.enrichers["document_discovery"].executor_config == {
            "workflow_config_file": "enrichers/document/document_workflow.yaml",
            "input_key": "source_path",
        }
        assert config.agents["math_agent"].dependencies == ["calculator_subagent"]
        assert config.agents["document_agent"].dependencies == ["markdown_extractor", "agile_mapper", "trello_publisher"]
        assert config.agents["document_agent"].required_pipeline == ["markdown_extractor", "agile_mapper", "trello_publisher"]
        assert config.subagents["agile_mapper"].dependencies == []
        assert not hasattr(config.agents["document_agent"], "output_model")
        assert not hasattr(config.subagents["markdown_extractor"], "input_model")
        assert not hasattr(config.subagents["markdown_extractor"], "output_model")
        assert not hasattr(config.subagents["agile_mapper"], "input_model")
        assert not hasattr(config.subagents["agile_mapper"], "output_model")
        assert not hasattr(config.subagents["trello_publisher"], "input_model")
        assert not hasattr(config.subagents["trello_publisher"], "output_model")
        assert config.subagents["trello_publisher"].mcp_dependencies == ["trello"]

    def test_agents_reference_valid_subagents(self):
        config = load_engine_config(CONFIGS_DIR)
        for agent_id, agent in config.agents.items():
            for dep in agent.dependencies:
                assert dep in config.subagents, (
                    f"Agent '{agent_id}' references missing subagent '{dep}'"
                )

    def test_subagents_reference_valid_tools(self):
        config = load_engine_config(CONFIGS_DIR)
        for sub_id, sub in config.subagents.items():
            for dep in sub.dependencies:
                assert dep in config.tools, (
                    f"Subagent '{sub_id}' references missing tool '{dep}'"
                )

    def test_node_configs_do_not_expose_model_contract_fields(self):
        config = load_engine_config(CONFIGS_DIR)
        assert "output_model" not in config.agents["document_agent"].model_dump()
        assert "input_model" not in config.subagents["markdown_extractor"].model_dump()
        assert "output_model" not in config.subagents["markdown_extractor"].model_dump()
        assert "input_model" not in config.subagents["agile_mapper"].model_dump()
        assert "output_model" not in config.subagents["agile_mapper"].model_dump()
        assert "input_model" not in config.subagents["trello_publisher"].model_dump()
        assert "output_model" not in config.subagents["trello_publisher"].model_dump()

    def test_load_engine_config_with_mcp_servers(self, tmp_path):
        (tmp_path / "tools").mkdir()
        (tmp_path / "subagents").mkdir()
        (tmp_path / "mcps").mkdir()

        (tmp_path / "tools" / "helper.yaml").write_text(
            """
id: helper
description: helper tool
parameters: {}
""".strip(),
            encoding="utf-8",
        )

        (tmp_path / "subagents" / "calc.yaml").write_text(
            """
id: calc
role_type: subagent
description: test subagent
system_prompt: test prompt
dependencies:
    - helper
mcp_dependencies:
    - remote_server
""".strip(),
            encoding="utf-8",
        )

        (tmp_path / "mcps" / "remote_server.yaml").write_text(
            """
id: remote_server
description: remote tool server
connection:
    transport: http
    url: https://example.com/mcp
""".strip(),
            encoding="utf-8",
        )

        config = load_engine_config(tmp_path)

        assert set(config.mcps) == {"remote_server"}
        assert config.subagents["calc"].mcp_dependencies == ["remote_server"]
        assert config.mcps["remote_server"].connection.url == "https://example.com/mcp"

    def test_invalid_mcp_reference_raises(self, tmp_path):
        (tmp_path / "tools").mkdir()
        (tmp_path / "subagents").mkdir()

        (tmp_path / "tools" / "helper.yaml").write_text(
            """
id: helper
description: helper tool
parameters: {}
""".strip(),
            encoding="utf-8",
        )

        (tmp_path / "subagents" / "calc.yaml").write_text(
            """
id: calc
role_type: subagent
description: test subagent
system_prompt: test prompt
dependencies:
    - helper
mcp_dependencies:
    - missing_server
""".strip(),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="unknown MCP server"):
            load_engine_config(tmp_path)

    def test_mcp_include_tools_invalid_server_ref_raises(self):
        """mcp_include_tools referencing a server not in mcp_dependencies should fail."""
        with pytest.raises(ValueError, match="mcp_include_tools references server"):
            NodeConfig(
                id="bad",
                role_type=RoleType.SUBAGENT,
                description="x",
                system_prompt="x",
                mcp_dependencies=["srv_a"],
                mcp_include_tools={"srv_b": ["tool1"]},
            )

    def test_mcp_include_tools_agent_cannot_declare(self):
        """Agents cannot use mcp_include_tools."""
        with pytest.raises(ValueError, match="cannot declare mcp_include_tools"):
            NodeConfig(
                id="bad_agent",
                role_type=RoleType.AGENT,
                description="x",
                system_prompt="x",
                dependencies=["sub1"],
                mcp_include_tools={"srv": ["tool"]},
            )

    def test_mcp_include_tools_valid_config_accepted(self):
        """Valid mcp_include_tools passes validation."""
        node = NodeConfig(
            id="ok",
            role_type=RoleType.SUBAGENT,
            description="x",
            system_prompt="x",
            mcp_dependencies=["srv"],
            mcp_include_tools={"srv": ["tool_a", "tool_b"]},
        )
        assert node.mcp_include_tools == {"srv": ["tool_a", "tool_b"]}

    def test_mcp_include_tools_loaded_from_yaml(self):
        """Verify mcp_include_tools round-trips through real config loading."""
        config = load_engine_config(CONFIGS_DIR)
        matcher = config.subagents["trello_task_matcher"]
        assert "trello" in matcher.mcp_include_tools
        assert "list_boards" in matcher.mcp_include_tools["trello"]
        operator = config.subagents["trello_task_operator"]
        assert "add_comment" in operator.mcp_include_tools["trello"]

    def test_invalid_yaml_missing_role_type(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "bad.yaml").write_text(
            "id: bad_agent\ndescription: test\nsystem_prompt: test\ndependencies: [x]\n"
        )
        with pytest.raises(Exception):
            load_engine_config(tmp_path)

    def test_invalid_reference_raises(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (tmp_path / "subagents").mkdir()
        (tmp_path / "tools").mkdir()
        (agents_dir / "a.yaml").write_text(
            "id: a\nrole_type: agent\ndescription: x\n"
            "system_prompt: x\ndependencies: [nonexistent]\n"
        )
        with pytest.raises(ValueError, match="unknown subagent"):
            load_engine_config(tmp_path)


# ─── Boundary Enforcement ───


class TestBoundaryEnforcement:
    def test_agent_can_call_allowed_subagent(self):
        enforce_agent_boundary(RoleType.AGENT, "sub1", ["sub1", "sub2"])

    def test_agent_cannot_call_unknown_subagent(self):
        with pytest.raises(BoundaryViolationError):
            enforce_agent_boundary(RoleType.AGENT, "unknown", ["sub1"])

    def test_subagent_can_call_allowed_tool(self):
        enforce_agent_boundary(RoleType.SUBAGENT, "add", ["add", "multiply"])

    def test_subagent_cannot_call_unknown_tool(self):
        with pytest.raises(BoundaryViolationError):
            enforce_agent_boundary(RoleType.SUBAGENT, "hack", ["add"])

    def test_agent_cannot_call_tool_directly(self):
        with pytest.raises(BoundaryViolationError):
            enforce_no_direct_tool_access(RoleType.AGENT, "add")

    def test_subagent_cannot_call_subagent(self):
        with pytest.raises(BoundaryViolationError):
            enforce_no_subagent_calling_subagent(RoleType.SUBAGENT, "other_sub")


# ─── Tool Registry ───


class TestToolRegistry:
    def test_math_tools_registered(self):
        registered = list_registered_tools()
        assert "add" in registered
        assert "multiply" in registered
        assert "subtract" in registered

    @pytest.mark.asyncio
    async def test_add_tool(self):
        fn = get_tool("add")
        result = await fn(a=2, b=3)
        assert result == "5"

    @pytest.mark.asyncio
    async def test_multiply_tool(self):
        fn = get_tool("multiply")
        result = await fn(a=4, b=5)
        assert result == "20"

    @pytest.mark.asyncio
    async def test_subtract_tool(self):
        fn = get_tool("subtract")
        result = await fn(a=10, b=3)
        assert result == "7"

    def test_get_unknown_tool_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            get_tool("nonexistent_tool")


# ─── LLM Client Configuration ───


class TestLLMClientConfiguration:
    def test_get_base_url_uses_precedence(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        monkeypatch.setenv("OPENAI_BASE_URL", "https://one.example/v1")
        monkeypatch.setenv("OPENAI_API_BASE", "https://two.example/v1")
        assert llm.get_base_url() == "https://one.example/v1"

        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        assert llm.get_base_url() == "https://two.example/v1"

    def test_raw_client_receives_base_url(self, monkeypatch):
        captured = {}

        class DummyClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
        monkeypatch.setattr("engine.llm.completion.AsyncOpenAI", DummyClient)

        client = llm.get_raw_client()

        assert isinstance(client, DummyClient)
        assert captured["api_key"] == "test-key"
        assert captured["base_url"] == "https://openrouter.ai/api/v1"

class TestTracingIntegration:
    def test_tracing_disabled_without_env(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)

        assert tracing.is_langfuse_enabled() is False
        assert tracing.get_langfuse() is None

    def test_langfuse_import_error_is_exposed(self, monkeypatch):
        import_error = RuntimeError("boom")
        monkeypatch.setattr(tracing, "get_client", None)
        monkeypatch.setattr(tracing, "_LANGFUSE_IMPORT_ERROR", import_error)

        assert tracing.get_langfuse_import_error() is import_error

    def test_langfuse_connection_status_logs_success(self, monkeypatch, caplog):
        class DummyClient:
            def auth_check(self):
                return True

        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:3000")
        monkeypatch.setattr(tracing, "get_client", lambda: DummyClient())

        with caplog.at_level("INFO"):
            assert tracing.log_langfuse_connection_status() is True

        assert "Langfuse connection established" in caplog.text

    @pytest.mark.asyncio
    async def test_chat_completion_records_generation_metadata(self, monkeypatch):
        calls = {}

        class DummyObservation:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self, **kwargs):
                calls.setdefault("updates", []).append(kwargs)

        def fake_observe(**kwargs):
            calls["observe"] = kwargs
            return DummyObservation()

        class DummyMessage:
            def model_dump(self):
                return {"role": "assistant", "content": "hi"}

        response = MagicMock()
        response.choices = [MagicMock(message=DummyMessage())]
        response.usage = MagicMock()
        response.usage.model_dump.return_value = {"input_tokens": 3, "output_tokens": 4}

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        monkeypatch.setattr("engine.llm.completion.observe", fake_observe)

        message = await llm.chat_completion(
            client=mock_client,
            messages=[
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "thinking",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 1, "b": 2}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "3"},
            ],
            model="gpt-test",
            trace_name="test-generation",
            trace_metadata={"run_id": "abc123"},
        )

        assert message.model_dump() == {"role": "assistant", "content": "hi"}
        assert calls["observe"]["name"] == "test-generation"
        assert isinstance(calls["observe"]["input"], str)
        assert "Trace kind: chat_completion" in calls["observe"]["input"]
        assert "Message count: 4" in calls["observe"]["input"]
        assert "SYSTEM" in calls["observe"]["input"]
        assert "USER" in calls["observe"]["input"]
        assert "hello" in calls["observe"]["input"]
        assert "Tool calls:" in calls["observe"]["input"]
        assert "add({\"a\": 1, \"b\": 2})" in calls["observe"]["input"]
        assert "Tool call id: call_1" in calls["observe"]["input"]
        assert calls["observe"]["metadata"] == {"run_id": "abc123"}
        usage_details = calls["updates"][0]["usage_details"]
        assert usage_details["input_tokens"] == 3
        assert usage_details["output_tokens"] == 4
        assert usage_details["total_tokens"] == 7

    @pytest.mark.asyncio
    async def test_structured_completion_uses_native_parse_api(self, monkeypatch):
        captured = {}

        class DummyObservation:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self, **kwargs):
                captured.setdefault("updates", []).append(kwargs)

        class DummyResponseModel(BaseModel):
            value: str

        class DummyMessage:
            content = None
            parsed = DummyResponseModel(value="ok")

        class DummyUsage:
            def model_dump(self):
                return {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                }

        class DummyResponse:
            choices = [MagicMock(message=DummyMessage())]
            usage = DummyUsage()

        class DummyCompletions:
            async def parse(self, **kwargs):
                captured.update(kwargs)
                return DummyResponse()

        class DummyClient:
            def __init__(self):
                self.responses = MagicMock(parse=DummyCompletions().parse)

        def fake_observe(**kwargs):
            captured["observe"] = kwargs
            return DummyObservation()

        monkeypatch.setattr("engine.llm.completion.observe", fake_observe)

        result = await llm.structured_completion(
            client=DummyClient(),
            messages=[{"role": "user", "content": "hello"}],
            response_model=DummyResponseModel,
            model="gpt-test",
        )

        assert isinstance(captured["observe"]["input"], str)
        assert "Trace kind: structured_completion" in captured["observe"]["input"]
        assert "Response model: DummyResponseModel" in captured["observe"]["input"]
        assert "hello" in captured["observe"]["input"]
        assert captured["text_format"] is DummyResponseModel
        assert captured["model"] == "gpt-test"
        usage_details = captured["updates"][0]["usage_details"]
        assert usage_details["input_tokens"] == 5
        assert usage_details["output_tokens"] == 2
        assert usage_details["total_tokens"] == 7
        assert result.value == "ok"

    @pytest.mark.asyncio
    async def test_structured_completion_falls_back_to_json_mode(self, monkeypatch):
        captured = {}

        class DummyObservation:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self, **kwargs):
                captured.setdefault("updates", []).append(kwargs)

        class DummyResponseModel(BaseModel):
            value: str

        class DummyMessage:
            content = '{"value":"ok"}'
            parsed = None

        class DummyUsage:
            def model_dump(self):
                return {
                    "prompt_tokens": 8,
                    "completion_tokens": 3,
                    "total_tokens": 11,
                }

        class DummyResponse:
            choices = [SimpleNamespace(message=DummyMessage())]
            usage = DummyUsage()

        class DummyBadRequestError(Exception):
            pass

        class DummyCompletions:
            async def parse(self, **kwargs):
                raise DummyBadRequestError(
                    "Provider returned error: response_format is invalid, recommended val is: must be text or json_object"
                )

            async def create(self, **kwargs):
                captured.update(kwargs)
                return DummyResponse()

        class DummyClient:
            def __init__(self):
                self.responses = SimpleNamespace(
                    parse=DummyCompletions().parse,
                    create=DummyCompletions().create,
                )

        def fake_observe(**kwargs):
            captured["observe"] = kwargs
            return DummyObservation()

        monkeypatch.setattr("engine.llm.completion.observe", fake_observe)
        monkeypatch.setattr("engine.llm.completion.BadRequestError", DummyBadRequestError)

        result = await llm.structured_completion(
            client=DummyClient(),
            messages=[{"role": "user", "content": "hello"}],
            response_model=DummyResponseModel,
            model="gpt-test",
        )

        assert isinstance(captured["observe"]["input"], str)
        assert "Trace kind: structured_completion" in captured["observe"]["input"]
        assert "Response model: DummyResponseModel" in captured["observe"]["input"]
        assert captured["text"] == {"format": {"type": "json_object"}}
        usage_details = captured["updates"][0]["usage_details"]
        assert usage_details["input_tokens"] == 8
        assert usage_details["output_tokens"] == 3
        assert usage_details["total_tokens"] == 11
        assert result.value == "ok"

    @pytest.mark.asyncio
    async def test_routing_decision_accepts_target_agent_alias(self):
        decision = RoutingDecision.model_validate_json(
            '{"target_agent": "math_agent", "task": "Compute 2 + 2"}'
        )

        assert decision.agent_id == "math_agent"
        assert decision.task == "Compute 2 + 2"

    def test_routing_decision_accepts_structured_input_json(self):
        decision = RoutingDecision.model_validate_json(
            '{"target_agent": "document_agent", "task": "Process document", "input_json": {"source_path": "docs/example.md"}}'
        )

        assert decision.input_json == {"source_path": "docs/example.md"}

    def test_provider_output_models_do_not_expose_input_json(self):
        assert "input_json" not in DelegationActionOutput.model_json_schema()["properties"]
        assert "input_json" not in AgentReActStepOutput.model_json_schema()["properties"]


# ─── Pydantic Response Models ───


class TestResponseModels:
    def test_routing_decision_valid(self):
        d = RoutingDecision(agent_id="math_agent", task="compute 2+2")
        assert d.agent_id == "math_agent"
        assert d.task == "compute 2+2"

    def test_delegation_action_valid(self):
        a = DelegationAction(subagent_id="calculator_subagent", task="add 2 and 3")
        assert a.subagent_id == "calculator_subagent"

    def test_delegation_action_coerces_stringified_input_json(self):
        a = DelegationAction(
            subagent_id="markdown_extractor",
            task="extract markdown",
            input_json='{"source_path": "docs/example.md"}',
        )

        assert a.input_json == {"source_path": "docs/example.md"}

    def test_agent_react_step_with_action(self):
        step = AgentReActStep(
            thought="I need to compute this",
            action=DelegationAction(subagent_id="calc", task="add 2+3"),
            final_answer=None,
        )
        assert step.action is not None
        assert step.final_answer is None

    def test_agent_react_step_with_final_answer(self):
        step = AgentReActStep(
            thought="I have the result",
            action=None,
            final_answer="The answer is 5",
        )
        assert step.action is None
        assert step.final_answer == "The answer is 5"

    def test_agent_react_step_serialization(self):
        step = AgentReActStep(
            thought="thinking",
            action=DelegationAction(subagent_id="sub1", task="do x"),
        )
        data = step.model_dump()
        assert data["thought"] == "thinking"
        assert data["action"]["subagent_id"] == "sub1"


# ─── Native ReAct Loop (for Subagent, unchanged) ───


class TestReActLoop:
    @pytest.mark.asyncio
    async def test_react_with_tool_call_then_final_answer(self):
        from engine.agents.react import ReActLoop

        mock_client = AsyncMock()

        # First call: LLM returns a tool call
        tool_call_msg = MagicMock()
        tool_call_msg.content = "I need to add numbers"
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "add"
        tc.function.arguments = json.dumps({"a": 2, "b": 3})
        tool_call_msg.tool_calls = [tc]
        tool_call_msg.model_dump.return_value = {
            "role": "assistant",
            "content": "I need to add numbers",
            "tool_calls": [{"id": "call_1", "function": {"name": "add", "arguments": '{"a":2,"b":3}'}, "type": "function"}],
        }

        # Second call: LLM returns final answer
        final_msg = MagicMock()
        final_msg.content = "The answer is 5"
        final_msg.tool_calls = None

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                MagicMock(choices=[MagicMock(message=tool_call_msg)]),
                MagicMock(choices=[MagicMock(message=final_msg)]),
            ]
        )

        async def handler(name, args):
            if name == "add":
                return str(args["a"] + args["b"])
            return "unknown"

        loop = ReActLoop(
            client=mock_client,
            system_prompt="test",
            tools_spec=[{"type": "function", "function": {"name": "add"}}],
            action_handler=handler,
            max_steps=5,
        )

        result = await loop.run("What is 2+3?")
        assert result == "The answer is 5"

    @pytest.mark.asyncio
    async def test_react_max_steps_exceeded(self):
        from engine.agents.react import ReActLoop, MaxStepsExceededError

        mock_client = AsyncMock()

        # Always returns tool calls, never a final answer
        tool_call_msg = MagicMock()
        tool_call_msg.content = "thinking"
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "add"
        tc.function.arguments = '{"a":1,"b":1}'
        tool_call_msg.tool_calls = [tc]
        tool_call_msg.model_dump.return_value = {
            "role": "assistant", "content": "thinking",
            "tool_calls": [{"id": "call_1", "function": {"name": "add", "arguments": '{"a":1,"b":1}'}, "type": "function"}],
        }

        mock_client.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=tool_call_msg)])
        )

        loop = ReActLoop(
            client=mock_client,
            system_prompt="test",
            tools_spec=[{"type": "function", "function": {"name": "add"}}],
            action_handler=AsyncMock(return_value="2"),
            max_steps=3,
        )

        with pytest.raises(MaxStepsExceededError):
            await loop.run("loop forever")


# ─── Per-Role Model Selection ───


class TestPerRoleModel:
    def test_node_config_with_model(self):
        nc = NodeConfig(
            id="test",
            role_type=RoleType.SUBAGENT,
            description="test",
            system_prompt="test",
            dependencies=["add"],
            model="gpt-4o-mini",
        )
        assert nc.model == "gpt-4o-mini"

    def test_node_config_without_model(self):
        nc = NodeConfig(
            id="test",
            role_type=RoleType.SUBAGENT,
            description="test",
            system_prompt="test",
            dependencies=["add"],
        )
        assert nc.model is None


class TestInstructorReActLoop:
    def _make_parse_response(self, parsed):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, content=None))]
        )

    def _make_parse_client(self, side_effect):
        client = SimpleNamespace()
        client.responses = SimpleNamespace(
            parse=AsyncMock(side_effect=side_effect),
            create=AsyncMock(),
        )
        return client

    @pytest.mark.asyncio
    async def test_instructor_react_delegation_then_final(self):
        from engine.agents.react import StructuredReActLoop

        # Step 1: LLM returns a delegation action
        step1 = AgentReActStep(
            thought="I need to calculate 2+3",
            action=DelegationAction(subagent_id="calculator_subagent", task="add 2 and 3"),
            final_answer=None,
        )
        # Step 2: LLM returns final answer
        step2 = AgentReActStep(
            thought="The subagent returned 5",
            action=None,
            final_answer="The answer is 5",
        )

        mock_instructor_client = self._make_parse_client([
            self._make_parse_response(step1),
            self._make_parse_response(step2),
        ])

        async def handler(name, args):
            assert name == "calculator_subagent"
            return "5"

        loop = StructuredReActLoop(
            client=mock_instructor_client,
            system_prompt="test agent",
            action_handler=handler,
            available_actions="- calculator_subagent: does math",
            max_steps=5,
        )

        result = await loop.run("What is 2+3?")
        assert result == "The answer is 5"

    @pytest.mark.asyncio
    async def test_instructor_react_immediate_final_answer(self):
        from engine.agents.react import StructuredReActLoop

        step = AgentReActStep(
            thought="This is trivial",
            action=None,
            final_answer="42",
        )

        mock_instructor_client = self._make_parse_client([self._make_parse_response(step)])

        loop = StructuredReActLoop(
            client=mock_instructor_client,
            system_prompt="test",
            action_handler=AsyncMock(),
            available_actions="- sub: does stuff",
            max_steps=5,
        )

        result = await loop.run("What is the meaning of life?")
        assert result == "42"

    @pytest.mark.asyncio
    async def test_instructor_react_strict_mode_requires_action_or_final(self):
        from engine.agents.react import StructuredReActLoop

        mock_instructor_client = self._make_parse_client([
            self._make_parse_response(
                AgentReActStep(
                    thought="I should delegate this",
                    action=DelegationAction(subagent_id="sub", task="do work"),
                    final_answer=None,
                )
            ),
            self._make_parse_response(
                AgentReActStep(
                    thought="The subagent finished",
                    action=None,
                    final_answer="ok",
                )
            ),
        ])

        loop = StructuredReActLoop(
            client=mock_instructor_client,
            system_prompt="test",
            action_handler=AsyncMock(return_value="ok"),
            available_actions="- sub: does stuff",
            max_steps=2,
            allow_thought_as_final=False,
        )

        result = await loop.run("do work")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_instructor_react_retries_on_multiple_tool_calls(self):
        from engine.agents.react import StructuredReActLoop

        step1 = AgentReActStep(
            thought="delegate once",
            action=DelegationAction(subagent_id="sub", task="do work"),
            final_answer=None,
        )
        step2 = AgentReActStep(
            thought="done",
            action=None,
            final_answer="finished",
        )
        mock_instructor_client = self._make_parse_client([
            Exception("Instructor does not support multiple tool calls, use List[Model] instead"),
            self._make_parse_response(step1),
            self._make_parse_response(step2),
        ])

        loop = StructuredReActLoop(
            client=mock_instructor_client,
            system_prompt="test",
            action_handler=AsyncMock(return_value="ok"),
            available_actions="- sub: does stuff",
            max_steps=2,
            allow_thought_as_final=False,
        )

        result = await loop.run("do work")
        assert result == "finished"
        assert mock_instructor_client.responses.parse.await_count == 3

    def test_instructor_react_prompt_is_single_action_only(self):
        from engine.agents.react import StructuredReActLoop

        loop = StructuredReActLoop(
            client=AsyncMock(),
            system_prompt="test",
            action_handler=AsyncMock(),
            available_actions="- sub: does stuff",
        )

        loop._init_messages("task")
        system_prompt = loop._messages[0]["content"]
        assert "emit exactly one action per response" in system_prompt
        assert "Do not batch multiple actions" in system_prompt

    def test_agent_react_step_rejects_literal_null(self):
        with pytest.raises(ValidationError, match="literal string 'null'"):
            AgentReActStep(
                thought="done",
                action=None,
                final_answer="null",
            )

    def test_agent_react_step_requires_action_or_final_answer(self):
        with pytest.raises(ValidationError, match="Either action or final_answer"):
            AgentReActStep(
                thought="done",
                action=None,
                final_answer=None,
            )

    def test_agent_react_step_coerces_stringified_action(self):
        step = AgentReActStep(
            thought="delegate",
            action='{"subagent": "calculator_subagent", "task": "add 2 and 3"}',
            final_answer=None,
        )
        assert isinstance(step.action, DelegationAction)
        assert step.action.subagent_id == "calculator_subagent"
        assert step.action.task == "add 2 and 3"

    @pytest.mark.asyncio
    async def test_instructor_react_strict_mode_rejects_thought_only(self):
        from engine.agents.react import StructuredReActLoop

        mock_instructor_client = self._make_parse_client([
            self._make_parse_response(SimpleNamespace(thought="maybe", action=None, final_answer=None))
        ])

        loop = StructuredReActLoop(
            client=mock_instructor_client,
            system_prompt="test",
            action_handler=AsyncMock(),
            available_actions="- sub: does stuff",
            max_steps=1,
            allow_thought_as_final=False,
        )

        with pytest.raises(ValueError, match="strict mode requires delegation"):
            await loop.run("do work")

    @pytest.mark.asyncio
    async def test_instructor_react_max_steps_exceeded(self):
        from engine.agents.react import StructuredReActLoop, MaxStepsExceededError

        # Always returns delegation, never final
        step = AgentReActStep(
            thought="Still thinking",
            action=DelegationAction(subagent_id="sub", task="keep going"),
            final_answer=None,
        )
        mock_instructor_client = self._make_parse_client(
            [self._make_parse_response(step)] * 3
        )

        loop = StructuredReActLoop(
            client=mock_instructor_client,
            system_prompt="test",
            action_handler=AsyncMock(return_value="some result"),
            available_actions="- sub: does stuff",
            max_steps=3,
        )

        with pytest.raises(MaxStepsExceededError):
            await loop.run("loop forever")
