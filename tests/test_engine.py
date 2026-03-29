import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from engine.core.loader import load_engine_config
import engine.core.llm as llm
import engine.core.tracing as tracing
from engine.core.models import (
    ActionPoint,
    AgentReActStep,
    ConclusionsSummary,
    DelegationAction,
    EngineConfig,
    Fact,
    FactsSummary,
    NodeConfig,
    RoleType,
    RoutingDecision,
    SourceReference,
)
from engine.core.security import (
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
        assert "math_agent" in config.agents
        assert "transcript_analyst" in config.agents
        assert "calculator_subagent" in config.subagents
        assert "facts_extractor" in config.subagents
        assert "conclusions_analyst" in config.subagents
        assert "action_point_generator" in config.subagents
        assert "add" in config.tools
        assert "read_transcript" in config.tools
        assert "write_markdown_file" in config.tools
        assert "create_output_directory" in config.tools
        assert "log_ticket_payload" in config.tools

    def test_transcript_agent_prompt_enforces_single_action(self):
        config = load_engine_config(CONFIGS_DIR)
        prompt = config.agents["transcript_analyst"].system_prompt
        assert "exactly one delegated subagent action" in prompt or "exactly one delegated subagent action" in prompt.lower()
        assert "Do not batch multiple stages, multiple subagent actions, or multi-step plans in one response." in prompt

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
        monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)

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

        monkeypatch.setattr(llm, "observe", fake_observe)

        message = await llm.chat_completion(
            client=mock_client,
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-test",
            trace_name="test-generation",
            trace_metadata={"run_id": "abc123"},
        )

        assert message.model_dump() == {"role": "assistant", "content": "hi"}
        assert calls["observe"]["name"] == "test-generation"
        assert calls["observe"]["metadata"] == {"run_id": "abc123"}
        assert calls["updates"][0]["usage_details"] == {"input_tokens": 3, "output_tokens": 4}

    @pytest.mark.asyncio
    async def test_structured_completion_uses_native_parse_api(self, monkeypatch):
        captured = {}

        class DummyObservation:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self, **kwargs):
                pass

        class DummyResponseModel(BaseModel):
            value: str

        class DummyMessage:
            content = None
            parsed = DummyResponseModel(value="ok")

        class DummyResponse:
            choices = [MagicMock(message=DummyMessage())]

        class DummyCompletions:
            async def parse(self, **kwargs):
                captured.update(kwargs)
                return DummyResponse()

        class DummyClient:
            def __init__(self):
                self.responses = MagicMock(parse=DummyCompletions().parse)

        monkeypatch.setattr(llm, "observe", lambda **kwargs: DummyObservation())

        result = await llm.structured_completion(
            client=DummyClient(),
            messages=[{"role": "user", "content": "hello"}],
            response_model=DummyResponseModel,
            model="gpt-test",
        )

        assert captured["text_format"] is DummyResponseModel
        assert captured["model"] == "gpt-test"
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
                pass

        class DummyResponseModel(BaseModel):
            value: str

        class DummyMessage:
            content = '{"value":"ok"}'
            parsed = None

        class DummyResponse:
            choices = [SimpleNamespace(message=DummyMessage())]

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

        monkeypatch.setattr(llm, "observe", lambda **kwargs: DummyObservation())
        monkeypatch.setattr(llm, "BadRequestError", DummyBadRequestError)

        result = await llm.structured_completion(
            client=DummyClient(),
            messages=[{"role": "user", "content": "hello"}],
            response_model=DummyResponseModel,
            model="gpt-test",
        )

        assert captured["text"] == {"format": {"type": "json_object"}}
        assert result.value == "ok"

    @pytest.mark.asyncio
    async def test_routing_decision_accepts_target_agent_alias(self):
        decision = RoutingDecision.model_validate_json(
            '{"target_agent": "transcript_analyst", "task": "Analyze the transcript"}'
        )

        assert decision.agent_id == "transcript_analyst"
        assert decision.task == "Analyze the transcript"


# ─── Pydantic Response Models ───


class TestResponseModels:
    def test_routing_decision_valid(self):
        d = RoutingDecision(agent_id="math_agent", task="compute 2+2")
        assert d.agent_id == "math_agent"
        assert d.task == "compute 2+2"

    def test_delegation_action_valid(self):
        a = DelegationAction(subagent_id="calculator_subagent", task="add 2 and 3")
        assert a.subagent_id == "calculator_subagent"

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
        from engine.core.react import ReActLoop

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
        from engine.core.react import ReActLoop, MaxStepsExceededError

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


# ─── Transcript Tools ───


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestTranscriptTools:
    def test_transcript_tools_registered(self):
        registered = list_registered_tools()
        assert "read_transcript" in registered
        assert "write_markdown_file" in registered
        assert "create_output_directory" in registered
        assert "log_ticket_payload" in registered

    @pytest.mark.asyncio
    async def test_read_transcript_tool(self):
        fn = get_tool("read_transcript")
        result = await fn(path=str(FIXTURES_DIR / "sample_transcript.txt"))
        assert "Sprint Retrospective" in result
        assert "Bob" in result

    @pytest.mark.asyncio
    async def test_read_transcript_missing_file(self):
        fn = get_tool("read_transcript")
        result = await fn(path="/nonexistent/file.txt")
        assert "ERROR" in result

    @pytest.mark.asyncio
    async def test_write_markdown_file_tool(self, tmp_path):
        fn = get_tool("write_markdown_file")
        out_file = tmp_path / "test_output.md"
        result = await fn(path=str(out_file), content="# Test\nHello")
        assert "File written" in result
        assert out_file.read_text() == "# Test\nHello"

    @pytest.mark.asyncio
    async def test_create_output_directory_tool(self, tmp_path):
        fn = get_tool("create_output_directory")
        result = await fn(base_dir=str(tmp_path), transcript_id="2026-03-28_14-00-00")
        assert "2026-03-28_14-00-00" in result
        assert (tmp_path / "2026-03-28_14-00-00" / "action-points").is_dir()

    @pytest.mark.asyncio
    async def test_log_ticket_payload_tool(self):
        fn = get_tool("log_ticket_payload")
        result = await fn(
            title="Fix flaky tests",
            description="Dedicate a sprint to test infra",
            acceptance_criteria='["CI time < 15min", "No flaky tests"]',
            definition_of_done='["All tests green", "Dockerized"]',
            priority="high",
            category="engineering",
            risk="CI blocked",
            dependencies='[]',
            estimate_effort="L",
        )
        assert "Ticket logged" in result


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

    def test_transcript_analyst_yaml_loads_model_field(self):
        config = load_engine_config(CONFIGS_DIR)
        for agent_id, agent in config.agents.items():
            assert hasattr(agent, "model")


# ─── Transcript Analysis Models ───


class TestTranscriptModels:
    def test_fact_model(self):
        f = Fact(
            statement="API v2 migration completed ahead of schedule",
            source=SourceReference(
                quote="we completed the API v2 migration ahead of schedule",
                context="Bob discussing sprint achievements",
            ),
        )
        assert f.statement
        assert f.source.quote

    def test_facts_summary(self):
        fs = FactsSummary(facts=[
            Fact(
                statement="47 story points delivered",
                source=SourceReference(quote="The team delivered 47 story points"),
            ),
        ])
        assert len(fs.facts) == 1

    def test_action_point_model(self):
        ap = ActionPoint(
            title="Build staging environment",
            description="Invest 2 sprints into building a proper staging env",
            acceptance_criteria=["Mirrors production data", "Auto-deploys on merge"],
            definition_of_done=["Staging env running", "Tested with prod-scale data"],
            priority="high",
            category="infrastructure",
            risk="Repeated unplanned downtime",
            dependencies=["Budget approval"],
            estimate_effort="XL",
            source_quotes=[
                SourceReference(
                    quote="I propose we invest 2 sprints into building a proper staging environment",
                    context="Bob's proposal",
                ),
            ],
        )
        assert ap.title == "Build staging environment"
        assert len(ap.acceptance_criteria) == 2
        assert len(ap.source_quotes) == 1

    def test_action_point_serialization(self):
        ap = ActionPoint(
            title="Test",
            description="Test desc",
            acceptance_criteria=["AC1"],
            definition_of_done=["DoD1"],
            priority="medium",
            category="test",
            source_quotes=[SourceReference(quote="some quote")],
        )
        data = ap.model_dump()
        assert data["title"] == "Test"
        assert data["source_quotes"][0]["quote"] == "some quote"


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
        from engine.core.react import StructuredReActLoop

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
        from engine.core.react import StructuredReActLoop

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
        from engine.core.react import StructuredReActLoop

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
        from engine.core.react import StructuredReActLoop

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
        from engine.core.react import StructuredReActLoop

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
            action='{"subagent": "facts_extractor", "task": "Read transcript"}',
            final_answer=None,
        )
        assert isinstance(step.action, DelegationAction)
        assert step.action.subagent_id == "facts_extractor"
        assert step.action.task == "Read transcript"

    @pytest.mark.asyncio
    async def test_instructor_react_strict_mode_rejects_thought_only(self):
        from engine.core.react import StructuredReActLoop

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
        from engine.core.react import StructuredReActLoop, MaxStepsExceededError

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
