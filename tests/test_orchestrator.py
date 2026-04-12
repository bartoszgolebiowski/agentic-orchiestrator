from __future__ import annotations

from unittest.mock import MagicMock

from engine.core.models import EngineConfig, NodeConfig, OrchestratorConfig, RoutingDecision, RoleType
from engine.roles.orchestrator import Orchestrator


def _build_engine_config() -> EngineConfig:
    return EngineConfig(
        orchestrator=OrchestratorConfig(),
        agents={
            "document_agent": NodeConfig(
                id="document_agent",
                role_type=RoleType.AGENT,
                description="Routes document work to the document pipeline.",
                system_prompt="You are the document agent.",
                dependencies=["markdown_extractor"],
            ),
        },
        subagents={
            "markdown_extractor": NodeConfig(
                id="markdown_extractor",
                role_type=RoleType.SUBAGENT,
                description="Extracts structure from markdown files.",
                system_prompt="You are the markdown extractor.",
            ),
        },
    )


def test_dynamic_routing_model_uses_enum_metadata_only_for_orchestrator() -> None:
    orchestrator = Orchestrator(engine_config=_build_engine_config(), client=MagicMock())

    assert RoutingDecision.model_fields["agent_id"].annotation is str

    routing_model = orchestrator._routing_model
    agent_enum = routing_model.model_fields["agent_id"].annotation
    member = agent_enum["document_agent"]

    assert routing_model.__name__ == "DynamicRoutingDecision"
    assert set(routing_model.model_fields) == {"agent_id", "task"}
    assert member.value == "document_agent"
    assert member.title == "Document Agent"
    assert member.description == "Routes document work to the document pipeline."

    schema = routing_model.model_json_schema()
    assert schema["additionalProperties"] is False
    assert "input_json" not in schema["properties"]

    agent_schema = schema["properties"]["agent_id"]
    assert agent_schema["description"] == "The ID of the selected agent."
    assert agent_schema["x-enumTitles"] == ["Document Agent"]
    assert agent_schema["x-enumDescriptions"] == ["Routes document work to the document pipeline."]
    assert agent_schema["x-agent-list"] == [
        "- Document Agent (document_agent): Routes document work to the document pipeline."
    ]