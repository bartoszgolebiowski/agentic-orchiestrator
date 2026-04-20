from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, AliasChoices, model_validator

from engine.mcp.models import McpServerConfig


class RoleType(str, Enum):
    AGENT = "agent"
    SUBAGENT = "subagent"


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module: str
    name: str

    @model_validator(mode="after")
    def validate_spec(self) -> "ModelSpec":
        if not self.module.strip():
            raise ValueError("ModelSpec.module cannot be empty")
        if not self.name.strip():
            raise ValueError("ModelSpec.name cannot be empty")
        return self


class ToolParameter(BaseModel):
    type: str
    description: str
    required: bool = True


class ToolDefinition(BaseModel):
    id: str
    description: str
    parameters: dict[str, ToolParameter] = Field(default_factory=dict)


class StructuredOutputContractConfig(BaseModel):
    id: str
    description: str
    schema_path: str
    instructions: str
    model: str | None = Field(default=None, description="Optional LLM model override for this contract.")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_attempts: int = Field(default=2, ge=1, le=5)


class NodeConfig(BaseModel):
    id: str
    role_type: RoleType
    description: str
    system_prompt: str
    dependencies: list[str] = Field(default_factory=list)
    mcp_dependencies: list[str] = Field(default_factory=list)
    mcp_include_tools: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Per-MCP-server tool allowlist. Maps server ID to a list of remote "
            "tool names this subagent may use. Servers not listed here expose all "
            "their tools. An empty list means no tools from that server."
        ),
    )
    mcp_skip_hitl_tools: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Per-MCP-server HITL bypass list. Maps server ID to a list of remote "
            "tool names that may skip human-in-the-loop confirmation. Tools not "
            "listed here still require HITL when a callback is configured. "
            "An empty dict means all MCP tools require HITL (default)."
        ),
    )
    required_pipeline: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of subagent IDs that MUST be invoked (in order) before "
            "the agent is allowed to produce a final answer. Empty means no enforcement."
        ),
    )
    tool_result_projection: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-tool projection applied to tool results before they are returned "
            "to the ReAct loop. Maps tool name (local or exposed MCP name) to either "
            "a JSONPath selector string or a field-map object. Field maps can also "
            "contain nested collection specs with 'path' and 'fields'."
        ),
    )
    max_steps: int = Field(default=10, ge=1, le=50)
    model: str | None = Field(default=None, description="LLM model override for this node. Falls back to env LLM_MODEL if not set.")

    @model_validator(mode="after")
    def validate_dependencies(self) -> "NodeConfig":
        if self.role_type == RoleType.AGENT and not self.dependencies:
            raise ValueError(
                f"Agent '{self.id}' must declare at least one subagent dependency"
            )
        if self.role_type == RoleType.AGENT and self.mcp_dependencies:
            raise ValueError(
                f"Agent '{self.id}' cannot declare mcp_dependencies"
            )
        if self.role_type == RoleType.AGENT and self.mcp_include_tools:
            raise ValueError(
                f"Agent '{self.id}' cannot declare mcp_include_tools"
            )
        if self.role_type == RoleType.AGENT and self.mcp_skip_hitl_tools:
            raise ValueError(
                f"Agent '{self.id}' cannot declare mcp_skip_hitl_tools"
            )
        if self.role_type == RoleType.AGENT and self.tool_result_projection:
            raise ValueError(
                f"Agent '{self.id}' cannot declare tool_result_projection"
            )
        for server_id in self.mcp_include_tools:
            if server_id not in self.mcp_dependencies:
                raise ValueError(
                    f"Node '{self.id}' mcp_include_tools references server "
                    f"'{server_id}' which is not in mcp_dependencies"
                )
        for server_id in self.mcp_skip_hitl_tools:
            if server_id not in self.mcp_dependencies:
                raise ValueError(
                    f"Node '{self.id}' mcp_skip_hitl_tools references server "
                    f"'{server_id}' which is not in mcp_dependencies"
                )
        for step in self.required_pipeline:
            if step not in self.dependencies:
                raise ValueError(
                    f"Node '{self.id}' required_pipeline references '{step}' "
                    f"which is not in dependencies"
                )
        # Deferred import prevents config<->tools registration cycles at module load time.
        from engine.tools.projection import validate_projection_spec
        for tool_name, expr in self.tool_result_projection.items():
            try:
                validate_projection_spec(expr)
            except Exception as exc:
                raise ValueError(
                    f"Node '{self.id}' tool_result_projection has invalid "
                    f"projection for tool '{tool_name}': {exc}"
                ) from exc
        return self


class OrchestratorConfig(BaseModel):
    system_prompt: str = Field(
        default=(
            "You are an orchestrator. Analyze the user query and select the single "
            "most competent agent to handle it. Respond with a JSON object containing "
            "'agent_id', 'task', and optional 'input_json' fields."
        )
    )
    max_steps: int = Field(default=1)
    model: str | None = Field(default=None, description="LLM model override for orchestrator.")


class EnricherConfig(BaseModel):
    """A config-driven pre-orchestration enricher loaded from configs/enrichers/*.yaml."""
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str = Field(description="Human-readable description shown to the LLM selector.")
    priority: int = Field(default=50, ge=0, le=100, description="Higher priority wins when multiple enrichers match. 0-100.")
    enabled: bool = Field(default=True, description="Set to false to disable without removing the file.")
    executor: str = Field(description="Executor type. Built-in: 'glob_file_discovery'.")
    executor_config: dict[str, Any] = Field(default_factory=dict, description="Executor-specific settings.")
    model: str | None = Field(default=None, description="LLM model override for the enrichment selector step.")

    @model_validator(mode="after")
    def validate_enricher(self) -> "EnricherConfig":
        if not self.id.strip():
            raise ValueError("Enricher id cannot be empty")
        if not self.executor.strip():
            raise ValueError(f"Enricher '{self.id}' must specify an executor")
        return self


class EnrichmentDecision(BaseModel):
    """Structured output from the LLM enrichment selector."""
    model_config = ConfigDict(extra="forbid")

    enricher_id: str | None = Field(
        default=None,
        description="The ID of the selected enricher, or null if no enrichment is needed.",
    )
    reason: str = Field(
        default="",
        description="Brief reason for the selection or why no enricher applies.",
    )


class EngineConfig(BaseModel):
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    agents: dict[str, NodeConfig] = Field(default_factory=dict)
    subagents: dict[str, NodeConfig] = Field(default_factory=dict)
    tools: dict[str, ToolDefinition] = Field(default_factory=dict)
    mcps: dict[str, McpServerConfig] = Field(default_factory=dict)
    enrichers: dict[str, EnricherConfig] = Field(default_factory=dict)


from engine.agents.models import (  # noqa: E402
    AgentReActStep,
    AgentReActStepOutput,
    DelegationAction,
    DelegationActionOutput,
    RoutingDecision,
    RoutingDecisionOutput,
)
