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


# ─── Instructor Response Models ───


class RoutingDecision(BaseModel):
    """Orchestrator's structured routing decision."""
    model_config = ConfigDict(extra="forbid")
    agent_id: str = Field(
        description="The ID of the selected agent.",
        validation_alias=AliasChoices("agent_id", "target_agent", "targetAgent"),
        serialization_alias="agent_id",
    )
    task: str = Field(
        default="",
        description="The task to pass to the selected agent, reformulated if needed.",
    )
    input_json: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured input to pass to the selected agent.",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_input_json(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        input_json = values.get("input_json")
        if isinstance(input_json, str):
            try:
                parsed = json.loads(input_json)
            except Exception:
                return values
            if isinstance(parsed, dict):
                values["input_json"] = parsed
        return values


class DelegationAction(BaseModel):
    """An action where an Agent delegates a sub-task to a Subagent."""
    model_config = ConfigDict(extra="forbid")
    subagent_id: str = Field(description="The ID of the subagent to delegate to.")
    task: str = Field(description="The specific sub-task to delegate.")
    input_json: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured input to pass to the subagent.",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_input_json(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        input_json = values.get("input_json")
        if isinstance(input_json, str):
            try:
                parsed = json.loads(input_json)
            except Exception:
                return values
            if isinstance(parsed, dict):
                values["input_json"] = parsed
        return values


class AgentReActStep(BaseModel):
    """A single ReAct reasoning step for an Agent (planning layer)."""
    model_config = ConfigDict(extra="forbid")
    thought: str = Field(description="Your reasoning about the current state and what to do next.")
    action: DelegationAction | None = Field(
        default=None,
        description="If you need to delegate a sub-task, specify the subagent and task. Leave null if you have the final answer.",
    )
    final_answer: str | None = Field(
        default=None,
        description="If you have enough information to answer, provide the final answer here. Leave null if you need to delegate.",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_action(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        action = values.get("action")
        if isinstance(action, str):
            try:
                parsed = json.loads(action)
            except Exception:
                return values

            if isinstance(parsed, dict):
                if "subagent" in parsed and "subagent_id" not in parsed:
                    parsed["subagent_id"] = parsed.pop("subagent")
                values["action"] = parsed

        elif isinstance(action, dict) and "subagent" in action and "subagent_id" not in action:
            action = dict(action)
            action["subagent_id"] = action.pop("subagent")
            if isinstance(action.get("input_json"), str):
                try:
                    parsed_input = json.loads(action["input_json"])
                except Exception:
                    parsed_input = None
                if isinstance(parsed_input, dict):
                    action["input_json"] = parsed_input
            values["action"] = action

        return values

    @model_validator(mode="after")
    def validate_step(self) -> "AgentReActStep":
        if self.final_answer is not None and self.final_answer.strip().lower() == "null":
            raise ValueError("final_answer must not be the literal string 'null'")
        if self.action is None and self.final_answer is None:
            raise ValueError("Either action or final_answer must be provided")
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
    required_pipeline: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of subagent IDs that MUST be invoked (in order) before "
            "the agent is allowed to produce a final answer. Empty means no enforcement."
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
        for step in self.required_pipeline:
            if step not in self.dependencies:
                raise ValueError(
                    f"Node '{self.id}' required_pipeline references '{step}' "
                    f"which is not in dependencies"
                )
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


class EngineConfig(BaseModel):
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    agents: dict[str, NodeConfig] = Field(default_factory=dict)
    subagents: dict[str, NodeConfig] = Field(default_factory=dict)
    tools: dict[str, ToolDefinition] = Field(default_factory=dict)
    mcps: dict[str, McpServerConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> "EngineConfig":
        for agent_id, agent in self.agents.items():
            if agent.role_type != RoleType.AGENT:
                raise ValueError(
                    f"Node '{agent_id}' is in agents/ but has role_type={agent.role_type}"
                )
            for dep in agent.dependencies:
                if dep not in self.subagents:
                    raise ValueError(
                        f"Agent '{agent_id}' references unknown subagent '{dep}'"
                    )

        for sub_id, sub in self.subagents.items():
            if sub.role_type != RoleType.SUBAGENT:
                raise ValueError(
                    f"Node '{sub_id}' is in subagents/ but has role_type={sub.role_type}"
                )
            for dep in sub.dependencies:
                if dep not in self.tools:
                    raise ValueError(
                        f"Subagent '{sub_id}' references unknown tool '{dep}'"
                    )
            for mcp_dep in sub.mcp_dependencies:
                if mcp_dep not in self.mcps:
                    raise ValueError(
                        f"Subagent '{sub_id}' references unknown MCP server '{mcp_dep}'"
                    )

        return self
