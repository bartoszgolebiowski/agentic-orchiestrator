from __future__ import annotations

import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, AliasChoices, model_validator


class RoleType(str, Enum):
    AGENT = "agent"
    SUBAGENT = "subagent"


# ─── Instructor Response Models ───


class RoutingDecision(BaseModel):
    """Orchestrator's structured routing decision."""
    agent_id: str = Field(
        description="The ID of the selected agent.",
        validation_alias=AliasChoices("agent_id", "target_agent", "targetAgent"),
        serialization_alias="agent_id",
    )
    task: str = Field(
        default="",
        description="The task to pass to the selected agent, reformulated if needed.",
    )


class DelegationAction(BaseModel):
    """An action where an Agent delegates a sub-task to a Subagent."""
    subagent_id: str = Field(description="The ID of the subagent to delegate to.")
    task: str = Field(description="The specific sub-task to delegate.")


class AgentReActStep(BaseModel):
    """A single ReAct reasoning step for an Agent (planning layer)."""
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
            values["action"] = action

        return values

    @model_validator(mode="after")
    def validate_step(self) -> "AgentReActStep":
        if self.final_answer is not None and self.final_answer.strip().lower() == "null":
            raise ValueError("final_answer must not be the literal string 'null'")
        if self.action is None and self.final_answer is None:
            raise ValueError("Either action or final_answer must be provided")
        return self


# ─── Transcript Analysis Models ───


class SourceReference(BaseModel):
    """A reference to a specific fragment of the transcript."""
    quote: str = Field(description="Full verbatim quote from the transcript.")
    context: str = Field(default="", description="Brief note about where in the transcript this appears (e.g. 'during budget discussion').")


class Fact(BaseModel):
    """A single extracted fact from the transcript."""
    statement: str = Field(description="A concise factual statement extracted from the transcript.")
    source: SourceReference = Field(description="Source quote and reference from the transcript.")


class FactsSummary(BaseModel):
    """Collection of all extracted facts from a transcript."""
    facts: list[Fact] = Field(description="List of extracted facts.")


class Conclusion(BaseModel):
    """A single conclusion, recommendation, or insight derived from the transcript."""
    statement: str = Field(description="The conclusion or recommendation.")
    type: str = Field(description="Category: one of 'strength', 'weakness', 'risk', 'opportunity', 'recommendation', 'vision'.")
    source: SourceReference = Field(description="Source quote and reference from the transcript.")


class ConclusionsSummary(BaseModel):
    """Collection of all conclusions drawn from the transcript."""
    conclusions: list[Conclusion] = Field(description="List of conclusions and recommendations.")


class ActionPoint(BaseModel):
    """A single ticket-ready action point derived from transcript analysis."""
    title: str = Field(description="Short, actionable title for the ticket.")
    description: str = Field(description="Detailed description of what needs to be done.")
    acceptance_criteria: list[str] = Field(description="List of measurable acceptance criteria.")
    definition_of_done: list[str] = Field(description="List of conditions that define when this task is complete.")
    priority: str = Field(description="Priority level: critical, high, medium, low.")
    category: str = Field(description="Category or domain this action point belongs to.")
    risk: str = Field(default="", description="Associated risks if this is not addressed.")
    dependencies: list[str] = Field(default_factory=list, description="Other action points or external dependencies.")
    estimate_effort: str = Field(default="", description="Estimated effort (e.g. 'S', 'M', 'L', 'XL' or hours).")
    source_quotes: list[SourceReference] = Field(description="Full source quotes from the transcript that led to this action point.")


class ToolParameter(BaseModel):
    type: str
    description: str
    required: bool = True


class ToolDefinition(BaseModel):
    id: str
    description: str
    parameters: dict[str, ToolParameter] = Field(default_factory=dict)


class NodeConfig(BaseModel):
    id: str
    role_type: RoleType
    description: str
    system_prompt: str
    dependencies: list[str] = Field(default_factory=list)
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
        if self.role_type == RoleType.SUBAGENT and not self.dependencies:
            raise ValueError(
                f"Subagent '{self.id}' must declare at least one tool dependency"
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
            "'agent_id' and 'task' fields."
        )
    )
    max_steps: int = Field(default=1)
    model: str | None = Field(default=None, description="LLM model override for orchestrator.")


class EngineConfig(BaseModel):
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    agents: dict[str, NodeConfig] = Field(default_factory=dict)
    subagents: dict[str, NodeConfig] = Field(default_factory=dict)
    tools: dict[str, ToolDefinition] = Field(default_factory=dict)

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

        return self
