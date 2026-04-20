from __future__ import annotations

import json
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class RoutingDecisionOutput(BaseModel):
    """Provider-safe orchestrator routing decision."""
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


class RoutingDecision(RoutingDecisionOutput):
    """Orchestrator's structured routing decision."""
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


class DelegationActionOutput(BaseModel):
    """Provider-safe action where an Agent delegates a sub-task to a Subagent."""
    model_config = ConfigDict(extra="forbid")
    subagent_id: str = Field(
        description="The ID of the subagent to delegate to.",
        validation_alias=AliasChoices("subagent_id", "subagent"),
        serialization_alias="subagent_id",
    )
    task: str = Field(description="The specific sub-task to delegate.")


class DelegationAction(DelegationActionOutput):
    """An action where an Agent delegates a sub-task to a Subagent."""
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


class AgentReActStepOutput(BaseModel):
    """Provider-safe single ReAct reasoning step for an Agent."""
    model_config = ConfigDict(extra="forbid")
    thought: str = Field(description="Your reasoning about the current state and what to do next.")
    action: DelegationActionOutput | None = Field(
        default=None,
        description="If you need to delegate a sub-task, specify the subagent and task. Leave null if you have the final answer.",
    )
    final_answer: str | None = Field(
        default=None,
        description="If you have enough information to answer, provide the final answer here. Leave null if you need to delegate.",
    )


class AgentReActStep(AgentReActStepOutput):
    """A single ReAct reasoning step for an Agent (planning layer)."""
    action: DelegationAction | None = Field(
        default=None,
        description="If you need to delegate a sub-task, specify the subagent and task. Leave null if you have the final answer.",
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
