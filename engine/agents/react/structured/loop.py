from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from engine.agents.models import AgentReActStep, AgentReActStepOutput
from engine.agents.react.base import (
    Action,
    ActionHandler,
    MaxStepsExceededError,
    StepResult,
    init_user_message,
    structured_payload_to_mapping,
)
from engine.agents.react.structured.payload_handler import PayloadHandler
from engine.agents.react.structured.pipeline_enforcer import PipelineEnforcer
from engine.config.contracts import validate_model_payload
from engine.events import EventType, emit_event
from engine.llm.client import (
    get_last_usage_details,
    merge_usage_summaries,
    structured_completion,
    summarize_usage,
)
from engine.pipeline import PipelineState, is_failure_observation

logger = logging.getLogger(__name__)


@dataclass
class StructuredReActLoop:
    """ReAct loop using native OpenAI structured outputs. Used by AgentPlanner."""

    client: AsyncOpenAI
    system_prompt: str
    action_handler: ActionHandler
    available_actions: str
    max_steps: int = 10
    model: str | None = None
    allow_thought_as_final: bool = True
    required_pipeline: list[str] = field(default_factory=list)
    final_response_model: type[BaseModel] | None = None

    _messages: list[dict[str, Any]] = field(default_factory=list, init=False)
    _steps: list[StepResult] = field(default_factory=list, init=False)
    _pipeline_enforcer: PipelineEnforcer = field(init=False)
    _payload_handler: PayloadHandler = field(init=False)

    def _init_messages(self, task: str, input_json: Any | None = None) -> None:
        pipeline_instruction = ""
        if self.required_pipeline:
            stages = ", ".join(f"'{stage}'" for stage in self.required_pipeline)
            pipeline_instruction = (
                f"\n\nMANDATORY PIPELINE: You MUST delegate to these subagents in this exact order: "
                f"{stages}. Do NOT produce a final_answer until ALL of these have been called and "
                f"returned successfully. Each stage must complete before starting the next one."
            )

        enhanced_prompt = (
            f"{self.system_prompt}\n\n"
            f"Available subagents you can delegate to:\n{self.available_actions}\n\n"
            f"Respond as structured JSON with exactly these fields:\n"
            f"- thought: string\n"
            f"- action: null or an object with at least {{\"subagent_id\": string, \"task\": string}}\n"
            f"- final_answer: null or string\n\n"
            f"Do not emit input_json in the structured response. The runtime will forward the current input_json payload to the delegated subagent automatically when present.\n"
            f"Use action as an object, not as a JSON string. Use the exact field name subagent_id.\n"
            f"Important: emit exactly one action per response at most. Do not batch multiple actions, "
            f"multiple subagents, or multiple stages in one turn. If more work is needed, wait for the next turn."
            f"{pipeline_instruction}"
        )

        self._messages = [
            {"role": "system", "content": enhanced_prompt},
            {"role": "user", "content": init_user_message(task, input_json=input_json)},
        ]

    def _strengthen_single_action_instruction(self) -> None:
        self._messages[0]["content"] += (
            "\n\nCRITICAL RECOVERY RULE: return exactly one delegated subagent action "
            "or one final answer per response. Never batch multiple tool calls, actions, "
            "or stages in a single response."
        )

    async def run(self, task: str, input_json: Any | None = None) -> str:
        self._init_messages(task, input_json=input_json)
        self._steps = []
        self._pipeline_enforcer = PipelineEnforcer(PipelineState(list(self.required_pipeline)))
        self._payload_handler = PayloadHandler(initial_payload=input_json)

        for step_num in range(1, self.max_steps + 1):
            logger.info(f"StructuredReAct step {step_num}/{self.max_steps}")
            llm_usage_details: dict[str, Any] | None = None
            llm_usage: dict[str, int] | None = None

            try:
                provider_step = await structured_completion(
                    client=self.client,
                    messages=self._messages,
                    response_model=AgentReActStepOutput,
                    model=self.model,
                    trace_name=f"agent-step-{step_num}",
                    trace_metadata={"loop": "instructor-react", "step": step_num, "max_steps": self.max_steps},
                )
                llm_usage_details = get_last_usage_details()
                llm_usage = summarize_usage(llm_usage_details)
                step: AgentReActStep = AgentReActStep.model_validate(
                    structured_payload_to_mapping(provider_step)
                )
            except Exception as exc:
                error_text = str(exc).lower()
                if "multiple tool calls" in error_text and not getattr(self, "_retried_single_action", False):
                    logger.warning("  Structured output returned multiple tool calls; retrying once with stricter single-action instruction")
                    initial_usage_details = get_last_usage_details()
                    initial_usage = summarize_usage(initial_usage_details)
                    self._retried_single_action = True
                    self._strengthen_single_action_instruction()
                    provider_step = await structured_completion(
                        client=self.client,
                        messages=self._messages,
                        response_model=AgentReActStepOutput,
                        model=self.model,
                        trace_name=f"agent-step-{step_num}-retry",
                        trace_metadata={"loop": "instructor-react", "step": step_num, "retry": True},
                    )
                    retry_usage_details = get_last_usage_details()
                    retry_usage = summarize_usage(retry_usage_details)
                    llm_usage = merge_usage_summaries(initial_usage, retry_usage)
                    llm_usage_details = retry_usage_details or initial_usage_details
                    step = AgentReActStep.model_validate(
                        structured_payload_to_mapping(provider_step)
                    )
                elif "either action or final_answer" in error_text:
                    if not self.allow_thought_as_final:
                        raise ValueError(
                            "Planner produced no action and no final answer; strict mode requires delegation or a final answer"
                        )
                    logger.warning("  LLM returned neither action nor final_answer; injecting correction and retrying")
                    self._messages.append({
                        "role": "user",
                        "content": (
                            "Your last response was invalid: you must provide either an action "
                            "(to delegate to a subagent) or a final_answer. You cannot leave both "
                            "as null. Please respond again now with one of these."
                        ),
                    })
                    continue
                else:
                    raise

            logger.info(f"  Thought: {step.thought[:200]}")

            emit_event(
                EventType.AGENT_STEP,
                step=step_num,
                thought=step.thought[:500],
                has_action=step.action is not None,
                has_final=step.final_answer is not None,
                llm_usage=llm_usage,
                llm_usage_details=llm_usage_details,
            )

            final_answer = step.final_answer
            if final_answer is not None and final_answer.strip().lower() == "null":
                logger.warning("  Final answer was literal null; treating as missing final answer")
                final_answer = None

            if final_answer is not None:
                redirected = self._pipeline_enforcer.enforce_before_final(
                    thought=step.thought,
                    messages=self._messages,
                    steps=self._steps,
                )
                if redirected:
                    continue

                logger.info(f"  Final answer: {final_answer[:200]}")
                self._messages.append({
                    "role": "assistant",
                    "content": f"Thought: {step.thought}\nFinal answer: {final_answer}",
                })

                if self.final_response_model is not None:
                    try:
                        validated = validate_model_payload(self.final_response_model, final_answer)
                        normalized = json.dumps(
                            validated.model_dump(mode="json"),
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                        self._steps.append(StepResult(
                            thought=step.thought,
                            action=None,
                            observation=None,
                            is_final=True,
                            final_answer=normalized,
                            llm_usage=llm_usage,
                            llm_usage_details=llm_usage_details,
                        ))
                        return normalized
                    except Exception:
                        self._messages.append({
                            "role": "user",
                            "content": (
                                "The previous response must be valid JSON matching the final output contract. "
                                "Return only valid JSON now."
                            ),
                        })
                        final_output = await structured_completion(
                            client=self.client,
                            messages=self._messages,
                            response_model=self.final_response_model,
                            model=self.model,
                            trace_name=f"agent-final:{self.final_response_model.__name__}",
                            trace_metadata={
                                "loop": "instructor-react",
                                "final_output": self.final_response_model.__name__,
                            },
                        )
                        final_usage_details = get_last_usage_details()
                        final_usage = summarize_usage(final_usage_details)
                        llm_usage = merge_usage_summaries(llm_usage, final_usage)
                        if final_usage_details is not None:
                            llm_usage_details = final_usage_details
                        normalized = json.dumps(
                            final_output.model_dump(mode="json"),
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                        self._steps.append(StepResult(
                            thought=step.thought,
                            action=None,
                            observation=None,
                            is_final=True,
                            final_answer=normalized,
                            llm_usage=llm_usage,
                            llm_usage_details=llm_usage_details,
                        ))
                        return normalized

                self._steps.append(StepResult(
                    thought=step.thought,
                    action=None,
                    observation=None,
                    is_final=True,
                    final_answer=final_answer,
                    llm_usage=llm_usage,
                    llm_usage_details=llm_usage_details,
                ))
                return final_answer

            if step.action is not None:
                action_name = step.action.subagent_id
                action_args = self._payload_handler.action_arguments(
                    task=step.action.task,
                    explicit_input_json=step.action.input_json,
                )

                logger.info(f"  Action: delegate to '{action_name}' with task: {step.action.task[:200]}")

                try:
                    observation = await self.action_handler(action_name, action_args)
                except Exception as exc:
                    observation = f"ERROR: {type(exc).__name__}: {exc}"
                    logger.error(f"  Action failed: {observation}")

                self._pipeline_enforcer.pipeline.observe_result(action_name, observation)

                logger.info(f"  Observation: {observation[:200]}")

                if observation.strip() and not is_failure_observation(observation):
                    self._payload_handler.observe(observation)

                self._messages.append({
                    "role": "assistant",
                    "content": f"Thought: {step.thought}\nAction: delegate to {action_name} with task: {step.action.task}",
                })
                self._messages.append({
                    "role": "user",
                    "content": f"Observation from {action_name}: {observation}",
                })

                self._steps.append(StepResult(
                    thought=step.thought,
                    action=Action(name=action_name, arguments=action_args),
                    observation=observation,
                    llm_usage=llm_usage,
                    llm_usage_details=llm_usage_details,
                ))
            else:
                if not self.allow_thought_as_final:
                    raise ValueError(
                        "Planner produced no action and no final answer; strict mode requires delegation or a final answer"
                    )
                logger.info("  No action or final_answer — using thought as answer")
                return step.thought

        raise MaxStepsExceededError(
            f"StructuredReAct loop exceeded {self.max_steps} steps without producing a final answer"
        )
