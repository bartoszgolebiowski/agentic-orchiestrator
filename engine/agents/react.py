from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI
from pydantic import BaseModel

from engine.config.contracts import normalize_handoff_payload, validate_model_payload
from engine.events import EventType, emit_event
from engine.llm.client import chat_completion, structured_completion
from engine.config.models import AgentReActStep, AgentReActStepOutput
from engine.pipeline import PipelineState, is_failure_observation

logger = logging.getLogger(__name__)


class MaxStepsExceededError(Exception):
    pass


@dataclass
class Action:
    name: str
    arguments: dict[str, Any]


@dataclass
class StepResult:
    thought: str | None
    action: Action | None
    observation: str | None
    is_final: bool = False
    final_answer: str | None = None


ActionHandler = Callable[[str, dict[str, Any]], Awaitable[str]]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: _to_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _structured_payload_to_mapping(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return {key: _to_jsonable(item) for key, item in payload.items()}
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    if hasattr(payload, "__dict__"):
        return {
            key: _to_jsonable(item)
            for key, item in vars(payload).items()
            if not key.startswith("_")
        }
    raise TypeError(f"Unsupported structured payload type: {type(payload)!r}")


@dataclass
class ReActLoop:
    """ReAct loop using native OpenAI tool calls. Used by SubagentExecutor."""

    client: AsyncOpenAI
    system_prompt: str
    tools_spec: list[dict[str, Any]]
    action_handler: ActionHandler
    max_steps: int = 10
    model: str | None = None
    final_response_model: type[BaseModel] | None = None

    _messages: list[dict[str, Any]] = field(default_factory=list, init=False)
    _steps: list[StepResult] = field(default_factory=list, init=False)

    def _init_messages(self, task: str, input_json: Any | None = None) -> None:
        if input_json is None:
            user_content = task
        else:
            user_content = json.dumps(
                {"task": task, "input_json": input_json},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        self._messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    async def run(self, task: str, input_json: Any | None = None) -> str:
        self._init_messages(task, input_json=input_json)
        self._steps = []

        for step_num in range(1, self.max_steps + 1):
            logger.info(f"ReAct step {step_num}/{self.max_steps}")

            response = await chat_completion(
                client=self.client,
                messages=self._messages,
                tools=self.tools_spec or None,
                model=self.model,
                trace_name=f"subagent-step-{step_num}",
                trace_metadata={"loop": "react", "step": step_num, "max_steps": self.max_steps},
            )

            if response.tool_calls:
                self._messages.append(response.model_dump())

                for tool_call in response.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    logger.info(f"  Action: {fn_name}({fn_args})")

                    emit_event(
                        EventType.SUBAGENT_STEP,
                        step=step_num,
                        action=fn_name,
                        arguments=fn_args,
                    )

                    try:
                        observation = await self.action_handler(fn_name, fn_args)
                    except Exception as e:
                        observation = f"ERROR: {type(e).__name__}: {e}"
                        logger.error(f"  Action failed: {observation}")

                    logger.info(f"  Observation: {observation[:200]}")

                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": observation,
                    })

                    self._steps.append(StepResult(
                        thought=response.content,
                        action=Action(name=fn_name, arguments=fn_args),
                        observation=observation,
                    ))
            else:
                final_answer = response.content or ""
                if not final_answer.strip():
                    logger.warning("  Final answer was empty; retrying")
                    self._messages.append(response.model_dump())
                    self._messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response was empty. Call one of the available tools "
                            "or return a non-empty final answer."
                        ),
                    })
                    continue
                logger.info(f"  Final answer: {final_answer[:200]}")
                self._messages.append(response.model_dump())

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
                            thought=final_answer,
                            action=None,
                            observation=None,
                            is_final=True,
                            final_answer=normalized,
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
                            trace_name=f"subagent-final:{self.final_response_model.__name__}",
                            trace_metadata={
                                "loop": "react",
                                "final_output": self.final_response_model.__name__,
                            },
                        )
                        normalized = json.dumps(
                            final_output.model_dump(mode="json"),
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                        self._steps.append(StepResult(
                            thought=final_answer,
                            action=None,
                            observation=None,
                            is_final=True,
                            final_answer=normalized,
                        ))
                        return normalized

                self._steps.append(StepResult(
                    thought=final_answer,
                    action=None,
                    observation=None,
                    is_final=True,
                    final_answer=final_answer,
                ))
                return final_answer

        raise MaxStepsExceededError(
            f"ReAct loop exceeded {self.max_steps} steps without producing a final answer"
        )


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
    _pipeline: PipelineState = field(init=False, default_factory=lambda: PipelineState([]))
    _handoff_payload: Any | None = field(default=None, init=False)

    def _init_messages(self, task: str, input_json: Any | None = None) -> None:
        pipeline_instruction = ""
        if self.required_pipeline:
            stages = ", ".join(f"'{s}'" for s in self.required_pipeline)
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
        if input_json is None:
            user_content = task
        else:
            user_content = json.dumps(
                {"task": task, "input_json": input_json},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        self._messages = [
            {"role": "system", "content": enhanced_prompt},
            {"role": "user", "content": user_content},
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
        self._handoff_payload = input_json
        self._pipeline = PipelineState(list(self.required_pipeline))

        for step_num in range(1, self.max_steps + 1):
            logger.info(f"StructuredReAct step {step_num}/{self.max_steps}")

            try:
                provider_step = await structured_completion(
                    client=self.client,
                    messages=self._messages,
                    response_model=AgentReActStepOutput,
                    model=self.model,
                    trace_name=f"agent-step-{step_num}",
                    trace_metadata={"loop": "instructor-react", "step": step_num, "max_steps": self.max_steps},
                )
                step: AgentReActStep = AgentReActStep.model_validate(
                    _structured_payload_to_mapping(provider_step)
                )
            except Exception as exc:
                error_text = str(exc).lower()
                if "multiple tool calls" in error_text and not getattr(self, "_retried_single_action", False):
                    logger.warning("  Structured output returned multiple tool calls; retrying once with stricter single-action instruction")
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
                    step = AgentReActStep.model_validate(
                        _structured_payload_to_mapping(provider_step)
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
            )

            final_answer = step.final_answer
            if final_answer is not None and final_answer.strip().lower() == "null":
                logger.warning("  Final answer was literal null; treating as missing final answer")
                final_answer = None

            if final_answer is not None:
                # Pipeline enforcement: reject premature final_answer
                if not self._pipeline.is_empty and not self._pipeline.is_complete:
                    missing = self._pipeline.missing_stages
                    next_stage = self._pipeline.next_stage
                    if missing and next_stage:
                        logger.warning(
                            f"  Pipeline enforcement: LLM tried to finish but {len(missing)} "
                            f"required stages remain: {missing}. Forcing delegation to '{next_stage}'."
                        )
                        self._messages.append({
                            "role": "assistant",
                            "content": f"Thought: {step.thought}",
                        })
                        self._messages.append({
                            "role": "user",
                            "content": (
                                f"PIPELINE VIOLATION: You cannot produce a final answer yet. "
                                f"The following mandatory stages have NOT been executed: {missing}. "
                                f"You MUST delegate to '{next_stage}' now. "
                                f"Continue the pipeline from where you left off."
                            ),
                        })
                        self._steps.append(StepResult(
                            thought=step.thought,
                            action=None,
                            observation=f"Pipeline enforcement: redirected to {next_stage}",
                        ))
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
                        ))
                        return normalized

                self._steps.append(StepResult(
                    thought=step.thought,
                    action=None,
                    observation=None,
                    is_final=True,
                    final_answer=final_answer,
                ))
                return final_answer

            if step.action is not None:
                action_name = step.action.subagent_id
                action_args = {"task": step.action.task}
                if step.action.input_json is not None:
                    action_args["input_json"] = step.action.input_json
                elif self._handoff_payload is not None:
                    action_args["input_json"] = self._handoff_payload

                logger.info(f"  Action: delegate to '{action_name}' with task: {step.action.task[:200]}")

                try:
                    observation = await self.action_handler(action_name, action_args)
                except Exception as e:
                    observation = f"ERROR: {type(e).__name__}: {e}"
                    logger.error(f"  Action failed: {observation}")

                self._pipeline.observe_result(action_name, observation)

                logger.info(f"  Observation: {observation[:200]}")

                if observation.strip() and not is_failure_observation(observation):
                    self._handoff_payload = normalize_handoff_payload(observation)

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
