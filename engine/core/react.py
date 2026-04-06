from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

from engine.core.llm import chat_completion, structured_completion
from engine.core.models import AgentReActStep

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


@dataclass
class ReActLoop:
    """ReAct loop using native OpenAI tool calls. Used by SubagentExecutor."""

    client: AsyncOpenAI
    system_prompt: str
    tools_spec: list[dict[str, Any]]
    action_handler: ActionHandler
    max_steps: int = 10
    model: str | None = None

    _messages: list[dict[str, Any]] = field(default_factory=list, init=False)
    _steps: list[StepResult] = field(default_factory=list, init=False)

    def _init_messages(self, task: str) -> None:
        self._messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

    async def run(self, task: str) -> str:
        self._init_messages(task)
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
                logger.info(f"  Final answer: {final_answer[:200]}")
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

    _messages: list[dict[str, Any]] = field(default_factory=list, init=False)
    _steps: list[StepResult] = field(default_factory=list, init=False)
    _completed_subagents: list[str] = field(default_factory=list, init=False)

    def _init_messages(self, task: str) -> None:
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
            f"- action: null or an object with exactly {{\"subagent_id\": string, \"task\": string}}\n"
            f"- final_answer: null or string\n\n"
            f"Use action as an object, not as a JSON string. Use the exact field name subagent_id.\n"
            f"Important: emit exactly one action per response at most. Do not batch multiple actions, "
            f"multiple subagents, or multiple stages in one turn. If more work is needed, wait for the next turn."
            f"{pipeline_instruction}"
        )
        self._messages = [
            {"role": "system", "content": enhanced_prompt},
            {"role": "user", "content": task},
        ]

    def _strengthen_single_action_instruction(self) -> None:
        self._messages[0]["content"] += (
            "\n\nCRITICAL RECOVERY RULE: return exactly one delegated subagent action "
            "or one final answer per response. Never batch multiple tool calls, actions, "
            "or stages in a single response."
        )

    async def run(self, task: str) -> str:
        self._init_messages(task)
        self._steps = []
        self._completed_subagents = []

        for step_num in range(1, self.max_steps + 1):
            logger.info(f"StructuredReAct step {step_num}/{self.max_steps}")

            try:
                step: AgentReActStep = await structured_completion(
                    client=self.client,
                    messages=self._messages,
                    response_model=AgentReActStep,
                    model=self.model,
                    trace_name=f"agent-step-{step_num}",
                    trace_metadata={"loop": "instructor-react", "step": step_num, "max_steps": self.max_steps},
                )
            except Exception as exc:
                error_text = str(exc).lower()
                if "multiple tool calls" in error_text and not getattr(self, "_retried_single_action", False):
                    logger.warning("  Structured output returned multiple tool calls; retrying once with stricter single-action instruction")
                    self._retried_single_action = True
                    self._strengthen_single_action_instruction()
                    step = await structured_completion(
                        client=self.client,
                        messages=self._messages,
                        response_model=AgentReActStep,
                        model=self.model,
                        trace_name=f"agent-step-{step_num}-retry",
                        trace_metadata={"loop": "instructor-react", "step": step_num, "retry": True},
                    )
                elif "either action or final_answer" in error_text:
                    # LLM returned both action and final_answer as null; inject a corrective message
                    # and retry so the loop can continue rather than crashing.
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

            final_answer = step.final_answer
            if final_answer is not None and final_answer.strip().lower() == "null":
                logger.warning("  Final answer was literal null; treating as missing final answer")
                final_answer = None

            if final_answer is not None:
                # Pipeline enforcement: reject premature final_answer
                if self.required_pipeline:
                    missing = [s for s in self.required_pipeline if s not in self._completed_subagents]
                    if missing:
                        next_stage = missing[0]
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

                logger.info(f"  Action: delegate to '{action_name}' with task: {step.action.task[:200]}")

                try:
                    observation = await self.action_handler(action_name, action_args)
                except Exception as e:
                    observation = f"ERROR: {type(e).__name__}: {e}"
                    logger.error(f"  Action failed: {observation}")

                # Track completed subagent for pipeline enforcement.
                # Only mark as completed when the observation doesn't indicate a failure.
                _failure_prefixes = ("error:", "i'm unable", "i am unable", "i cannot", "could not", "failed to")
                _obs_lower = observation.lower()
                _is_failure = any(_obs_lower.startswith(p) for p in _failure_prefixes)
                if action_name in self.required_pipeline and action_name not in self._completed_subagents:
                    if not _is_failure:
                        self._completed_subagents.append(action_name)
                        remaining = [s for s in self.required_pipeline if s not in self._completed_subagents]
                        logger.info(
                            f"  Pipeline progress: completed '{action_name}'. "
                            f"Remaining: {remaining if remaining else 'none (pipeline complete)'}"
                        )
                    else:
                        logger.warning(
                            f"  Pipeline stage '{action_name}' reported a failure; not marking as complete."
                        )

                logger.info(f"  Observation: {observation[:200]}")

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