from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from engine.agents.react.base import (
    Action,
    ActionHandler,
    MaxStepsExceededError,
    StepResult,
    init_user_message,
)
from engine.config.contracts import validate_model_payload
from engine.events import EventType, emit_event
from engine.llm.client import (
    chat_completion,
    get_last_usage_details,
    merge_usage_summaries,
    structured_completion,
    summarize_usage,
)

logger = logging.getLogger(__name__)


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
        self._messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": init_user_message(task, input_json=input_json)},
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
            llm_usage_details = get_last_usage_details()
            llm_usage = summarize_usage(llm_usage_details)

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
                        llm_usage=llm_usage,
                        llm_usage_details=llm_usage_details,
                    )

                    try:
                        observation = await self.action_handler(fn_name, fn_args)
                    except Exception as exc:
                        observation = f"ERROR: {type(exc).__name__}: {exc}"
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
                        llm_usage=llm_usage,
                        llm_usage_details=llm_usage_details,
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
                emit_event(
                    EventType.SUBAGENT_STEP,
                    step=step_num,
                    action=None,
                    arguments={},
                    is_final=True,
                    llm_usage=llm_usage,
                    llm_usage_details=llm_usage_details,
                )

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
                            trace_name=f"subagent-final:{self.final_response_model.__name__}",
                            trace_metadata={
                                "loop": "react",
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
                            thought=final_answer,
                            action=None,
                            observation=None,
                            is_final=True,
                            final_answer=normalized,
                            llm_usage=llm_usage,
                            llm_usage_details=llm_usage_details,
                        ))
                        return normalized

                self._steps.append(StepResult(
                    thought=final_answer,
                    action=None,
                    observation=None,
                    is_final=True,
                    final_answer=final_answer,
                    llm_usage=llm_usage,
                    llm_usage_details=llm_usage_details,
                ))
                return final_answer

        raise MaxStepsExceededError(
            f"ReAct loop exceeded {self.max_steps} steps without producing a final answer"
        )
