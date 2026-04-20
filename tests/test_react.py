from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from engine.events import EventType
from engine.agents.react import ReActLoop


@dataclass
class _DummyMessage:
    content: str

    @property
    def tool_calls(self) -> list[object]:
        return []

    def model_dump(self) -> dict[str, str]:
        return {"role": "assistant", "content": self.content}


@pytest.mark.asyncio
async def test_react_loop_retries_empty_final_answer(monkeypatch):
    captured_messages: list[list[dict[str, object]]] = []
    responses = [
        _DummyMessage(""),
        _DummyMessage("{\"ok\": true}"),
    ]

    async def fake_action_handler(name: str, arguments: dict[str, object]) -> str:
        raise AssertionError("action_handler should not be called for blank final-answer retries")

    async def fake_chat_completion(*, client, messages, tools, model, trace_name, trace_metadata, temperature=0.0):
        captured_messages.append([dict(message) for message in messages])
        return responses.pop(0)

    loop = ReActLoop(
        client=object(),
        system_prompt="system prompt",
        tools_spec=[],
        action_handler=fake_action_handler,
        max_steps=3,
    )

    monkeypatch.setattr("engine.agents.react.native.chat_completion", fake_chat_completion)

    result = await loop.run("extract markdown")

    assert result == '{"ok": true}'
    assert len(captured_messages) == 2
    assert "previous response was empty" in str(captured_messages[1][-1]["content"]).lower()


@pytest.mark.asyncio
async def test_react_loop_emits_step_usage_for_tool_and_final_steps(monkeypatch):
    captured_events: list[tuple[EventType, dict[str, object]]] = []
    usage_snapshots = iter([
        {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
        {"input_tokens": 16, "output_tokens": 7, "total_tokens": 23},
    ])

    class _ToolMessage:
        content = "I should call a tool"

        def __init__(self) -> None:
            self.tool_calls = [
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(
                        name="sample_tool",
                        arguments=json.dumps({"value": 1}),
                    ),
                )
            ]

        def model_dump(self) -> dict[str, object]:
            return {
                "role": "assistant",
                "content": self.content,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "sample_tool",
                            "arguments": json.dumps({"value": 1}),
                        },
                    }
                ],
            }

    responses = [_ToolMessage(), _DummyMessage('{"done": true}')]

    async def fake_action_handler(name: str, arguments: dict[str, object]) -> str:
        assert name == "sample_tool"
        assert arguments == {"value": 1}
        return "tool result"

    async def fake_chat_completion(*, client, messages, tools, model, trace_name, trace_metadata, temperature=0.0):
        return responses.pop(0)

    def fake_emit_event(event_type: EventType, **data: object) -> None:
        captured_events.append((event_type, data))

    monkeypatch.setattr("engine.agents.react.native.chat_completion", fake_chat_completion)
    monkeypatch.setattr("engine.agents.react.native.emit_event", fake_emit_event)
    monkeypatch.setattr("engine.agents.react.native.get_last_usage_details", lambda: next(usage_snapshots))

    loop = ReActLoop(
        client=object(),
        system_prompt="system prompt",
        tools_spec=[{"type": "function", "function": {"name": "sample_tool"}}],
        action_handler=fake_action_handler,
        max_steps=3,
    )

    result = await loop.run("use the tool")

    assert result == '{"done": true}'
    tool_step_event = captured_events[0]
    assert tool_step_event[0] == EventType.SUBAGENT_STEP
    assert tool_step_event[1]["action"] == "sample_tool"
    assert tool_step_event[1]["llm_usage"] == {
        "input_tokens": 12,
        "output_tokens": 5,
        "total_tokens": 17,
    }

    final_events = [event for event in captured_events if event[1].get("is_final") is True]
    assert len(final_events) == 1
    assert final_events[0][1]["llm_usage"] == {
        "input_tokens": 16,
        "output_tokens": 7,
        "total_tokens": 23,
    }