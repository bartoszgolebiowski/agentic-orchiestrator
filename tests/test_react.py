from __future__ import annotations

from dataclasses import dataclass

import pytest

from engine.core.react import ReActLoop


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

    monkeypatch.setattr("engine.agents.react.chat_completion", fake_chat_completion)

    result = await loop.run("extract markdown")

    assert result == '{"ok": true}'
    assert len(captured_messages) == 2
    assert "previous response was empty" in str(captured_messages[1][-1]["content"]).lower()