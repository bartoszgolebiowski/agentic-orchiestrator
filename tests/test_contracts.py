from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from engine.contracts.document import MarkdownExtractionInput
from engine.config.contracts import describe_model_contract, normalize_handoff_payload
from engine.agents.react import StructuredReActLoop


def test_describe_model_contract_is_compact() -> None:
    summary = describe_model_contract(MarkdownExtractionInput)

    assert "MarkdownExtractionInput" in summary
    assert "Additional properties: not allowed." in summary
    assert "Required:" in summary
    assert "source_path" in summary
    assert "$defs" not in summary
    assert '"properties"' not in summary


def test_normalize_handoff_payload_parses_json_strings() -> None:
    assert normalize_handoff_payload('{"source_path": "docs/example.md"}') == {
        "source_path": "docs/example.md"
    }


def test_structured_react_loop_mentions_input_json() -> None:
    loop = StructuredReActLoop(
        client=MagicMock(),
        system_prompt="System prompt",
        action_handler=AsyncMock(),
        available_actions="- markdown_extractor: example",
    )

    loop._init_messages("analyze docs", input_json={"source_path": "docs/example.md"})

    prompt = loop._messages[0]["content"]
    assert "input_json" in prompt
    assert "subagent_id" in prompt
    assert "final_answer" in prompt