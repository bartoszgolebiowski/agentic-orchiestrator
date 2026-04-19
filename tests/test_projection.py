from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.config.loader import load_engine_config
from engine.config.models import NodeConfig, RoleType
from engine.tools.projection import (
    compile_projection,
    project_tool_result,
    project_tool_result_strict,
    validate_projection_spec,
)


CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class TestCompileProjection:
    def test_valid_expression_compiles(self) -> None:
        selector = compile_projection("$.id")
        assert selector is not None

    def test_invalid_expression_raises(self) -> None:
        from jsonpath_ng.exceptions import JsonPathParserError

        with pytest.raises(JsonPathParserError):
            compile_projection("[[[invalid")


class TestProjectSelector:
    def test_single_field_selector(self) -> None:
        raw = _json({"id": "abc123", "name": "Card", "desc": "long text"})
        result = project_tool_result(raw, "$.id")
        assert json.loads(result) == "abc123"

    def test_missing_selector_returns_raw(self) -> None:
        raw = _json({"name": "Card"})
        result = project_tool_result(raw, "$.missing")
        assert result == raw

    def test_invalid_jsonpath_returns_raw(self) -> None:
        raw = _json({"id": "123"})
        result = project_tool_result(raw, "[[[bad")
        assert result == raw


class TestStrictProjectSelector:
    def test_rejects_non_json_output(self) -> None:
        with pytest.raises(ValueError, match="requires JSON output"):
            project_tool_result_strict("plain text response", {"id": "$.id"})

    def test_rejects_missing_projection_result(self) -> None:
        raw = _json({"unexpected": "payload"})

        with pytest.raises(ValueError, match="produced no result"):
            project_tool_result_strict(raw, {"id": "$.id"})

    def test_preserves_error_envelope(self) -> None:
        raw = _json({"status": "error", "hint": "board not selected"})

        result = project_tool_result_strict(raw, {"id": "$.id"})

        assert result == raw


class TestProjectFieldMap:
    def test_card_subset_projection(self) -> None:
        raw = _json(
            {
                "id": "card-1",
                "name": "Build login",
                "desc": "Implement login page",
                "due": "2026-04-12T12:00:00.000Z",
                "dueComplete": False,
                "idList": "list-9",
                "labels": [
                    {"id": "lbl-1", "name": "backend", "color": "blue"},
                    {"id": "lbl-2", "name": "urgent", "color": "red"},
                ],
                "url": "https://trello.example/card/1",
                "shortUrl": "https://trello.example/c/1",
                "badges": {"votes": 2},
            }
        )
        spec = {
            "id": "$.id",
            "name": "$.name",
            "desc": "$.desc",
            "due": "$.due",
            "dueComplete": "$.dueComplete",
            "idList": "$.idList",
            "labels": "$.labels[*].name",
            "url": "$.url",
            "shortUrl": "$.shortUrl",
        }

        result = json.loads(project_tool_result(raw, spec))

        assert result == {
            "desc": "Implement login page",
            "due": "2026-04-12T12:00:00.000Z",
            "dueComplete": False,
            "id": "card-1",
            "idList": "list-9",
            "labels": ["backend", "urgent"],
            "name": "Build login",
            "shortUrl": "https://trello.example/c/1",
            "url": "https://trello.example/card/1",
        }

    def test_board_subset_projection_with_name_alias(self) -> None:
        raw = _json(
            {
                "id": "board-1",
                "displayName": "Delivery",
                "desc": "Team board",
                "url": "https://trello.example/b/1",
                "closed": False,
                "prefs": {"background": "blue"},
            }
        )
        spec = {
            "id": "$.id",
            "name": ["$.name", "$.displayName"],
            "desc": "$.desc",
            "url": "$.url",
            "closed": "$.closed",
        }

        result = json.loads(project_tool_result(raw, spec))

        assert result == {
            "closed": False,
            "desc": "Team board",
            "id": "board-1",
            "name": "Delivery",
            "url": "https://trello.example/b/1",
        }


class TestProjectArray:
    def test_list_projection(self) -> None:
        raw = _json(
            [
                {"id": "list-1", "name": "Todo", "closed": False, "pos": 1000, "cards": [1]},
                {"id": "list-2", "name": "Doing", "closed": False, "pos": 2000, "cards": [2]},
            ]
        )
        spec = {
            "id": "$.id",
            "name": "$.name",
            "closed": "$.closed",
            "pos": "$.pos",
        }

        result = json.loads(project_tool_result(raw, spec))

        assert result == [
            {"closed": False, "id": "list-1", "name": "Todo", "pos": 1000},
            {"closed": False, "id": "list-2", "name": "Doing", "pos": 2000},
        ]

    def test_card_array_projection(self) -> None:
        raw = _json(
            [
                {
                    "id": "card-1",
                    "name": "Fix login",
                    "desc": "first card",
                    "due": None,
                    "dueComplete": False,
                    "idList": "todo",
                    "labels": [{"name": "backend"}],
                    "url": "https://trello.example/card/1",
                    "shortUrl": "https://trello.example/c/1",
                    "badges": {"votes": 3},
                },
                {
                    "id": "card-2",
                    "name": "Write docs",
                    "desc": "second card",
                    "due": "2026-04-15T09:00:00.000Z",
                    "dueComplete": True,
                    "idList": "doing",
                    "labels": [],
                    "url": "https://trello.example/card/2",
                    "shortUrl": "https://trello.example/c/2",
                    "badges": {"votes": 0},
                },
            ]
        )
        spec = {
            "id": "$.id",
            "name": "$.name",
            "desc": "$.desc",
            "due": "$.due",
            "dueComplete": "$.dueComplete",
            "idList": "$.idList",
            "labels": "$.labels[*].name",
            "url": "$.url",
            "shortUrl": "$.shortUrl",
        }

        result = json.loads(project_tool_result(raw, spec))

        assert result[0]["id"] == "card-1"
        assert result[0]["labels"] == ["backend"]
        assert result[1]["dueComplete"] is True
        assert result[1]["shortUrl"] == "https://trello.example/c/2"


class TestProjectNestedCollection:
    def test_checklist_projection(self) -> None:
        raw = _json(
            {
                "id": "checklist-1",
                "name": "Acceptance Criteria",
                "percentComplete": 50,
                "checkItems": [
                    {"id": "item-1", "text": "login works", "state": "complete", "limits": {}},
                    {"id": "item-2", "name": "signup works", "state": "incomplete", "extra": True},
                ],
                "other": "noise",
            }
        )
        spec = {
            "id": "$.id",
            "name": ["$.name", "$.text"],
            "percentComplete": "$.percentComplete",
            "checkItems": {
                "path": "$.checkItems[*]",
                "fields": {
                    "id": "$.id",
                    "text": ["$.text", "$.name"],
                    "state": "$.state",
                },
            },
        }

        result = json.loads(project_tool_result(raw, spec))

        assert result == {
            "checkItems": [
                {"id": "item-1", "state": "complete", "text": "login works"},
                {"id": "item-2", "state": "incomplete", "text": "signup works"},
            ],
            "id": "checklist-1",
            "name": "Acceptance Criteria",
            "percentComplete": 50,
        }


class TestProjectionConfigValidation:
    def test_valid_projection_accepted(self) -> None:
        node = NodeConfig(
            id="proj_sub",
            role_type=RoleType.SUBAGENT,
            description="test",
            system_prompt="test",
            tool_result_projection={
                "some_tool": {
                    "id": "$.id",
                    "name": ["$.name", "$.displayName"],
                }
            },
        )
        assert node.tool_result_projection["some_tool"]["id"] == "$.id"

    def test_nested_collection_projection_validates(self) -> None:
        validate_projection_spec(
            {
                "id": "$.id",
                "items": {
                    "path": "$.checkItems[*]",
                    "fields": {"id": "$.id", "state": "$.state"},
                },
            }
        )

    def test_invalid_jsonpath_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid projection"):
            NodeConfig(
                id="bad_proj",
                role_type=RoleType.SUBAGENT,
                description="test",
                system_prompt="test",
                tool_result_projection={"some_tool": {"id": "[[[bad"}},
            )

    def test_agent_cannot_declare_projection(self) -> None:
        with pytest.raises(ValueError, match="cannot declare tool_result_projection"):
            NodeConfig(
                id="bad_agent",
                role_type=RoleType.AGENT,
                description="test",
                system_prompt="test",
                dependencies=["sub1"],
                tool_result_projection={"tool": {"id": "$.id"}},
            )


class TestProjectionYAMLLoading:
    def test_trello_publisher_has_projections(self) -> None:
        config = load_engine_config(CONFIGS_DIR)
        publisher = config.subagents["trello_publisher"]

        assert publisher.tool_result_projection["trello__add_card_to_list"]["id"] == "$.id"
        assert publisher.tool_result_projection["trello__add_card_to_list"]["labels"] == "$.labels[*].name"
        assert publisher.tool_result_projection["trello__get_checklist_by_name"]["checkItems"]["path"] == "$.checkItems[*]"

    def test_trello_intake_parser_exposes_no_trello_tools(self) -> None:
        config = load_engine_config(CONFIGS_DIR)
        parser = config.subagents["trello_intake_parser"]

        assert parser.mcp_dependencies == ["trello"]
        assert parser.mcp_include_tools["trello"] == []
        assert parser.mcp_skip_hitl_tools["trello"] == []
        assert parser.tool_result_projection == {}

    def test_mcp_included_tools_have_projection_entries(self) -> None:
        config = load_engine_config(CONFIGS_DIR)
        missing: dict[str, list[str]] = {}

        for subagent_id, subagent in config.subagents.items():
            if not subagent.mcp_include_tools:
                continue

            expected_projection_keys: set[str] = set()
            for server_id, remote_names in subagent.mcp_include_tools.items():
                server_config = config.mcps[server_id]
                expected_projection_keys.update(
                    server_config.exposed_tool_name(remote_name) for remote_name in remote_names
                )

            missing_keys = sorted(expected_projection_keys - set(subagent.tool_result_projection))
            if missing_keys:
                missing[subagent_id] = missing_keys

        assert missing == {}, f"Missing projection coverage: {missing}"

    def test_trello_task_matcher_has_board_and_card_projections(self) -> None:
        config = load_engine_config(CONFIGS_DIR)
        matcher = config.subagents["trello_task_matcher"]

        assert matcher.tool_result_projection["trello__list_workspaces"]["name"] == ["$.displayName", "$.name"]
        assert matcher.tool_result_projection["trello__get_cards_by_list_id"]["labels"] == "$.labels[*].name"

    def test_trello_task_operator_has_comment_and_checklist_projections(self) -> None:
        config = load_engine_config(CONFIGS_DIR)
        operator = config.subagents["trello_task_operator"]

        assert operator.tool_result_projection["trello__add_comment"]["text"] == "$.data.text"
        assert operator.tool_result_projection["trello__get_checklist_by_name"]["checkItems"]["fields"]["state"] == "$.state"
