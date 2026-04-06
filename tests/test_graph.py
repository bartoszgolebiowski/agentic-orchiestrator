from __future__ import annotations

from pathlib import Path

import pytest

from engine.core.loader import load_engine_config
from engine.core.graph import (
    ConfigIssue,
    build_dependency_dict,
    build_dependency_tree,
    validate_config_graph,
)


CONFIGS_DIR = Path(__file__).parent.parent / "configs"


class TestValidateConfigGraph:
    def test_real_config_has_no_errors(self):
        config = load_engine_config(CONFIGS_DIR)
        issues = validate_config_graph(config)
        errors = [i for i in issues if i.level == "error"]
        assert errors == [], f"Unexpected errors: {errors}"

    def test_detects_cross_namespace_collision(self, tmp_path):
        (tmp_path / "agents").mkdir()
        (tmp_path / "subagents").mkdir()
        (tmp_path / "tools").mkdir()

        (tmp_path / "tools" / "helper.yaml").write_text(
            "id: helper\ndescription: helper\nparameters: {}\n",
            encoding="utf-8",
        )
        # Use 'helper' as both a subagent and a tool ID
        (tmp_path / "subagents" / "helper.yaml").write_text(
            "id: helper\nrole_type: subagent\ndescription: dup\n"
            "system_prompt: x\ndependencies: [helper]\n",
            encoding="utf-8",
        )
        (tmp_path / "agents" / "a.yaml").write_text(
            "id: a\nrole_type: agent\ndescription: x\n"
            "system_prompt: x\ndependencies: [helper]\n",
            encoding="utf-8",
        )

        config = load_engine_config(tmp_path)
        issues = validate_config_graph(config)
        collision_errors = [i for i in issues if "collides" in i.message]
        assert len(collision_errors) == 1
        assert "helper" in collision_errors[0].message

    def test_detects_orphaned_subagent(self, tmp_path):
        (tmp_path / "agents").mkdir()
        (tmp_path / "subagents").mkdir()
        (tmp_path / "tools").mkdir()

        (tmp_path / "tools" / "t1.yaml").write_text(
            "id: t1\ndescription: tool\nparameters: {}\n",
            encoding="utf-8",
        )
        (tmp_path / "subagents" / "used.yaml").write_text(
            "id: used\nrole_type: subagent\ndescription: x\n"
            "system_prompt: x\ndependencies: [t1]\n",
            encoding="utf-8",
        )
        (tmp_path / "subagents" / "orphan.yaml").write_text(
            "id: orphan\nrole_type: subagent\ndescription: x\n"
            "system_prompt: x\ndependencies: [t1]\n",
            encoding="utf-8",
        )
        (tmp_path / "agents" / "a.yaml").write_text(
            "id: a\nrole_type: agent\ndescription: x\n"
            "system_prompt: x\ndependencies: [used]\n",
            encoding="utf-8",
        )

        config = load_engine_config(tmp_path)
        issues = validate_config_graph(config)
        orphan_warnings = [i for i in issues if "orphan" in i.message.lower()]
        assert any("orphan" in w.message for w in orphan_warnings)

    def test_detects_orphaned_tool(self, tmp_path):
        (tmp_path / "agents").mkdir()
        (tmp_path / "subagents").mkdir()
        (tmp_path / "tools").mkdir()

        (tmp_path / "tools" / "used.yaml").write_text(
            "id: used\ndescription: tool\nparameters: {}\n",
            encoding="utf-8",
        )
        (tmp_path / "tools" / "unused.yaml").write_text(
            "id: unused\ndescription: tool\nparameters: {}\n",
            encoding="utf-8",
        )
        (tmp_path / "subagents" / "s.yaml").write_text(
            "id: s\nrole_type: subagent\ndescription: x\n"
            "system_prompt: x\ndependencies: [used]\n",
            encoding="utf-8",
        )
        (tmp_path / "agents" / "a.yaml").write_text(
            "id: a\nrole_type: agent\ndescription: x\n"
            "system_prompt: x\ndependencies: [s]\n",
            encoding="utf-8",
        )

        config = load_engine_config(tmp_path)
        issues = validate_config_graph(config)
        tool_warnings = [i for i in issues if "unused" in i.message and "Tool" in i.message]
        assert len(tool_warnings) == 1


class TestBuildDependencyTree:
    def test_tree_contains_all_agents(self):
        config = load_engine_config(CONFIGS_DIR)
        tree = build_dependency_tree(config)
        assert "Orchestrator" in tree
        for agent_id in config.agents:
            assert agent_id in tree
        assert "trello_publisher" in tree

    def test_tree_contains_pipeline_annotation(self):
        config = load_engine_config(CONFIGS_DIR)
        tree = build_dependency_tree(config)
        assert "[pipeline:" in tree

    def test_tree_omits_contract_annotations(self):
        config = load_engine_config(CONFIGS_DIR)
        tree = build_dependency_tree(config)
        assert "[input:" not in tree
        assert "[output:" not in tree

    def test_tree_shows_mcp_dependencies(self, tmp_path):
        (tmp_path / "agents").mkdir()
        (tmp_path / "subagents").mkdir()
        (tmp_path / "tools").mkdir()
        (tmp_path / "mcps").mkdir()

        (tmp_path / "tools" / "t1.yaml").write_text(
            "id: t1\ndescription: tool\nparameters: {}\n",
            encoding="utf-8",
        )
        (tmp_path / "mcps" / "remote.yaml").write_text(
            "id: remote\ndescription: remote server\n"
            "connection:\n  transport: http\n  url: https://example.com/mcp\n",
            encoding="utf-8",
        )
        (tmp_path / "subagents" / "s.yaml").write_text(
            "id: s\nrole_type: subagent\ndescription: x\n"
            "system_prompt: x\ndependencies: [t1]\nmcp_dependencies: [remote]\n",
            encoding="utf-8",
        )
        (tmp_path / "agents" / "a.yaml").write_text(
            "id: a\nrole_type: agent\ndescription: x\n"
            "system_prompt: x\ndependencies: [s]\n",
            encoding="utf-8",
        )

        config = load_engine_config(tmp_path)
        tree = build_dependency_tree(config)
        assert "MCP: remote" in tree


class TestBuildDependencyDict:
    def test_dict_is_machine_readable(self):
        config = load_engine_config(CONFIGS_DIR)
        graph = build_dependency_dict(config)
        assert "orchestrator" in graph
        for agent_id in config.agents:
            assert agent_id in graph["orchestrator"]
            entry = graph["orchestrator"][agent_id]
            assert "subagents" in entry
            assert "required_pipeline" in entry
            assert "input_model" not in entry
            assert "output_model" not in entry
            for subagent_entry in entry["subagents"].values():
                assert "input_model" not in subagent_entry
                assert "output_model" not in subagent_entry
