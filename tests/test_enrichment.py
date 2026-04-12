from __future__ import annotations

from pathlib import Path

import pytest

from engine.core.loader import load_engine_config
from engine.core.models import EnricherConfig, EnrichmentDecision


CONFIGS_DIR = Path(__file__).parent.parent / "configs"


# ─── Config Loading ───


class TestEnricherLoading:
    def test_enrichers_loaded_from_configs_dir(self):
        config = load_engine_config(CONFIGS_DIR)
        assert "document_discovery" in config.enrichers
        enricher = config.enrichers["document_discovery"]
        assert enricher.executor == "glob_file_discovery"
        assert enricher.enabled is True
        assert enricher.priority == 50

    def test_empty_enrichers_dir(self, tmp_path):
        (tmp_path / "enrichers").mkdir()
        config = load_engine_config(tmp_path)
        assert config.enrichers == {}

    def test_no_enrichers_dir(self, tmp_path):
        config = load_engine_config(tmp_path)
        assert config.enrichers == {}

    def test_duplicate_enricher_id_raises(self, tmp_path):
        enrichers_dir = tmp_path / "enrichers"
        enrichers_dir.mkdir()
        (enrichers_dir / "a.yaml").write_text(
            "id: dup\ndescription: first\nexecutor: glob_file_discovery\n",
            encoding="utf-8",
        )
        (enrichers_dir / "b.yaml").write_text(
            "id: dup\ndescription: second\nexecutor: glob_file_discovery\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Duplicate enricher id"):
            load_engine_config(tmp_path)


# ─── Model Validation ───


class TestEnricherConfigValidation:
    def test_valid_enricher(self):
        e = EnricherConfig(
            id="test",
            description="test enricher",
            executor="glob_file_discovery",
        )
        assert e.id == "test"
        assert e.priority == 50
        assert e.enabled is True
        assert e.model is None

    def test_empty_id_raises(self):
        with pytest.raises(ValueError, match="empty"):
            EnricherConfig(id="  ", description="x", executor="glob_file_discovery")

    def test_empty_executor_raises(self):
        with pytest.raises(ValueError, match="executor"):
            EnricherConfig(id="test", description="x", executor="  ")

    def test_priority_bounds(self):
        with pytest.raises(Exception):
            EnricherConfig(id="x", description="x", executor="e", priority=101)

    def test_disabled_enricher(self):
        e = EnricherConfig(
            id="x",
            description="x",
            executor="glob_file_discovery",
            enabled=False,
        )
        assert e.enabled is False


class TestEnrichmentDecision:
    def test_no_enricher_selected(self):
        d = EnrichmentDecision(enricher_id=None, reason="No enricher needed")
        assert d.enricher_id is None

    def test_enricher_selected(self):
        d = EnrichmentDecision(
            enricher_id="document_discovery",
            reason="Query asks for document processing",
        )
        assert d.enricher_id == "document_discovery"


# ─── Graph Validation ───


class TestEnricherGraphValidation:
    def test_real_config_has_no_enricher_errors(self):
        from engine.core.graph import validate_config_graph

        config = load_engine_config(CONFIGS_DIR)
        issues = validate_config_graph(config)
        errors = [i for i in issues if i.level == "error"]
        assert errors == [], f"Unexpected errors: {errors}"

    def test_enricher_collision_with_agent(self, tmp_path):
        from engine.core.graph import validate_config_graph

        (tmp_path / "agents").mkdir()
        (tmp_path / "subagents").mkdir()
        (tmp_path / "tools").mkdir()
        (tmp_path / "enrichers").mkdir()

        (tmp_path / "tools" / "t.yaml").write_text(
            "id: t\ndescription: tool\nparameters: {}\n",
            encoding="utf-8",
        )
        (tmp_path / "subagents" / "s.yaml").write_text(
            "id: s\nrole_type: subagent\ndescription: x\n"
            "system_prompt: x\ndependencies: [t]\n",
            encoding="utf-8",
        )
        (tmp_path / "agents" / "collider.yaml").write_text(
            "id: collider\nrole_type: agent\ndescription: x\n"
            "system_prompt: x\ndependencies: [s]\n",
            encoding="utf-8",
        )
        (tmp_path / "enrichers" / "collider.yaml").write_text(
            "id: collider\ndescription: clash\nexecutor: glob_file_discovery\n",
            encoding="utf-8",
        )

        config = load_engine_config(tmp_path)
        issues = validate_config_graph(config)
        collision_errors = [i for i in issues if "collider" in i.message and "collides" in i.message]
        assert len(collision_errors) >= 1
