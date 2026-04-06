from __future__ import annotations

from pathlib import Path

from engine.workflows.document import (
    build_document_input,
    discover_markdown_sources,
    load_document_workflow_config,
    should_run_document_workflow,
)


CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def test_load_document_workflow_config() -> None:
    workflow = load_document_workflow_config(CONFIGS_DIR)

    assert workflow is not None
    assert workflow.scan_pattern == "**/*.md"
    assert workflow.source_dir.name == "documents"


def test_discover_markdown_sources() -> None:
    workflow = load_document_workflow_config(CONFIGS_DIR)
    assert workflow is not None

    sources = discover_markdown_sources(workflow.source_dir, workflow.scan_pattern)

    assert len(sources) >= 4
    assert all(path.suffix.lower() == ".md" for path in sources)


def test_should_run_document_workflow() -> None:
    assert should_run_document_workflow("analyze documents and create epics/stories/tasks")
    assert should_run_document_workflow("extract markdown notes")
    assert not should_run_document_workflow("calculate 2 + 2")


def test_build_document_input() -> None:
    payload = build_document_input(Path("docs/example.md"))

    assert payload == {"source_path": str(Path("docs/example.md").resolve())}
