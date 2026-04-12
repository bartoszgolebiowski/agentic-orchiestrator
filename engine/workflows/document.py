from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentWorkflowConfig:
    source_dir: Path
    output_dir: Path | None = None
    scan_pattern: str = "**/*.md"
    watch_interval_seconds: int = 10
    trello: dict[str, Any] = field(default_factory=dict)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_handle:
        data = yaml.safe_load(file_handle)
    return data or {}


def _expand_env_values(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_values(item) for key, item in value.items()}
    return value


def _resolve_relative_path(base_dir: Path, raw_value: str | None) -> Path | None:
    if raw_value is None:
        return None
    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def load_document_workflow_config(
    config_dir: Path,
    workflow_config_file: str | None = None,
) -> DocumentWorkflowConfig | None:
    if workflow_config_file is None:
        config_path = config_dir / "document_workflow.yaml"
        if not config_path.exists():
            matches = sorted(config_dir.rglob("document_workflow.yaml"))
            if not matches:
                return None
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple document_workflow.yaml files found under {config_dir}"
                )
            config_path = matches[0]
    else:
        config_path = (config_dir / Path(workflow_config_file)).resolve()
        if not config_path.exists():
            return None

    raw = _expand_env_values(_load_yaml(config_path))
    source_dir = _resolve_relative_path(config_dir, raw.get("source_dir"))
    if source_dir is None:
        raise ValueError("document_workflow.yaml must define source_dir")

    output_dir = _resolve_relative_path(config_dir, raw.get("output_dir"))

    return DocumentWorkflowConfig(
        source_dir=source_dir,
        output_dir=output_dir,
        scan_pattern=str(raw.get("scan_pattern", "**/*.md")),
        watch_interval_seconds=int(raw.get("watch_interval_seconds", 10)),
        trello=dict(raw.get("trello", {}) or {}),
    )


def discover_markdown_sources(source_dir: Path, scan_pattern: str) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Document source directory does not exist: {source_dir}")

    candidates = [
        path.resolve()
        for path in source_dir.glob(scan_pattern)
        if path.is_file() and path.suffix.lower() == ".md"
    ]

    return sorted(dict.fromkeys(candidates))


def should_run_document_workflow(query: str) -> bool:
    lowered = query.lower()
    keywords = (
        "document",
        "documents",
        "markdown",
        "notes",
        "epic",
        "epics",
        "story",
        "stories",
    )
    # Exclude queries that target existing Trello card updates rather than
    # document-to-Trello publishing so the orchestrator can route them to
    # the trello_update_agent instead.
    update_signals = (
        "update card",
        "move card",
        "add comment",
        "log progress",
        "change status",
        "update trello",
        "update task",
        "confirmed_card_id",
        "card_id",
    )
    if any(signal in lowered for signal in update_signals):
        return False
    return any(keyword in lowered for keyword in keywords)


def build_document_input(source_path: Path) -> dict[str, str]:
    return {"source_path": str(source_path.resolve())}