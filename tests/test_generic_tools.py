from __future__ import annotations

import json

import pytest

from engine.contracts.document import DocumentPlan
from engine.core.contracts import resolve_model, validate_model_payload
from engine.core.models import ModelSpec
from engine.tools.markdown_tools import read_markdown_structure


@pytest.mark.asyncio
async def test_read_markdown_structure_extracts_sections(tmp_path):
    source = tmp_path / "plan.md"
    source.write_text(
        """
# Product Plan

## Goals
- Improve reliability
- Improve onboarding
""".strip(),
        encoding="utf-8",
    )

    payload_json = await read_markdown_structure(source_path=str(source))
    payload = json.loads(payload_json)

    assert payload["title"] == "Product Plan"
    assert payload["line_count"] >= 3
    assert payload["sections"][0]["heading"] == "Product Plan"
    assert "outline" in payload


def test_model_reference_resolves_document_plan():
    model_cls = resolve_model(ModelSpec(module="engine.contracts.document", name="DocumentPlan"))

    assert model_cls is DocumentPlan


def test_document_plan_model_validation():
    payload = {
        "source_path": "docs/example.md",
        "title": "Example Plan",
        "summary": "",
        "labels": ["Q3"],
        "epics": [],
    }

    parsed = validate_model_payload(DocumentPlan, payload)

    assert parsed.title == "Example Plan"
    assert parsed.source_path == "docs/example.md"
