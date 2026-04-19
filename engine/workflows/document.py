"""Backward-compat shim — use engine.enrichment.document directly."""
from engine.enrichment.document import (
    DocumentWorkflowConfig,
    build_document_input,
    discover_markdown_sources,
    load_document_workflow_config,
    should_run_document_workflow,
)

__all__ = [
    "DocumentWorkflowConfig",
    "build_document_input",
    "discover_markdown_sources",
    "load_document_workflow_config",
    "should_run_document_workflow",
]
