"""Backward-compat shim — use engine.config.graph directly."""
from engine.config.graph import (
    ConfigIssue,
    build_dependency_dict,
    build_dependency_tree,
    validate_config_graph,
)

__all__ = ["ConfigIssue", "build_dependency_dict", "build_dependency_tree", "validate_config_graph"]
