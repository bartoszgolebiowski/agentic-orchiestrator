"""Per-tool result projection using JSONPath selectors.

Projection specs are intentionally small and declarative:

- A plain string is treated as a JSONPath selector that extracts a scalar or
  list value.
- A mapping is treated as a field map that builds a smaller JSON object.
- A mapping with ``path`` and ``fields`` projects a nested collection.

If projection fails or the raw value is not valid JSON, the original string is
returned unchanged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any, TypeAlias

from jsonpath_ng import parse as jsonpath_parse
from jsonpath_ng.exceptions import JsonPathParserError

logger = logging.getLogger(__name__)

ProjectionSelector: TypeAlias = str | Sequence[str]
ProjectionSpec: TypeAlias = str | Mapping[str, Any]


@lru_cache(maxsize=256)
def compile_projection(expression: str) -> Any:
    """Compile a JSONPath expression.

    Raises ``JsonPathParserError`` on bad syntax.
    """
    return jsonpath_parse(expression)


def _normalize_matches(matches: list[Any], *, force_list: bool = False) -> Any:
    if force_list:
        return [match.value for match in matches]
    if len(matches) == 1:
        return matches[0].value
    return [match.value for match in matches]


def _resolve_selector(data: Any, selector_spec: ProjectionSelector) -> Any | None:
    selectors = [selector_spec] if isinstance(selector_spec, str) else list(selector_spec)
    if not selectors:
        return None

    for expression in selectors:
        try:
            selector = compile_projection(expression)
        except Exception:
            logger.warning("Invalid JSONPath expression %r — returning raw result", expression)
            return None

        matches = selector.find(data)
        if matches:
            return _normalize_matches(matches, force_list="[*]" in expression)

    return None


def _project_spec(data: Any, spec: Any) -> Any | None:
    if spec is None:
        return None

    if isinstance(spec, str):
        return _resolve_selector(data, spec)

    if isinstance(spec, Sequence) and not isinstance(spec, (str, Mapping)):
        return _resolve_selector(data, spec)

    if isinstance(spec, Mapping):
        if "path" in spec and "fields" in spec:
            return _project_collection(data, spec)
        return _project_field_map(data, spec)

    return None


def _project_field_map(data: Any, field_map: Mapping[str, Any]) -> Any | None:
    if isinstance(data, list):
        projected_items: list[Any] = []
        projected_any = False
        for item in data:
            projected_item = _project_field_map(item, field_map)
            if projected_item is None:
                projected_items.append(item)
            else:
                projected_items.append(projected_item)
                projected_any = True
        return projected_items if projected_any else None

    if not isinstance(data, Mapping):
        return None

    projected: dict[str, Any] = {}
    for field_name, field_spec in field_map.items():
        if field_name in {"path", "fields"}:
            continue
        value = _project_spec(data, field_spec)
        if value is not None:
            projected[field_name] = value

    return projected if projected else None


def _project_collection(data: Any, collection_spec: Mapping[str, Any]) -> Any | None:
    path_spec = collection_spec.get("path")
    fields_spec = collection_spec.get("fields")
    if not isinstance(fields_spec, Mapping):
        return None

    selected = _project_spec(data, path_spec)
    if selected is None:
        return None

    if isinstance(selected, list):
        projected_items: list[Any] = []
        projected_any = False
        for item in selected:
            projected_item = _project_field_map(item, fields_spec)
            if projected_item is None:
                projected_items.append(item)
            else:
                projected_items.append(projected_item)
                projected_any = True
        return projected_items if projected_any else None

    return _project_field_map(selected, fields_spec)


def validate_projection_spec(spec: ProjectionSpec) -> None:
    """Validate a projection spec by compiling every JSONPath selector it uses."""

    def validate(value: Any) -> None:
        if isinstance(value, str):
            compile_projection(value)
            return

        if isinstance(value, Sequence) and not isinstance(value, (str, Mapping)):
            if not value:
                raise ValueError("Projection selector list cannot be empty")
            for item in value:
                if not isinstance(item, str):
                    raise ValueError("Projection selector lists must contain strings only")
                compile_projection(item)
            return

        if isinstance(value, Mapping):
            if "path" in value or "fields" in value:
                if set(value) != {"path", "fields"}:
                    raise ValueError("Collection projection specs must contain only 'path' and 'fields'")
                validate(value["path"])
                fields = value["fields"]
                if not isinstance(fields, Mapping):
                    raise ValueError("Collection projection 'fields' must be a mapping")
                if not fields:
                    raise ValueError("Collection projection 'fields' cannot be empty")
                validate(fields)
                return

            if not value:
                raise ValueError("Projection field maps cannot be empty")

            for field_name, field_spec in value.items():
                if not isinstance(field_name, str) or not field_name.strip():
                    raise ValueError("Projection field names must be non-empty strings")
                validate(field_spec)
            return

        raise ValueError(f"Unsupported projection spec type: {type(value)!r}")

    validate(spec)


def project_tool_result(raw: str, projection: ProjectionSpec) -> str:
    """Project *raw* tool output through a projection spec.

    Rules
    -----
    * String selectors behave like regular JSONPath and return the extracted
      value.
    * Field maps return a compact JSON object with the selected fields.
    * Collection specs project the selected array/object items with a nested
      field map.
    * If parsing, compilation, or extraction fails at any point, return *raw*
      unchanged.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw

    try:
        projected = _project_spec(data, projection)
    except (JsonPathParserError, Exception):
        logger.warning("Projection failed — returning raw result", exc_info=True)
        return raw

    if projected is None:
        return raw

    return json.dumps(projected, ensure_ascii=False, indent=2, sort_keys=True)


def project_tool_result_strict(raw: str, projection: ProjectionSpec) -> str:
    """Project *raw* tool output and fail closed when projection cannot be applied.

    This variant is for LLM-facing tool responses. It never falls back to the
    original raw payload, because that would leak verbose tool output back into
    the ReAct loop.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Configured tool projection requires JSON output") from exc

    try:
        projected = _project_spec(data, projection)
    except (JsonPathParserError, Exception) as exc:
        logger.warning("Strict projection failed", exc_info=True)
        raise ValueError("Configured tool projection could not be applied") from exc

    if projected is None:
        raise ValueError("Configured tool projection produced no result")

    return json.dumps(projected, ensure_ascii=False, indent=2, sort_keys=True)
