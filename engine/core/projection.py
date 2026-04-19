from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any, TypeAlias

from jsonpath_ng import parse as jsonpath_parse

ProjectionSelector: TypeAlias = str | Sequence[str]
ProjectionSpec: TypeAlias = str | Mapping[str, Any]


@lru_cache(maxsize=256)
def compile_projection(expression: str) -> Any:
    return jsonpath_parse(expression)


def validate_projection_spec(spec: ProjectionSpec) -> None:
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
