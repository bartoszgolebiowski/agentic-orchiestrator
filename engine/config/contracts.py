from __future__ import annotations

import inspect
import importlib
import json
import types
from functools import lru_cache
from typing import Any, Annotated, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from engine.config.models import ModelSpec


@lru_cache(maxsize=256)
def _resolve_model(module_name: str, model_name: str) -> type[BaseModel]:
    module = importlib.import_module(module_name)
    candidate = getattr(module, model_name, None)
    if candidate is None:
        raise ImportError(f"Model '{model_name}' was not found in module '{module_name}'")
    if not isinstance(candidate, type) or not issubclass(candidate, BaseModel):
        raise TypeError(
            f"Reference '{module_name}:{model_name}' does not resolve to a Pydantic BaseModel"
        )
    return candidate


def resolve_model(spec: ModelSpec) -> type[BaseModel]:
    return _resolve_model(spec.module, spec.name)


def resolve_model_or_none(spec: ModelSpec | None) -> type[BaseModel] | None:
    if spec is None:
        return None
    return resolve_model(spec)


def format_model_spec(spec: ModelSpec | None) -> str:
    if spec is None:
        return "(none)"
    return f"{spec.module}:{spec.name}"


def _format_annotation(annotation: Any) -> str:
    if annotation is Any:
        return "any"
    if annotation is None or annotation is type(None):
        return "null"
    if isinstance(annotation, str):
        return annotation

    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type):
            if issubclass(annotation, BaseModel):
                return annotation.__name__
            if annotation.__module__ == "builtins":
                return annotation.__name__
        return str(annotation).replace("typing.", "")

    if origin in (types.UnionType, Union):
        return " | ".join(_format_annotation(arg) for arg in get_args(annotation))

    if origin is Annotated:
        args = get_args(annotation)
        return _format_annotation(args[0]) if args else "any"

    if origin is Literal:
        return "Literal[" + ", ".join(repr(arg) for arg in get_args(annotation)) + "]"

    if origin in (list, set, frozenset):
        args = get_args(annotation)
        inner = ", ".join(_format_annotation(arg) for arg in args) if args else "any"
        return f"{origin.__name__}[{inner}]"

    if origin is tuple:
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return f"tuple[{_format_annotation(args[0])}, ...]"
        inner = ", ".join(_format_annotation(arg) for arg in args) if args else "any"
        return f"tuple[{inner}]"

    if origin is dict:
        args = get_args(annotation)
        key = _format_annotation(args[0]) if len(args) > 0 else "any"
        value = _format_annotation(args[1]) if len(args) > 1 else "any"
        return f"dict[{key}, {value}]"

    return str(annotation).replace("typing.", "")


def _format_default(field: Any) -> str | None:
    if getattr(field, "default_factory", None) is not None:
        factory = field.default_factory
        if factory is list:
            return "[]"
        if factory is dict:
            return "{}"
        if factory is set:
            return "set()"
        if factory is tuple:
            return "()"
        factory_name = getattr(factory, "__name__", None) or "callable"
        return f"<factory:{factory_name}>"

    if getattr(field, "is_required", None) is not None and field.is_required():
        return None

    default = getattr(field, "default", None)
    if default is None:
        return "null"
    if isinstance(default, str):
        return json.dumps(default, ensure_ascii=False)
    return repr(default)


def describe_model_contract(model_cls: type[BaseModel]) -> str:
    lines: list[str] = [model_cls.__name__]

    doc = inspect.getdoc(model_cls)
    if doc:
        first_line = doc.splitlines()[0].strip()
        if first_line and first_line != model_cls.__name__:
            lines.append(f"Description: {first_line}")

    extra_mode = getattr(model_cls, "model_config", {}).get("extra")
    if extra_mode == "forbid":
        lines.append("Additional properties: not allowed.")

    required_lines: list[str] = []
    optional_lines: list[str] = []

    for field_name, field in model_cls.model_fields.items():
        annotation_text = _format_annotation(getattr(field, "annotation", Any))
        line = f"- {field_name}: {annotation_text}"

        default_text = _format_default(field)
        if default_text is not None and not field.is_required():
            line += f" (default={default_text})"

        description = getattr(field, "description", None)
        if description:
            line += f" - {description.strip()}"

        if field.is_required():
            required_lines.append(line)
        else:
            optional_lines.append(line)

    if required_lines:
        lines.append("Required:")
        lines.extend(required_lines)

    if optional_lines:
        lines.append("Optional:")
        lines.extend(optional_lines)

    return "\n".join(lines)


def model_schema_json(model_cls: type[BaseModel]) -> str:
    return json.dumps(model_cls.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True)


def validate_model_payload(model_cls: type[BaseModel], payload: Any) -> BaseModel:
    if isinstance(payload, model_cls):
        return payload
    if isinstance(payload, str):
        return model_cls.model_validate_json(payload)
    return model_cls.model_validate(payload)


def normalize_handoff_payload(payload: Any) -> Any:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:
            return payload
    return payload
