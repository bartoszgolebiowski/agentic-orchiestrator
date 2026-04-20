from __future__ import annotations

from typing import Any

from engine.config.models import ToolDefinition
from engine.tools.entities import ToolEntity
from engine.tools.models import ToolSpec


class ToolEntityFactory:
    @staticmethod
    def from_definition(tool_id: str, definition: ToolDefinition, call_fn: Any) -> ToolEntity:
        parameter_map = {
            name: {
                "type": param.type,
                "description": param.description,
                "required": param.required,
            }
            for name, param in definition.parameters.items()
        }
        return ToolEntity(
            id=tool_id,
            description=definition.description,
            call_fn=call_fn,
            spec=ToolSpec(id=tool_id, description=definition.description, parameters=parameter_map),
        )

    @staticmethod
    def to_openai_spec(entity: ToolEntity) -> dict[str, Any]:
        required = [
            name for name, param in entity.spec.parameters.items()
            if bool(param.get("required", True))
        ]
        properties = {
            name: {
                "type": param.get("type", "string"),
                "description": param.get("description", ""),
            }
            for name, param in entity.spec.parameters.items()
        }
        return {
            "type": "function",
            "function": {
                "name": entity.id,
                "description": entity.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
