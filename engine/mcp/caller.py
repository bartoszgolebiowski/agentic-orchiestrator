from __future__ import annotations

import json
from typing import Any

from mcp import ClientSession
from mcp.types import CallToolResult


def _format_content_block(content_block: Any) -> str:
    if getattr(content_block, "type", None) == "text":
        return getattr(content_block, "text", "")

    if hasattr(content_block, "model_dump"):
        try:
            payload = content_block.model_dump(mode="json")
        except TypeError:
            payload = content_block.model_dump()
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    return str(content_block)


def serialize_error_response(message: str) -> str:
    return json.dumps({"status": "error", "hint": message}, ensure_ascii=False, indent=2)


def format_error_hint(exc: Exception) -> str:
    if isinstance(exc, KeyError) and exc.args:
        message = str(exc.args[0]).strip()
    else:
        message = str(exc).strip()
    return message or type(exc).__name__


def serialize_call_tool_result(result: CallToolResult) -> str:
    parts: list[str] = []

    if result.structuredContent is not None:
        parts.append(json.dumps(result.structuredContent, ensure_ascii=False, indent=2, sort_keys=True))

    for content_block in result.content:
        content_text = _format_content_block(content_block).strip()
        if content_text:
            parts.append(content_text)

    result_text = "\n".join(parts).strip()
    if result.isError:
        return serialize_error_response(result_text or "MCP tool call failed")
    return result_text


class McpToolCaller:
    async def call(self, session: ClientSession, remote_name: str, arguments: dict[str, Any]) -> str:
        result = await session.call_tool(remote_name, arguments=arguments)
        return serialize_call_tool_result(result)
