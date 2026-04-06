from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


_IDENTIFIER_RE = re.compile(r"[^0-9A-Za-z_]+")


def normalize_identifier(value: str) -> str:
    normalized = _IDENTIFIER_RE.sub("_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "mcp"


class McpBaseConnectionConfig(BaseModel):
    startup_timeout_seconds: float = Field(default=30.0, gt=0)
    request_timeout_seconds: float = Field(default=60.0, gt=0)


class McpStdioConnectionConfig(McpBaseConnectionConfig):
    transport: Literal["stdio"] = "stdio"
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: Path | None = None
    encoding: str = "utf-8"
    encoding_error_handler: Literal["strict", "ignore", "replace"] = "strict"

    def resolved_cwd(self, base_dir: Path | None = None) -> Path | None:
        if self.cwd is None:
            return None
        if self.cwd.is_absolute() or base_dir is None:
            return self.cwd
        return (base_dir / self.cwd).resolve()


class McpHttpConnectionConfig(McpBaseConnectionConfig):
    transport: Literal["http"] = "http"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    terminate_on_close: bool = True


McpConnectionConfig = Annotated[
    McpStdioConnectionConfig | McpHttpConnectionConfig,
    Field(discriminator="transport"),
]


class McpServerConfig(BaseModel):
    id: str
    description: str
    connection: McpConnectionConfig
    tool_prefix: str | None = None
    include_tools: list[str] = Field(default_factory=list)
    exclude_tools: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_config(self) -> "McpServerConfig":
        if not self.id.strip():
            raise ValueError("MCP server id cannot be empty")
        if self.tool_prefix is not None and not self.tool_prefix.strip():
            raise ValueError(f"MCP server '{self.id}' tool_prefix cannot be empty")
        overlap = set(self.include_tools) & set(self.exclude_tools)
        if overlap:
            raise ValueError(
                f"MCP server '{self.id}' cannot both include and exclude the same tools: {sorted(overlap)}"
            )
        return self

    @property
    def exposed_tool_prefix(self) -> str:
        return normalize_identifier(self.tool_prefix or f"mcp_{self.id}")

    def exposed_tool_name(self, remote_tool_name: str) -> str:
        return f"{self.exposed_tool_prefix}__{normalize_identifier(remote_tool_name)}"
