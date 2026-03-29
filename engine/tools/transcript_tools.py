import json
import logging
import os
from pathlib import Path

from engine.core.models import ToolParameter
from engine.tools.registry import tool

logger = logging.getLogger(__name__)


@tool(
    tool_id="read_transcript",
    description="Reads a transcript file from the given path and returns its full text content.",
    parameters={
        "path": ToolParameter(type="string", description="Absolute or relative path to the transcript file."),
    },
)
async def read_transcript(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"ERROR: File not found: {path}"
    return p.read_text(encoding="utf-8")


@tool(
    tool_id="write_markdown_file",
    description="Writes content to a Markdown file at the given path. Creates parent directories if needed.",
    parameters={
        "path": ToolParameter(type="string", description="Path where the Markdown file should be written."),
        "content": ToolParameter(type="string", description="Full Markdown content to write."),
    },
)
async def write_markdown_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"File written: {p}"


@tool(
    tool_id="create_output_directory",
    description="Creates an output directory for a transcript analysis run. Returns the created path.",
    parameters={
        "base_dir": ToolParameter(type="string", description="Base output directory (e.g. 'output')."),
        "transcript_id": ToolParameter(type="string", description="Unique identifier for this transcript (timestamp-based)."),
    },
)
async def create_output_directory(base_dir: str, transcript_id: str) -> str:
    p = Path(base_dir) / transcript_id
    p.mkdir(parents=True, exist_ok=True)
    (p / "action-points").mkdir(exist_ok=True)
    return str(p)


@tool(
    tool_id="log_ticket_payload",
    description="Logs a ticket payload to console as a formatted JSON object. This is the ticket system integration adapter (currently console-only).",
    parameters={
        "title": ToolParameter(type="string", description="Ticket title."),
        "description": ToolParameter(type="string", description="Ticket description."),
        "acceptance_criteria": ToolParameter(type="string", description="JSON array of acceptance criteria."),
        "definition_of_done": ToolParameter(type="string", description="JSON array of definition of done items."),
        "priority": ToolParameter(type="string", description="Priority level."),
        "category": ToolParameter(type="string", description="Ticket category."),
        "risk": ToolParameter(type="string", description="Associated risk.", required=False),
        "dependencies": ToolParameter(type="string", description="JSON array of dependencies.", required=False),
        "estimate_effort": ToolParameter(type="string", description="Effort estimate.", required=False),
    },
)
async def log_ticket_payload(
    title: str,
    description: str,
    acceptance_criteria: str,
    definition_of_done: str,
    priority: str,
    category: str,
    risk: str = "",
    dependencies: str = "[]",
    estimate_effort: str = "",
) -> str:
    try:
        ac_list = json.loads(acceptance_criteria)
    except json.JSONDecodeError:
        ac_list = [acceptance_criteria]
    try:
        dod_list = json.loads(definition_of_done)
    except json.JSONDecodeError:
        dod_list = [definition_of_done]
    try:
        dep_list = json.loads(dependencies)
    except json.JSONDecodeError:
        dep_list = [dependencies] if dependencies else []

    payload = {
        "title": title,
        "description": description,
        "acceptance_criteria": ac_list,
        "definition_of_done": dod_list,
        "priority": priority,
        "category": category,
        "risk": risk,
        "dependencies": dep_list,
        "estimate_effort": estimate_effort,
    }

    logger.info(f"\n{'='*60}\n TICKET PAYLOAD\n{'='*60}")
    logger.info(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(f"{'='*60}\n")

    return f"Ticket logged: {title}"
