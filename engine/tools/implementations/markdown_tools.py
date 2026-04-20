from __future__ import annotations

import json
import re
from pathlib import Path
from textwrap import indent
from typing import Any

from engine.config.models import ToolParameter
from engine.tools.registry import tool


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
BULLET_RE = re.compile(r"^\s*(?:[-*+]|(?:\d+\.))\s+(.*\S)\s*$")
CODE_FENCE_RE = re.compile(r"^\s*```")


def _build_markdown_sections(raw_text: str, source_stem: str) -> tuple[str, list[dict[str, Any]]]:
    lines = raw_text.splitlines()
    sections: list[dict[str, Any]] = []
    title = source_stem

    heading_stack: list[str] = []
    current_heading: str | None = None
    current_level = 0
    current_path: list[str] = [source_stem]
    current_start = 1
    current_lines: list[str] = []
    current_bullets: list[str] = []
    in_code_block = False

    def flush_section(end_line: int) -> None:
        nonlocal current_lines, current_bullets
        if not current_lines and current_heading is None:
            return
        sections.append(
            {
                "heading": current_heading,
                "level": current_level,
                "path": list(current_path),
                "start_line": current_start,
                "end_line": max(current_start, end_line),
                "text": "\n".join(current_lines).strip(),
                "bullets": list(current_bullets),
            }
        )
        current_lines = []
        current_bullets = []

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()

        if CODE_FENCE_RE.match(stripped):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        if not in_code_block:
            heading_match = HEADING_RE.match(line)
            if heading_match:
                flush_section(line_number - 1)
                current_level = len(heading_match.group(1))
                current_heading = heading_match.group(2).strip()
                heading_stack = heading_stack[: max(current_level - 1, 0)] + [current_heading]
                current_path = list(heading_stack)
                current_start = line_number
                current_lines = [line]
                current_bullets = []
                if current_level == 1 and title == source_stem:
                    title = current_heading
                continue

        if not current_lines and current_heading is None:
            current_start = line_number

        current_lines.append(line)
        bullet_match = BULLET_RE.match(line)
        if bullet_match:
            current_bullets.append(bullet_match.group(1).strip())

    flush_section(len(lines))

    if not sections:
        sections = [
            {
                "heading": None,
                "level": 0,
                "path": [title],
                "start_line": 1,
                "end_line": max(1, len(lines)),
                "text": raw_text.strip(),
                "bullets": [],
            }
        ]

    return title, sections


def _build_outline(sections: list[dict[str, Any]], max_section_chars: int = 900) -> str:
    entries: list[str] = []
    for index, section in enumerate(sections, start=1):
        path = " > ".join(section["path"])
        excerpt = (section["text"] or "").strip()
        if len(excerpt) > max_section_chars:
            excerpt = excerpt[:max_section_chars].rstrip() + "..."
        entries.append(
            f"{index}. {path} | lines {section['start_line']}-{section['end_line']}\n"
            f"{indent(excerpt or '(empty section)', '   ')}"
        )
    return "\n".join(entries)


@tool(
    tool_id="read_markdown_structure",
    description="Reads markdown and returns a structured JSON representation of sections and bullets.",
    parameters={
        "source_path": ToolParameter(type="string", description="Path to the markdown file."),
        "include_raw": ToolParameter(type="boolean", description="Include raw markdown text in output.", required=False),
        "include_outline": ToolParameter(type="boolean", description="Include generated outline text in output.", required=False),
    },
)
async def read_markdown_structure(
    source_path: str,
    include_raw: bool = True,
    include_outline: bool = True,
) -> str:
    source = Path(source_path).expanduser().resolve()
    raw_text = source.read_text(encoding="utf-8")
    title, sections = _build_markdown_sections(raw_text, source.stem)

    payload: dict[str, Any] = {
        "source_path": str(source),
        "title": title,
        "line_count": len(raw_text.splitlines()),
        "sections": sections,
    }
    if include_raw:
        payload["raw_text"] = raw_text
    if include_outline:
        payload["outline"] = _build_outline(sections)

    return json.dumps(payload, ensure_ascii=False, indent=2)
