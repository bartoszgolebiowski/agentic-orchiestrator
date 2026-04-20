from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MarkdownExtractionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(description="Absolute or workspace-relative path to the markdown file.")


class TaskModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    description: str = ""
    labels: list[str] = Field(default_factory=list)


class StoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    description: str = ""
    tasks: list[TaskModel] = Field(default_factory=list)


class EpicModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    summary: str = ""
    stories: list[StoryModel] = Field(default_factory=list)


class DocumentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str
    title: str
    summary: str = ""
    labels: list[str] = Field(default_factory=list)
    epics: list[EpicModel] = Field(default_factory=list)
