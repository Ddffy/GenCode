"""Pydantic argument models for all registered tools.

Each model is the single source of truth for a tool's argument structure,
defaults, and pure value-level constraints (type, range, non-empty).
Workspace-aware checks (path safety, file existence, patch uniqueness) still
live in validate_tool() since they require access to the agent.
"""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator


class ListFilesArgs(BaseModel):
    path: str = "."


class ReadFileArgs(BaseModel):
    path: str
    start: int = 1
    end: int = 200

    @field_validator("start")
    @classmethod
    def start_ge_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("start must be >= 1")
        return v

    @model_validator(mode="after")
    def end_ge_start(self) -> "ReadFileArgs":
        if self.end < self.start:
            raise ValueError("invalid line range")
        return self


class SearchArgs(BaseModel):
    pattern: str
    path: str = "."

    @field_validator("pattern")
    @classmethod
    def pattern_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("pattern must not be empty")
        return v


class KnowledgeReadArgs(BaseModel):
    """On-demand read of an approved Wiki page or one of its sections."""

    id: str
    kind: str = "wiki"
    section: str = ""
    max_chars: int = 12_000

    @field_validator("id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("id must not be empty")
        return v

    @field_validator("kind")
    @classmethod
    def wiki_only(cls, v: str) -> str:
        if v.strip().lower() != "wiki":
            raise ValueError("knowledge_read only reads kind=wiki")
        return v

    @field_validator("max_chars")
    @classmethod
    def max_chars_in_range(cls, v: int) -> int:
        if v < 1 or v > 24_000:
            raise ValueError("max_chars must be in [1, 24000]")
        return v


class InspectImageArgs(BaseModel):
    path: str
    question: str
    profile: str = "general"
    output_schema: str = ""

    @field_validator("path")
    @classmethod
    def path_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("path must not be empty")
        return v

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v


class RunShellArgs(BaseModel):
    command: str
    timeout: int = 20

    @field_validator("command")
    @classmethod
    def command_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("command must not be empty")
        return v

    @field_validator("timeout")
    @classmethod
    def timeout_in_range(cls, v: int) -> int:
        if v < 1 or v > 120:
            raise ValueError("timeout must be in [1, 120]")
        return v


class WriteFileArgs(BaseModel):
    path: str
    content: str


class PatchFileArgs(BaseModel):
    path: str
    old_text: str
    new_text: str

    @field_validator("old_text")
    @classmethod
    def old_text_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("old_text must not be empty")
        return v


class TodoAddArgs(BaseModel):
    content: str
    status: str = "pending"
    priority: str = "normal"
    note: str = ""

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v


class TodoUpdateArgs(BaseModel):
    model_config = ConfigDict(extra="allow")
    todo_id: str
    status: Optional[str] = None
    content: Optional[str] = None
    priority: Optional[str] = None
    note: Optional[str] = None

    @field_validator("todo_id")
    @classmethod
    def todo_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("todo_id must not be empty")
        return v


class TodoListArgs(BaseModel):
    pass


class AgentArgs(BaseModel):
    description: str
    prompt: str
    subagent_type: str = "worker"
    write_scope: Union[List[str], str, None] = None

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("description must not be empty")
        return v

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be empty")
        return v

    @field_validator("subagent_type")
    @classmethod
    def valid_subagent_type(cls, v: str) -> str:
        if v not in {"worker", "Explore"}:
            raise ValueError("subagent_type must be worker or Explore")
        return v

    @field_validator("write_scope", mode="before")
    @classmethod
    def valid_write_scope(cls, v: object) -> object:
        if v is not None and not isinstance(v, (list, str)):
            raise ValueError("write_scope must be a list of workspace paths")
        return v


class SendMessageArgs(BaseModel):
    to: str
    message: str

    @field_validator("to")
    @classmethod
    def to_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("to must not be empty")
        return v

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be empty")
        return v


class TaskStopArgs(BaseModel):
    task_id: str

    @field_validator("task_id")
    @classmethod
    def task_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("task_id must not be empty")
        return v


class EnterPlanModeArgs(BaseModel):
    topic: str
    path: Optional[str] = None

    @field_validator("topic")
    @classmethod
    def topic_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("topic must not be empty")
        return v


class ExitPlanModeArgs(BaseModel):
    pass


class AskUserArgs(BaseModel):
    question: str
    choices: Optional[List[str]] = None

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v

    @field_validator("choices", mode="before")
    @classmethod
    def choices_must_be_list(cls, v: object) -> object:
        if v is not None and not isinstance(v, list):
            raise ValueError("choices must be a list")
        return v


class RepoMapArgs(BaseModel):
    query: str = ""
    limit_chars: int = 6000

    @field_validator("limit_chars")
    @classmethod
    def limit_chars_in_range(cls, v: int) -> int:
        if v < 200 or v > 20000:
            raise ValueError("limit_chars must be between 200 and 20000")
        return v


def first_error_message(exc: "ValidationError") -> str:  # type: ignore[name-defined]
    """Extract a clean single-line message from a Pydantic ValidationError."""
    errors = exc.errors(include_url=False)
    if not errors:
        return str(exc)
    err = errors[0]
    msg = str(err.get("msg", "")).removeprefix("Value error, ")
    # For missing required fields, reproduce the old KeyError repr: "'fieldname'"
    # so callers that checked for that format continue to work.
    if err.get("type") == "missing":
        loc = err.get("loc", ())
        field = loc[-1] if loc else ""
        if field:
            return f"'{field}'"
    return msg
