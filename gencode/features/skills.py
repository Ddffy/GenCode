"""Skill discovery, prompt expansion, and slash workflow execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from .skill_io import read_skill_body as _read_skill_body, read_skill_metadata as _read_skill_metadata

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
SKILL_FILE_CREATION_GUIDE = """When creating GenCode skill files at .gencode/skills/<name>/SKILL.md or skills/<name>/SKILL.md, use frontmatter:
---
name: audit
description: Audit a file
user-invocable: true
---
Audit $ARGUMENTS for risky changes."""
@dataclass(frozen=True)
class Skill:
    name: str
    description: str = ""
    prompt: str = ""
    source: str = "builtin"
    skill_root: str = ""
    when_to_use: str = ""
    context: str = "inline"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str = ""
    user_invocable: bool = True
    disable_model_invocation: bool = False
    model: str = ""
    paths: tuple[str, ...] = ()
    prompt_fn: Callable[[str], str] | None = None
    prompt_loader: Callable[[], str] | None = None
    def render(self, arguments=""):
        if self.prompt_fn:
            text = self.prompt_fn(str(arguments))
        elif self.prompt_loader:
            text = self.prompt_loader()
        else:
            text = self.prompt
        replacements = {
            "$ARGUMENTS": str(arguments),
            "${GENCODE_SKILL_DIR}": self.skill_root,
            "${CLAUDE_SKILL_DIR}": self.skill_root,
        }
        if self.argument_hint:
            replacements[f"${{{self.argument_hint}}}"] = str(arguments)
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text.strip()

    def metadata(self):
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "context": self.context,
            "allowed_tools": list(self.allowed_tools),
            "paths": list(self.paths),
            "user_invocable": self.user_invocable,
            "disable_model_invocation": self.disable_model_invocation,
            "model": self.model,
        }


def discover_skills(root, home=None):
    from .skills_bundled import bundled_skills

    skills = {skill.name: skill for skill in bundled_skills()}
    if home is None:
        try:
            home = Path.home()
        except RuntimeError:
            # A stripped CI environment may intentionally omit HOME/USERPROFILE.
            # Project and built-in skills remain usable without a user directory.
            home = Path(root)
    search_roots = [
        (Path(home) / ".gencode" / "skills", "user"),
        (Path(root) / "skills", "project"),
        (Path(root) / ".gencode" / "skills", "project"),
        # Approved durable skills live beside Wiki/Spec records.  Candidate,
        # rejected and quarantined records are filtered by load_skill_file.
        (Path(root) / ".gencode" / "knowledge" / "skills", "knowledge"),
    ]
    for directory, source in search_roots:
        for skill in load_skills_from_dir(directory, source=source):
            skills[skill.name] = skill
    return dict(sorted(skills.items()))
def load_skills_from_dir(skills_dir, source):
    skills_dir = Path(skills_dir).expanduser()
    if not skills_dir.exists():
        return []
    files = []
    for path in sorted(skills_dir.iterdir()):
        if path.is_dir() and (path / "SKILL.md").is_file():
            files.append(path / "SKILL.md")
        elif path.is_file() and path.suffix.lower() == ".md":
            files.append(path)
    return [skill for path in files if (skill := load_skill_file(path, source=source))]
def load_skill_file(path, source):
    path = Path(path)
    metadata = _read_skill_metadata(path)
    default_name = path.parent.name if path.name == "SKILL.md" else path.stem
    name = str(metadata.get("name") or default_name).strip().lstrip("/")
    if not name:
        return None
    status = _string(metadata.get("status")).strip().lower()
    if status and status != "active":
        return None
    return Skill(
        name=name,
        description=_string(metadata.get("description")),
        when_to_use=_string(metadata.get("when_to_use")),
        context=_string(metadata.get("context"), "inline") or "inline",
        allowed_tools=tuple(_list_value(metadata.get("allowed_tools"))),
        argument_hint=_string(metadata.get("arguments") or metadata.get("argument_hint")),
        user_invocable=_bool_value(metadata.get("user_invocable", True)),
        disable_model_invocation=_bool_value(metadata.get("disable_model_invocation", False)),
        model=_string(metadata.get("model")),
        paths=tuple(_list_value(metadata.get("paths"))),
        source=source,
        skill_root=str(path.parent),
        prompt="",
        prompt_loader=lambda path=path: _read_skill_body(path),
    )
def parse_frontmatter(text):
    match = FRONTMATTER_RE.match(str(text))
    if not match:
        return {}, str(text)
    metadata = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip().lower().replace("-", "_")] = _parse_value(value.strip())
    return metadata, str(text)[match.end() :]


def render_prompt_section(skills):
    visible = [skill for skill in list_skills(skills, user_invocable_only=False) if _should_show_in_prompt(skill)]
    if not visible:
        return "Available skills:\n- none"
    lines = ["Available skills:"]
    for skill in visible:
        description = skill.description or skill.when_to_use or "No description"
        suffix = f" — {skill.when_to_use}" if skill.when_to_use else ""
        lines.append(f"- /{skill.name}: {description}{suffix}")
    return "\n".join(lines)


def render_skills_list(skills):
    lines = []
    for skill in list_skills(skills):
        description = skill.description or skill.when_to_use or "No description"
        hint = f" [{skill.argument_hint}]" if skill.argument_hint else ""
        lines.append(f"/{skill.name}{hint:<10} {description} [{skill.source}]")
    return "\n".join(lines)


def list_skills(skills, user_invocable_only=True):
    items = [skills[name] for name in sorted(skills)]
    if user_invocable_only:
        items = [skill for skill in items if skill.user_invocable]
    return sorted(items, key=lambda skill: (skill.source != "builtin", skill.name))


def parse_slash_command(text):
    text = str(text).strip()
    if not text.startswith("/") or text == "/":
        return "", ""
    command, _, arguments = text[1:].partition(" ")
    return command.strip(), arguments.strip()


def _parse_value(value):
    value = value.strip()
    if value.startswith(("[", "{", '"')):
        try:
            return __import__("json").loads(value)
        except (TypeError, ValueError):
            pass
    value = value.strip("\"'")
    if value.lower() in {"true", "yes"}:
        return True
    if value.lower() in {"false", "no"}:
        return False
    if "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


def _list_value(value):
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _string(value, default=""):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _bool_value(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _should_show_in_prompt(skill):
    return skill.user_invocable or bool(skill.paths)
