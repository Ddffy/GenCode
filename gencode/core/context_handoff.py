"""LLM-backed handoff summary helpers for context compaction."""

from __future__ import annotations

from dataclasses import dataclass

from ..providers.base import complete_model


HANDOFF_PROMPT_TEMPLATE = """\
You are a context compactor for a coding agent. Your job is to produce a structured
handoff summary so the agent can continue its task without re-reading the full history.

Below is the conversation delta and optionally a prior summary. Produce a handoff
summary in this format exactly:

## Goal
<one sentence: what the user wants to accomplish>

## Constraints
- <user-stated constraint>

## Files Read
- <path>

## Files Modified
- <path>

## Key Decisions
- <decision and reason>

## Blockers
- <blocker or open question>

## Next Steps
- <what the agent should do next>

Rules:
- Keep each section concise. Omit empty optional sections.
- Goal and Next Steps are required.
- Preserve exact file paths, variable names, error messages, and test names.
- Do not add commentary outside these sections.
- If there is a prior summary, merge its content rather than duplicating.

{prior_summary_block}

## Conversation Delta

{delta_text}
"""


@dataclass(frozen=True)
class HandoffSummary:
    """Structured summary produced by the LLM handoff compactor."""

    goal: str
    constraints: tuple[str, ...] = ()
    files_read: tuple[str, ...] = ()
    files_modified: tuple[str, ...] = ()
    key_decisions: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()
    raw_text: str = ""


class HandoffParser:
    """Parses structured markdown LLM output into a HandoffSummary."""

    FIELD_BY_HEADER = {
        "goal": "goal",
        "constraints": "constraints",
        "files read": "files_read",
        "files modified": "files_modified",
        "key decisions": "key_decisions",
        "blockers": "blockers",
        "next steps": "next_steps",
    }

    def parse(self, raw_text: str) -> HandoffSummary:
        raw = str(raw_text or "")
        sections = self._sections(raw)
        goal = self._paragraph(sections.get("goal", ""))
        return HandoffSummary(
            goal=goal,
            constraints=self._bullets(sections.get("constraints", "")),
            files_read=self._bullets(sections.get("files_read", "")),
            files_modified=self._bullets(sections.get("files_modified", "")),
            key_decisions=self._bullets(sections.get("key_decisions", "")),
            blockers=self._bullets(sections.get("blockers", "")),
            next_steps=self._bullets(sections.get("next_steps", "")),
            raw_text=raw,
        )

    def _sections(self, raw_text: str) -> dict[str, str]:
        sections = {}
        current = None
        lines = []
        for line in raw_text.splitlines():
            if line.startswith("## "):
                if current:
                    sections[current] = "\n".join(lines).strip()
                header = line[3:].strip().lower()
                current = self.FIELD_BY_HEADER.get(header)
                lines = []
            elif current:
                lines.append(line)
        if current:
            sections[current] = "\n".join(lines).strip()
        return sections

    @staticmethod
    def _paragraph(text: str) -> str:
        for line in str(text or "").splitlines():
            stripped = line.strip()
            if stripped:
                return stripped.removeprefix("- ").strip()
        return ""

    @staticmethod
    def _bullets(text: str) -> tuple[str, ...]:
        items = []
        for line in str(text or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                value = stripped[2:].strip()
                if value:
                    items.append(value)
        return tuple(items)


class HandoffAdapter:
    """Generates a handoff summary through the normalized model boundary."""

    def __init__(self, model_client, max_summary_tokens=1024):
        self.model_client = model_client
        self.max_summary_tokens = int(max_summary_tokens)
        self.parser = HandoffParser()
        self.last_usage = None

    def generate(self, delta_text: str, prior_summary_text: str = "") -> HandoffSummary | None:
        prior_block = ""
        if str(prior_summary_text or "").strip():
            prior_block = "## Prior Summary (merge into your output)\n\n" + str(prior_summary_text).strip()
        prompt = HANDOFF_PROMPT_TEMPLATE.format(
            prior_summary_block=prior_block,
            delta_text=str(delta_text or ""),
        )
        try:
            result = complete_model(self.model_client, prompt, self.max_summary_tokens)
        except Exception:
            self.last_usage = None
            return None

        self.last_usage = self._usage(result.metadata)
        summary = self.parser.parse(result.text)
        if not summary.goal or not summary.next_steps:
            return None
        return summary

    @staticmethod
    def _usage(metadata):
        meta = dict(metadata or {})
        input_tokens = _optional_int(meta.get("input_tokens"))
        output_tokens = _optional_int(meta.get("output_tokens"))
        total_tokens = _optional_int(meta.get("total_tokens")) or input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": _optional_int(meta.get("cached_tokens")),
            "model": str(meta.get("provider_model", "")),
            "provider": str(meta.get("provider_protocol", "")),
        }


def render_handoff_summary(summary: HandoffSummary) -> str:
    """Render a handoff summary as session compact_summary text."""

    lines = ["## Goal", str(summary.goal).strip()]
    _append_section(lines, "Constraints", summary.constraints)
    _append_section(lines, "Files Read", summary.files_read)
    _append_section(lines, "Files Modified", summary.files_modified)
    _append_section(lines, "Key Decisions", summary.key_decisions)
    _append_section(lines, "Blockers", summary.blockers)
    _append_section(lines, "Next Steps", summary.next_steps)
    return "\n".join(lines).strip()


def render_delta_for_handoff(delta_items, *, max_chars=20_000):
    """Render compact delta items as bounded input for the handoff compactor."""

    parts = []
    for item in delta_items:
        role = str(item.get("role", ""))
        content = str(item.get("content", ""))
        kind = str(item.get("kind", ""))
        label = str(item.get("name") or item.get("tool_name") or kind or role or "event")
        if role == "tool":
            snippet = _truncate(content, 3_000)
            parts.append(f"[Tool:{label}]: {snippet}")
        elif role == "user":
            parts.append(f"[User]: {_truncate(content, 2_000)}")
        elif role == "assistant":
            parts.append(f"[Assistant]: {_truncate(content, 2_000)}")
        elif kind == "compact_summary":
            parts.append(f"[Prior Summary]: {_truncate(content, 2_000)}")
    text = "\n\n".join(part for part in parts if part.strip())
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _append_section(lines, title, items):
    if not items:
        return
    lines.extend(["", f"## {title}"])
    lines.extend(f"- {item}" for item in items)


def _truncate(text, limit):
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... ({len(text)} chars total, truncated)"


def _optional_int(value):
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
