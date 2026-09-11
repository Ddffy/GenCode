"""Read-only prompt context report construction."""

from __future__ import annotations

from dataclasses import dataclass

from ..features import skills as skillslib
from .context_sections import CURRENT_REQUEST_SECTION, SECTION_ORDER
from .context_usage import ContextUsageAnalyzer

RELEVANT_MEMORY_LIMIT = 3


@dataclass(frozen=True)
class ContextReportBuilder:
    agent: object
    total_budget: int
    reduction_order: tuple[str, ...]

    def build(self, prompt, rendered, budgets, reduction_log, selected_notes, user_message, section_texts):
        section_metadata = {}
        for section in SECTION_ORDER[:-1]:
            section_metadata[section] = {
                "raw_chars": rendered[section].raw_chars,
                "budget_chars": int(budgets.get(section, 0)),
                "rendered_chars": rendered[section].rendered_chars,
            }
        section_metadata[CURRENT_REQUEST_SECTION] = {
            "raw_chars": len(section_texts[CURRENT_REQUEST_SECTION]),
            "budget_chars": None,
            "rendered_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
        }
        return {
            "prompt_chars": len(prompt),
            "prompt_budget_chars": int(self.total_budget),
            "prompt_over_budget": len(prompt) > int(self.total_budget),
            "section_order": list(SECTION_ORDER),
            "section_budgets": {
                section: (None if section == CURRENT_REQUEST_SECTION else int(budgets.get(section, 0)))
                for section in SECTION_ORDER
            },
            "sections": section_metadata,
            "budget_reductions": list(reduction_log),
            "reduction_order": list(self.reduction_order),
            "relevant_memory": self._relevant_memory_metadata(rendered, selected_notes),
            "history": self._history_metadata(rendered),
            "skills": self._skills_metadata(),
            "current_request": {
                "text": user_message,
                "raw_chars": len(user_message),
                "rendered_chars": len(user_message),
                "section_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
            },
            "context_usage": ContextUsageAnalyzer(self.agent).analyze(rendered),
        }

    def _relevant_memory_metadata(self, rendered, selected_notes):
        relevant = rendered["relevant_memory"]
        details = relevant.details or {}
        return {
            "limit": RELEVANT_MEMORY_LIMIT,
            "selected_count": len(selected_notes),
            "selected_notes": [note["text"] for note in selected_notes],
            "selected_sources": [str(note.get("source", "")).strip() for note in selected_notes],
            "selected_kinds": [str(note.get("kind", "episodic")).strip() or "episodic" for note in selected_notes],
            "selected_durable_count": sum(
                1 for note in selected_notes if (str(note.get("kind", "episodic")).strip() or "episodic") == "durable"
            ),
            "raw_chars": relevant.raw_chars,
            "rendered_chars": relevant.rendered_chars,
            "rendered_notes": list(details.get("rendered_notes", [])),
            "rendered_count": int(details.get("rendered_count", 0)),
        }

    def _history_metadata(self, rendered):
        history = rendered["history"]
        details = history.details or {}
        return {
            "raw_chars": history.raw_chars,
            "rendered_chars": history.rendered_chars,
            "older_entries_count": int(details.get("older_entries_count", 0)),
            "recent_window": int(details.get("recent_window", 0)),
            "old_turn_line_limit": int(details.get("old_turn_line_limit", 0)),
            "collapsed_duplicate_reads": int(details.get("collapsed_duplicate_reads", 0)),
            "reused_file_summary_count": int(details.get("reused_file_summary_count", 0)),
            "summarized_tool_count": int(details.get("summarized_tool_count", 0)),
            "rendered_turns": int(details.get("rendered_turns", 0)),
            "microcompact_artifact_refs": list(details.get("microcompact_artifact_refs", [])),
            "microcompact_saved_chars": int(details.get("microcompact_saved_chars", 0)),
            "replacement_cache_hits": int(details.get("replacement_cache_hits", 0)),
            "replacement_records_created": int(details.get("replacement_records_created", 0)),
            "replacement_saved_chars": int(details.get("replacement_saved_chars", 0)),
            "proposed_replacements": list(details.get("proposed_replacements", [])),
        }

    def _skills_metadata(self):
        skills = getattr(self.agent, "skills", {})
        items = [skill.metadata() for skill in skillslib.list_skills(skills, user_invocable_only=False)]
        return {
            "available_count": len(items),
            "user_invocable_count": sum(1 for item in items if item["user_invocable"]),
            "items": items,
        }
