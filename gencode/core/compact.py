"""Session compaction boundary."""

from dataclasses import asdict, dataclass

from .context_handoff import HandoffAdapter, render_delta_for_handoff, render_handoff_summary
from .compact_summary import summarize_compact_items
from .context_usage import estimate_tokens
from .workspace import now


@dataclass(frozen=True)
class CompactPlan:
    trigger: str
    keep_recent_turns: int
    previous_boundary_event_id: str | None
    prior_summary_event_id: str | None
    delta_event_ids: tuple[str, ...]
    protected_event_ids: tuple[str, ...]
    no_op_reason: str | None

    def to_dict(self):
        return asdict(self)


class CompactManager:
    def __init__(self, agent):
        self.agent = agent

    def plan(self, trigger="manual", keep_recent_turns=2):
        selected = self._select(keep_recent_turns)
        metadata = dict(self.agent.session.get("context_summary", {}) or {})
        prior = selected["prior_summary"]
        no_op_reason = None
        if not selected["delta_items"]:
            no_op_reason = (
                "protected_recent_turns_cover_history"
                if selected["compactable_items"]
                else "no_events_before_protected_recent_turns"
            )
        return CompactPlan(
            trigger=str(trigger),
            keep_recent_turns=int(keep_recent_turns),
            previous_boundary_event_id=metadata.get("last_included_event_id") or None,
            prior_summary_event_id=str((prior or {}).get("event_id", "")) or None,
            delta_event_ids=tuple(self._event_key(item, index) for index, item in selected["delta_items"]),
            protected_event_ids=tuple(self._event_key(item, index) for index, item in selected["protected_items"]),
            no_op_reason=no_op_reason,
        )

    def compact(self, trigger="manual", keep_recent_turns=2, summary_mode="deterministic"):
        plan = self.plan(trigger=trigger, keep_recent_turns=keep_recent_turns)
        history = list(self.agent.session.get("history", []))
        selected = self._select(keep_recent_turns)
        if self.agent.current_task_state:
            self.agent.emit_trace(
                self.agent.current_task_state,
                "compaction_started",
                {"trigger": trigger, "pre_tokens": self._tokens(history), "plan": plan.to_dict()},
            )
        if not plan.delta_event_ids:
            summary = self._summary(
                trigger, history, history, "", plan, summary_called=False, summary_mode=summary_mode
            )
            self.agent.session_event_bus.emit("compaction_created", summary)
            if self.agent.current_task_state:
                self.agent.emit_trace(self.agent.current_task_state, "compaction_finished", summary)
            return summary

        delta_items = [item for _, item in selected["delta_items"]]
        kept_items = [item for _, item in selected["protected_items"]]
        prior_text = str((selected["prior_summary"] or {}).get("content", "")).strip()
        summary_text, summary_mode, compact_call_usage = self._compact_summary_text(
            delta_items, prior_text, summary_mode
        )
        summary_item = self.agent.turn_history.enrich(
            {
                "role": "system",
                "kind": "compact_summary",
                "content": summary_text,
                "created_at": now(),
                "source": "compact",
                "turn_id": str((selected["prior_summary"] or {}).get("turn_id", "compact_summary")),
            }
        )
        self.agent.session["history"] = [summary_item, *kept_items]
        self.agent.session["context_summary"] = self._context_summary(plan, summary_item, selected)
        summary = self._summary(
            trigger,
            history,
            self.agent.session["history"],
            summary_text,
            plan,
            summary_mode=summary_mode,
            compact_call_usage=compact_call_usage,
        )
        self.agent.session.setdefault("compactions", []).append(self._persistent_summary(summary))
        self.agent.session_path = self.agent.session_store.save(self.agent.session)
        self.agent.session_event_bus.emit("compaction_created", summary)
        if self.agent.current_task_state:
            self.agent.emit_trace(self.agent.current_task_state, "compaction_finished", summary)
        return summary

    @staticmethod
    def _group(history):
        groups = []
        by_id = {}
        for item in history:
            turn_id = str(item.get("turn_id") or "legacy")
            if turn_id not in by_id:
                by_id[turn_id] = []
                groups.append((turn_id, by_id[turn_id]))
            by_id[turn_id].append(item)
        return groups

    def _select(self, keep_recent_turns):
        keep_recent_turns = int(keep_recent_turns)
        history = list(self.agent.session.get("history", []))
        groups = self._group(history)
        protected_turns = groups[-keep_recent_turns:] if keep_recent_turns > 0 else []
        protected_ids = {id(item) for _, items in protected_turns for item in items}
        compactable = [(index, item) for index, item in enumerate(history) if id(item) not in protected_ids]
        protected = [(index, item) for index, item in enumerate(history) if id(item) in protected_ids]
        prior_summary = self._prior_summary(history)
        boundary = str((self.agent.session.get("context_summary", {}) or {}).get("last_included_event_id", ""))
        boundary_seen = not boundary or not any(self._event_key(item, index) == boundary for index, item in compactable)
        delta = []
        for index, item in compactable:
            if item.get("kind") == "compact_summary":
                continue
            event_key = self._event_key(item, index)
            if not boundary_seen:
                boundary_seen = event_key == boundary
                continue
            delta.append((index, item))
        return {"compactable_items": compactable, "protected_items": protected, "delta_items": delta, "prior_summary": prior_summary}

    def _prior_summary(self, history):
        metadata = dict(self.agent.session.get("context_summary", {}) or {})
        summary_event_id = str(metadata.get("summary_event_id", ""))
        summaries = [item for item in history if item.get("kind") == "compact_summary"]
        for item in summaries:
            if summary_event_id and item.get("event_id") == summary_event_id:
                return item
        return summaries[-1] if summaries else None

    def _context_summary(self, plan, summary_item, selected):
        previous = dict(self.agent.session.get("context_summary", {}) or {})
        last_index, last_item = selected["delta_items"][-1]
        return {
            "summary_event_id": str(summary_item.get("event_id", "")),
            "last_included_event_id": self._event_key(last_item, last_index),
            "protected_recent_turns": plan.keep_recent_turns,
            "source_event_count": int(previous.get("source_event_count", 0) or 0) + len(selected["delta_items"]),
            "updated_at": now(),
        }

    def _compact_summary_text(self, delta_items, prior_text, summary_mode):
        if summary_mode != "llm":
            return summarize_compact_items(delta_items, prior_text=prior_text), "deterministic", None
        adapter = HandoffAdapter(self.agent.model_client)
        handoff = adapter.generate(render_delta_for_handoff(delta_items), prior_text)
        if handoff is None:
            return summarize_compact_items(delta_items, prior_text=prior_text), "deterministic_fallback", adapter.last_usage
        return render_handoff_summary(handoff), "llm", adapter.last_usage

    def _summary(
        self,
        trigger,
        before,
        after,
        summary_text,
        plan,
        summary_called=True,
        summary_mode="deterministic",
        compact_call_usage=None,
    ):
        pre_chars = sum(len(str(item.get("content", ""))) for item in before)
        post_chars = sum(len(str(item.get("content", ""))) for item in after)
        context_summary = dict(self.agent.session.get("context_summary", {}) or {})
        return {
            "trigger": str(trigger),
            "created_at": now(),
            "pre_tokens": estimate_tokens(pre_chars),
            "post_tokens": estimate_tokens(post_chars),
            "pre_items": len(before),
            "post_items": len(after),
            "summary_chars": len(summary_text),
            "summary_called": bool(summary_called),
            "summary_mode": str(summary_mode),
            "summary_has_next_steps": "## Next Steps" in summary_text,
            "summary_has_file_references": "## Files Read" in summary_text or "## Files Modified" in summary_text,
            "compact_call_usage": compact_call_usage,
            "no_op_reason": plan.no_op_reason,
            "summary_event_id": context_summary.get("summary_event_id", plan.prior_summary_event_id),
            "last_included_event_id": context_summary.get("last_included_event_id", plan.previous_boundary_event_id),
            "protected_recent_turns": plan.keep_recent_turns,
            "source_event_count": int(context_summary.get("source_event_count", 0) or 0),
            "delta_event_count": len(plan.delta_event_ids),
            "protected_event_count": len(plan.protected_event_ids),
            "plan": plan.to_dict(),
        }

    @staticmethod
    def _tokens(history):
        return estimate_tokens(sum(len(str(item.get("content", ""))) for item in history))

    @staticmethod
    def _event_key(item, index):
        return str(item.get("event_id") or f"legacy_{index:06d}")

    @staticmethod
    def _persistent_summary(summary):
        persisted = dict(summary)
        persisted.pop("compact_call_usage", None)
        return persisted
