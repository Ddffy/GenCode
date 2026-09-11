"""Context build facade for prompt pressure, compaction, and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field

CHECKPOINT_NONE_STATUS = "no-checkpoint"


@dataclass(frozen=True)
class ContextSnapshot:
    request: str
    prefix: str
    session: dict
    memory: object
    selected_skills: tuple
    last_completion_metadata: dict | None
    model_context_window: int
    prefix_refresh: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ContextBuildResult:
    prompt: str
    metadata: dict
    should_compact: bool
    compact_trigger: str | None


class ContextOrchestrator:
    version = "local-v1"

    def __init__(self, agent):
        self.agent = agent

    def snapshot(self, request, prefix_refresh=None):
        client = self.agent.model_client
        return ContextSnapshot(
            request=str(request),
            prefix=str(self.agent.prefix),
            session=self.agent.session,
            memory=self.agent.memory,
            selected_skills=tuple(getattr(self.agent, "skills", {}).values()),
            last_completion_metadata=getattr(self.agent, "last_completion_metadata", None),
            model_context_window=int(getattr(client, "context_window", 0) or 0),
            prefix_refresh=dict(prefix_refresh or {}),
        )

    def build(self, snapshot):
        prompt, metadata = self.agent.context_manager.build(snapshot.request)
        plan = None
        summary = None
        should_compact = False
        skip_reason = ""
        compact_metrics = {}
        compact_trigger, summary_mode, skip_reason = self._compact_request(metadata, snapshot)
        if compact_trigger and len(snapshot.session.get("history", [])) <= 4:
            skip_reason = "history_too_short_for_auto_compaction"
            metadata["auto_compacted"] = False
            metadata["auto_compaction_skip_reason"] = skip_reason
        elif compact_trigger:
            pre_compact_estimated_tokens = int(
                (metadata.get("context_usage", {}) or {}).get("total_estimated_tokens", 0) or 0
            )
            plan = self.agent.compact_manager.plan(trigger=compact_trigger)
            summary = self.agent.compact_history(
                trigger=plan.trigger,
                keep_recent_turns=plan.keep_recent_turns,
                summary_mode=summary_mode,
            )
            should_compact = bool(summary.get("summary_called", True))
            if should_compact:
                prompt, metadata = self.agent.context_manager.build(snapshot.request)
            post_compact_estimated_tokens = int(
                (metadata.get("context_usage", {}) or {}).get("total_estimated_tokens", 0) or 0
            )
            compact_metrics = {
                "pre_compact_estimated_tokens": pre_compact_estimated_tokens,
                "post_compact_estimated_tokens": post_compact_estimated_tokens,
            }
            metadata.update(
                {
                    "auto_compacted": should_compact,
                    "auto_compaction_plan": plan.to_dict(),
                    "auto_compaction_summary": summary,
                }
            )
        self._attach_metadata(metadata, snapshot, plan, summary, should_compact, skip_reason, compact_metrics)
        self._emit_decision(metadata)
        self._emit_usage(metadata)
        return ContextBuildResult(
            prompt=prompt,
            metadata=metadata,
            should_compact=should_compact,
            compact_trigger=(plan.trigger if plan else None),
        )

    def _compact_request(self, metadata, snapshot):
        if metadata.get("prompt_over_budget"):
            return "auto_prompt_over_budget", "deterministic", ""
        usage = dict(metadata.get("context_usage", {}) or {})
        if usage.get("pressure_tier") != "tier3_summary":
            return None, "deterministic", ""
        delta_count = self._count_delta_events(
            snapshot.session.get("history", []),
            (snapshot.session.get("context_summary") or {}).get("last_included_event_id"),
        )
        if delta_count < 4:
            return None, "deterministic", "delta_too_small_for_tier3_compaction"
        return "auto_pressure_compact", "llm", ""

    @staticmethod
    def _count_delta_events(history, last_boundary_event_id):
        if not last_boundary_event_id:
            return len(history)
        for index, item in enumerate(history):
            if item.get("event_id") == last_boundary_event_id:
                return len(history) - index - 1
        return len(history)

    def _attach_metadata(self, metadata, snapshot, plan, summary, should_compact, skip_reason, compact_metrics=None):
        agent = self.agent
        refresh = snapshot.prefix_refresh
        metadata.update(
            {
                "prefix_chars": len(agent.prefix),
                "workspace_chars": len(agent.workspace.text()),
                "memory_chars": len(agent.memory_text()),
                "history_chars": len(agent.history_text()),
                "request_chars": len(snapshot.request),
                "tool_count": len(agent.tools),
                "workspace_docs": len(agent.workspace.project_docs),
                "recent_commits": len(agent.workspace.recent_commits),
                "prefix_hash": agent.prefix_state.hash,
                "prompt_cache_key": agent.prefix_state.hash,
                "workspace_fingerprint": agent.prefix_state.workspace_fingerprint,
                "tool_signature": agent.prefix_state.tool_signature,
                "workspace_changed": refresh.get("workspace_changed", False),
                "prefix_changed": refresh.get("prefix_changed", False),
                "prompt_cache_supported": bool(
                    getattr(agent.model_client, "supports_prompt_cache", False)
                ),
                "resume_status": agent.resume_state.get("status", CHECKPOINT_NONE_STATUS),
                "stale_summary_invalidations": int(
                    agent.resume_state.get("stale_summary_invalidations", 0)
                ),
                "stale_paths": list(agent.resume_state.get("stale_paths", [])),
                "runtime_identity_mismatch_fields": list(
                    agent.resume_state.get("runtime_identity_mismatch_fields", [])
                ),
            }
        )
        metadata.update(agent.detected_secret_env_summary())
        metadata["context_orchestrator"] = self._orchestrator_metadata(
            metadata, plan, summary, should_compact, skip_reason, compact_metrics
        )

    def _orchestrator_metadata(self, metadata, plan, summary, should_compact, skip_reason, compact_metrics=None):
        usage = dict(metadata.get("context_usage", {}) or {})
        history = dict(metadata.get("history", {}) or {})
        summary = dict(summary or {})
        payload = {
            "version": self.version,
            "pressure_tier": str(usage.get("pressure_tier", "")),
            "usage_source": str(usage.get("usage_source", "")),
            "retention_decisions": [],
            "replacement_cache_hits": int(history.get("replacement_cache_hits", 0) or 0),
            "replacement_records_created": int(history.get("replacement_records_created", 0) or 0),
            "replacement_saved_chars": int(history.get("replacement_saved_chars", 0) or 0),
            "replacement_ledger_enabled": True,
            "summary_delta_event_count": int(summary.get("delta_event_count", 0) or 0),
            "summary_called": bool(summary.get("summary_called", False)),
            "summary_mode": str(summary.get("summary_mode", "")),
            "compact_summary_has_next_steps": summary.get("summary_has_next_steps"),
            "compact_summary_has_file_references": summary.get("summary_has_file_references"),
            "compact_call_usage": summary.get("compact_call_usage"),
            "should_compact": bool(should_compact),
            "compact_trigger": plan.trigger if plan else None,
            "skip_reason": str(skip_reason or ""),
        }
        payload.update(dict(compact_metrics or {}))
        return payload

    def _emit_decision(self, metadata):
        payload = {
            "run_id": getattr(getattr(self.agent, "current_task_state", None), "run_id", ""),
            "context_orchestrator": metadata.get("context_orchestrator", {}),
            "context_usage": metadata.get("context_usage", {}),
        }
        self.agent.session_event_bus.emit("context_orchestrator_decision", payload)
        task_state = getattr(self.agent, "current_task_state", None)
        if task_state:
            self.agent.emit_trace(task_state, "context_orchestrator_decision", payload)

    def _emit_usage(self, metadata):
        self.agent.session_event_bus.emit(
            "context_usage_recorded",
            {
                "run_id": getattr(getattr(self.agent, "current_task_state", None), "run_id", ""),
                "context_usage": metadata.get("context_usage", {}),
            },
        )
