"""Evidence and tool-state checks for final-readiness decisions."""

from .final_readiness_artifacts import summarize_required_artifacts
from .final_readiness_context import compact_net_negative
from .final_readiness_context import compact_summary_quality_low
from .final_readiness_context import context_pressure_without_reduction
from .final_readiness_context import context_pressure_compaction_failed
from .final_readiness_context import provider_usage_unavailable
from .final_readiness_context import replacement_ledger_disabled_under_pressure
from .final_readiness_context import tier3_summary_without_delta

UNRESOLVED_TODO_STATUS = {"pending", "in_progress"}


def readiness_reasons(task_state, workspace_root=None):
    summaries = task_state.evidence_summaries or {}
    reasons = []
    required_artifacts = summarize_required_artifacts(task_state, workspace_root)
    if required_artifacts.get("missing_paths"):
        summaries = dict(task_state.evidence_summaries or {})
        summaries["required_artifact_summary"] = required_artifacts
        task_state.evidence_summaries = summaries
        reasons.append("missing_required_artifact")
    verification = dict(summaries.get("verification_signal", {}) or {})
    if task_state.changed_paths and verification.get("state") != "passed":
        reasons.append("changed_paths_without_verification")
    if verification.get("state") == "failed":
        reasons.append("failed_verification")
    if _has_partial_success_workspace_change(task_state):
        reasons.append("partial_success_workspace_changed")
    governance = dict(summaries.get("governance_summary", {}) or {})
    if int(governance.get("deny_count", 0) or 0):
        reasons.append("governance_denial")
    if _has_unresolved_high_priority_todo(task_state):
        reasons.append("unresolved_high_priority_todo")
    context = dict(summaries.get("context_budget_summary", {}) or {})
    if context_pressure_without_reduction(context):
        reasons.append("context_pressure_without_reduction")
    if tier3_summary_without_delta(context):
        reasons.append("tier3_summary_without_delta")
    if replacement_ledger_disabled_under_pressure(context):
        reasons.append("replacement_ledger_disabled_under_pressure")
    if provider_usage_unavailable(context):
        reasons.append("provider_real_token_usage_unavailable")
    if compact_net_negative(context):
        reasons.append("compact_net_negative")
    if compact_summary_quality_low(context):
        reasons.append("compact_summary_quality_low")
    if context_pressure_compaction_failed(context):
        reasons.append("context_pressure_compaction_failed")
    return reasons


def _has_unresolved_high_priority_todo(task_state):
    latest = {}
    for change in task_state.todo_changes or []:
        todo = dict(change.get("todo", {}) or {})
        todo_id = str(todo.get("id", ""))
        if todo_id:
            latest[todo_id] = todo
    return any(
        todo.get("priority") == "high" and todo.get("status") in UNRESOLVED_TODO_STATUS
        for todo in latest.values()
    )


def _has_partial_success_workspace_change(task_state):
    return any(
        item.get("status") == "partial_success"
        and item.get("workspace_changed") is True
        for item in task_state.runtime_reminders or []
    )
