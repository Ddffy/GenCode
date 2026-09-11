"""Per-run governance evidence for tool decisions.

Tool execution records allow, warn, and deny decisions here so reports can
explain what the runtime permitted or blocked. This module summarizes decisions
but does not enforce policy itself.
"""

GOVERNANCE_SUMMARY_SCHEMA = "gencode.governance_summary.v1"


def record_governance_decision(
    agent,
    tool_name,
    args,
    *,
    decision,
    reason_code,
    decision_type,
    original_reason="",
    security_event_type="",
    effects=None,
    source="tool_executor",
):
    task_state = getattr(agent, "current_task_state", None)
    if task_state is None:
        return None
    return agent.emit_trace(
        task_state,
        "governance_decision",
        {
            "decision": str(decision),
            "decision_type": str(decision_type),
            "reason_code": str(reason_code),
            "original_reason": str(original_reason or reason_code),
            "security_event_type": str(security_event_type),
            "effects": list(effects or []),
            "tool_name": str(tool_name),
            "tool_profile": getattr(agent.active_tool_profile, "name", ""),
            "read_only": bool(getattr(agent, "read_only", False)),
            "args": args or {},
            "source": source,
        },
    )


def reduce_governance_summary(summary, event):
    summary = dict(summary or {})
    summary.setdefault("schema_version", GOVERNANCE_SUMMARY_SCHEMA)
    decision = str(event.get("decision", ""))
    reason = str(event.get("reason_code", ""))
    decision_type = str(event.get("decision_type", ""))
    key = f"{decision}_count"
    summary[key] = int(summary.get(key, 0) or 0) + 1
    for missing in ("allow_count", "deny_count", "warn_count"):
        summary.setdefault(missing, 0)
    type_counts = dict(summary.get("decision_type_counts", {}) or {})
    type_counts[decision_type] = type_counts.get(decision_type, 0) + 1
    summary["decision_type_counts"] = type_counts
    reasons = dict(summary.get("reasons", {}) or {})
    reasons[reason] = reasons.get(reason, 0) + 1
    summary["reasons"] = reasons
    if decision == "deny":
        summary["last_denied_reason"] = reason
    return summary
