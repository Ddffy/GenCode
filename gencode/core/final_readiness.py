"""Final-answer readiness gate over TaskState evidence."""

import hashlib

from .final_readiness_artifacts import (
    extract_required_artifact_paths as extract_required_artifact_paths,
    summarize_required_artifacts as summarize_required_artifacts,
)
from .final_readiness_reasons import (
    FINAL_READINESS_SUMMARY_SCHEMA,
    reason_message,
    reason_severity,
)
from .final_readiness_tools import readiness_reasons

VALID_MODES = {"off", "warn", "soft", "strict"}


def evaluate_final_readiness(task_state, mode, workspace_root=None):
    mode = str(mode or "warn")
    if mode not in VALID_MODES:
        mode = "warn"
    reasons = readiness_reasons(task_state, workspace_root=workspace_root)
    signature = _reason_signature(reasons)
    state = _state(task_state)
    reminded = set(state.get("reminded_reason_signatures", []))
    already_sent = bool(signature and signature in reminded)
    decision = "allow"
    action = "none"
    if reasons and mode == "warn":
        decision = "warn"
    elif reasons and mode == "soft":
        decision, action = ("warn", "none") if already_sent else ("remind", "runtime_notice")
        if not already_sent:
            reminded.add(signature)
    elif reasons and mode == "strict":
        decision, action = (
            ("block", "block") if any(reason_severity(reason) == "hard" for reason in reasons) else ("warn", "none")
        )
    state["reminded_reason_signatures"] = sorted(reminded)
    return {
        "mode": mode,
        "decision": decision,
        "reasons": reasons,
        "reason_signature": signature,
        "reminder_already_sent": already_sent,
        "action": action,
        "required_artifact_summary": dict(
            (task_state.evidence_summaries or {}).get("required_artifact_summary", {})
            or {}
        ),
    }


def readiness_notice(decision):
    messages = [reason_message(reason) for reason in decision.get("reasons", [])]
    text = "\n".join(f"- {message}" for message in messages) or "- Readiness warning."
    if decision.get("action") == "block":
        return f"Final answer blocked by runtime readiness gate:\n{text}"
    return (
        "Before final answer, address this runtime readiness issue:\n"
        f"{text}\nReturn final again only after addressing it or explaining why it is unavailable."
    )


def reduce_final_readiness_summary(summary, event):
    summary = dict(summary or {})
    summary.setdefault("schema_version", FINAL_READINESS_SUMMARY_SCHEMA)
    decision = str(event.get("decision", ""))
    summary[f"{decision}_count"] = int(summary.get(f"{decision}_count", 0) or 0) + 1
    for missing in ("allow_count", "warn_count", "remind_count", "block_count"):
        summary.setdefault(missing, 0)
    summary["last_decision"] = decision
    summary["last_reasons"] = list(event.get("reasons", []) or [])
    return summary


def _reason_signature(reasons):
    if not reasons:
        return ""
    return hashlib.sha256("|".join(sorted(reasons)).encode("utf-8")).hexdigest()[:16]


def _state(task_state):
    summaries = dict(task_state.evidence_summaries or {})
    state = dict(summaries.get("final_readiness_state", {}) or {})
    summaries["final_readiness_state"] = state
    task_state.evidence_summaries = summaries
    return state
