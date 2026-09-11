"""Loop transition contracts and summary reduction.

Engine records real control-flow transitions here as trace events, then report
consumers reduce them into a compact summary. These contracts describe the loop;
they do not drive it.
"""

CONTINUE_PROVIDER_RETRY = "provider_retry"
CONTINUE_PARSE_RETRY = "parse_retry"
CONTINUE_TOOL_BATCH_EXECUTED = "tool_batch_executed"
CONTINUE_PLAN_NOTICE = "plan_notice"
CONTINUE_FINAL_READINESS_NOTICE = "final_readiness_notice"
TERMINAL_FINAL_ANSWER_RETURNED = "final_answer_returned"
TERMINAL_ABORTED = "aborted"
TERMINAL_MODEL_ERROR = "model_error"
TERMINAL_STEP_LIMIT_REACHED = "step_limit_reached"
TERMINAL_RETRY_LIMIT_REACHED = "retry_limit_reached"
TERMINAL_FINAL_GATE_BLOCKED = "final_gate_blocked"
CONTINUE_KIND = "continue"
TERMINAL_KIND = "terminal"
TRANSITION_SUMMARY_SCHEMA = "gencode.transition_summary.v1"


def build_transition(*, kind, reason, attempt_index, tool_call_count=0, tool_requested_count=0, tool_executed_count=0, stop_reason=""):
    payload = {
        "kind": str(kind),
        "reason": str(reason),
        "attempt_index": int(attempt_index),
    }
    for key, value in {
        "tool_call_count": tool_call_count,
        "tool_requested_count": tool_requested_count,
        "tool_executed_count": tool_executed_count,
    }.items():
        if value:
            payload[key] = int(value)
    if stop_reason:
        payload["stop_reason"] = str(stop_reason)
    return payload


def reduce_transition_summary(summary, transition):
    summary = dict(summary or {})
    summary.setdefault("schema_version", TRANSITION_SUMMARY_SCHEMA)
    kind = str(transition.get("kind", ""))
    reason = str(transition.get("reason", ""))
    reasons = dict(summary.get("reasons", {}) or {})
    reasons[reason] = reasons.get(reason, 0) + 1
    summary["reasons"] = reasons
    summary["max_attempt_index"] = max(
        int(summary.get("max_attempt_index", 0) or 0),
        int(transition.get("attempt_index", 0) or 0),
    )
    if kind == CONTINUE_KIND:
        summary["continue_count"] = int(summary.get("continue_count", 0) or 0) + 1
        summary.setdefault("terminal_count", 0)
        summary["tool_requested_count"] = int(summary.get("tool_requested_count", 0) or 0) + int(transition.get("tool_requested_count", 0) or 0)
        summary["tool_executed_count"] = int(summary.get("tool_executed_count", 0) or 0) + int(transition.get("tool_executed_count", 0) or 0)
        return summary
    if kind == TERMINAL_KIND:
        if int(summary.get("terminal_count", 0) or 0) >= 1:
            raise ValueError("run already has a terminal transition")
        summary["terminal_count"] = 1
        summary.setdefault("continue_count", 0)
        summary["terminal_reason"] = str(transition.get("stop_reason") or transition.get("reason") or "")
        return summary
    return summary


def emit_transition(agent, task_state, *, kind, reason, tool_call_count=0, tool_requested_count=0, tool_executed_count=0, stop_reason=""):
    payload = build_transition(
        kind=kind,
        reason=reason,
        attempt_index=task_state.attempts,
        tool_call_count=tool_call_count,
        tool_requested_count=tool_requested_count,
        tool_executed_count=tool_executed_count,
        stop_reason=stop_reason,
    )
    return agent.emit_trace(task_state, "loop_transition", payload)

def emit_continue_transition(agent, task_state, reason, **evidence):
    return emit_transition(agent, task_state, kind=CONTINUE_KIND, reason=reason, **evidence)

def emit_terminal_transition(agent, task_state, reason, **evidence):
    return emit_transition(agent, task_state, kind=TERMINAL_KIND, reason=reason, **evidence)
