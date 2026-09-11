"""Structured trace event helpers.

Runtime events are normalized here before they enter the run trace. The helper
sets stable trace, turn, phase, and span fields but leaves event-specific
semantics to the caller.
"""

from .workspace import now

PHASE_BY_EVENT = {
    "run_started": "runtime",
    "context_orchestrator_decision": "prompt",
    "prompt_built": "prompt",
    "model_requested": "model",
    "model_parsed": "parse",
    "loop_transition": "loop",
    "governance_decision": "governance",
    "final_readiness_decision": "final_gate",
    "tool_executed": "tool",
    "checkpoint_created": "checkpoint",
    "compaction_started": "compact",
    "compaction_finished": "compact",
    "runtime_identity_mismatch": "runtime",
    "run_finished": "runtime",
}


def build_runtime_event(runtime, task_state, event, payload):
    payload = dict(payload or {})
    payload["event"] = str(event)
    payload["created_at"] = now()
    payload.setdefault("trace_id", task_state.run_id)
    payload.setdefault("turn_id", task_state.task_id)
    payload.setdefault("phase", PHASE_BY_EVENT.get(str(event), "runtime"))
    payload.setdefault("status", _status_for(event, payload))
    payload.setdefault("duration_ms", int(payload.get("duration_ms", 0) or 0))
    payload.setdefault("input_chars", int(payload.get("input_chars", 0) or 0))
    payload.setdefault("output_chars", int(payload.get("output_chars", 0) or 0))
    payload.setdefault("estimated_input_tokens", int(payload.get("estimated_input_tokens", 0) or 0))
    payload.setdefault("estimated_output_tokens", int(payload.get("estimated_output_tokens", 0) or 0))
    payload.setdefault("artifact_paths", list(payload.get("affected_paths", []) or []))
    payload.setdefault("error_type", _error_type(payload))
    payload.setdefault("parent_span_id", runtime._last_trace_span_id.get(task_state.run_id, ""))
    runtime._trace_seq += 1
    payload.setdefault("span_id", f"span_{runtime._trace_seq:06d}")
    runtime._last_trace_span_id[task_state.run_id] = payload["span_id"]
    return payload


def _status_for(event, payload):
    if "status" in payload:
        return str(payload.get("status") or "")
    if event == "tool_executed":
        return str(payload.get("tool_status") or "ok")
    if event == "run_finished":
        return str(payload.get("status") or "completed")
    if str(payload.get("tool_error_code", "")):
        return "error"
    return "ok"


def _error_type(payload):
    return str(payload.get("tool_error_code") or payload.get("security_event_type") or payload.get("error_type") or "")
