"""Control-loop tool and retry helpers shared by Engine.

These helpers execute tool payloads and handle retry summaries while Engine
keeps the turn loop shape visible. Terminal-state policy lives in
completion_governance.
"""

import time

from ..providers.base import complete_model
from ..providers.errors import ProviderError
from .native_messages import build_native_messages
from .workspace import clip, now


def complete_runtime_model(agent, prompt, user_message, max_new_tokens, tools=None, **kwargs):
    """Dispatch native-capable clients through structured messages."""
    messages = None
    if tools and hasattr(agent.model_client, "complete_messages"):
        messages = build_native_messages(agent, user_message, prompt=prompt)
    return complete_model(
        agent.model_client,
        prompt,
        max_new_tokens,
        tools=tools,
        messages=messages,
        **kwargs,
    )


def handle_prompt_checkpoints(engine, task_state, user_message, prompt_metadata):
    """Persist recovery checkpoints implied by the freshly built prompt."""
    agent = engine.runtime

    def create(trigger):
        checkpoint = agent.create_checkpoint(task_state, user_message, trigger=trigger)
        agent.run_store.write_task_state(task_state)
        agent.emit_trace(
            task_state,
            "checkpoint_created",
            {"checkpoint_id": checkpoint["checkpoint_id"], "trigger": trigger},
        )

    status = prompt_metadata.get("resume_status")
    if status == "partial-stale":
        create("freshness_mismatch")
    elif status == "workspace-mismatch":
        agent.emit_trace(
            task_state,
            "runtime_identity_mismatch",
            {"fields": list(prompt_metadata.get("runtime_identity_mismatch_fields", []))},
        )
        create("workspace_mismatch")
    pressure = prompt_metadata.get("pressure", {}).get("tier", "tier0_observe")
    if prompt_metadata.get("budget_reductions") or pressure != "tier0_observe":
        create("context_reduction")


def execute_tool_payload(engine, task_state, user_message, payload):
    agent = engine.runtime
    name = payload.get("name", "")
    args = payload.get("args", {})
    task_state.record_tool(name)
    tool_started_at = time.monotonic()
    agent.session_event_bus.emit(
        "tool_started", {"run_id": task_state.run_id, "tool_name": name, "args": args}
    )
    yield {"type": "tool_call", "run_id": task_state.run_id, "name": name, "args": args}

    tool_result = agent.run_tool(name, args)
    tool_metadata = dict(agent._last_tool_result_metadata or {})
    tool_duration_ms = int((time.monotonic() - tool_started_at) * 1000)
    agent.session_event_bus.emit(
        "tool_finished",
        {
            "run_id": task_state.run_id,
            "tool_name": name,
            "status": tool_metadata.get("tool_status", ""),
            "tool_error_code": tool_metadata.get("tool_error_code", ""),
            "workspace_changed": bool(tool_metadata.get("workspace_changed", False)),
            "affected_paths": list(tool_metadata.get("affected_paths", [])),
            "duration_ms": tool_duration_ms,
        },
    )
    history_item = {
        "role": "tool",
        "name": name,
        "args": args,
        "content": tool_result,
        "created_at": now(),
        "tool_status": str(tool_metadata.get("tool_status", "")),
        "tool_error_code": str(tool_metadata.get("tool_error_code", "")),
        "workspace_changed": bool(tool_metadata.get("workspace_changed", False)),
        "affected_paths": list(tool_metadata.get("affected_paths", []) or []),
    }
    # Native providers require the assistant call and its tool result to share
    # the provider-issued ID.  Text-protocol calls simply omit this field and
    # keep their legacy history shape.
    if payload.get("id"):
        history_item["tool_call_id"] = str(payload["id"])
    if tool_metadata.get("full_output_artifact"):
        history_item.update(
            {
                "artifact_ref": tool_metadata["full_output_artifact"],
                "original_chars": int(tool_metadata.get("original_chars", 0) or 0),
                "content_sha256": str(tool_metadata.get("content_sha256", "")),
            }
        )
    if tool_metadata.get("media_refs"):
        history_item["media_refs"] = list(tool_metadata.get("media_refs", []) or [])
    agent.record(history_item)
    for notification in engine.drain_worker_notifications():
        yield {
            "type": "worker_notification",
            "run_id": getattr(agent, "current_run_id", ""),
            "content": notification,
        }
    agent.run_store.write_task_state(task_state)
    agent.emit_trace(
        task_state,
        "tool_executed",
        {
            "name": name,
            "args": args,
            "result": clip(tool_result, 500),
            "duration_ms": tool_duration_ms,
            **tool_metadata,
        },
    )
    checkpoint = agent.create_checkpoint(
        task_state, user_message, trigger="tool_executed"
    )
    agent.run_store.write_task_state(task_state)
    agent.emit_trace(
        task_state,
        "checkpoint_created",
        {"checkpoint_id": checkpoint["checkpoint_id"], "trigger": "tool_executed"},
    )
    yield {
        "type": "tool_result",
        "run_id": task_state.run_id,
        "name": name,
        "content": tool_result,
        "metadata": tool_metadata,
    }


SKIPPED_TOOL_MESSAGE = "error: skipped, this call did not run (per-turn tool budget or abort)"


def record_skipped_tool_call(engine, task_state, payload, *, reason):
    """Record a result for a call that was never executed.

    A budget or abort check happens per call, so a batch of tool calls can be cut
    short in the middle. Every tool_use block must be followed by a matching
    tool_result, or providers reject the next request outright ("tool_use ids were
    found without tool_result blocks immediately after"), which leaves the whole
    session unusable rather than failing just the current turn. So the remaining
    calls still get a recorded result: it keeps the history pairable and tells the
    model why the results it asked for are missing.
    """
    agent = engine.runtime
    name = str(payload.get("name", ""))
    args = payload.get("args", {}) or {}
    metadata = {
        "tool_status": "rejected",
        "tool_error_code": reason,
        "workspace_changed": False,
        "affected_paths": [],
        "read_only": False,
        "risk_level": "low",
        "security_event_type": "",
        "diff_summary": [],
    }
    history_item = {
        "role": "tool",
        "name": name,
        "args": args,
        "content": SKIPPED_TOOL_MESSAGE,
        "created_at": now(),
        **metadata,
    }
    if payload.get("id"):
        history_item["tool_call_id"] = str(payload["id"])
    agent.record(history_item)
    agent.emit_trace(
        task_state, "tool_skipped", {"name": name, "args": args, "reason": reason}
    )
    yield {
        "type": "tool_result",
        "run_id": task_state.run_id,
        "name": name,
        "content": SKIPPED_TOOL_MESSAGE,
        "metadata": metadata,
    }


def should_retry_model_error(exc, provider_retries):
    if not isinstance(exc, ProviderError):
        return False
    code = str(getattr(exc, "code", "") or "")
    if code not in {"empty_response"}:
        return False
    return provider_retries.get(code, 0) < 1


_STEP_LIMIT_SUMMARY_NOTICE = (
    "You have hit the per-turn tool budget (max_steps). Do not call any more tools. "
    "Right now, return a single <final>...</final> answer in the user's language that "
    "briefly covers: (1) what you accomplished this turn, (2) what remains undone, "
    "(3) how the user can continue (e.g., `/resume` then `继续`). Keep it concise."
)


def request_step_limit_summary(engine, task_state, user_message):
    """Ask the model to write a graceful step-limit summary.

    Returns the final text, or None if the model fails or refuses to comply.
    Side effects: emits a trace event but does NOT mutate session history —
    the caller decides whether to record the resulting final.
    """
    agent = engine.runtime
    started_at = time.monotonic()
    try:
        prompt, _ = agent._build_prompt_and_metadata(_STEP_LIMIT_SUMMARY_NOTICE)
        result = complete_model(agent.model_client, prompt, agent.max_new_tokens)
    except Exception as exc:
        agent.emit_trace(
            task_state,
            "step_limit_summary_failed",
            {"error": clip(str(exc), 200)},
        )
        return None
    raw = (result.text or "").strip() if result else ""
    if (result.metadata or {}).get("native_tool_calling") and raw:
        kind, payload = "final", raw
    else:
        kind, payload = agent.parse(raw)
    duration_ms = int((time.monotonic() - started_at) * 1000)
    agent.emit_trace(
        task_state,
        "step_limit_summary",
        {"kind": kind, "duration_ms": duration_ms, "produced": bool(kind == "final")},
    )
    if kind == "final" and payload:
        return str(payload).strip()
    return None
