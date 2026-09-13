"""Terminal-state governance for GenCode turns."""

import time

from .before_final_hooks import run_before_final_hooks
from .final_readiness import evaluate_final_readiness, readiness_notice
from .knowledge_governance import maintain_knowledge_safely
from .turn_transitions import emit_terminal_transition
from .workspace import clip, now


def final_readiness_action(engine, task_state, proposed_final=""):
    """Return the action GenCode should take before accepting a final answer.

    Engine owns the turn loop. This module owns the terminal-readiness policy
    that can ask the model for more work, block unsafe finals, or allow them.
    """
    agent = engine.runtime
    hook_decision = run_before_final_hooks(agent, task_state, proposed_final)
    if hook_decision.get("hook_count"):
        agent.emit_trace(task_state, "before_final_hook_decision", hook_decision)
        action = str(hook_decision.get("action", "allow"))
        if action == "runtime_notice":
            notice = (
                str(hook_decision.get("message", ""))
                or "Before-final hook requested more work."
            )
            _record_runtime_notice(agent, task_state, notice)
            return action, notice
        if action == "block":
            return (
                action,
                str(hook_decision.get("message", ""))
                or "Before-final hook blocked the final answer.",
            )

    decision = evaluate_final_readiness(
        task_state, getattr(agent, "final_readiness_mode", "warn")
    )
    if decision["mode"] == "off":
        return "allow", ""
    agent.emit_trace(task_state, "final_readiness_decision", decision)
    action = str(decision.get("action", "none"))
    if action == "runtime_notice":
        notice = readiness_notice(decision)
        _record_runtime_notice(agent, task_state, notice)
        return action, notice
    if action == "block":
        return action, readiness_notice(decision)
    return "allow", ""


def _record_runtime_notice(agent, task_state, notice):
    agent.record({"role": "assistant", "content": notice, "created_at": now()})
    agent.session_event_bus.emit(
        "assistant_message",
        {
            "run_id": task_state.run_id,
            "kind": "runtime_notice",
            "content": notice,
        },
    )
    agent.run_store.write_task_state(task_state)


def finish_successful_run(engine, task_state, user_message, final, run_started_at):
    agent = engine.runtime
    agent.record({"role": "assistant", "content": final, "created_at": now()})
    if agent.runtime_mode == "plan":
        agent.exit_plan_mode()
    agent.session_event_bus.emit(
        "assistant_message",
        {"run_id": task_state.run_id, "kind": "final", "content": clip(final, 500)},
    )
    task_state.finish_success(final)
    worker_events = _emit_terminal_artifacts(
        engine,
        task_state,
        user_message,
        final,
        run_started_at,
        checkpoint_trigger="run_finished",
    )
    yield from worker_events
    yield {"type": "final", "run_id": task_state.run_id, "content": final}
    yield _turn_finished_event(task_state)


def finish_stopped_run(
    engine, task_state, user_message, final, stop_reason, run_started_at
):
    agent = engine.runtime
    task_state.stop(stop_reason, final_answer=final)
    agent.abort_requested = False
    agent.record({"role": "assistant", "content": final, "created_at": now()})
    agent.session_event_bus.emit(
        "assistant_message",
        {"run_id": task_state.run_id, "kind": "stop", "content": clip(final, 500)},
    )
    agent.run_store.write_task_state(task_state)
    _emit_terminal_artifacts(
        engine,
        task_state,
        user_message,
        final,
        run_started_at,
        checkpoint_trigger=stop_reason,
        maintain_memory=False,
        drain_workers=False,
    )
    yield {"type": "stop", "run_id": task_state.run_id, "content": final}
    yield _turn_finished_event(task_state)


def finish_limited_run(engine, task_state, user_message, final, run_started_at):
    agent = engine.runtime
    agent.record({"role": "assistant", "content": final, "created_at": now()})
    agent.session_event_bus.emit(
        "assistant_message",
        {"run_id": task_state.run_id, "kind": "stop", "content": clip(final, 500)},
    )
    agent.run_store.write_task_state(task_state)
    _emit_terminal_artifacts(
        engine,
        task_state,
        user_message,
        final,
        run_started_at,
        checkpoint_trigger=task_state.stop_reason or "run_stopped",
    )
    yield {"type": "stop", "run_id": task_state.run_id, "content": final}
    yield _turn_finished_event(task_state)


def _emit_terminal_artifacts(
    engine,
    task_state,
    user_message,
    final,
    run_started_at,
    *,
    checkpoint_trigger,
    maintain_memory=True,
    drain_workers=True,
):
    agent = engine.runtime
    emit_terminal_transition(
        agent,
        task_state,
        reason=task_state.stop_reason,
        stop_reason=task_state.stop_reason,
    )
    if maintain_memory:
        agent.promote_durable_memory(user_message, final)
        maintain_memory_safely(agent, task_state, final)
    checkpoint = agent.create_checkpoint(
        task_state, user_message, trigger=checkpoint_trigger
    )
    agent.run_store.write_task_state(task_state)
    agent.emit_trace(
        task_state,
        "checkpoint_created",
        {"checkpoint_id": checkpoint["checkpoint_id"], "trigger": checkpoint_trigger},
    )
    duration_ms = int((time.monotonic() - run_started_at) * 1000)
    agent.emit_trace(
        task_state,
        "run_finished",
        {
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "final_answer": final,
            "run_duration_ms": duration_ms,
        },
    )
    agent.session_event_bus.emit(
        "turn_finished",
        {
            "run_id": task_state.run_id,
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "duration_ms": duration_ms,
        },
    )
    agent.run_store.write_report(
        task_state, agent.redact_artifact(agent.build_report(task_state))
    )
    if drain_workers:
        worker_events = [
            {
                "type": "worker_notification",
                "run_id": getattr(agent, "current_run_id", ""),
                "content": notification,
            }
            for notification in engine.drain_worker_notifications()
        ]
    else:
        worker_events = []
    agent.current_turn_id = ""
    agent.current_run_id = ""
    return worker_events


def maintain_memory_safely(agent, task_state, final_answer):
    try:
        agent.maintain_memory_after_turn(final_answer)
    except Exception as exc:
        audit = getattr(agent, "last_memory_maintenance", {"errors": []})
        errors = audit.setdefault("errors", [])
        errors.append(str(exc))
        agent.last_memory_maintenance = audit
        agent.session_event_bus.emit(
            "memory_maintenance_failed",
            {"run_id": task_state.run_id, "error": clip(str(exc), 300)},
        )
        agent.emit_trace(
            task_state, "memory_maintenance_failed", {"error": clip(str(exc), 300)}
        )
    maintain_knowledge_safely(agent, task_state, final_answer)


def _turn_finished_event(task_state):
    return {
        "type": "turn_finished",
        "run_id": task_state.run_id,
        "status": task_state.status,
        "stop_reason": task_state.stop_reason,
    }
