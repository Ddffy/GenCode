"""Failure-contained terminal maintenance for typed durable knowledge."""

from .workspace import clip


def maintain_knowledge_safely(agent, task_state, final_answer):
    try:
        audit = agent.maintain_knowledge_after_turn(final_answer)
        agent.emit_trace(task_state, "knowledge.maintenance", dict(audit))
    except Exception as exc:  # noqa: BLE001 - maintenance must not fail a turn
        audit = getattr(
            agent,
            "last_knowledge_maintenance",
            {"candidates": [], "quarantined": [], "errors": []},
        )
        audit.setdefault("errors", []).append(str(exc))
        agent.last_knowledge_maintenance = audit
        payload = {"run_id": task_state.run_id, "error": clip(str(exc), 300)}
        agent.session_event_bus.emit("knowledge_maintenance_failed", payload)
        agent.emit_trace(task_state, "knowledge_maintenance_failed", payload)
