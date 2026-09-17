"""Failure-contained terminal maintenance for typed durable knowledge."""

from .citation_report import build_citation_report
from .workspace import clip


def maintain_knowledge_safely(agent, task_state, final_answer):
    validate_turn_citations_safely(agent, task_state, final_answer)
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


def validate_turn_citations_safely(agent, task_state, final_answer):
    """Record whether the answer cited evidence that retrieval never returned.

    Validation is deliberately observational: it records the verdict, traces it,
    and emits a session event, but does not rewrite or reject the answer. Whether
    an unsupported citation should force a retry or an abstention is a product
    decision, so it stays out of this hook.
    """
    try:
        report = build_citation_report(
            final_answer,
            code_retrieval=getattr(agent, "last_code_retrieval", None),
            knowledge_retrieval=getattr(agent, "last_knowledge_retrieval", None),
        )
    except Exception as exc:  # noqa: BLE001 - validation must not fail a turn
        agent.last_citation_validation = {"checked": False, "error": clip(str(exc), 300)}
        agent.emit_trace(
            task_state, "citation_validation_failed", {"error": clip(str(exc), 300)}
        )
        return
    agent.last_citation_validation = report
    agent.emit_trace(task_state, "citation.validation", dict(report))
    if report.get("invalid"):
        payload = {"invalid": report["invalid"], "used": report["used"]}
        agent.emit_trace(task_state, "citation.invalid", payload)
        agent.session_event_bus.emit("citation_validation_failed", payload)
