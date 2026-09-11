"""Runtime consumers that derive state from trace events.

Consumers update TaskState views such as artifact graphs, verifier suggestions,
reminders, and evidence summaries as events arrive. They should remain
side-effect-light and avoid becoming a second control loop.
"""

from .artifacts import build_artifact_graph, build_verifier_suggestions
from .evidence_summaries import update_evidence_summaries
from .workspace import clip


class ArtifactGraphConsumer:
    def handle(self, runtime, task_state, event):
        if event.get("event") not in {"tool_executed", "run_finished", "checkpoint_created"}:
            return
        if not task_state.changed_paths and not event.get("artifact_paths"):
            return
        graph = build_artifact_graph(runtime.root, task_state.changed_paths)
        task_state.artifact_graph = graph


class VerifierSuggestionConsumer:
    def handle(self, runtime, task_state, event):
        if event.get("event") not in {"tool_executed", "run_finished", "checkpoint_created"}:
            return
        graph = task_state.artifact_graph or build_artifact_graph(runtime.root, task_state.changed_paths)
        task_state.verifier_suggestions = build_verifier_suggestions(runtime.root, graph)


class ReminderConsumer:
    def handle(self, runtime, task_state, event):
        if event.get("event") != "tool_executed":
            return
        status = str(event.get("status", ""))
        if status in {"", "ok"}:
            return
        reminder = {
            "event": "tool_executed",
            "tool": str(event.get("name", "")),
            "status": status,
            "error_type": str(event.get("error_type", "")),
            "message": clip(str(event.get("result", "")), 240),
            "workspace_changed": bool(event.get("workspace_changed", False)),
            "affected_paths": list(event.get("affected_paths", []) or []),
            "source_span_id": event.get("span_id", ""),
            "created_at": event.get("created_at", ""),
        }
        task_state.runtime_reminders.append(reminder)


class EvidenceSummaryConsumer:
    critical = True

    def handle(self, runtime, task_state, event):
        task_state.evidence_summaries = update_evidence_summaries(
            task_state.evidence_summaries, event, changed_paths=task_state.changed_paths
        )


def default_runtime_consumers():
    return [
        ArtifactGraphConsumer(),
        VerifierSuggestionConsumer(),
        ReminderConsumer(),
        EvidenceSummaryConsumer(),
    ]
