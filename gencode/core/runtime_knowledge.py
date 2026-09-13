"""Runtime integration for typed Skill, Wiki, and Spec knowledge."""

from ..features import knowledge as knowledgelib
from ..features import skills as skillslib


class RuntimeKnowledgeMixin:
    """Keep durable-knowledge lifecycle policy out of the runtime object graph."""

    def initialize_knowledge(self):
        self.knowledge_store = knowledgelib.KnowledgeStore(
            self.root / ".gencode" / "knowledge",
            self.root,
            event_sink=self.session_event_bus.emit,
        )
        self.last_knowledge_retrieval = None
        self.last_knowledge_maintenance = {
            "candidates": [],
            "quarantined": [],
            "errors": [],
        }

    def ensure_knowledge_session_shape(self):
        knowledge = self.session.setdefault("knowledge", {"active_specs": []})
        if not isinstance(knowledge, dict):
            knowledge = {"active_specs": []}
            self.session["knowledge"] = knowledge
        knowledge.setdefault("active_specs", [])

    def knowledge_command_text(self):
        return self.knowledge_store.command_text()

    def approve_knowledge(self, record_id, kind=None):
        record = self.knowledge_store.approve(record_id, kind=kind)
        self.skills = skillslib.discover_skills(self.root)
        return f"Approved {record['kind']}:{record['id']} v{record['version']}."

    def reject_knowledge(self, record_id, kind=None):
        record = self.knowledge_store.reject(record_id, kind=kind)
        self.skills = skillslib.discover_skills(self.root)
        return f"Rejected {record['kind']}:{record['id']}."

    def active_spec_ids(self):
        self._ensure_session_shape()
        return list(self.session["knowledge"].get("active_specs", []))

    def bind_spec(self, record_id):
        record = self.knowledge_store.get(record_id, kind="spec", include_inactive=True)
        if not record:
            raise ValueError(f"unknown spec: {record_id}")
        selected, rejected = self.knowledge_store.bound_specs([record_id])
        if not selected:
            reason = rejected[0]["reject_reason"] if rejected else "not_active"
            raise ValueError(f"spec cannot be bound: {reason}")
        specs = [item for item in self.active_spec_ids() if item != record["id"]]
        specs.append(record["id"])
        self.session["knowledge"]["active_specs"] = specs
        self.session_path = self.session_store.save(self.session)
        self.session_event_bus.emit(
            "spec_bound", {"spec_id": record["id"], "version": record["version"]}
        )
        return f"Bound spec:{record['id']} v{record['version']}."

    def clear_spec(self, record_id=""):
        current = self.active_spec_ids()
        if record_id:
            normalized = knowledgelib.normalize_slug(record_id)
            current = [item for item in current if item != normalized]
        else:
            current = []
        self.session["knowledge"]["active_specs"] = current
        self.session_path = self.session_store.save(self.session)
        self.session_event_bus.emit(
            "spec_unbound",
            {"spec_id": str(record_id or "*"), "remaining": current},
        )
        return "Bound specs: " + (", ".join(current) if current else "none")

    def maintain_knowledge_after_turn(self, final_answer):
        if not self.feature_enabled("typed_knowledge"):
            audit = {
                "candidates": [],
                "quarantined": [],
                "errors": [],
                "skip_reason": "disabled",
            }
            self.last_knowledge_maintenance = audit
            return audit
        audit = self.knowledge_store.maintain_from_final(
            final_answer,
            session_id=self.session.get("id", ""),
            run_id=getattr(self, "current_run_id", ""),
        )
        self.last_knowledge_maintenance = audit
        return audit

    def knowledge_report_fields(self):
        retrieval = (
            self.knowledge_store.trace_retrieval(self.last_knowledge_retrieval)
            if self.last_knowledge_retrieval
            else {}
        )
        return {
            "knowledge_retrieval": retrieval,
            "knowledge_maintenance": dict(self.last_knowledge_maintenance),
        }

    def reset_knowledge_state(self):
        self.session["knowledge"] = {"active_specs": []}
        self.last_knowledge_retrieval = None
        self.last_knowledge_maintenance = {
            "candidates": [],
            "quarantined": [],
            "errors": [],
        }
