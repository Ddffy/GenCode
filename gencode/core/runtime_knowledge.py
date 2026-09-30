"""Runtime integration for typed Skill, Wiki, and Spec knowledge."""

from ..config import resolve_project_retrieval_config
from ..features import knowledge as knowledgelib
from ..features.knowledge_proposal import propose_candidate
from ..features import skills as skillslib


def _empty_maintenance_audit():
    return {"candidates": [], "quarantined": [], "errors": [], "auto_dream": {}}


class RuntimeKnowledgeMixin:
    """Keep durable-knowledge lifecycle policy out of the runtime object graph."""

    def initialize_knowledge(self):
        self.knowledge_store = knowledgelib.KnowledgeStore(
            self.root / ".gencode" / "knowledge",
            self.root,
            event_sink=self.session_event_bus.emit,
            retrieval_config=resolve_project_retrieval_config(start=self.root),
        )
        self.last_knowledge_retrieval = None
        self.last_code_retrieval = None
        self.last_citation_validation = {}
        self.last_knowledge_maintenance = _empty_maintenance_audit()
        self.knowledge_source_sessions = {str(self.session.get("id", ""))}
        self.knowledge_proposal_source = "assistant_tool"

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

    def propose_knowledge_candidate(self, args):
        return propose_candidate(self, args)

    def maintain_knowledge_after_turn(self, final_answer=""):
        # Final answers are ordinary user-facing prose. Candidate creation uses
        # the structured knowledge_propose tool; no XML tag scraping here.
        from ..features import dream as dreamlib

        audit = dict(self.last_knowledge_maintenance or {})
        audit.setdefault("candidates", [])
        audit.setdefault("quarantined", [])
        audit.setdefault("errors", [])
        auto = dreamlib.maintain_after_turn(self)
        audit["auto_dream"] = auto.get("auto_dream", {})
        audit["errors"].extend(auto.get("errors", []))
        self.last_knowledge_maintenance = audit
        return audit

    def run_knowledge_dream(self, *, quiet=False, session_ids=None, extra_notes=()):
        from ..features import dream as dreamlib

        return dreamlib.run_dream(
            self, quiet=quiet, session_ids=session_ids, extra_notes=extra_notes
        )

    def knowledge_report_fields(self):
        retrieval = (
            self.knowledge_store.trace_retrieval(self.last_knowledge_retrieval)
            if self.last_knowledge_retrieval
            else {}
        )
        return {
            "knowledge_retrieval": retrieval,
            "code_retrieval": dict(self.last_code_retrieval or {}),
            "knowledge_maintenance": dict(self.last_knowledge_maintenance),
            "citation_validation": dict(self.last_citation_validation or {}),
        }

    def reset_knowledge_state(self):
        self.session["knowledge"] = {"active_specs": []}
        self.last_knowledge_retrieval = None
        self.last_code_retrieval = None
        self.last_citation_validation = {}
        self.last_knowledge_maintenance = _empty_maintenance_audit()
