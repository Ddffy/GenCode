"""Session-scoped replacement ledger for rendered history stubs."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ReplacementRecord:
    event_id: str
    content_sha256: str
    replacement_text: str
    saved_chars: int
    tool_name: str = ""
    artifact_ref: str = ""
    render_mode: str = "artifact_stub"
    original_chars: int = 0
    created_at: str = ""
    pressure_tier: str = ""

    @classmethod
    def from_dict(cls, data):
        return cls(
            event_id=str(data.get("event_id", "")),
            content_sha256=str(data.get("content_sha256", "")),
            replacement_text=str(data.get("replacement_text", "")),
            saved_chars=int(data.get("saved_chars", 0) or 0),
            tool_name=str(data.get("tool_name", "")),
            artifact_ref=str(data.get("artifact_ref", "")),
            render_mode=str(data.get("render_mode", "artifact_stub") or "artifact_stub"),
            original_chars=int(data.get("original_chars", 0) or 0),
            created_at=str(data.get("created_at", "")),
            pressure_tier=str(data.get("pressure_tier", "")),
        )

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ProposedReplacement:
    record: ReplacementRecord

    def to_dict(self):
        return self.record.to_dict()


class ReplacementLedger:
    def __init__(self, records=None):
        self._records = {}
        for record in records or []:
            if record.event_id and record.content_sha256:
                self._records[record.event_id] = record

    @classmethod
    def from_session(cls, session):
        payload = dict(session or {}).get("context_replacements", {}) or {}
        if isinstance(payload, list):
            raw_records = payload
        elif isinstance(payload, dict):
            raw_records = _records_from_mapping(payload)
        else:
            raw_records = []
        return cls(ReplacementRecord.from_dict(record) for record in raw_records if isinstance(record, dict))

    def matching_record(self, item):
        event_id = str(item.get("event_id", ""))
        content_sha256 = str(item.get("content_sha256", ""))
        if not event_id or not content_sha256:
            return None
        record = self._records.get(event_id)
        if record and record.content_sha256 == content_sha256:
            return record
        return None

    def proposed_record(self, item, replacement_text):
        event_id = str(item.get("event_id", ""))
        content_sha256 = str(item.get("content_sha256", ""))
        if not event_id or not content_sha256:
            return None
        saved_chars = max(0, len(str(item.get("content", ""))) - len(str(replacement_text)))
        original_chars = int(item.get("original_chars", 0) or 0)
        if original_chars <= 0:
            original_chars = len(str(item.get("content", "")))
        return ProposedReplacement(
            ReplacementRecord(
                event_id=event_id,
                content_sha256=content_sha256,
                replacement_text=str(replacement_text),
                saved_chars=saved_chars,
                tool_name=str(item.get("name", "")),
                artifact_ref=str(item.get("artifact_ref", "")),
                render_mode="artifact_stub" if item.get("artifact_ref") else "one_line_summary",
                original_chars=original_chars,
                created_at=str(item.get("created_at", "")),
                pressure_tier=str(item.get("pressure_tier", "")),
            )
        )


def commit_proposed_replacements(session, metadata):
    history = dict(metadata.get("history", {}) or {})
    proposals = history.get("proposed_replacements", []) or []
    if not proposals:
        return 0
    ledger = session.setdefault("context_replacements", {})
    if not isinstance(ledger, dict):
        ledger = {}
        session["context_replacements"] = ledger
    committed = 0
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        record = ReplacementRecord.from_dict(proposal)
        if not record.event_id or not record.content_sha256:
            continue
        current = ledger.get(record.event_id)
        if isinstance(current, dict) and current.get("content_sha256") == record.content_sha256:
            continue
        ledger[record.event_id] = _session_record_payload(record)
        committed += 1
    return committed


def _session_record_payload(record):
    payload = record.to_dict()
    payload.pop("event_id", None)
    return payload


def _records_from_mapping(payload):
    records = payload.get("records")
    raw_records = list(records) if isinstance(records, list) else []
    raw_records.extend(_keyed_records(payload))
    return raw_records


def _keyed_records(payload):
    keyed_records = []
    for event_id, record in payload.items():
        if event_id == "records":
            continue
        if not isinstance(record, dict):
            continue
        keyed_record = dict(record)
        keyed_record.setdefault("event_id", str(event_id))
        keyed_records.append(keyed_record)
    return keyed_records
