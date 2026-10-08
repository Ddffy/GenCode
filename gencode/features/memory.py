"""Session-scoped working memory for the Agent runtime.

Cross-session durable knowledge lives exclusively in :mod:`knowledge` as
Skill/Wiki/Spec records. This module must not read or write ``.gencode/memory``.
"""

import hashlib
import re
import subprocess
from datetime import datetime
from pathlib import Path

from ..core.runtime.workspace_context import clip, now

WORKING_FILE_LIMIT = 8
EPISODIC_NOTE_LIMIT = 12
FILE_SUMMARY_LIMIT = 6
MAX_ANCHOR_HASH_BYTES = 10 * 1024 * 1024
_WORKSPACE_FINGERPRINT_CACHE = {}


def default_memory_state():
    """Return the small, serializable state owned by one Session."""
    return {
        "working": {"task_summary": "", "recent_files": []},
        "episodic_notes": [],
        "file_summaries": {},
        # Compatibility aliases for older session JSON. They remain scoped to
        # this session and are normalized from the canonical fields below.
        "task": "",
        "files": [],
        "notes": [],
        "next_note_index": 0,
    }


def _ensure_list(value):
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def resolve_workspace_path(raw_path, workspace_root=None):
    path = Path(str(raw_path))
    if workspace_root is None:
        return path
    root = Path(workspace_root).resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def canonicalize_path(raw_path, workspace_root=None):
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None:
        return Path(str(raw_path)).as_posix()
    if workspace_root is None:
        return Path(str(raw_path)).as_posix()
    return resolved.relative_to(Path(workspace_root).resolve()).as_posix()


def file_freshness(raw_path, workspace_root=None):
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        return None
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def compute_anchor_hash(path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    if path.stat().st_size > MAX_ANCHOR_HASH_BYTES:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workspace_fingerprint(workspace_root):
    root = str(Path(workspace_root).resolve())
    cached = _WORKSPACE_FINGERPRINT_CACHE.get(root)
    if cached:
        return cached
    try:
        git_root = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        ).strip()
        normalized_root = str(Path(git_root.decode("utf-8", "replace")).resolve())
        fingerprint = hashlib.sha256(normalized_root.encode("utf-8")).hexdigest()[:12]
    except Exception:  # noqa: BLE001 - non-Git workspaces use their resolved path
        fingerprint = hashlib.sha256(root.encode("utf-8")).hexdigest()[:12]
    _WORKSPACE_FINGERPRINT_CACHE[root] = fingerprint
    return fingerprint


def _normalize_note(note, index):
    if isinstance(note, dict):
        raw_text = note.get("text", "")
    else:
        raw_text = note
        note = {}
    text = clip(str(raw_text).strip(), 500)
    if not text:
        return None
    tags = _dedupe_preserve_order(
        [str(tag).strip() for tag in _ensure_list(note.get("tags", [])) if str(tag).strip()]
    )
    kind = str(note.get("kind", "episodic")).strip() or "episodic"
    # Old sessions called some session-local notes "durable". They never
    # represented the current cross-session knowledge store; normalize them
    # into the only valid scope here: episodic/session-local.
    if kind == "durable":
        kind = "episodic"
    normalized = {
        "text": text,
        "tags": tags,
        "source": str(note.get("source", "")).strip(),
        "created_at": str(note.get("created_at", "")).strip() or now(),
        "note_index": int(note.get("note_index", index)),
        "kind": kind,
    }
    for key in (
        "note_id", "status", "supersedes", "evidence", "scope",
        "stale_evidence", "scope_mismatch",
    ):
        if key in note:
            normalized[key] = note[key]
    return normalized


def normalize_memory_state(state, workspace_root=None):
    """Normalize legacy/current Session JSON without consulting disk memory."""
    if state is None:
        state = default_memory_state()
    elif not isinstance(state, dict):
        raise TypeError("memory state must be a mapping")

    working = state.get("working")
    if not isinstance(working, dict):
        working = {}
    task_summary = str(working.get("task_summary", "") or state.get("task", "")).strip()
    recent_files = working.get("recent_files") or state.get("files", [])
    working["task_summary"] = clip(task_summary, 300)
    working["recent_files"] = _dedupe_preserve_order(
        [
            canonicalize_path(path, workspace_root)
            for path in _ensure_list(recent_files)
            if str(path).strip()
        ]
    )[-WORKING_FILE_LIMIT:]
    state["working"] = working

    raw_notes = state.get("episodic_notes")
    if not isinstance(raw_notes, list):
        raw_notes = []
    if not raw_notes and state.get("notes"):
        raw_notes = _ensure_list(state["notes"])
    notes = []
    for index, raw_note in enumerate(raw_notes):
        normalized = _normalize_note(raw_note, index)
        if normalized:
            notes.append(normalized)
    notes = notes[-EPISODIC_NOTE_LIMIT:]
    state["episodic_notes"] = notes

    summaries = state.get("file_summaries")
    if not isinstance(summaries, dict):
        summaries = {}
    normalized_summaries = {}
    for raw_path, value in summaries.items():
        path = canonicalize_path(raw_path, workspace_root)
        if isinstance(value, dict):
            summary = clip(str(value.get("summary", "")).strip(), 500)
            created_at = str(value.get("created_at", "")).strip() or now()
            freshness = value.get("freshness")
        else:
            summary = clip(str(value).strip(), 500)
            created_at = now()
            freshness = None
        if summary:
            normalized_summaries[path] = {
                "summary": summary,
                "created_at": created_at,
                "freshness": str(freshness).strip() if freshness else None,
            }
    state["file_summaries"] = normalized_summaries

    try:
        next_note_index = max(0, int(state.get("next_note_index", 0)))
    except (TypeError, ValueError):
        next_note_index = 0
    state["next_note_index"] = max(
        next_note_index, max((note["note_index"] + 1 for note in notes), default=0)
    )
    state["task"] = working["task_summary"]
    state["files"] = list(working["recent_files"])
    state["notes"] = [note["text"] for note in notes]
    # Discard the old cross-session topic cache; Skill/Wiki/Spec are the sole
    # source of durable retrieval.
    state.pop("durable_topics", None)
    return state


def set_task_summary(state, summary, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    state["working"]["task_summary"] = clip(str(summary).strip(), 300)
    state["task"] = state["working"]["task_summary"]
    return state


def remember_file(state, path, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    if path:
        files = [item for item in state["working"]["recent_files"] if item != path]
        state["working"]["recent_files"] = (files + [path])[-WORKING_FILE_LIMIT:]
        state["files"] = list(state["working"]["recent_files"])
    return state


def append_note(state, text, tags=(), source="", created_at=None, workspace_root=None, kind="episodic"):
    state = normalize_memory_state(state, workspace_root)
    text = clip(str(text).strip(), 500)
    if not text:
        return state
    normalized_tags = _dedupe_preserve_order(
        [str(tag).strip() for tag in _ensure_list(tags) if str(tag).strip()]
    )
    note = {
        "text": text,
        "tags": normalized_tags,
        "source": str(source).strip(),
        "created_at": str(created_at).strip() if created_at else now(),
        "note_index": int(state.get("next_note_index", 0)),
        "kind": "episodic" if str(kind).strip() == "durable" else (str(kind).strip() or "episodic"),
    }
    state["next_note_index"] = note["note_index"] + 1
    notes = [item for item in state["episodic_notes"] if item["text"] != text]
    state["episodic_notes"] = (notes + [note])[-EPISODIC_NOTE_LIMIT:]
    state["notes"] = [item["text"] for item in state["episodic_notes"]]
    return state


def set_file_summary(state, path, summary, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    summary = clip(str(summary).strip(), 500)
    if path and summary:
        state["file_summaries"][path] = {
            "summary": summary,
            "created_at": now(),
            "freshness": file_freshness(path, workspace_root),
        }
    return state


def invalidate_file_summary(state, path, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    state["file_summaries"].pop(canonicalize_path(path, workspace_root).strip(), None)
    return state


def invalidate_stale_file_summaries(state, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    invalidated = []
    for path, summary in list(state["file_summaries"].items()):
        if summary.get("freshness") == file_freshness(path, workspace_root):
            continue
        invalidated.append(path)
        state["file_summaries"].pop(path, None)
    return state, invalidated


def summarize_read_result(result, limit=180):
    lines = [line.strip() for line in str(result).splitlines() if line.strip()]
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return clip(" | ".join(lines[:3]), limit) if lines else "(empty)"


def _tokenize(text):
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", str(text))}


def _parse_timestamp(value):
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _query_hash(query):
    return hashlib.sha256(str(query).encode("utf-8")).hexdigest()[:12]


def _retrieval_note_id(note):
    explicit = str(note.get("note_id", "")).strip()
    if explicit:
        return explicit
    source = str(note.get("source", "")).strip() or str(note.get("kind", "episodic"))
    return hashlib.sha256(f"{source}\n{note.get('text', '')}".encode()).hexdigest()[:12]


def _source_path_for_evidence(workspace_root, source_path):
    path = Path(str(source_path or "").strip())
    return path if path.is_absolute() or workspace_root is None else Path(workspace_root) / path


def _apply_evidence_staleness(note, workspace_root):
    evidence = note.get("evidence") if isinstance(note.get("evidence"), dict) else {}
    stored_hash = str(evidence.get("evidence_anchor_hash", "") or "").strip()
    source_path = evidence.get("source_path")
    if not stored_hash or not source_path:
        return note
    current_hash = compute_anchor_hash(_source_path_for_evidence(workspace_root, source_path))
    if current_hash and current_hash != stored_hash:
        note = dict(note)
        note["stale_evidence"] = True
    return note


def _retrieval_reject_reason(note, workspace_root=None):
    status = str(note.get("status", "active")).strip() or "active"
    if status in {"quarantined", "superseded"}:
        return status
    if note.get("stale_evidence"):
        return "stale_evidence"
    scope = str(note.get("scope", "")).strip()
    if scope and scope not in {"workspace_fingerprint", "global"}:
        return "scope_mismatch"
    if note.get("scope_mismatch"):
        return "scope_mismatch"
    return ""


def _ranked_retrieval_notes(state, query, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    query_tokens = _tokenize(query)
    ranked = []
    for raw_note in state["episodic_notes"]:
        note = _apply_evidence_staleness(dict(raw_note), workspace_root)
        tags = {str(tag).lower() for tag in note.get("tags", [])}
        note_tokens = _tokenize(note.get("text", "")) | _tokenize(note.get("source", "")) | tags
        exact_tag_match = int(bool(query_tokens & tags))
        keyword_overlap = len(query_tokens & note_tokens)
        if exact_tag_match == 0 and keyword_overlap == 0:
            continue
        recency = _parse_timestamp(note.get("created_at"))
        note_index = int(note.get("note_index", 0))
        score = exact_tag_match * 1000 + keyword_overlap * 10 + recency / 1_000_000 + note_index / 1_000_000_000
        ranked.append(((exact_tag_match, keyword_overlap, recency, note_index), score, note))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def retrieval_view_structured(state, query, limit=3, workspace_root=None):
    selected = []
    rejected = []
    for _, score, note in _ranked_retrieval_notes(state, query, workspace_root):
        reject_reason = _retrieval_reject_reason(note, workspace_root)
        record = dict(note)
        record.update({
            "note_id": _retrieval_note_id(note),
            "layer": str(note.get("kind", "episodic") or "episodic"),
            "score": float(score),
        })
        if reject_reason:
            record["reject_reason"] = reject_reason
            rejected.append(record)
        elif len(selected) < max(0, int(limit)):
            selected.append(record)
        else:
            record["reject_reason"] = "below_limit"
            rejected.append(record)
    return {"selected": selected, "rejected": rejected, "query_hash": _query_hash(query)}


def retrieval_candidates(state, query, limit=3, workspace_root=None):
    return retrieval_view_structured(state, query, limit, workspace_root)["selected"]


def retrieval_view(state, query, limit=3, workspace_root=None):
    candidates = retrieval_candidates(state, query, limit, workspace_root)
    if not candidates:
        return "Relevant memory:\n- none"
    return "\n".join(["Relevant memory:"] + [f"- {note['text']}" for note in candidates])


def render_memory_text(state, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    lines = [
        "Memory:",
        f"- task: {state['working']['task_summary'] or '-'}",
        f"- recent_files: {', '.join(state['working']['recent_files']) or '-'}",
    ]
    summaries = []
    for path in state["working"]["recent_files"][:FILE_SUMMARY_LIMIT]:
        summary = state["file_summaries"].get(path, {})
        if summary.get("summary") and summary.get("freshness") == file_freshness(path, workspace_root):
            summaries.append(f"- {path}: {summary['summary']}")
    lines.append("- file_summaries:")
    lines.extend(f"  {line}" for line in summaries) if summaries else lines.append("  -")
    lines.append(f"- episodic_notes: {len(state['episodic_notes'])}")
    return "\n".join(lines)


def is_effectively_empty(state, workspace_root=None):
    state = normalize_memory_state(state, workspace_root)
    return not (
        state["working"]["task_summary"]
        or state["working"]["recent_files"]
        or state["episodic_notes"]
        or state["file_summaries"]
    )


class LayeredMemory:
    """Mutable facade over the current Session's bounded working-memory state."""

    def __init__(self, state=None, workspace_root=None):
        self.workspace_root = workspace_root
        self.state = normalize_memory_state(state, workspace_root)
        self.last_retrieval = None

    def to_dict(self):
        self.state = normalize_memory_state(self.state, self.workspace_root)
        return self.state

    def canonical_path(self, path):
        return canonicalize_path(path, self.workspace_root)

    def set_task_summary(self, summary):
        self.state = set_task_summary(self.state, summary, self.workspace_root)
        return self

    def remember_file(self, path):
        self.state = remember_file(self.state, path, self.workspace_root)
        return self

    def append_note(self, text, tags=(), source="", created_at=None, kind="episodic"):
        self.state = append_note(
            self.state, text, tags, source, created_at, self.workspace_root, kind
        )
        return self

    def set_file_summary(self, path, summary):
        self.state = set_file_summary(self.state, path, summary, self.workspace_root)
        return self

    def invalidate_file_summary(self, path):
        self.state = invalidate_file_summary(self.state, path, self.workspace_root)
        return self

    def invalidate_stale_file_summaries(self):
        self.state, invalidated = invalidate_stale_file_summaries(self.state, self.workspace_root)
        return invalidated

    def retrieval_candidates(self, query, limit=3):
        self.last_retrieval = retrieval_view_structured(
            self.state, query, limit=limit, workspace_root=self.workspace_root
        )
        return self.last_retrieval["selected"]

    def retrieval_view_structured(self, query, limit=3):
        self.last_retrieval = retrieval_view_structured(
            self.state, query, limit=limit, workspace_root=self.workspace_root
        )
        return self.last_retrieval

    def retrieval_view(self, query, limit=3):
        return retrieval_view(self.state, query, limit=limit, workspace_root=self.workspace_root)

    def render_memory_text(self):
        return render_memory_text(self.state, self.workspace_root)
