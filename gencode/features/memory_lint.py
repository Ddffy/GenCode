"""Static memory lint primitives shared by future memory validators."""

import argparse
import json
import re
import sys
from pathlib import Path

_KEYWORD = r"(?:key|token|secret|password|api)"
_LONG_HEX = r"[A-Fa-f0-9]{32,}"
_LONG_BASE64 = r"[A-Za-z0-9+/]{40,}={0,2}"

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{36,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(
        rf"(?i)(?:{_KEYWORD}.{{0,20}}(?:{_LONG_HEX}|{_LONG_BASE64})|(?:{_LONG_HEX}|{_LONG_BASE64}).{{0,20}}{_KEYWORD})"
    ),
]

RELATIVE_DATE_PATTERN = re.compile(r"(?i)\b(tomorrow|yesterday|next week|last week)\b|今天|明天|昨天|下周|上周")


def _load_topic_metadata(path):
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        note_id = str(row.get("note_id", "")).strip()
        if note_id:
            rows[note_id] = row
    return rows


def _load_topic_notes(topic_path):
    topic = topic_path.stem
    lines = topic_path.read_text(encoding="utf-8").splitlines()
    notes = []
    capture = False
    for raw in lines:
        line = raw.strip()
        if line == "## Notes":
            capture = True
            continue
        if capture and line.startswith("- "):
            text = line[2:].strip()
            notes.append({"topic": topic, "text": text, "note_id": _note_id_for(topic, text)})
    return notes


def _tokenize(text):
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", str(text))}


def _note_id_for(topic_slug, note_text):
    import hashlib

    return hashlib.sha256(f"{topic_slug}\n{note_text}".encode("utf-8")).hexdigest()[:12]


def _subject_key(text):
    text = str(text).strip()
    patterns = (
        r"^(.+?)\s+is\s+.+$",
        r"^(.+?)\s+are\s+.+$",
        r"^(.+?)\s+uses?\s+.+$",
        r"^(.+?)\s+should\s+.+$",
        r"^(.+?)是.+$",
        r"^(.+?)使用.+$",
    )
    for pattern in patterns:
        match = re.match(pattern, text, re.I)
        if match:
            subject = " ".join(_tokenize(match.group(1)))
            return subject or None
    return None


def _finding(rule, topic, note_id="", text="", **extra):
    payload = {"rule": rule, "topic": topic}
    if note_id:
        payload["note_id"] = note_id
    if text:
        payload["text"] = text
    payload.update(extra)
    return payload


def lint_memory_dir(memory_dir):
    memory_dir = Path(memory_dir)
    topics_dir = memory_dir / "topics"
    if not topics_dir.exists():
        return []

    findings = []
    subject_rows = {}
    known_note_ids = set()
    all_metadata = {}
    for topic_path in sorted(topics_dir.glob("*.md")):
        topic = topic_path.stem
        metadata = _load_topic_metadata(topics_dir / f"{topic}.metadata.jsonl")
        all_metadata[topic] = metadata
        for note in _load_topic_notes(topic_path):
            known_note_ids.add(note["note_id"])
            row = metadata.get(note["note_id"], {})
            status = str(row.get("status", "active")).strip() or "active"
            if status == "active":
                subject = _subject_key(note["text"])
                if subject:
                    subject_rows.setdefault((topic, subject), []).append(note)
                evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
                if not str(evidence.get("session_id", "")).strip():
                    findings.append(_finding("missing_evidence", topic, note["note_id"], note["text"]))
            if RELATIVE_DATE_PATTERN.search(note["text"]):
                findings.append(_finding("relative_date", topic, note["note_id"], note["text"]))
            if any(pattern.search(note["text"]) for pattern in SECRET_PATTERNS):
                findings.append(_finding("secret_shaped", topic, note["note_id"], note["text"]))

    for (topic, _subject), notes in sorted(subject_rows.items()):
        if len(notes) > 1:
            findings.append(
                _finding(
                    "duplicate_active_subject",
                    topic,
                    text=notes[0]["text"],
                    note_ids=[note["note_id"] for note in notes],
                )
            )

    for topic, metadata in all_metadata.items():
        for row in metadata.values():
            supersedes = row.get("supersedes")
            if supersedes and str(supersedes) not in known_note_ids:
                findings.append(_finding("orphan_supersede", topic, str(row.get("note_id", "")), supersedes=str(supersedes)))

    return sorted(findings, key=lambda item: (item["rule"], item.get("topic", ""), item.get("note_id", "")))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Lint GenCode durable memory files.")
    parser.add_argument("memory_dir", help="Memory directory containing topics/*.md and sidecars.")
    args = parser.parse_args(argv)
    findings = lint_memory_dir(args.memory_dir)
    for finding in findings:
        print(json.dumps(finding, ensure_ascii=False, sort_keys=True))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
