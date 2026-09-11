"""Rule-based deterministic compact summary extraction."""

import re

CONSTRAINT_PATTERNS = (
    "不要", "不能", "必须", "只", "不改", "保持", "除了",
    "don't", "must", "only", "keep", "never", "always",
    "do not", "without changing", "preserve",
)
DECISION_PATTERNS = (
    "decided", "选择", "因为", "approach", "改用", "放弃",
    "instead", "rather than", "switched to",
)
ERROR_PATTERNS = (
    "Error", "error:", "failed", "失败", "Traceback", "FAILED",
    "AssertionError", "TypeError", "KeyError",
)
REJECTED_PATTERNS = (
    "不行", "reverted", "doesn't work", "didn't work", "does not work",
)


def summarize_compact_items(items, prior_text=""):
    evidence = _collect_evidence(items)
    summary = "\n".join(
        [
            "Compacted session summary:",
            f"- Goal: {evidence['goal']}",
            f"- User constraints: {_joined(evidence['user_constraints'])}",
            f"- Files read: {', '.join(sorted(set(evidence['files_read']))) or '-'}",
            f"- Files modified: {', '.join(sorted(set(evidence['files_modified']))) or '-'}",
            f"- Key decisions: {_joined(evidence['key_decisions'])}",
            f"- Rejected paths: {_joined(evidence['rejected_paths'])}",
            f"- Last error context: {evidence['last_error_context']}",
            f"- Critical artifacts: {', '.join(evidence['critical_artifacts']) or '-'}",
            f"- Current progress: compacted {len(items)} history items",
            "- Next step: continue from the latest preserved turn",
        ]
    )
    if prior_text:
        summary = prior_text + "\n\nIncremental compacted delta:\n" + "\n".join(summary.splitlines()[1:])
    return summary[:1997].rstrip() + "..." if len(summary) > 2000 else summary


def _collect_evidence(items):
    evidence = {
        "goal": "-",
        "user_constraints": [],
        "files_read": [],
        "files_modified": [],
        "key_decisions": [],
        "rejected_paths": [],
        "last_error_context": "-",
        "critical_artifacts": [],
    }
    for item in items:
        _collect_artifact(item, evidence)
        if item.get("role") == "user":
            evidence["goal"] = str(item.get("content", "")).strip() or evidence["goal"]
            _collect_user_constraints(item, evidence)
        elif item.get("role") == "assistant":
            _collect_assistant_evidence(item, evidence)
        elif item.get("role") == "tool":
            _collect_tool_evidence(item, evidence)
    for item in reversed(items):
        content = str(item.get("content", "")).strip()
        if item.get("role") == "tool" and content and _matches(content, ERROR_PATTERNS):
            evidence["last_error_context"] = content[:200]
            break
    return evidence


def _collect_artifact(item, evidence):
    artifact_ref = str(item.get("artifact_ref", "")).strip()
    if artifact_ref and artifact_ref not in evidence["critical_artifacts"]:
        evidence["critical_artifacts"].append(artifact_ref)


def _collect_user_constraints(item, evidence):
    for sentence in _sentences(item.get("content", "")):
        if _matches(sentence, CONSTRAINT_PATTERNS):
            _add_unique(evidence["user_constraints"], sentence, 5)


def _collect_assistant_evidence(item, evidence):
    for sentence in _sentences(item.get("content", "")):
        if _matches(sentence, DECISION_PATTERNS):
            _add_unique(evidence["key_decisions"], sentence, 3)
        lowered = sentence.lower()
        if ("tried" in lowered and "but" in lowered) or _matches(sentence, REJECTED_PATTERNS):
            _add_unique(evidence["rejected_paths"], sentence, 3)


def _collect_tool_evidence(item, evidence):
    path = str(item.get("args", {}).get("path", "")).strip()
    if item.get("name") == "read_file" and path:
        evidence["files_read"].append(path)
    if item.get("name") in {"write_file", "patch_file"} and path:
        evidence["files_modified"].append(path)


def _sentences(text):
    parts = re.split(r"[。！？!?]+|\n+|\.(?:\s+|$)", str(text))
    return [part.strip(" \t\r\n:;,.，；、") for part in parts if part.strip()]


def _matches(text, patterns):
    lowered = text.lower()
    return any(str(pattern).lower() in lowered for pattern in patterns)


def _add_unique(values, value, limit):
    value = value.strip()
    if value and value not in values and len(values) < limit:
        values.append(value)


def _joined(values):
    return "; ".join(values) if values else "-"
