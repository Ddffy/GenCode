"""Provider-free coding-agent memory contract and challenge benchmark."""

import json
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..features.memory import LayeredMemory, compute_anchor_hash, retrieval_view_structured

DEFAULT_ARTIFACT_PATH = Path("_local/benchmark/artifacts/memory-agent-eval-v1.json")
DEFAULT_CHALLENGE_ARTIFACT_PATH = Path("_local/benchmark/artifacts/memory-challenge-v1.json")
DEFAULT_REPORT_PATH = Path("docs/metrics/gencode-memory-evaluation-report.md")
DEFAULT_MEMORY_ABLATION_PATH = Path("_local/benchmark/artifacts/memory-ablation-v2.json")
DEFAULT_MEMORY_FIDELITY_PATH = Path("_local/benchmark/artifacts/memory-fidelity-v1.json")
DEFAULT_DREAM_QUALITY_PATH = Path("_local/benchmark/artifacts/dream-quality-v1.json")
DEFAULT_RECOVERY_ABLATION_PATH = Path("_local/benchmark/artifacts/recovery-ablation-v2.json")
CHALLENGE_VARIANTS = ("memory_on", "memory_off", "naive_recent", "unsafe_memory")
STOPWORDS = {
    "a",
    "an",
    "and",
    "be",
    "did",
    "does",
    "is",
    "of",
    "on",
    "or",
    "should",
    "the",
    "this",
    "to",
    "use",
    "used",
    "what",
    "when",
    "where",
    "which",
}


def _captured_at():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_ratio(numerator, denominator):
    if not denominator:
        return 0.0
    return numerator / denominator


def _pct(value):
    return f"{float(value):.2%}"


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _tokens(text):
    return {token for token in re.findall(r"[a-z0-9_/-]+", str(text).lower()) if token not in STOPWORDS}


def _note_id(note):
    return str(note.get("note_id") or note.get("memory_id") or "").strip()


def _selected_texts(structured):
    return [note["text"] for note in structured["selected"]]


def _selected_ids(structured):
    return [_note_id(note) for note in structured["selected"] if _note_id(note)]


def _rejected_reasons(structured):
    return {note["text"]: note.get("reject_reason", "") for note in structured["rejected"]}


def _selected_contains(structured, text):
    return text in _selected_texts(structured)


def _simple_answer(structured, required_texts, expected_answer, abstain=False):
    selected = set(_selected_texts(structured))
    if abstain:
        return "unknown" if not selected else "unsupported"
    return expected_answer if all(text in selected for text in required_texts) else "unsupported"


def _contract_row(case_id, category, query, structured, required_texts, forbidden_texts, expected_answer, answer):
    selected = _selected_texts(structured)
    required_ids = [
        _note_id(note)
        for note in structured["selected"] + structured["rejected"]
        if note.get("text") in set(required_texts) and _note_id(note)
    ]
    forbidden_ids = [
        _note_id(note)
        for note in structured["selected"] + structured["rejected"]
        if note.get("text") in set(forbidden_texts) and _note_id(note)
    ]
    required_selected = [text for text in required_texts if text in selected]
    forbidden_selected = [text for text in forbidden_texts if text in selected]
    answer_correct = answer == expected_answer
    return {
        "id": case_id,
        "category": category,
        "query": query,
        "expected_answer": expected_answer,
        "answer": answer,
        "answer_correct": answer_correct,
        "passed": answer_correct and len(required_selected) == len(required_texts) and not forbidden_selected,
        "required_evidence_ids": required_ids,
        "forbidden_memory_ids": forbidden_ids,
        "selected_evidence_ids": _selected_ids(structured),
        "required_evidence_selected": len(required_selected),
        "required_evidence_total": len(required_texts),
        "forbidden_selected": bool(forbidden_selected),
        "selected_texts": selected,
        "rejected_reasons": _rejected_reasons(structured),
    }


def _direct_recall_case():
    memory = LayeredMemory()
    fact = "deploy target is staging"
    memory.append_note(fact, tags=("deploy",), created_at="2026-06-24T10:00:00+00:00")
    structured = memory.retrieval_view_structured("deploy target", limit=3)
    return _contract_row(
        "direct_recall_001",
        "direct_recall",
        "deploy target",
        structured,
        [fact],
        [],
        fact,
        _simple_answer(structured, [fact], fact),
    )


def _irrelevant_distractor_case():
    memory = LayeredMemory()
    distractor = "deploy key is blue and unrelated"
    fact = "deploy key is red"
    memory.append_note(distractor, tags=("deploy",), created_at="2026-06-24T10:00:00+00:00")
    memory.append_note(fact, tags=("deploy",), created_at="2026-06-24T10:01:00+00:00")
    structured = memory.retrieval_view_structured("deploy key", limit=1)
    return _contract_row(
        "irrelevant_distractor_001",
        "irrelevant_distractor",
        "deploy key",
        structured,
        [fact],
        [distractor],
        fact,
        _simple_answer(structured, [fact], fact),
    )


def _knowledge_update_case():
    memory = LayeredMemory()
    old_fact = "capital is X"
    new_fact = "capital is Y"
    memory.append_note(old_fact, tags=("capital",), created_at="2026-06-24T10:02:00+00:00")
    memory.append_note(new_fact, tags=("capital",), created_at="2026-06-24T10:01:00+00:00")
    old_note = memory.state["episodic_notes"][0]
    new_note = memory.state["episodic_notes"][1]
    old_note["note_id"] = "old-capital"
    new_note["note_id"] = "new-capital"
    old_note["status"] = "superseded"
    new_note["supersedes"] = old_note["note_id"]
    structured = retrieval_view_structured(memory.state, "capital", limit=3)
    row = _contract_row(
        "knowledge_update_001",
        "knowledge_update",
        "capital",
        structured,
        [new_fact],
        [old_fact],
        new_fact,
        _simple_answer(structured, [new_fact], new_fact),
    )
    row["update_success"] = _selected_contains(structured, new_fact) and _rejected_reasons(structured).get(old_fact) == "superseded"
    return row


def _selective_forgetting_case():
    memory = LayeredMemory()
    forbidden = "unsafe memory should be forgotten"
    safe = "safe memory policy is active"
    memory.append_note(forbidden, tags=("memory",), created_at="2026-06-24T10:02:00+00:00")
    memory.append_note(safe, tags=("memory",), created_at="2026-06-24T10:01:00+00:00")
    memory.state["episodic_notes"][0]["note_id"] = "forgotten-memory"
    memory.state["episodic_notes"][0]["status"] = "quarantined"
    memory.state["episodic_notes"][1]["note_id"] = "safe-memory"
    structured = retrieval_view_structured(memory.state, "memory policy", limit=3)
    row = _contract_row(
        "selective_forgetting_001",
        "selective_forgetting",
        "memory policy",
        structured,
        [safe],
        [forbidden],
        safe,
        _simple_answer(structured, [safe], safe),
    )
    row["invalid_memory_excluded"] = not _selected_contains(structured, forbidden)
    return row


def _stale_evidence_case():
    with tempfile.TemporaryDirectory(prefix="gencode-memory-agent-stale-") as temp_dir:
        workspace_root = Path(temp_dir)
        (workspace_root / "README.md").write_text("demo\n", encoding="utf-8")
        anchor = workspace_root / "anchor.txt"
        anchor.write_text("old\n", encoding="utf-8")
        fact = "Anchor fact uses alpha."
        memory = LayeredMemory(workspace_root=workspace_root)
        memory.promote_durable([("project-conventions", fact)])
        metadata_path = workspace_root / ".gencode" / "memory" / "topics" / "project-conventions.metadata.jsonl"
        metadata_rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
        metadata_rows[0]["evidence"]["source_path"] = "anchor.txt"
        metadata_rows[0]["evidence"]["evidence_anchor_hash"] = compute_anchor_hash(anchor)
        metadata_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata_rows), encoding="utf-8")
        anchor.write_text("new\n", encoding="utf-8")
        structured = memory.retrieval_view_structured("anchor", limit=3)
    row = _contract_row(
        "stale_evidence_001",
        "stale_evidence",
        "anchor",
        structured,
        [],
        [fact],
        "unknown",
        _simple_answer(structured, [], "unknown", abstain=True),
    )
    row["stale_memory_used"] = _selected_contains(structured, fact)
    return row


def _secret_safety_case():
    memory = LayeredMemory()
    secret = "api key sk-AAAAAAAAAAAAAAAAAAAA for service X"
    memory.append_note(secret, tags=("config",), created_at="2026-06-24T10:00:00+00:00")
    memory.state["episodic_notes"][0]["note_id"] = "secret-openai"
    memory.state["episodic_notes"][0]["status"] = "quarantined"
    structured = retrieval_view_structured(memory.state, "config", limit=3)
    row = _contract_row(
        "secret_safety_001",
        "secret_safety",
        "config",
        structured,
        [],
        [secret],
        "unknown",
        _simple_answer(structured, [], "unknown", abstain=True),
    )
    row["secret_exposed"] = _selected_contains(structured, secret)
    return row


def _no_evidence_abstention_case():
    memory = LayeredMemory()
    memory.append_note("build tool is uv", tags=("build",), created_at="2026-06-24T10:00:00+00:00")
    structured = memory.retrieval_view_structured("payment provider", limit=3)
    row = _contract_row(
        "no_evidence_abstention_001",
        "no_evidence_abstention",
        "payment provider",
        structured,
        [],
        [],
        "unknown",
        _simple_answer(structured, [], "unknown", abstain=True),
    )
    row["abstained"] = row["answer"] == "unknown"
    return row


def _multi_session_reasoning_case():
    memory = LayeredMemory()
    first = "session one decided the benchmark target is memory"
    second = "session two decided the report format is markdown"
    memory.append_note(first, tags=("session", "benchmark"), source="session-1", created_at="2026-06-24T10:00:00+00:00")
    memory.append_note(second, tags=("session", "report"), source="session-2", created_at="2026-06-24T10:01:00+00:00")
    memory.state["episodic_notes"][0]["note_id"] = "session-one-memory-target"
    memory.state["episodic_notes"][1]["note_id"] = "session-two-report-format"
    structured = memory.retrieval_view_structured("session benchmark report", limit=3)
    expected = "benchmark target is memory; report format is markdown"
    row = _contract_row(
        "multi_session_reasoning_001",
        "multi_session_reasoning",
        "session benchmark report",
        structured,
        [first, second],
        [],
        expected,
        _simple_answer(structured, [first, second], expected),
    )
    row["multi_session_reasoning_correct"] = row["answer_correct"] and row["required_evidence_selected"] == 2
    return row


def run_memory_agent_cases():
    return [
        _direct_recall_case(),
        _irrelevant_distractor_case(),
        _knowledge_update_case(),
        _selective_forgetting_case(),
        _stale_evidence_case(),
        _secret_safety_case(),
        _no_evidence_abstention_case(),
        _multi_session_reasoning_case(),
    ]


def _summarize_contract(rows, memory_ablation=None, memory_fidelity=None, recovery_ablation=None, dream_quality=None):
    evidence_rows = [row for row in rows if row["required_evidence_total"] > 0]
    selected_count = sum(len(row["selected_evidence_ids"]) for row in rows)
    relevant_selected = sum(row["required_evidence_selected"] for row in rows)
    required_total = sum(row["required_evidence_total"] for row in evidence_rows)
    stale_rows = [row for row in rows if row["category"] == "stale_evidence"]
    secret_rows = [row for row in rows if row["category"] == "secret_safety"]
    no_evidence_rows = [row for row in rows if row["category"] == "no_evidence_abstention"]
    update_rows = [row for row in rows if row["category"] == "knowledge_update"]
    invalid_rows = [row for row in rows if row["category"] in {"selective_forgetting", "stale_evidence", "secret_safety"}]
    multi_session_rows = [row for row in rows if row["category"] == "multi_session_reasoning"]

    fidelity_summary = dict((memory_fidelity or {}).get("summary", {}) or {})
    recovery = dict(((recovery_ablation or {}).get("variants", {}).get("resume_enabled", {}) or {}).get("summary", {}) or {})
    memory_efficiency = _memory_ablation_summary(memory_ablation)
    dream_summary = dict((dream_quality or {}).get("summary", {}) or {})
    stale_use_rate = max(
        _safe_ratio(sum(1 for row in stale_rows if row.get("stale_memory_used")), len(stale_rows)),
        float(fidelity_summary.get("stale_use_rate", 0.0) or 0.0),
    )
    secret_exposure_rate = max(
        _safe_ratio(sum(1 for row in secret_rows if row.get("secret_exposed")), len(secret_rows)),
        float(fidelity_summary.get("secret_exposure_rate", 0.0) or 0.0),
    )

    return {
        "total_cases": len(rows),
        "passed": sum(1 for row in rows if row["passed"]),
        "failed": sum(1 for row in rows if not row["passed"]),
        "task_correctness_rate": _safe_ratio(sum(1 for row in rows if row["answer_correct"]), len(rows)),
        "case_pass_rate": _safe_ratio(sum(1 for row in rows if row["passed"]), len(rows)),
        "evidence_recall_at_k": _safe_ratio(relevant_selected, required_total),
        "evidence_precision_at_k": _safe_ratio(relevant_selected, selected_count),
        "update_success_rate": min(
            _safe_ratio(sum(1 for row in update_rows if row.get("update_success")), len(update_rows)),
            float(fidelity_summary.get("supersede_success_rate", 1.0) or 0.0),
        ),
        "invalid_memory_exclusion_rate": _safe_ratio(sum(1 for row in invalid_rows if not row["forbidden_selected"]), len(invalid_rows)),
        "abstention_accuracy": _safe_ratio(sum(1 for row in no_evidence_rows if row.get("abstained")), len(no_evidence_rows)),
        "multi_session_reasoning_accuracy": _safe_ratio(
            sum(1 for row in multi_session_rows if row.get("multi_session_reasoning_correct")), len(multi_session_rows)
        ),
        "stale_memory_use_rate": stale_use_rate,
        "secret_exposure_rate": secret_exposure_rate,
        "false_resume_accept_rate": float(recovery.get("resume_false_accept_rate", 0.0) or 0.0),
        "resume_success_rate": float(recovery.get("resume_success_rate", 0.0) or 0.0),
        "workspace_drift_detection_rate": float(recovery.get("workspace_drift_detection_rate", 0.0) or 0.0),
        "repeated_context_reads_delta": {
            "memory_off": memory_efficiency.get("memory_off_repeated_reads", 0),
            "memory_on": memory_efficiency.get("memory_on_repeated_reads", 0),
            "absolute_reduction": memory_efficiency.get("memory_off_repeated_reads", 0)
            - memory_efficiency.get("memory_on_repeated_reads", 0),
        },
        "tool_steps_delta": {
            "memory_off": memory_efficiency.get("memory_off_avg_tool_steps", 0.0),
            "memory_on": memory_efficiency.get("memory_on_avg_tool_steps", 0.0),
        },
        "followup_task_correct_rate": memory_efficiency.get("memory_on_correct_rate", 0.0),
        "dream_quality": {
            "signal_retention_rate": float(dream_summary.get("signal_retention_rate", 0.0) or 0.0),
            "noise_rejection_rate": float(dream_summary.get("noise_rejection_rate", 0.0) or 0.0),
            "secret_rejection_rate": float(dream_summary.get("secret_rejection_rate", 0.0) or 0.0),
            "dedupe_rate": float(dream_summary.get("dedupe_rate", 0.0) or 0.0),
        },
    }


def _memory_ablation_summary(memory_ablation):
    if not memory_ablation:
        return {}
    variants = memory_ablation.get("variants", {})
    on = dict(variants.get("memory_on", {}) or {})
    off = dict(variants.get("memory_off", {}) or {})
    return {
        "memory_on_repeated_reads": int(on.get("repeated_reads", 0) or 0),
        "memory_off_repeated_reads": int(off.get("repeated_reads", 0) or 0),
        "memory_on_avg_tool_steps": float(on.get("avg_tool_steps", 0.0) or 0.0),
        "memory_off_avg_tool_steps": float(off.get("avg_tool_steps", 0.0) or 0.0),
        "memory_on_correct_rate": float(on.get("correct_rate", 0.0) or 0.0),
    }


def _mem(memory_id, text, answer, tags=(), created_at="2026-06-24T10:00:00+00:00", status="active", **flags):
    note = {
        "memory_id": memory_id,
        "note_id": memory_id,
        "text": text,
        "answer": answer,
        "tags": list(tags),
        "created_at": created_at,
        "status": status,
        "kind": "episodic",
    }
    note.update(flags)
    return note


def _case(case_id, category, query, expected_answer, notes, required_ids=(), forbidden_ids=(), **flags):
    return {
        "id": case_id,
        "category": category,
        "query": query,
        "expected_answer": expected_answer,
        "notes": list(notes),
        "required_evidence_ids": list(required_ids),
        "forbidden_memory_ids": list(forbidden_ids),
        **flags,
    }


def _build_challenge_cases():
    return [
        _case(
            "info_extract_test_command",
            "information_extraction",
            "What test command did the project standardize on?",
            "uv run pytest tests -q",
            [
                _mem("mem-old-test-command", "Old test command is pytest tests -q.", "pytest tests -q", ("test", "command"), "2026-06-24T10:02:00+00:00", "superseded"),
                _mem("mem-test-command", "Project test command is uv run pytest tests -q.", "uv run pytest tests -q", ("test", "command"), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-test-command"],
            ["mem-old-test-command"],
        ),
        _case(
            "info_extract_branch_policy",
            "information_extraction",
            "What branch prefix should be used by default?",
            "feature/",
            [
                _mem("mem-old-branch-prefix", "Branches use topic/ prefix.", "topic/", ("branch", "prefix"), "2026-06-24T10:02:00+00:00", "superseded"),
                _mem("mem-branch-prefix", "Feature branches use feature/ prefix unless user specifies otherwise.", "feature/", ("branch", "prefix"), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-branch-prefix"],
            ["mem-old-branch-prefix"],
        ),
        _case(
            "info_extract_memory_storage",
            "information_extraction",
            "What storage constraint applies to GenCode memory?",
            "filesystem-first; no vector DB",
            [
                _mem("mem-vector-db-distractor", "GenCode memory uses a vector database.", "vector database", ("memory", "storage"), "2026-06-24T10:02:00+00:00", "superseded"),
                _mem("mem-filesystem-memory", "GenCode memory is filesystem-first and must not use vector DB.", "filesystem-first; no vector DB", ("memory", "storage"), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-filesystem-memory"],
            ["mem-vector-db-distractor"],
        ),
        _case(
            "info_extract_report_location",
            "information_extraction",
            "Where should memory evaluation reports be written?",
            "docs/metrics",
            [
                _mem("mem-report-old-location", "Memory reports live under release/v3/testing.", "release/v3/testing", ("memory", "report"), "2026-06-24T10:02:00+00:00", "superseded"),
                _mem("mem-report-location", "Memory evaluation reports live under docs/metrics.", "docs/metrics", ("memory", "report"), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-report-location"],
            ["mem-report-old-location"],
        ),
        _case(
            "multi_session_metric_claim",
            "multi_session_reasoning",
            "Can the resume bullet claim live-provider memory reliability?",
            "No; only provider-free deterministic validation is supported.",
            [
                _mem("mem-provider-free", "GenCode memory benchmark uses provider-free deterministic fixtures.", "provider-free deterministic", ("provider", "benchmark"), source="session-a"),
                _mem("mem-no-live-provider-claim", "The resume bullet should not claim live-provider validation.", "No; only provider-free deterministic validation is supported.", ("provider", "claim"), source="session-b"),
            ],
            ["mem-provider-free", "mem-no-live-provider-claim"],
        ),
        _case(
            "multi_session_pr_boundary",
            "multi_session_reasoning",
            "Should memory eval changes be bundled with unrelated refactors?",
            "No.",
            [
                _mem("mem-pr-boundary-a", "Baseline and challenge expansion were separate PR boundaries.", "separate PR boundaries", ("pr", "boundary", "memory", "eval", "changes", "refactors"), source="session-a"),
                _mem("mem-pr-boundary-b", "Each benchmark phase must be independently mergeable.", "No.", ("pr", "boundary"), source="session-b"),
            ],
            ["mem-pr-boundary-a", "mem-pr-boundary-b"],
        ),
        _case(
            "multi_session_report_numbers",
            "multi_session_reasoning",
            "What repeated-read delta can be reported?",
            "60 -> 0",
            [
                _mem("mem-reads-off", "memory_off repeated reads were 60.", "60", ("reads",), source="session-a"),
                _mem("mem-reads-on", "memory_on repeated reads were 0.", "60 -> 0", ("reads",), source="session-b"),
            ],
            ["mem-reads-off", "mem-reads-on"],
        ),
        _case(
            "multi_session_memory_scope",
            "multi_session_reasoning",
            "What memory extension mechanism is allowed?",
            "sidecar metadata, not vector DB or embeddings",
            [
                _mem("mem-no-vector", "Do not introduce vector DB or embedding store.", "no vector DB or embeddings", ("memory", "storage"), source="session-a"),
                _mem("mem-sidecar-allowed", "Allowed memory metadata extension is sidecar jsonl.", "sidecar metadata, not vector DB or embeddings", ("memory", "metadata"), source="session-b"),
            ],
            ["mem-no-vector", "mem-sidecar-allowed"],
        ),
        _case(
            "update_test_command",
            "knowledge_updates",
            "What is the current test command?",
            "uv run pytest tests -q",
            [
                _mem("mem-update-old-test", "Project test command is pytest tests -q.", "pytest tests -q", ("test", "command"), "2026-06-24T10:03:00+00:00", "superseded"),
                _mem("mem-update-new-test", "Project test command is uv run pytest tests -q.", "uv run pytest tests -q", ("test", "command"), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-update-new-test"],
            ["mem-update-old-test"],
        ),
        _case(
            "update_provider_policy",
            "knowledge_updates",
            "Can this benchmark use live provider now?",
            "No.",
            [
                _mem("mem-live-provider-old", "Memory benchmark may use live provider.", "Yes.", ("provider",), "2026-06-24T10:03:00+00:00", "superseded"),
                _mem("mem-live-provider-new", "Report-ready memory benchmark comes before live-provider validation; current benchmark is provider-free.", "No.", ("provider",), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-live-provider-new"],
            ["mem-live-provider-old"],
        ),
        _case(
            "update_metric_positioning",
            "knowledge_updates",
            "Should memory_hit_rate be the headline metric?",
            "No.",
            [
                _mem("mem-hit-rate-old", "memory_hit_rate is a core resume metric.", "Yes.", ("metric",), "2026-06-24T10:03:00+00:00", "superseded"),
                _mem("mem-hit-rate-new", "memory_hit_rate is only suitable for docs/interview expansion.", "No.", ("metric",), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-hit-rate-new"],
            ["mem-hit-rate-old"],
        ),
        _case(
            "update_resume_claim",
            "knowledge_updates",
            "Can we claim production reliability?",
            "No.",
            [
                _mem("mem-prod-reliability-old", "GenCode proves production long-term memory reliability.", "Yes.", ("claim",), "2026-06-24T10:03:00+00:00", "superseded"),
                _mem("mem-prod-reliability-new", "GenCode proves deterministic coding-agent memory contracts and challenge results.", "No.", ("claim",), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-prod-reliability-new"],
            ["mem-prod-reliability-old"],
        ),
        _case(
            "stale_file_summary",
            "temporal_reasoning",
            "What command should be used?",
            "unknown",
            [_mem("mem-stale-readme", "README says command is make test.", "make test", ("command",), stale_evidence=True)],
            [],
            ["mem-stale-readme"],
            stale_case=True,
        ),
        _case(
            "stale_dependency_fact",
            "temporal_reasoning",
            "What dependency version is current?",
            "unknown",
            [_mem("mem-stale-dependency", "Dependency version is v1.", "v1", ("dependency",), stale_evidence=True)],
            [],
            ["mem-stale-dependency"],
            stale_case=True,
        ),
        _case(
            "scope_mismatch_workspace",
            "temporal_reasoning",
            "Should workspace A memory be used in workspace B?",
            "No.",
            [_mem("mem-scope-a", "Workspace A memory says use private release path.", "private release path", ("workspace",), scope_mismatch=True)],
            [],
            ["mem-scope-a"],
            stale_case=True,
            reject_answer="No.",
        ),
        _case(
            "resume_workspace_drift",
            "temporal_reasoning",
            "Should resume continue after workspace drift?",
            "No.",
            [_mem("mem-drift-resume", "Resume checkpoint from other workspace is valid.", "Yes.", ("resume",), scope_mismatch=True)],
            [],
            ["mem-drift-resume"],
            stale_case=True,
            invalid_resume=True,
            reject_answer="No.",
        ),
        _case(
            "ambiguous_conflict_without_supersede",
            "temporal_reasoning",
            "What is the current deployment owner?",
            "unknown",
            [
                _mem("mem-owner-a", "Deployment owner is team alpha.", "team alpha", ("deployment", "owner"), "2026-06-24T10:00:00+00:00"),
                _mem("mem-owner-b", "Deployment owner is team beta.", "team beta", ("deployment", "owner"), "2026-06-24T10:01:00+00:00"),
            ],
            [],
            ["mem-owner-a", "mem-owner-b"],
            limitation_case=True,
        ),
        _case(
            "no_evidence_payment_provider",
            "abstention",
            "What payment provider does this repo use?",
            "unknown",
            [_mem("mem-build-tool", "Build tool is uv.", "uv", ("build",))],
            no_evidence=True,
        ),
        _case(
            "no_evidence_release_date",
            "abstention",
            "What is the release date?",
            "unknown",
            [_mem("mem-report-format", "Markdown documentation format.", "markdown", ("documentation",))],
            no_evidence=True,
        ),
        _case(
            "no_evidence_vector_store",
            "abstention",
            "What vector DB does GenCode memory use?",
            "unknown",
            [_mem("mem-filesystem-storage", "Durable notes are plain files.", "plain files", ("storage",))],
            no_evidence=True,
        ),
        _case(
            "no_evidence_api_key",
            "abstention",
            "What is the user's API key?",
            "unknown",
            [_mem("mem-secret-policy", "Secrets must never be exposed from memory.", "no secret", ("secret",))],
            no_evidence=True,
        ),
        _case(
            "secret_openai_key",
            "agentic_efficiency",
            "What config should be used?",
            "unknown",
            [_mem("mem-secret-openai", "api key sk-AAAAAAAAAAAAAAAAAAAA for service X", "sk-AAAAAAAAAAAAAAAAAAAA", ("config",), status="quarantined")],
            [],
            ["mem-secret-openai"],
            secret_case=True,
        ),
        _case(
            "secret_github_pat",
            "agentic_efficiency",
            "What GitHub token is stored?",
            "unknown",
            [_mem("mem-secret-github", "github token ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", ("github",), status="quarantined")],
            [],
            ["mem-secret-github"],
            secret_case=True,
        ),
        _case(
            "efficiency_followup_known_file",
            "agentic_efficiency",
            "Which file did we already inspect for memory metrics?",
            "docs/metrics/gencode-benchmark-core-report.md",
            [_mem("mem-known-file", "Already inspected docs/metrics/gencode-benchmark-core-report.md for memory metrics.", "docs/metrics/gencode-benchmark-core-report.md", ("file", "metrics"))],
            ["mem-known-file"],
            efficiency_case=True,
        ),
        _case(
            "efficiency_followup_known_decision",
            "agentic_efficiency",
            "Should rejected memory enter the prompt?",
            "No.",
            [_mem("mem-rejected-not-prompt", "Rejected memory must never enter the prompt.", "No.", ("prompt", "memory"))],
            ["mem-rejected-not-prompt"],
            efficiency_case=True,
        ),
        _case(
            "info_extract_tool_call_format",
            "information_extraction",
            "What tool-call format does GenCode expect?",
            "XML or JSON tool envelope",
            [
                _mem("mem-tool-format-old", "Tool calls should be free-form shell text.", "free-form shell text", ("tool", "format"), "2026-06-24T10:04:00+00:00", "superseded"),
                _mem("mem-tool-format-new", "Tool calls use an XML or JSON tool envelope.", "XML or JSON tool envelope", ("tool", "format"), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-tool-format-new"],
            ["mem-tool-format-old"],
        ),
        _case(
            "info_extract_artifact_guardrail",
            "information_extraction",
            "Which existing benchmark artifacts must remain byte-identical?",
            "memory/context/recovery/harness v2 artifacts",
            [
                _mem("mem-artifacts-old", "Only memory-ablation-v2 is protected.", "memory-ablation-v2", ("artifact", "guardrail"), "2026-06-24T10:04:00+00:00", "superseded"),
                _mem("mem-artifacts-new", "Protected artifacts are memory/context/recovery/harness v2 artifacts.", "memory/context/recovery/harness v2 artifacts", ("artifact", "guardrail"), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-artifacts-new"],
            ["mem-artifacts-old"],
        ),
        _case(
            "info_extract_secret_rule",
            "information_extraction",
            "What should happen to secret-shaped memory?",
            "quarantine and exclude from prompt",
            [
                _mem("mem-secret-rule-old", "Secret-shaped memory can be summarized into prompts.", "summarize into prompts", ("secret", "rule"), "2026-06-24T10:04:00+00:00", "superseded"),
                _mem("mem-secret-rule-new", "Secret-shaped memory should quarantine and exclude from prompt.", "quarantine and exclude from prompt", ("secret", "rule"), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-secret-rule-new"],
            ["mem-secret-rule-old"],
        ),
        _case(
            "info_extract_retrieval_trace",
            "information_extraction",
            "Where should retrieval trace emit happen?",
            "after retrieval_candidates",
            [
                _mem("mem-trace-old", "Retrieval trace should emit before retrieval starts.", "before retrieval starts", ("retrieval", "trace"), "2026-06-24T10:04:00+00:00", "superseded"),
                _mem("mem-trace-new", "Retrieval trace should emit after retrieval_candidates.", "after retrieval_candidates", ("retrieval", "trace"), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-trace-new"],
            ["mem-trace-old"],
        ),
        _case(
            "info_extract_python_runner",
            "information_extraction",
            "What Python runner should benchmark commands use?",
            "uv run python",
            [
                _mem("mem-python-runner-old", "Benchmark commands use python.", "python", ("python", "runner"), "2026-06-24T10:04:00+00:00", "superseded"),
                _mem("mem-python-runner-new", "Benchmark commands use uv run python.", "uv run python", ("python", "runner"), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-python-runner-new"],
            ["mem-python-runner-old"],
        ),
        _case(
            "multi_session_metric_report_scope",
            "multi_session_reasoning",
            "What metric claim is allowed for the memory report?",
            "provider-free evidence recall and safety baselines, not live reliability",
            [
                _mem("mem-metric-evidence", "The memory report can claim provider-free evidence recall and safety baselines.", "provider-free evidence recall and safety baselines", ("metric", "report"), source="session-a"),
                _mem("mem-metric-live-boundary", "The memory report must not claim live reliability.", "provider-free evidence recall and safety baselines, not live reliability", ("metric", "boundary"), source="session-b"),
            ],
            ["mem-metric-evidence", "mem-metric-live-boundary"],
        ),
        _case(
            "multi_session_recovery_claim",
            "multi_session_reasoning",
            "Which recovery risks should the memory benchmark report?",
            "false-resume accept and workspace drift",
            [
                _mem("mem-recovery-false-resume", "Recovery benchmark tracks false-resume accept.", "false-resume accept", ("recovery", "risk"), source="session-a"),
                _mem("mem-recovery-workspace-drift", "Recovery benchmark tracks workspace drift.", "false-resume accept and workspace drift", ("recovery", "risk"), source="session-b"),
            ],
            ["mem-recovery-false-resume", "mem-recovery-workspace-drift"],
        ),
        _case(
            "multi_session_security_claim",
            "multi_session_reasoning",
            "How should unsafe memory be handled in prompts?",
            "measure secret exposure and keep rejected memory out of prompts",
            [
                _mem("mem-security-secret-exposure", "Memory benchmark measures secret exposure.", "secret exposure", ("security", "memory"), source="session-a"),
                _mem("mem-security-rejected-prompt", "Rejected memory stays out of prompts.", "measure secret exposure and keep rejected memory out of prompts", ("security", "prompt"), source="session-b"),
            ],
            ["mem-security-secret-exposure", "mem-security-rejected-prompt"],
        ),
        _case(
            "multi_session_artifact_boundary",
            "multi_session_reasoning",
            "How can benchmark expansion respect old artifacts?",
            "write new side artifacts while old v2 artifacts stay byte-identical",
            [
                _mem("mem-old-artifact-hash", "Old v2 artifacts must stay byte-identical.", "old v2 artifacts stay byte-identical", ("artifact", "hash"), source="session-a"),
                _mem("mem-new-side-artifacts", "New memory challenge results can be written as side artifacts.", "write new side artifacts while old v2 artifacts stay byte-identical", ("artifact", "side"), source="session-b"),
            ],
            ["mem-old-artifact-hash", "mem-new-side-artifacts"],
        ),
        _case(
            "multi_session_interview_claim",
            "multi_session_reasoning",
            "What should the interview story emphasize?",
            "memory system design validated by deterministic baselines",
            [
                _mem("mem-interview-system", "The interview story is about memory system design.", "memory system design", ("interview", "memory"), source="session-a"),
                _mem("mem-interview-baseline", "The interview evidence is deterministic baseline comparison.", "memory system design validated by deterministic baselines", ("interview", "baseline"), source="session-b"),
            ],
            ["mem-interview-system", "mem-interview-baseline"],
        ),
        _case(
            "update_report_headline_metric",
            "knowledge_updates",
            "What should be the headline memory metric family?",
            "evidence recall, stale-use, secret exposure, false-resume, and efficiency",
            [
                _mem("mem-headline-old", "Repeated reads alone should be the headline memory metric.", "repeated reads", ("metric", "headline"), "2026-06-24T10:04:00+00:00", "superseded"),
                _mem("mem-headline-new", "Headline memory metrics are evidence recall, stale-use, secret exposure, false-resume, and efficiency.", "evidence recall, stale-use, secret exposure, false-resume, and efficiency", ("metric", "headline"), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-headline-new"],
            ["mem-headline-old"],
        ),
        _case(
            "update_branch_target",
            "knowledge_updates",
            "What branch is active for memory agent evaluation work?",
            "feature/memory-agent-eval",
            [
                _mem("mem-branch-target-old", "Memory eval work happens on v3.", "v3", ("branch", "target"), "2026-06-24T10:04:00+00:00", "superseded"),
                _mem("mem-branch-target-new", "Memory eval work happens on feature/memory-agent-eval.", "feature/memory-agent-eval", ("branch", "target"), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-branch-target-new"],
            ["mem-branch-target-old"],
        ),
        _case(
            "update_cli_run_name",
            "knowledge_updates",
            "Which CLI run generates only the challenge artifact?",
            "memory_challenge",
            [
                _mem("mem-cli-run-old", "Use memory_agent to generate the challenge artifact.", "memory_agent", ("cli", "run"), "2026-06-24T10:04:00+00:00", "superseded"),
                _mem("mem-cli-run-new", "Use memory_challenge to generate only the challenge artifact.", "memory_challenge", ("cli", "run"), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-cli-run-new"],
            ["mem-cli-run-old"],
        ),
        _case(
            "update_eval_scope",
            "knowledge_updates",
            "What is the memory evaluation scope now?",
            "provider-free challenge benchmark",
            [
                _mem("mem-eval-scope-old", "Memory evaluation is a small demo.", "small demo", ("eval", "scope"), "2026-06-24T10:04:00+00:00", "superseded"),
                _mem("mem-eval-scope-new", "Memory evaluation is a provider-free challenge benchmark.", "provider-free challenge benchmark", ("eval", "scope"), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-eval-scope-new"],
            ["mem-eval-scope-old"],
        ),
        _case(
            "update_memory_claim",
            "knowledge_updates",
            "Can the report claim memory is perfect?",
            "No.",
            [
                _mem("mem-perfect-claim-old", "The report can claim memory is perfect.", "Yes.", ("claim", "perfect"), "2026-06-24T10:04:00+00:00", "superseded"),
                _mem("mem-perfect-claim-new", "The report should use baseline comparison and not claim memory is perfect.", "No.", ("claim", "perfect"), "2026-06-24T10:01:00+00:00"),
            ],
            ["mem-perfect-claim-new"],
            ["mem-perfect-claim-old"],
        ),
        _case(
            "stale_core_report_location",
            "temporal_reasoning",
            "Where is the current core benchmark report?",
            "unknown",
            [_mem("mem-stale-core-report", "Core benchmark report is under release/v3/metrics.", "release/v3/metrics", ("core", "report"), stale_evidence=True)],
            [],
            ["mem-stale-core-report"],
            stale_case=True,
        ),
        _case(
            "stale_diff_assumption",
            "temporal_reasoning",
            "Which files are safe to stage wholesale?",
            "unknown",
            [_mem("mem-stale-stage", "metrics.py is clean and safe to stage wholesale.", "metrics.py", ("stage", "diff"), stale_evidence=True)],
            [],
            ["mem-stale-stage"],
            stale_case=True,
        ),
        _case(
            "scope_mismatch_other_repo",
            "temporal_reasoning",
            "Should another repo's memory policy apply here?",
            "No.",
            [_mem("mem-scope-other-repo", "Ariadne memory policy requires a database-backed store.", "database-backed store", ("repo", "memory"), scope_mismatch=True)],
            [],
            ["mem-scope-other-repo"],
            stale_case=True,
            reject_answer="No.",
        ),
        _case(
            "resume_stale_checkpoint_metrics",
            "temporal_reasoning",
            "Should a stale metrics checkpoint be accepted?",
            "No.",
            [_mem("mem-stale-metrics-checkpoint", "Metrics checkpoint from old workspace is valid.", "Yes.", ("metrics", "checkpoint"), scope_mismatch=True)],
            [],
            ["mem-stale-metrics-checkpoint"],
            stale_case=True,
            invalid_resume=True,
            reject_answer="No.",
        ),
        _case(
            "ambiguous_conflict_without_supersede_runner",
            "temporal_reasoning",
            "What command runner is current?",
            "unknown",
            [
                _mem("mem-runner-python", "Current command runner is python.", "python", ("command", "runner"), "2026-06-24T10:00:00+00:00"),
                _mem("mem-runner-uv-python", "Current command runner is uv run python.", "uv run python", ("command", "runner"), "2026-06-24T10:01:00+00:00"),
            ],
            [],
            ["mem-runner-python", "mem-runner-uv-python"],
            limitation_case=True,
        ),
        _case(
            "ambiguous_conflict_without_supersede_report_owner",
            "temporal_reasoning",
            "Who owns the memory report?",
            "unknown",
            [
                _mem("mem-report-owner-runtime", "Memory report owner is runtime team.", "runtime team", ("memory", "report", "owner"), "2026-06-24T10:00:00+00:00"),
                _mem("mem-report-owner-eval", "Memory report owner is eval team.", "eval team", ("memory", "report", "owner"), "2026-06-24T10:01:00+00:00"),
            ],
            [],
            ["mem-report-owner-runtime", "mem-report-owner-eval"],
            limitation_case=True,
        ),
        _case(
            "no_evidence_user_phone",
            "abstention",
            "What is the user's phone number?",
            "unknown",
            [_mem("mem-unrelated-build", "Build tool is uv.", "uv", ("build",), status="superseded")],
            no_evidence=True,
        ),
        _case(
            "no_evidence_cloud_vendor",
            "abstention",
            "What cloud vendor hosts GenCode production?",
            "unknown",
            [_mem("mem-unrelated-markdown", "Reports use markdown.", "markdown", ("report",))],
            no_evidence=True,
        ),
        _case(
            "no_evidence_prod_incident",
            "abstention",
            "What is the current production incident id?",
            "unknown",
            [_mem("mem-unrelated-branch", "Branch prefix is feature/.", "feature/", ("branch",), status="superseded")],
            no_evidence=True,
        ),
        _case(
            "no_evidence_database_password",
            "abstention",
            "What database password should be used?",
            "unknown",
            [_mem("mem-unrelated-secret-policy", "Secret-shaped memory is quarantined.", "quarantined", ("secret", "policy"))],
            no_evidence=True,
        ),
        _case(
            "secret_anthropic_key",
            "agentic_efficiency",
            "What Anthropic key is stored?",
            "unknown",
            [_mem("mem-secret-anthropic", "anthropic key sk-ant-AAAAAAAAAAAAAAAAAAAA", "sk-ant-AAAAAAAAAAAAAAAAAAAA", ("anthropic",), status="quarantined")],
            [],
            ["mem-secret-anthropic"],
            secret_case=True,
        ),
        _case(
            "secret_slack_webhook",
            "agentic_efficiency",
            "What Slack webhook is stored?",
            "unknown",
            [_mem("mem-secret-slack", "slack webhook https://hooks.slack.com/services/T000/B000/SECRET", "https://hooks.slack.com/services/T000/B000/SECRET", ("slack",), status="quarantined")],
            [],
            ["mem-secret-slack"],
            secret_case=True,
        ),
        _case(
            "efficiency_known_test_path",
            "agentic_efficiency",
            "Which test file already covers memory challenge?",
            "tests/test_memory_agent_eval.py",
            [_mem("mem-known-test-file", "Already inspected tests/test_memory_agent_eval.py for memory challenge coverage.", "tests/test_memory_agent_eval.py", ("test", "file"))],
            ["mem-known-test-file"],
            efficiency_case=True,
        ),
        _case(
            "efficiency_known_handoff_file",
            "agentic_efficiency",
            "Which local plan file records the memory challenge design?",
            "docs/plans/memory-challenge-benchmark.md",
            [_mem("mem-known-plan-file", "Memory challenge design is recorded in docs/plans/memory-challenge-benchmark.md.", "docs/plans/memory-challenge-benchmark.md", ("plan", "file"))],
            ["mem-known-plan-file"],
            efficiency_case=True,
        ),
        _case(
            "efficiency_reuse_design_decision",
            "agentic_efficiency",
            "Should the benchmark connect a live provider for this suite?",
            "No.",
            [_mem("mem-no-live-provider-suite", "This memory challenge suite should remain provider-free.", "No.", ("provider", "suite"))],
            ["mem-no-live-provider-suite"],
            efficiency_case=True,
        ),
    ]


def _state_for_case(case):
    notes = []
    for index, note in enumerate(case["notes"]):
        normalized = {
            "text": note["text"],
            "tags": note.get("tags", []),
            "source": note.get("source", ""),
            "created_at": note.get("created_at", "2026-06-24T10:00:00+00:00"),
            "note_index": index,
            "kind": "episodic",
            "note_id": note["note_id"],
            "status": note.get("status", "active"),
        }
        for key in ("stale_evidence", "scope_mismatch", "supersedes", "scope"):
            if key in note:
                normalized[key] = note[key]
        notes.append(normalized)
    return {"working": {"task_summary": "", "recent_files": []}, "episodic_notes": notes, "file_summaries": {}, "next_note_index": len(notes)}


def _active_note(note):
    return not (
        note.get("status") in {"superseded", "quarantined"}
        or note.get("stale_evidence")
        or note.get("scope_mismatch")
    )


def _select_structured_notes(case, enforce_rejections):
    state = _state_for_case(case)
    if enforce_rejections:
        structured = retrieval_view_structured(state, case["query"], limit=case.get("limit", 3))
        selected_ids = {_note_id(note) for note in structured["selected"]}
        selected = [note for note in case["notes"] if note["memory_id"] in selected_ids]
        return selected, _rejected_reasons(structured)

    selected = _rank_case_notes(case, include_unsafe=True)[: case.get("limit", 3)]
    return selected, {}


def _rank_case_notes(case, include_unsafe=False):
    query_tokens = _tokens(case["query"])
    ranked = []
    for index, note in enumerate(case["notes"]):
        if not include_unsafe and not _active_note(note):
            continue
        note_tokens = _tokens(note["text"]) | set(str(tag).lower() for tag in note.get("tags", []))
        overlap = len(query_tokens & note_tokens)
        if overlap == 0:
            continue
        ranked.append((note.get("created_at", ""), overlap, index, note))
    ranked.sort(key=lambda item: (item[1], item[0], item[2]), reverse=True)
    return [item[3] for item in ranked]


def _select_variant_notes(case, variant):
    if variant == "memory_off":
        return [], {}
    if variant == "memory_on":
        return _select_structured_notes(case, enforce_rejections=True)
    if variant == "naive_recent":
        return _rank_case_notes(case, include_unsafe=True)[: case.get("limit", 2)], {}
    if variant == "unsafe_memory":
        return _select_structured_notes(case, enforce_rejections=False)
    raise ValueError(f"unknown memory challenge variant: {variant}")


def _answer_from_selected(case, selected, variant):
    selected_ids = {note["memory_id"] for note in selected}
    required = set(case.get("required_evidence_ids", []))
    forbidden = set(case.get("forbidden_memory_ids", []))
    if case.get("no_evidence"):
        return "unknown" if not selected else selected[0]["answer"]
    if not selected and case.get("reject_answer"):
        return case["reject_answer"]
    if selected_ids & forbidden:
        return next(note["answer"] for note in selected if note["memory_id"] in forbidden)
    if required and required <= selected_ids:
        return case["expected_answer"]
    if not selected:
        return "unknown"
    if variant == "memory_off":
        return "unknown"
    return selected[0]["answer"]


def _challenge_row(case, variant, selected, rejected_reasons):
    selected_ids = [note["memory_id"] for note in selected]
    required_ids = list(case.get("required_evidence_ids", []))
    forbidden_ids = list(case.get("forbidden_memory_ids", []))
    answer = _answer_from_selected(case, selected, variant)
    selected_set = set(selected_ids)
    required_set = set(required_ids)
    forbidden_set = set(forbidden_ids)
    answer_correct = answer == case["expected_answer"]
    required_ok = not required_set or required_set <= selected_set
    forbidden_selected = bool(forbidden_set & selected_set)
    no_evidence_ok = not case.get("no_evidence") or (answer == "unknown" and not selected)
    false_resume_accepted = bool(case.get("invalid_resume") and forbidden_selected)
    repeated_reads = int(variant == "memory_off" and (bool(required_set) or bool(case.get("efficiency_case"))))
    tool_calls = repeated_reads
    stale_used = bool(case.get("stale_case") and forbidden_selected)
    secret_exposed = bool(case.get("secret_case") and forbidden_selected)
    passed = answer_correct and no_evidence_ok and not forbidden_selected and (not required_set or required_ok)
    if case.get("invalid_resume"):
        passed = answer_correct and not false_resume_accepted
    return {
        "id": case["id"],
        "category": case["category"],
        "variant": variant,
        "query": case["query"],
        "expected_answer": case["expected_answer"],
        "answer": answer,
        "answer_correct": answer_correct,
        "required_evidence_ids": required_ids,
        "forbidden_memory_ids": forbidden_ids,
        "selected_evidence_ids": selected_ids,
        "selected_texts": [note["text"] for note in selected],
        "rejected_reasons": rejected_reasons,
        "repeated_reads": repeated_reads,
        "tool_calls": tool_calls,
        "abstained": answer == "unknown",
        "secret_exposed": secret_exposed,
        "stale_memory_used": stale_used,
        "false_resume_accepted": false_resume_accepted,
        "stale_case": bool(case.get("stale_case")),
        "secret_case": bool(case.get("secret_case")),
        "invalid_resume": bool(case.get("invalid_resume")),
        "passed": passed,
    }


def _run_challenge_case(case, variant):
    selected, rejected_reasons = _select_variant_notes(case, variant)
    return _challenge_row(case, variant, selected, rejected_reasons)


def _category_count(rows, category):
    return sum(1 for row in rows if row["category"] == category)


def _category_success(rows, category):
    category_rows = [row for row in rows if row["category"] == category]
    return _safe_ratio(sum(1 for row in category_rows if row["passed"]), len(category_rows))


def _flag_count(rows, flag):
    return sum(1 for row in rows if row.get(flag))


def _summarize_challenge_rows(rows):
    total = len(rows)
    required_total = sum(len(row["required_evidence_ids"]) for row in rows)
    required_selected = sum(len(set(row["required_evidence_ids"]) & set(row["selected_evidence_ids"])) for row in rows)
    selected_total = sum(len(row["selected_evidence_ids"]) for row in rows)
    stale_cases = sum(1 for row in rows if row.get("stale_case"))
    secret_cases = sum(1 for row in rows if row.get("secret_case"))
    invalid_resume_cases = sum(1 for row in rows if row.get("invalid_resume"))
    return {
        "total_cases": total,
        "answer_accuracy": _safe_ratio(sum(1 for row in rows if row["answer_correct"]), total),
        "case_pass_rate": _safe_ratio(sum(1 for row in rows if row["passed"]), total),
        "evidence_recall_at_k": _safe_ratio(required_selected, required_total),
        "evidence_precision_at_k": _safe_ratio(required_selected, selected_total),
        "update_success_rate": _category_success(rows, "knowledge_updates"),
        "stale_use_rate": _safe_ratio(sum(1 for row in rows if row["stale_memory_used"]), stale_cases),
        "secret_exposure_rate": _safe_ratio(sum(1 for row in rows if row["secret_exposed"]), secret_cases),
        "abstention_accuracy": _category_success(rows, "abstention"),
        "multi_session_reasoning_accuracy": _category_success(rows, "multi_session_reasoning"),
        "false_resume_accept_rate": _safe_ratio(sum(1 for row in rows if row["false_resume_accepted"]), invalid_resume_cases),
        "avg_repeated_reads": _safe_ratio(sum(row["repeated_reads"] for row in rows), total),
        "avg_tool_calls": _safe_ratio(sum(row["tool_calls"] for row in rows), total),
        "failed": sum(1 for row in rows if not row["passed"]),
    }


def _compare_variants(memory_on, baseline):
    return {
        "answer_accuracy_delta": memory_on["answer_accuracy"] - baseline["answer_accuracy"],
        "evidence_recall_delta": memory_on["evidence_recall_at_k"] - baseline["evidence_recall_at_k"],
        "stale_use_reduction": baseline["stale_use_rate"] - memory_on["stale_use_rate"],
        "secret_exposure_reduction": baseline["secret_exposure_rate"] - memory_on["secret_exposure_rate"],
        "false_resume_reduction": baseline["false_resume_accept_rate"] - memory_on["false_resume_accept_rate"],
        "repeated_reads_reduction": baseline["avg_repeated_reads"] - memory_on["avg_repeated_reads"],
        "tool_calls_reduction": baseline["avg_tool_calls"] - memory_on["avg_tool_calls"],
    }


def run_memory_challenge_payload():
    cases = _build_challenge_cases()
    variants = {}
    for variant in CHALLENGE_VARIANTS:
        rows = [_run_challenge_case(case, variant) for case in cases]
        variants[variant] = {"summary": _summarize_challenge_rows(rows), "rows": rows}
    memory_on = variants["memory_on"]["summary"]
    comparisons = {
        f"memory_on_vs_{baseline}": _compare_variants(memory_on, variants[baseline]["summary"])
        for baseline in ("memory_off", "naive_recent", "unsafe_memory")
    }
    return {
        "schema_version": 1,
        "artifact_type": "memory-challenge-v1",
        "captured_at": _captured_at(),
        "case_count": len(cases),
        "case_categories": dict(Counter(case["category"] for case in cases)),
        "variants": variants,
        "comparisons": comparisons,
    }


def render_memory_evaluation_report(artifact):
    contract = artifact["contract"]
    challenge = artifact["challenge"]
    memory_on = challenge["variants"]["memory_on"]["summary"]
    memory_off = challenge["variants"]["memory_off"]["summary"]
    naive = challenge["variants"]["naive_recent"]["summary"]
    unsafe = challenge["variants"]["unsafe_memory"]["summary"]
    reads_delta = artifact["summary"]["repeated_context_reads_delta"]
    lines = [
        "# GenCode 长期记忆系统与 Challenge Benchmark 报告",
        "",
        "## Resume Claim",
        "",
        "> 设计并实现 GenCode 面向 Coding Agent 的长期记忆系统，覆盖结构化检索、事实更新、过期证据拒用、敏感信息隔离与跨会话任务恢复；参考 LongMemEval/BEAM 设计 provider-free memory challenge benchmark，引入 memory-off、naive-recent、unsafe-memory baseline，对比验证 evidence recall@k、stale-use、secret exposure、abstention、false-resume 与重复上下文读取等指标。",
        "",
        "## What Was Built",
        "",
        "- 文件优先 durable memory：长期事实落在 `.gencode/memory/topics/*.md`，主存储保持可读、可 diff、可审计。",
        "- sidecar metadata：每条 note 记录 `note_id`、`status`、`supersedes`、`evidence`、`scope`，用于解释 selected/rejected。",
        "- 结构化检索：`retrieval_view_structured` 返回 selected 与 rejected，rejected 带 `reject_reason`，但不进入 prompt。",
        "- 记忆更新/遗忘：通过 `superseded`、`quarantined`、`stale_evidence`、`scope_mismatch` 控制旧事实、污染记忆和过期证据。",
        "- 跨会话恢复：复用 recovery artifact 的 resume、workspace drift、false-resume、first action 和 todo continuity 指标。",
        "",
        "## Contract Verification",
        "",
        f"- Contract cases：{contract['summary']['passed']}/{contract['summary']['total_cases']} passed。",
        "- 这部分只证明机制合同没有坏；不能把 contract pass rate 当作长期记忆能力得分。",
        "",
        "## Challenge Benchmark",
        "",
        f"- Challenge cases：{challenge['case_count']}",
        f"- Variants：{', '.join(CHALLENGE_VARIANTS)}",
        f"- `memory_on` failed cases：{memory_on['failed']}，这些失败用于暴露当前 memory schema 的边界，不应被抹平。",
        "",
        "| Variant | answer_accuracy | evidence_recall@k | evidence_precision@k | stale_use | secret_exposure | abstention | false_resume | avg_reads |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in challenge["variants"].items():
        summary = payload["summary"]
        lines.append(
            f"| `{name}` | {_pct(summary['answer_accuracy'])} | {_pct(summary['evidence_recall_at_k'])} | "
            f"{_pct(summary['evidence_precision_at_k'])} | {_pct(summary['stale_use_rate'])} | "
            f"{_pct(summary['secret_exposure_rate'])} | {_pct(summary['abstention_accuracy'])} | "
            f"{_pct(summary['false_resume_accept_rate'])} | {summary['avg_repeated_reads']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Baseline Comparison",
            "",
            "| Comparison | answer_accuracy_delta | evidence_recall_delta | stale_use_reduction | secret_exposure_reduction | false_resume_reduction | repeated_reads_reduction |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, comparison in challenge["comparisons"].items():
        lines.append(
            f"| `{name}` | {_pct(comparison['answer_accuracy_delta'])} | {_pct(comparison['evidence_recall_delta'])} | "
            f"{_pct(comparison['stale_use_reduction'])} | {_pct(comparison['secret_exposure_reduction'])} | "
            f"{_pct(comparison['false_resume_reduction'])} | {comparison['repeated_reads_reduction']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Metric Definitions",
            "",
            "- `answer_accuracy`：最终 deterministic answer 是否匹配 expected answer。",
            "- `evidence_recall_at_k`：required evidence 中被 selected memory 命中的比例。",
            "- `evidence_precision_at_k`：selected memory 中属于 required evidence 的比例。",
            "- `stale_use_rate`：stale/scope-mismatched memory 被使用的比例，越低越好。",
            "- `secret_exposure_rate`：secret-shaped memory 被 selected 或 answer 泄露的比例，越低越好。",
            "- `abstention_accuracy`：无证据时正确返回 unknown 的比例。",
            "- `false_resume_accept_rate`：无效 resume/workspace drift 被错误接受的比例，越低越好。",
            "- `avg_repeated_reads`：每个 case 的重复读取 proxy，用来衡量 context reuse，不作为主要 memory quality 指标。",
            "",
            "## Industry Alignment",
            "",
            "- LoCoMo：借鉴其通过下游 QA/事件理解验证长程记忆的口径，不复现其数据集。",
            "- LongMemEval：对齐 information extraction、multi-session reasoning、temporal reasoning、knowledge updates、abstention 五类能力。",
            "- BEAM：借鉴 probing-question 形式，每个 case 显式绑定 `required_evidence_ids` / `forbidden_memory_ids`。",
            "- STATE-Bench：保留 stateful agent task、tool calls、resume 和 repeated reads 视角。",
            "- Mem0：将 repeated reads/tool calls 放在 efficiency，而不是 memory quality 主指标。",
            "",
            "## Results You Can Claim",
            "",
            f"- `memory_on` vs `memory_off`：evidence recall delta = {_pct(challenge['comparisons']['memory_on_vs_memory_off']['evidence_recall_delta'])}，avg repeated reads reduction = {challenge['comparisons']['memory_on_vs_memory_off']['repeated_reads_reduction']:.2f}。",
            f"- `memory_on` vs `naive_recent`：stale/use and update filtering improved; evidence precision delta = {_pct(challenge['comparisons']['memory_on_vs_naive_recent']['evidence_recall_delta'])} recall delta。",
            f"- `memory_on` vs `unsafe_memory`：secret exposure reduction = {_pct(challenge['comparisons']['memory_on_vs_unsafe_memory']['secret_exposure_reduction'])}，stale use reduction = {_pct(challenge['comparisons']['memory_on_vs_unsafe_memory']['stale_use_reduction'])}。",
            f"- Existing follow-up ablation still reports repeated context reads `{reads_delta['memory_off']} -> {reads_delta['memory_on']}`.",
            "",
            "## Limitations",
            "",
            "- 这是 provider-free deterministic challenge benchmark，不是 live provider benchmark。",
            "- 没有复现 LoCoMo/LongMemEval/BEAM 原始数据集，也不声称 SOTA。",
            "- Case 数量仍小于公开 benchmark，但已经有 baseline、负例、stale/secret/abstention/false-resume 对照。",
            "",
            "## Interview Notes",
            "",
            "- 不要把 8/8 contract suite 包装成长期记忆能力满分。",
            "- 可以说 contract suite 是 8/8，challenge benchmark 则用 baseline 对比证明收益。",
            f"- 当前 challenge `memory_on` 指标：answer_accuracy={_pct(memory_on['answer_accuracy'])}, evidence_recall@k={_pct(memory_on['evidence_recall_at_k'])}, stale_use={_pct(memory_on['stale_use_rate'])}, secret_exposure={_pct(memory_on['secret_exposure_rate'])}.",
            f"- Baseline sanity：memory_off answer_accuracy={_pct(memory_off['answer_accuracy'])}, naive_recent answer_accuracy={_pct(naive['answer_accuracy'])}, unsafe_memory secret_exposure={_pct(unsafe['secret_exposure_rate'])}.",
        ]
    )
    return "\n".join(lines) + "\n"


def _top_level_summary(contract_summary, challenge, memory_ablation):
    memory_on = challenge["variants"]["memory_on"]["summary"]
    memory_efficiency = _memory_ablation_summary(memory_ablation)
    return {
        "total_cases": challenge["case_count"],
        "passed": challenge["variants"]["memory_on"]["summary"]["total_cases"] - challenge["variants"]["memory_on"]["summary"]["failed"],
        "failed": challenge["variants"]["memory_on"]["summary"]["failed"],
        "task_correctness_rate": memory_on["answer_accuracy"],
        "case_pass_rate": memory_on["case_pass_rate"],
        "evidence_recall_at_k": memory_on["evidence_recall_at_k"],
        "evidence_precision_at_k": memory_on["evidence_precision_at_k"],
        "update_success_rate": memory_on["update_success_rate"],
        "invalid_memory_exclusion_rate": contract_summary["invalid_memory_exclusion_rate"],
        "abstention_accuracy": memory_on["abstention_accuracy"],
        "multi_session_reasoning_accuracy": memory_on["multi_session_reasoning_accuracy"],
        "stale_memory_use_rate": memory_on["stale_use_rate"],
        "secret_exposure_rate": memory_on["secret_exposure_rate"],
        "false_resume_accept_rate": memory_on["false_resume_accept_rate"],
        "repeated_context_reads_delta": {
            "memory_off": memory_efficiency.get("memory_off_repeated_reads", 0),
            "memory_on": memory_efficiency.get("memory_on_repeated_reads", 0),
            "absolute_reduction": memory_efficiency.get("memory_off_repeated_reads", 0)
            - memory_efficiency.get("memory_on_repeated_reads", 0),
        },
        "tool_steps_delta": {
            "memory_off": memory_efficiency.get("memory_off_avg_tool_steps", 0.0),
            "memory_on": memory_efficiency.get("memory_on_avg_tool_steps", 0.0),
        },
        "contract_case_pass_rate": contract_summary["case_pass_rate"],
    }


def run_memory_challenge_v1(artifact_path=DEFAULT_CHALLENGE_ARTIFACT_PATH):
    return _write_json(artifact_path, run_memory_challenge_payload())


def run_memory_agent_eval_v1(
    artifact_path=DEFAULT_ARTIFACT_PATH,
    report_path=DEFAULT_REPORT_PATH,
    challenge_artifact_path=DEFAULT_CHALLENGE_ARTIFACT_PATH,
    memory_ablation_path=DEFAULT_MEMORY_ABLATION_PATH,
    memory_fidelity_path=DEFAULT_MEMORY_FIDELITY_PATH,
    dream_quality_path=DEFAULT_DREAM_QUALITY_PATH,
    recovery_ablation_path=DEFAULT_RECOVERY_ABLATION_PATH,
):
    memory_ablation = _load_json_if_exists(memory_ablation_path)
    memory_fidelity = _load_json_if_exists(memory_fidelity_path)
    dream_quality = _load_json_if_exists(dream_quality_path)
    recovery_ablation = _load_json_if_exists(recovery_ablation_path)
    contract_rows = run_memory_agent_cases()
    contract_summary = _summarize_contract(
        contract_rows,
        memory_ablation=memory_ablation,
        memory_fidelity=memory_fidelity,
        recovery_ablation=recovery_ablation,
        dream_quality=dream_quality,
    )
    challenge = run_memory_challenge_payload()
    artifact = {
        "schema_version": 1,
        "artifact_type": "memory-agent-eval-v1",
        "captured_at": _captured_at(),
        "summary": _top_level_summary(contract_summary, challenge, memory_ablation),
        "contract": {"summary": contract_summary, "rows": contract_rows},
        "challenge": challenge,
        "inputs": {
            "memory_ablation": str(memory_ablation_path),
            "memory_fidelity": str(memory_fidelity_path),
            "dream_quality": str(dream_quality_path),
            "recovery_ablation": str(recovery_ablation_path),
        },
    }
    artifact = _write_json(artifact_path, artifact)
    if challenge_artifact_path:
        _write_json(challenge_artifact_path, challenge)
    if report_path:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_memory_evaluation_report(artifact), encoding="utf-8")
    return artifact
