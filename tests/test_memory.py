import hashlib
import subprocess

from gencode.features.memory_lint import SECRET_PATTERNS
from gencode.features.memory import (
    LayeredMemory,
    compute_anchor_hash,
    retrieval_view_structured,
    workspace_fingerprint,
)


def test_working_memory_tracks_summary_and_recent_files():
    memory = LayeredMemory()

    memory.set_task_summary("Investigate flaky tests")
    memory.remember_file("README.md")
    memory.remember_file("src/app.py")
    memory.remember_file("README.md")

    snapshot = memory.to_dict()

    assert snapshot["working"]["task_summary"] == "Investigate flaky tests"
    assert snapshot["working"]["recent_files"] == ["src/app.py", "README.md"]
    assert snapshot["task"] == "Investigate flaky tests"
    assert snapshot["files"] == ["src/app.py", "README.md"]


def test_episodic_notes_append_and_retrieve_deterministically():
    memory = LayeredMemory()

    memory.append_note("Exact tag note", tags=("recall",), created_at="2026-04-07T10:00:00+00:00")
    memory.append_note("Keyword overlap note about memory", created_at="2026-04-07T10:01:00+00:00")
    memory.append_note("Newest unrelated note", created_at="2026-04-07T10:02:00+00:00")
    memory.append_note("Older unrelated note", created_at="2026-04-07T09:59:00+00:00")

    snapshot = memory.to_dict()
    assert [note["text"] for note in snapshot["episodic_notes"]] == [
        "Exact tag note",
        "Keyword overlap note about memory",
        "Newest unrelated note",
        "Older unrelated note",
    ]
    assert snapshot["notes"] == [
        "Exact tag note",
        "Keyword overlap note about memory",
        "Newest unrelated note",
        "Older unrelated note",
    ]

    lines = [line for line in memory.retrieval_view("recall memory", limit=4).splitlines() if line.startswith("- ")]
    assert lines == [
        "- Exact tag note",
        "- Keyword overlap note about memory",
    ]


def test_retrieval_view_structured_reports_selected_and_rejected_reasons():
    memory = LayeredMemory()

    memory.append_note("alpha selected note", tags=("alpha",), created_at="2026-04-07T10:04:00+00:00")
    memory.append_note("alpha below limit note", tags=("alpha",), created_at="2026-04-07T10:03:00+00:00")
    memory.append_note("alpha quarantined note", tags=("alpha",), created_at="2026-04-07T10:02:00+00:00")
    memory.append_note("alpha superseded note", tags=("alpha",), created_at="2026-04-07T10:01:00+00:00")
    memory.state["episodic_notes"][2]["status"] = "quarantined"
    memory.state["episodic_notes"][3]["status"] = "superseded"

    structured = retrieval_view_structured(memory.state, "alpha", limit=1)

    assert set(structured) == {"selected", "rejected", "query_hash"}
    assert len(structured["query_hash"]) == 12
    assert [note["text"] for note in structured["selected"]] == ["alpha selected note"]
    reject_reasons = {note["reject_reason"] for note in structured["rejected"]}
    assert reject_reasons >= {"below_limit", "quarantined", "superseded"}
    for note in structured["rejected"]:
        assert set(note) >= {"note_id", "layer", "score", "reject_reason"}
    assert "alpha below limit note" not in memory.retrieval_view("alpha", limit=1)


def test_structured_retrieval_rejects_stale_evidence_and_scope_mismatch():
    memory = LayeredMemory()
    memory.append_note("alpha valid note", tags=("alpha",), created_at="2026-04-07T10:03:00+00:00")
    memory.append_note("alpha stale evidence note", tags=("alpha",), created_at="2026-04-07T10:02:00+00:00")
    memory.append_note("alpha wrong scope note", tags=("alpha",), created_at="2026-04-07T10:01:00+00:00")
    memory.state["episodic_notes"][1]["stale_evidence"] = True
    memory.state["episodic_notes"][2]["scope"] = "other-workspace"

    structured = retrieval_view_structured(memory.state, "alpha", limit=3)

    assert [note["text"] for note in structured["selected"]] == ["alpha valid note"]
    rejected = {note["text"]: note["reject_reason"] for note in structured["rejected"]}
    assert rejected["alpha stale evidence note"] == "stale_evidence"
    assert rejected["alpha wrong scope note"] == "scope_mismatch"


def test_file_summaries_use_canonical_paths_and_freshness(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("alpha\n", encoding="utf-8")
    memory = LayeredMemory(workspace_root=tmp_path)

    memory.set_file_summary("./sample.txt", "sample.txt: alpha")
    memory.remember_file("./sample.txt")
    snapshot = memory.to_dict()["file_summaries"]["sample.txt"]

    assert snapshot["summary"] == "sample.txt: alpha"
    assert snapshot["freshness"]

    assert "sample.txt: alpha" in memory.render_memory_text()
    file_path.write_text("beta\n", encoding="utf-8")
    assert "sample.txt: alpha" not in memory.render_memory_text()

    memory.invalidate_file_summary("sample.txt")

    assert "sample.txt" not in memory.to_dict()["file_summaries"]


def test_workspace_fingerprint_uses_git_root_when_available(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "nested").mkdir()
    expected = hashlib.sha256(str(tmp_path.resolve()).encode("utf-8")).hexdigest()[:12]

    assert workspace_fingerprint(tmp_path / "nested" / "..") == expected


def test_workspace_fingerprint_uses_resolved_path_for_non_git_dir(tmp_path):
    expected = hashlib.sha256(str(tmp_path.resolve()).encode("utf-8")).hexdigest()[:12]

    assert workspace_fingerprint(tmp_path) == expected


def test_anchor_hash_returns_hash_for_files_at_or_below_size_limit(tmp_path):
    path = tmp_path / "nine-mib.bin"
    payload = b"a" * (9 * 1024 * 1024)
    path.write_bytes(payload)

    assert compute_anchor_hash(path) == hashlib.sha256(payload).hexdigest()


def test_anchor_hash_returns_none_for_large_or_missing_files(tmp_path):
    large_path = tmp_path / "eleven-mib.bin"
    large_path.write_bytes(b"a" * (11 * 1024 * 1024))

    assert compute_anchor_hash(large_path) is None
    assert compute_anchor_hash(tmp_path / "missing.txt") is None


def test_secret_patterns_match_supported_secret_shapes():
    positives = [
        "sk-" + "A" * 20,
        "AKIA" + "0" * 16,
        "ghp_" + "A" * 36,
        "xoxb-" + "A" * 10,
        "api key " + "a" * 40,
    ]

    for candidate in positives:
        assert any(pattern.search(candidate) for pattern in SECRET_PATTERNS), candidate


def test_secret_patterns_do_not_match_short_or_context_free_random_text():
    negatives = [
        "abc123",
        "A" * 40,
        "deadbeef" * 4,
    ]

    for candidate in negatives:
        assert not any(pattern.search(candidate) for pattern in SECRET_PATTERNS), candidate


def test_process_notes_keep_kind_and_latest_duplicate_wins():
    memory = LayeredMemory()

    memory.append_note(
        "Shell partial success on README.md; inspect diff before retry",
        tags=("process", "partial_success"),
        created_at="2026-04-07T10:00:00+00:00",
        kind="process",
    )
    memory.append_note(
        "Shell partial success on README.md; inspect diff before retry",
        tags=("process", "partial_success"),
        created_at="2026-04-07T10:01:00+00:00",
        kind="process",
    )

    notes = memory.to_dict()["episodic_notes"]

    assert len(notes) == 1
    assert notes[0]["kind"] == "process"
    assert notes[0]["created_at"] == "2026-04-07T10:01:00+00:00"


def test_layered_memory_ignores_legacy_cross_session_topic_store(tmp_path):
    memory_root = tmp_path / ".gencode" / "memory"
    topics_dir = memory_root / "topics"
    topics_dir.mkdir(parents=True)
    (memory_root / "MEMORY.md").write_text(
        "# Durable Memory Index\n\n"
        "- [project-conventions](topics/project-conventions.md): Project Conventions\n"
        "  - summary: Stable repository conventions.\n"
        "  - tags: convention\n",
        encoding="utf-8",
    )
    (topics_dir / "project-conventions.md").write_text(
        "# Project Conventions\n\n"
        "- topic: project-conventions\n"
        "- summary: Stable repository conventions.\n"
        "- tags: convention\n"
        "- updated_at: 2026-04-12T08:14:49+00:00\n\n"
        "## Notes\n"
        "- Use constrained tools instead of guessing.\n"
        "- Preserve local agent state under .gencode/.\n",
        encoding="utf-8",
    )
    topic_path = topics_dir / "project-conventions.md"
    original = topic_path.read_text(encoding="utf-8")

    memory = LayeredMemory(workspace_root=tmp_path)

    assert "durable_topics" not in memory.to_dict()
    assert "Use constrained tools instead of guessing" not in memory.retrieval_view("constrained tools")
    assert topic_path.read_text(encoding="utf-8") == original
