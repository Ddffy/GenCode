import json
import hashlib
import subprocess
import sys

from gencode.features.memory_lint import lint_memory_dir


def _write_memory_fixture(root, note_texts, metadata_rows):
    topics = root / "topics"
    topics.mkdir(parents=True)
    (topics / "topic.md").write_text(
        "# Topic\n\n"
        "- topic: topic\n"
        "- summary: Test topic.\n"
        "- tags: test\n"
        "- updated_at: 2026-06-24T00:00:00+00:00\n\n"
        "## Notes\n"
        + "".join(f"- {text}\n" for text in note_texts),
        encoding="utf-8",
    )
    (topics / "topic.metadata.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata_rows),
        encoding="utf-8",
    )


def _note_id(text, topic="topic"):
    return hashlib.sha256(f"{topic}\n{text}".encode("utf-8")).hexdigest()[:12]


def _row(text, *, status="active", supersedes=None, session_id="fixture"):
    return {
        "note_id": _note_id(text),
        "status": status,
        "supersedes": supersedes,
        "evidence": {
            "session_id": session_id,
            "source_path": None,
            "created_at": "2026-06-24T00:00:00+00:00",
            "evidence_anchor_hash": None,
        },
        "scope": "workspace_fingerprint",
    }


def test_memory_lint_dirty_fixture_reports_exactly_five_findings():
    findings = lint_memory_dir("tests/fixtures/memory_lint_dirty")

    assert [finding["rule"] for finding in findings] == [
        "duplicate_active_subject",
        "missing_evidence",
        "orphan_supersede",
        "relative_date",
        "secret_shaped",
    ]


def test_memory_lint_cli_returns_one_for_findings():
    result = subprocess.run(
        [sys.executable, "-m", "gencode.features.memory_lint", "tests/fixtures/memory_lint_dirty"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout.count("\n") == 5


def test_memory_lint_duplicate_active_subject_positive_and_negative(tmp_path):
    _write_memory_fixture(
        tmp_path,
        ["GenCode uses pytest.", "GenCode uses unittest."],
        [_row("GenCode uses pytest."), _row("GenCode uses unittest.", status="superseded")],
    )
    assert not [finding for finding in lint_memory_dir(tmp_path) if finding["rule"] == "duplicate_active_subject"]

    _write_memory_fixture(
        tmp_path / "dirty",
        ["GenCode uses pytest.", "GenCode uses unittest."],
        [_row("GenCode uses pytest."), _row("GenCode uses unittest.")],
    )
    assert [finding for finding in lint_memory_dir(tmp_path / "dirty") if finding["rule"] == "duplicate_active_subject"]


def test_memory_lint_relative_date_positive_and_negative(tmp_path):
    _write_memory_fixture(tmp_path, ["Review the audit on 2026-06-24."], [_row("Review the audit on 2026-06-24.")])
    assert not [finding for finding in lint_memory_dir(tmp_path) if finding["rule"] == "relative_date"]

    _write_memory_fixture(tmp_path / "dirty", ["Review the audit tomorrow."], [_row("Review the audit tomorrow.")])
    assert [finding for finding in lint_memory_dir(tmp_path / "dirty") if finding["rule"] == "relative_date"]


def test_memory_lint_secret_shaped_positive_and_negative(tmp_path):
    _write_memory_fixture(tmp_path, ["Store safe config placeholder."], [_row("Store safe config placeholder.")])
    assert not [finding for finding in lint_memory_dir(tmp_path) if finding["rule"] == "secret_shaped"]

    _write_memory_fixture(tmp_path / "dirty", ["Store api key sk-AAAAAAAAAAAAAAAAAAAAAA."], [_row("Store api key sk-AAAAAAAAAAAAAAAAAAAAAA.")])
    assert [finding for finding in lint_memory_dir(tmp_path / "dirty") if finding["rule"] == "secret_shaped"]


def test_memory_lint_orphan_supersede_positive_and_negative(tmp_path):
    _write_memory_fixture(tmp_path, ["GenCode uses pytest."], [_row("GenCode uses pytest.", supersedes=None)])
    assert not [finding for finding in lint_memory_dir(tmp_path) if finding["rule"] == "orphan_supersede"]

    _write_memory_fixture(tmp_path / "dirty", ["GenCode uses pytest."], [_row("GenCode uses pytest.", supersedes="missing")])
    assert [finding for finding in lint_memory_dir(tmp_path / "dirty") if finding["rule"] == "orphan_supersede"]


def test_memory_lint_missing_evidence_positive_and_negative(tmp_path):
    _write_memory_fixture(tmp_path, ["GenCode should keep evidence."], [_row("GenCode should keep evidence.")])
    assert not [finding for finding in lint_memory_dir(tmp_path) if finding["rule"] == "missing_evidence"]

    _write_memory_fixture(tmp_path / "dirty", ["GenCode should keep evidence."], [_row("GenCode should keep evidence.", session_id="")])
    assert [finding for finding in lint_memory_dir(tmp_path / "dirty") if finding["rule"] == "missing_evidence"]
