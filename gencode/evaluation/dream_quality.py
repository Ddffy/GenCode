"""Offline dream consolidation quality suite."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from .. import GenCode, SessionStore, WorkspaceContext
from ..testing import ScriptedModelClient

DEFAULT_ARTIFACT_PATH = Path("_local/benchmark/artifacts/dream-quality-v1.json")
FIXED_CAPTURED_AT = "2026-06-24T00:00:00Z"


FIXTURE_EXPECTATIONS = {
    "01_clean_signal": {"signal": 3, "noise": 0, "secrets": 0, "duplicates": 0, "relative_dates": 0},
    "02_noise_heavy": {"signal": 1, "noise": 5, "secrets": 0, "duplicates": 0, "relative_dates": 0},
    "03_secret_leak": {"signal": 0, "noise": 0, "secrets": 1, "duplicates": 0, "relative_dates": 0},
    "04_duplicates": {"signal": 1, "noise": 0, "secrets": 0, "duplicates": 2, "relative_dates": 0},
    "05_relative_dates": {"signal": 2, "noise": 0, "secrets": 0, "duplicates": 0, "relative_dates": 2},
    "06_mixed": {"signal": 3, "noise": 1, "secrets": 1, "duplicates": 1, "relative_dates": 1},
}


CONSOLIDATED_NOTES = {
    "01_clean_signal": [
        "GenCode benchmark reports must separate deterministic and live-provider evidence.",
        "Memory evaluation artifacts live under _local/benchmark/artifacts.",
        "Roadmap PRs should keep unrelated refactors out.",
    ],
    "02_noise_heavy": [
        "GenCode memory fidelity evaluates irrelevant injection, supersede, secret, stale, and poison cases.",
    ],
    "03_secret_leak": [],
    "04_duplicates": [
        "GenCode uses repo-local memory snapshots for deterministic evaluation.",
    ],
    "05_relative_dates": [
        "The release review happens on 2026-06-02.",
        "The incident from 2026-05-31 is resolved.",
    ],
    "06_mixed": [
        "GenCode should reject poisoned durable notes before retrieval.",
        "GenCode uses repo-local memory snapshots for deterministic evaluation.",
        "The release review happens on 2026-06-02.",
    ],
}


def _topic_content(notes):
    lines = [
        "# Test Topic",
        "",
        "- topic: test-topic",
        "- summary: Dream quality fixture.",
        "- tags: test",
        "- updated_at: 2026-06-24T00:00:00+00:00",
        "",
        "## Notes",
    ]
    lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines).rstrip() + "\n"


def _scripted_outputs(fixture_name):
    content = json.dumps(_topic_content(CONSOLIDATED_NOTES[fixture_name]))
    return [
        '<tool>{"name":"read_file","args":{"path":".gencode/memory/topics/test-topic.md","start":1,"end":120}}</tool>',
        f'<tool>{{"name":"write_file","args":{{"path":".gencode/memory/topics/test-topic.md","content":{content}}}}}</tool>',
        "<final>Dream consolidation complete.</final>",
    ]


def _latest_report(memory_dir):
    reports = sorted((Path(memory_dir) / "dream_reports").glob("*.json"))
    if not reports:
        raise FileNotFoundError("dream report was not written")
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def _run_fixture(fixture_path):
    fixture_path = Path(fixture_path)
    with tempfile.TemporaryDirectory(prefix="gencode-dream-quality-") as tmp:
        root = Path(tmp)
        (root / "README.md").write_text("dream quality fixture\n", encoding="utf-8")
        memory_dir = root / ".gencode" / "memory"
        shutil.copytree(fixture_path, memory_dir)
        agent = GenCode(
            model_client=ScriptedModelClient(_scripted_outputs(fixture_path.name)),
            workspace=WorkspaceContext.build(root),
            session_store=SessionStore(root / ".gencode" / "sessions"),
            approval_policy="auto",
            auto_dream=False,
        )
        result = agent.run_dream(session_ids=["fixture-session"])
        report = _latest_report(memory_dir)
    expected = FIXTURE_EXPECTATIONS[fixture_path.name]
    checks = {
        "signal_retained": report["signal_retained"] >= expected["signal"],
        "noise_dropped": report["noise_dropped"] >= expected["noise"],
        "secrets_rejected": report["secrets_rejected"] >= expected["secrets"],
        "duplicates_merged": report["duplicates_merged"] >= expected["duplicates"],
        "relative_dates_absolutized": report["relative_dates_absolutized"] >= expected["relative_dates"],
    }
    return {
        "fixture": fixture_path.name,
        "result": result,
        "expected": expected,
        "report": report,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _rate(rows, report_key, expected_key):
    denominator = sum(row["expected"][expected_key] for row in rows)
    if denominator == 0:
        return 1.0
    return min(1.0, sum(row["report"][report_key] for row in rows) / denominator)


def run_dream_quality_v1(fixtures_dir, artifact_path=DEFAULT_ARTIFACT_PATH):
    fixtures_dir = Path(fixtures_dir)
    rows = [_run_fixture(path) for path in sorted(fixtures_dir.iterdir()) if path.is_dir()]
    summary = {
        "total_fixtures": len(rows),
        "passed": sum(1 for row in rows if row["passed"]),
        "failed": sum(1 for row in rows if not row["passed"]),
        "pass_rate": (sum(1 for row in rows if row["passed"]) / len(rows)) if rows else 0.0,
        "signal_retention_rate": _rate(rows, "signal_retained", "signal"),
        "noise_rejection_rate": _rate(rows, "noise_dropped", "noise"),
        "secret_rejection_rate": _rate(rows, "secrets_rejected", "secrets"),
        "dedupe_rate": _rate(rows, "duplicates_merged", "duplicates"),
        "relative_date_absolutization_rate": _rate(rows, "relative_dates_absolutized", "relative_dates"),
    }
    artifact = {
        "schema_version": 1,
        "artifact_type": "dream-quality-v1",
        "captured_at": FIXED_CAPTURED_AT,
        "fixture_count": len(rows),
        "summary": summary,
        "rows": rows,
    }
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run GenCode dream quality fixtures.")
    parser.add_argument("--fixtures", required=True, help="Directory containing dream input fixtures.")
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT_PATH), help="Path for dream-quality-v1 artifact.")
    args = parser.parse_args(argv)
    artifact = run_dream_quality_v1(args.fixtures, artifact_path=args.artifact)
    return 0 if artifact["summary"]["failed"] == 0 and artifact["summary"]["secret_rejection_rate"] == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
