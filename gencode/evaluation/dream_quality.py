"""Deterministic checks for Dream's Skill/Wiki/Spec candidate boundary."""

import argparse
import json
import tempfile
from pathlib import Path

from .. import GenCode, SessionStore, WorkspaceContext
from ..testing import ScriptedModelClient

DEFAULT_ARTIFACT_PATH = Path("_local/benchmark/artifacts/dream-quality-v1.json")
FIXED_CAPTURED_AT = "2026-09-29T00:00:00Z"


def _agent(root, outputs):
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("Dream evaluation workspace.\n", encoding="utf-8")
    store = SessionStore(root / ".gencode" / "sessions")
    return GenCode(
        model_client=ScriptedModelClient(outputs),
        workspace=WorkspaceContext.build(root, repo_root_override=root),
        session_store=store,
        approval_policy="auto",
        auto_dream=False,
    )


def _save_history(agent, messages):
    agent.session["history"] = [dict(message) for message in messages]
    agent.session_path = agent.session_store.save(agent.session)
    return agent.session["id"]


def _safe_candidate_case(root):
    outputs = [
        (
            '<tool>{"name":"knowledge_propose","args":{"kind":"wiki",'
            '"id":"benchmark-rule","title":"Benchmark rule",'
            '"description":"Stable evaluation guidance",'
            '"body":"Separate deterministic harness contracts from live-model scores.",'
            '"tags":["benchmark"]}}</tool>'
        ),
        "<final>Created the Wiki candidate for review.</final>",
    ]
    agent = _agent(root, outputs)
    session_id = _save_history(
        agent,
        [
            {"role": "user", "content": "Remember: separate deterministic harness contracts from live-model scores."},
            {"role": "assistant", "content": "That is a stable benchmark-design rule."},
        ],
    )
    agent.run_dream(session_ids=[session_id])
    record = agent.knowledge_store.get("benchmark-rule", kind="wiki", include_inactive=True)
    passed = bool(
        record
        and record["status"] == "candidate"
        and record["provenance"].get("source_sessions") == [session_id]
        and not agent.knowledge_store.retrieve("benchmark design") ["wiki"]
    )
    return {"id": "safe_candidate", "passed": passed, "status": record.get("status") if record else "missing"}


def _secret_quarantine_case(root):
    secret = "sk-AAAAAAAAAAAAAAAAAAAA"
    payload = {
        "name": "knowledge_propose",
        "args": {
            "kind": "wiki",
            "id": "secret-note",
            "title": "Credential",
            "description": "Secret-shaped test input",
            "body": f"API key is {secret}.",
        },
    }
    output = f"<tool>{json.dumps(payload)}</tool>"
    agent = _agent(root, [output, "<final>Unsafe candidate quarantined.</final>"])
    session_id = _save_history(
        agent, [{"role": "user", "content": "The test input includes a fake credential."}]
    )
    agent.run_dream(session_ids=[session_id])
    record = agent.knowledge_store.get("secret-note", kind="wiki", include_inactive=True)
    path = root / ".gencode" / "knowledge" / "wiki" / "secret-note.md"
    content = path.read_text(encoding="utf-8") if path.is_file() else ""
    passed = bool(
        record
        and record["status"] == "quarantined"
        and "secret_shaped" in record.get("quality_reasons", [])
        and secret not in content
    )
    return {"id": "secret_quarantine", "passed": passed, "status": record.get("status") if record else "missing"}


def _noise_skip_case(root):
    agent = _agent(root, ["<final>No reusable knowledge in this conversation.</final>"])
    session_id = _save_history(
        agent,
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hi!"},
        ],
    )
    agent.run_dream(session_ids=[session_id])
    records = agent.knowledge_store.list_records(include_inactive=True)
    return {"id": "noise_only", "passed": records == [], "candidate_count": len(records)}


def _duplicate_skip_case(root):
    agent = _agent(root, ["<final>The existing active Wiki already covers this fact.</final>"])
    original = agent.knowledge_store.upsert(
        "wiki",
        "stable-rule",
        title="Stable rule",
        description="A verified project rule",
        body="Use the repository verifier before reporting success.",
        status="active",
        trusted=True,
    )
    session_id = _save_history(
        agent,
        [
            {"role": "user", "content": "Remember: use the repository verifier before reporting success."},
            {"role": "assistant", "content": "That is the existing project rule."},
        ],
    )
    agent.run_dream(session_ids=[session_id])
    current = agent.knowledge_store.get("stable-rule", kind="wiki", include_inactive=True)
    records = agent.knowledge_store.list_records(kind="wiki", include_inactive=True)
    passed = bool(
        current
        and current["status"] == "active"
        and current["content_hash"] == original["content_hash"]
        and len(records) == 1
    )
    return {"id": "active_duplicate", "passed": passed, "record_count": len(records)}


def run_dream_quality_v1(fixtures_dir=None, artifact_path=DEFAULT_ARTIFACT_PATH):
    """Exercise signal, duplicate, and secret boundaries without a live provider.

    ``fixtures_dir`` remains accepted for callers of the earlier benchmark API,
    but transcript fixtures are now created in isolated temporary workspaces.
    """
    rows = []
    with tempfile.TemporaryDirectory(prefix="gencode-dream-quality-") as temp_dir:
        root = Path(temp_dir)
        rows.append(_safe_candidate_case(root / "safe"))
        rows.append(_secret_quarantine_case(root / "secret"))
        rows.append(_noise_skip_case(root / "noise"))
        rows.append(_duplicate_skip_case(root / "duplicate"))
    passed = sum(bool(row["passed"]) for row in rows)
    summary = {
        "total_cases": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "signal_retention_rate": float(rows[0]["passed"]),
        "noise_rejection_rate": float(rows[2]["passed"]),
        "secret_rejection_rate": float(rows[1]["passed"]),
        "dedupe_rate": float(rows[3]["passed"]),
        "relative_date_absolutization_rate": 0.0,
    }
    artifact = {
        "schema_version": 2,
        "artifact_type": "dream-quality-v1",
        "captured_at": FIXED_CAPTURED_AT,
        "fixture_count": len(rows),
        "summary": summary,
        "rows": rows,
    }
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run deterministic typed-knowledge Dream checks.")
    parser.add_argument("--fixtures", default=None, help="Deprecated; the benchmark now uses isolated built-in transcripts.")
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT_PATH), help="Path for the Dream quality artifact.")
    args = parser.parse_args(argv)
    artifact = run_dream_quality_v1(args.fixtures, artifact_path=args.artifact)
    return 0 if artifact["summary"]["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
