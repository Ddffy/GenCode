#!/usr/bin/env python3
"""Run one real-provider Goal through planning, Worker, verifier, and Critic."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gencode.cli import (
    build_agent,
    build_arg_parser,
    handle_repl_command_async,
)
from gencode.config import load_project_env, resolve_provider_config


def run_live_acceptance(
    output_dir=None,
    *,
    timeout_seconds=1800,
    max_steps=100,
    provider=None,
    workspace=None,
    objective=None,
):
    load_project_env(ROOT, override=False)
    config = resolve_provider_config(provider, start=ROOT)
    if not config.api_key:
        raise RuntimeError(f"No API key is configured for provider {config.name}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
    base = Path(output_dir or ROOT / ".gencode" / "evaluations" / "goal-live").resolve()
    run_dir = base / run_id
    fixture_mode = workspace is None
    if not fixture_mode and not str(objective or "").strip():
        raise ValueError("--objective is required when using --workspace")
    if fixture_mode:
        workspace = run_dir / "workspace"
        workspace.mkdir(parents=True)
        _initialize_fixture(workspace)
    else:
        workspace = Path(workspace).expanduser().resolve()
        if not workspace.is_dir():
            raise ValueError(f"Goal workspace is not a directory: {workspace}")
        run_dir.mkdir(parents=True, exist_ok=True)

    args = build_arg_parser().parse_args(
        ["--cwd", str(workspace), "--approval", "auto", "--max-steps", str(max_steps)]
    )
    args.provider = config.name
    args.model = config.model
    args.base_url = config.base_url
    args.api_key = config.api_key
    old_protocol = os.environ.get("GENCODE_PROTOCOL")
    if config.protocol:
        os.environ["GENCODE_PROTOCOL"] = config.protocol
    started_at = time.monotonic()
    goal_id = ""
    agent = None
    try:
        agent = build_agent(args)
        objective = str(objective or "").strip() or (
            "Make the smallest code change needed: append exactly `GOAL_LIVE_ACCEPTED` to README.md. "
            "Do not change any other tracked file. The independent verifier is `python -m pytest -q`."
        )
        handled, _, response = asyncio.run(_start_and_wait(agent, objective, timeout_seconds))
        if not handled:
            raise RuntimeError(f"/goal interface did not handle the request: {response}")
        goal_id = response.split()[1]
        goal = agent.goal_manager.store.load(goal_id)
        summary = _make_summary(
            workspace,
            goal_id,
            goal,
            config,
            agent.model_client,
            time.monotonic() - started_at,
            fixture_mode=fixture_mode,
        )
        summary["start_response"] = response
        summary["wait_response"] = agent.goal_manager.status_text(goal_id)
        summary["status"] = "passed" if _acceptance_passed(summary) else "failed"
        _write_summary(run_dir, summary)
        return summary
    finally:
        if old_protocol is None:
            os.environ.pop("GENCODE_PROTOCOL", None)
        else:
            os.environ["GENCODE_PROTOCOL"] = old_protocol


async def _start_and_wait(agent, objective, timeout_seconds):
    handled, should_exit, response = await handle_repl_command_async(agent, f"/goal {objective}")
    if should_exit:
        return handled, should_exit, response
    if not handled or not response.startswith("Goal "):
        return handled, should_exit, response
    goal_id = response.split()[1]
    task = agent.goal_manager._active.get(goal_id)
    if task is None:
        return handled, should_exit, response
    done, _ = await asyncio.wait({task}, timeout=timeout_seconds)
    if not done:
        await agent.goal_manager.cancel(goal_id)
        raise TimeoutError(f"live Goal did not finish within {timeout_seconds} seconds")
    await asyncio.gather(task, return_exceptions=True)
    return handled, should_exit, response


def _initialize_fixture(workspace):
    (workspace / "README.md").write_text("Goal live acceptance fixture.\n", encoding="utf-8")
    (workspace / "test_goal_live.py").write_text(
        "from pathlib import Path\n\n"
        "def test_goal_marker_is_present():\n"
        "    assert 'GOAL_LIVE_ACCEPTED' in Path('README.md').read_text(encoding='utf-8')\n",
        encoding="utf-8",
    )
    for args in (
        ("init", "-q"),
        ("config", "user.email", "gencode-goal-live@example.com"),
        ("config", "user.name", "GenCode Goal Live Acceptance"),
        ("add", "README.md", "test_goal_live.py"),
        ("commit", "-qm", "goal live acceptance baseline"),
    ):
        subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True, text=True)


def _git(workspace, *args):
    result = subprocess.run(
        ["git", *args], cwd=workspace, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _make_summary(
    workspace,
    goal_id,
    goal,
    provider_config,
    model_client,
    elapsed_seconds,
    *,
    fixture_mode,
):
    requests = 0
    completed = 0
    durations_ms = []
    provider_identities = set()
    run_records = []
    for trace_path in sorted((workspace / ".gencode" / "runs").glob("*/trace.jsonl")):
        run_records.append(trace_path.parent.name)
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "model_requested":
                requests += 1
            if event.get("event") == "model_parsed":
                completed += 1
                durations_ms.append(int(event.get("duration_ms", 0)))
                metadata = event.get("completion_metadata") or {}
                provider_identities.add(
                    (
                        str(metadata.get("provider") or ""),
                        str(metadata.get("model") or ""),
                    )
                )
    verifier_runs = goal.get("verifier_runs", [])
    integration_worktree = Path(goal.get("integration_worktree", ""))
    integrated_attempts = [item for item in goal.get("attempts", []) if item.get("status") == "integrated"]
    source_head = _git(workspace, "rev-parse", "HEAD")
    candidate_readme = integration_worktree / "README.md"
    candidate_files = goal.get("candidate_files", [])
    return {
        "schema_version": 1,
        "goal_id": goal_id,
        "goal_status": goal.get("status"),
        "provider": provider_config.name,
        "protocol": provider_config.protocol,
        "model": provider_config.model,
        "workspace": str(workspace),
        "api_key_configured": bool(provider_config.api_key),
        "model_client_class": type(model_client).__name__,
        "model_client_module": type(model_client).__module__,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "max_runtime_seconds": goal.get("max_runtime_seconds", 0),
        "graph_revision": goal.get("graph_revision", 0),
        "replan_count": goal.get("replan_count", 0),
        "attempt_count": len(goal.get("attempts", [])),
        "completed_nodes": sum(node.get("status") == "completed" for node in goal.get("nodes", [])),
        "model_requests": requests,
        "completed_model_responses": completed,
        "model_round_durations_ms": durations_ms,
        "provider_identities_in_usage_metadata": sorted(
            [identity for identity in provider_identities if any(identity)]
        ),
        "run_ids": run_records,
        "verifier_runs": [
            {
                "passed": item.get("passed"),
                "summary": item.get("summary", ""),
                "commands": item.get("commands", []),
            }
            for item in verifier_runs
        ],
        "critic_rounds": goal.get("critic_rounds", 0),
        "critic_findings": goal.get("critic_findings", []),
        "candidate_files": candidate_files,
        "candidate_files_exist": bool(candidate_files)
        and all((integration_worktree / item).is_file() for item in candidate_files),
        "fixture_mode": fixture_mode,
        "integration_branch": goal.get("integration_branch", ""),
        "integration_commit": goal.get("integration_commit", ""),
        "source_head_after_goal": source_head,
        "base_commit": goal.get("base_commit", ""),
        "source_base_unchanged": source_head == goal.get("base_commit"),
        "candidate_contains_marker": candidate_readme.is_file()
        and "GOAL_LIVE_ACCEPTED" in candidate_readme.read_text(encoding="utf-8"),
        "worker_worktrees_cleaned": bool(integrated_attempts)
        and all(item.get("worktree_cleaned") is True for item in integrated_attempts),
        "failure": goal.get("failure", ""),
        "evidence_dir": str(workspace / ".gencode" / "goals" / goal_id),
    }


def _acceptance_passed(summary):
    verifiers = summary.get("verifier_runs", [])
    common = (
        summary.get("goal_status") == "completed"
        and summary.get("api_key_configured") is True
        and summary.get("model_client_module", "").startswith("gencode.providers")
        and summary.get("max_runtime_seconds", 0) >= 1800
        and summary.get("model_requests", 0) >= 3
        and summary.get("completed_model_responses", 0) >= 3
        and bool(verifiers)
        and all(item.get("passed") is True for item in verifiers)
        and summary.get("critic_rounds", 0) >= 1
        and not summary.get("critic_findings")
        and summary.get("source_base_unchanged") is True
        and summary.get("worker_worktrees_cleaned") is True
    )
    if not common:
        return False
    if summary.get("fixture_mode"):
        return (
            summary.get("candidate_contains_marker") is True
            and summary.get("candidate_files") == ["README.md"]
        )
    return summary.get("candidate_files_exist") is True


def _write_summary(run_dir, summary):
    path = run_dir / "summary.json"
    summary["summary_path"] = str(path)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="Evidence root; defaults to .gencode/evaluations/goal-live")
    parser.add_argument("--timeout", type=int, default=1800, help="Maximum live scenario time in seconds")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum tool/model steps per Goal Worker turn")
    parser.add_argument("--provider", help="Use a configured provider profile; defaults to project configuration")
    parser.add_argument("--workspace", help="Run the live Goal in an existing clean Git workspace instead of a generated fixture")
    parser.add_argument("--objective", help="Goal objective for an existing workspace")
    args = parser.parse_args(argv)
    if args.timeout < 1 or args.timeout > 1800:
        parser.error("--timeout must be in [1, 1800]")
    if args.max_steps < 1 or args.max_steps > 200:
        parser.error("--max-steps must be in [1, 200]")
    try:
        summary = run_live_acceptance(
            args.output_dir,
            timeout_seconds=args.timeout,
            max_steps=args.max_steps,
            provider=args.provider,
            workspace=args.workspace,
            objective=args.objective,
        )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
