#!/usr/bin/env python3
"""Run deterministic real-session acceptance scenarios for GenCode."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gencode.testing import ScriptedModelClient  # noqa: E402
from gencode import GenCode, SessionStore, WorkspaceContext  # noqa: E402
from gencode.config import resolve_provider_config  # noqa: E402
from gencode.features.skills_runtime import invoke_skill  # noqa: E402
from gencode.providers import AnthropicCompatibleModelClient, OpenAICompatibleModelClient, ProviderError  # noqa: E402

SUMMARY_JSON = "gate8-real-session-acceptance.json"
SUMMARY_MARKDOWN = "gate8-real-session-acceptance.md"
LIVE_ENV_FLAG = "GENCODE_ACCEPTANCE_LIVE"


def run_acceptance(output_dir, include_live=None):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [
        _run_scenario(output_dir, "bugfix_pytest", _scenario_bugfix_pytest),
        _run_scenario(output_dir, "plan_todo_explore", _scenario_plan_todo_explore),
        _run_scenario(output_dir, "skill_inline", _scenario_skill_inline),
        _run_scenario(output_dir, "worker_write_scope", _scenario_worker_write_scope),
        _run_scenario(output_dir, "resume_continuation", _scenario_resume_continuation),
        _run_scenario(output_dir, "security_rejection", _scenario_security_rejection),
        _run_scenario(output_dir, "context_pressure", _scenario_context_pressure),
        _run_scenario(output_dir, "provider_error_recovery", _scenario_provider_error_recovery),
        _run_scenario(
            output_dir,
            "live_provider_smoke",
            lambda root, workspace: _scenario_live_provider_smoke(root, workspace, include_live=include_live),
            optional=True,
            include_live=include_live,
        ),
    ]
    required_ok = all(item["status"] == "passed" for item in scenarios if not item.get("optional"))
    optional_ok = all(item["status"] in {"passed", "skipped"} for item in scenarios if item.get("optional"))
    summary = {
        "status": "passed" if required_ok and optional_ok else "failed",
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / SUMMARY_MARKDOWN).write_text(render_markdown(summary) + "\n", encoding="utf-8")
    return summary


def render_markdown(summary):
    lines = [
        "# Gate8 Real Session Acceptance",
        "",
        f"- status: `{summary['status']}`",
        f"- scenarios: `{summary['scenario_count']}`",
        "",
        "| Scenario | Status | Required | Report | Trace | Events |",
        "|---|---|---:|---|---|---|",
    ]
    for scenario in summary["scenarios"]:
        lines.append(
            "| {id} | {status} | {required} | `{report}` | `{trace}` | `{events}` |".format(
                id=scenario["id"],
                status=scenario["status"],
                required="no" if scenario.get("optional") else "yes",
                report=scenario.get("report_path", ""),
                trace=scenario.get("trace_path", ""),
                events=scenario.get("session_event_path", ""),
            )
        )
    return "\n".join(lines)


def _run_scenario(output_dir, scenario_id, runner, optional=False, include_live=None):
    workspace = output_dir / "workspaces" / scenario_id
    if workspace.exists():
        _remove_tree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        if optional and include_live is False:
            return _skipped_record(output_dir, workspace, scenario_id, "live provider smoke disabled")
        record = runner(output_dir, workspace)
        record["optional"] = bool(optional)
        if record.get("status") == "skipped":
            return record
        record["status"] = "passed" if all(check["status"] == "passed" for check in record["checks"]) else "failed"
        return record
    except Exception as exc:
        return {
            "id": scenario_id,
            "status": "failed",
            "optional": bool(optional),
            "workspace_relpath": _relpath(workspace, output_dir),
            "error": str(exc),
            "checks": [{"name": "scenario_exception", "status": "failed", "detail": str(exc)}],
        }


def _scenario_bugfix_pytest(output_dir, workspace):
    src_dir = workspace / "src"
    tests_dir = workspace / "tests"
    src_dir.mkdir()
    tests_dir.mkdir()
    (src_dir / "calculator.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tests_dir / "test_calculator.py").write_text(
        "from src.calculator import add\n\n\ndef test_adds_numbers():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    pytest_command = subprocess.list2cmdline([sys.executable, "-m", "pytest", "-q"])
    agent = _build_agent(
        workspace,
        [
            '<tool>{"name":"read_file","args":{"path":"tests/test_calculator.py","start":1,"end":20}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"src/calculator.py","start":1,"end":20}}</tool>',
            '<tool name="patch_file" path="src/calculator.py"><old_text>return a - b</old_text><new_text>return a + b</new_text></tool>',
            f'<tool>{{"name":"run_shell","args":{{"command":{json.dumps(pytest_command)},"timeout":60}}}}</tool>',
            "<final>Bug fixed and tests pass.</final>",
        ],
        max_steps=6,
    )
    answer = agent.ask("Fix the failing calculator test and verify it.")
    return _finalize(
        output_dir,
        workspace,
        agent,
        "bugfix_pytest",
        checks=[
            _check("answer", answer == "Bug fixed and tests pass.", answer),
            _check(
                "fixed_file",
                (workspace / "src" / "calculator.py").read_text(encoding="utf-8")
                == "def add(a, b):\n    return a + b\n",
            ),
            _check(
                "pytest_ran",
                any(
                    item["role"] == "tool" and item["name"] == "run_shell" and "passed" in item["content"]
                    for item in agent.session["history"]
                ),
            ),
        ],
    )


def _scenario_plan_todo_explore(output_dir, workspace):
    _write_readme(workspace, "Gate8 plan fixture.\n")
    agent = _build_agent(
        workspace,
        [
            '<tool>{"name":"todo_add","args":{"content":"Draft Gate8 plan","status":"in_progress","priority":"high"}}</tool>',
            '<tool>{"name":"agent","args":{"description":"Inspect fixture","prompt":"Read README.md","subagent_type":"Explore"}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":1}}</tool>',
            "<final>Fixture inspected.</final>",
            '<tool>{"name":"todo_update","args":{"todo_id":"todo_1","status":"done","note":"plan written"}}</tool>',
            '<tool name="write_file" path=".gencode/plans/gate8-plan.md"><content># Gate8 Plan\n- Evidence harness\n</content></tool>',
            "<final>Gate8 plan ready.</final>",
        ],
        max_steps=6,
    )
    agent.enter_plan_mode("gate8")
    answer = agent.ask("Plan Gate8 with todo and Explore evidence")
    return _finalize(
        output_dir,
        workspace,
        agent,
        "plan_todo_explore",
        checks=[
            _check("answer", answer == "Gate8 plan ready.", answer),
            _check("plan_file", (workspace / ".gencode" / "plans" / "gate8-plan.md").is_file()),
            _check("todo_done", agent.session["todos"]["items"][0]["status"] == "done"),
            _check("explore_worker", agent.session["workers"]["items"][0]["subagent_type"] == "Explore"),
        ],
    )


def _scenario_skill_inline(output_dir, workspace):
    _write_readme(workspace, "Gate8 skill fixture.\n")
    skill_dir = workspace / ".gencode" / "skills" / "evidence"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: evidence
description: Inspect evidence target
allowed-tools: read_file
---
Inspect $ARGUMENTS and report the evidence path.
""",
        encoding="utf-8",
    )
    agent = _build_agent(
        workspace,
        [
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":1}}</tool>',
            "<final>Skill evidence checked.</final>",
        ],
        max_steps=4,
    )
    answer = invoke_skill(agent, "evidence", "README.md")
    events = _read_events(agent)
    return _finalize(
        output_dir,
        workspace,
        agent,
        "skill_inline",
        checks=[
            _check("answer", answer == "Skill evidence checked.", answer),
            _check("skill_invoked", any(event["event"] == "skill_invoked" for event in events)),
            _check("skill_completed", any(event["event"] == "skill_completed" for event in events)),
        ],
    )


def _scenario_worker_write_scope(output_dir, workspace):
    _write_readme(workspace, "Gate8 worker fixture.\n")
    agent = _build_agent(
        workspace,
        [
            '<tool>{"name":"agent","args":{"description":"Write scoped notes","prompt":"Create first note","subagent_type":"worker","write_scope":["notes"]}}</tool>',
            '<tool name="write_file" path="notes/first.txt"><content>first\n</content></tool>',
            "<final>First note written.</final>",
            '<tool>{"name":"send_message","args":{"to":"agent_1","message":"Create second note"}}</tool>',
            '<tool name="write_file" path="notes/second.txt"><content>second\n</content></tool>',
            "<final>Second note written.</final>",
            "<final>Scoped worker completed.</final>",
        ],
        max_steps=6,
    )
    answer = agent.ask("Use a scoped worker twice")
    return _finalize(
        output_dir,
        workspace,
        agent,
        "worker_write_scope",
        checks=[
            _check("answer", answer == "Scoped worker completed.", answer),
            _check("first_note", (workspace / "notes" / "first.txt").read_text(encoding="utf-8") == "first\n"),
            _check("second_note", (workspace / "notes" / "second.txt").read_text(encoding="utf-8") == "second\n"),
            _check("write_scope", agent.session["workers"]["items"][0]["write_scope"] == ["notes"]),
        ],
    )


def _scenario_resume_continuation(output_dir, workspace):
    _write_readme(workspace, "Gate8 resume fixture.\n")
    store = SessionStore(workspace / ".gencode" / "sessions")
    first = GenCode(
        model_client=ScriptedModelClient(
            [
                '<tool>{"name":"todo_add","args":{"content":"Resume continuation","status":"in_progress","priority":"high"}}</tool>',
                "<final>Paused after first step.</final>",
            ]
        ),
        workspace=_scenario_workspace(workspace),
        session_store=store,
        approval_policy="auto",
        max_steps=3,
    )
    first_answer = first.ask("Start a resumable task")
    resumed = GenCode.from_session(
        model_client=ScriptedModelClient(
            [
                '<tool name="write_file" path="notes/resume.txt"><content>continued\n</content></tool>',
                '<tool>{"name":"todo_update","args":{"todo_id":"todo_1","status":"done","note":"continued after resume"}}</tool>',
                "<final>Resumed task completed.</final>",
            ]
        ),
        workspace=_scenario_workspace(workspace),
        session_store=store,
        session_id=first.session["id"],
        approval_policy="auto",
        max_steps=4,
    )
    answer = resumed.ask("Resume and finish the task")
    events = _read_events(resumed)
    return _finalize(
        output_dir,
        workspace,
        resumed,
        "resume_continuation",
        checks=[
            _check("first_answer", first_answer == "Paused after first step.", first_answer),
            _check("answer", answer == "Resumed task completed.", answer),
            _check("todo_persisted", resumed.session["todos"]["items"][0]["status"] == "done"),
            _check("resume_file", (workspace / "notes" / "resume.txt").read_text(encoding="utf-8") == "continued\n"),
            _check("session_continued", any(event["event"] == "turn_started" for event in events)),
        ],
    )


def _scenario_security_rejection(output_dir, workspace):
    _write_readme(workspace, "Gate8 security fixture.\n")
    old_secret = os.environ.get("GENCODE_ACCEPTANCE_SECRET")
    os.environ["GENCODE_ACCEPTANCE_SECRET"] = "gencode-secret-value-123"
    agent = _build_agent(
        workspace,
        [
            '<tool>{"name":"read_file","args":{"path":"../outside.txt","start":1,"end":1}}</tool>',
            '<tool>{"name":"agent","args":{"description":"Bad scoped write","prompt":"Write outside scope","subagent_type":"worker","write_scope":["allowed"]}}</tool>',
            '<tool name="write_file" path="blocked/out.txt"><content>nope\n</content></tool>',
            "<final>Blocked scoped write.</final>",
            '<tool>{"name":"run_shell","args":{"command":"echo $GENCODE_ACCEPTANCE_SECRET","timeout":5}}</tool>',
            "<final>Path escape blocked.</final>",
        ],
        max_steps=5,
    )
    try:
        answer = agent.ask("Try unsafe workspace and secret operations")
        events = _read_events(agent)
        trace_text = (agent.current_run_dir / "trace.jsonl").read_text(encoding="utf-8")
        worker_error_codes = agent.session["workers"]["items"][0].get("tool_error_codes", [])
        return _finalize(
            output_dir,
            workspace,
            agent,
            "security_rejection",
            checks=[
                _check("answer", answer == "Path escape blocked.", answer),
                _check("invalid_arguments", any(event.get("tool_error_code") == "invalid_arguments" for event in events)),
                _check("write_scope_blocked", "write_scope_mismatch" in worker_error_codes),
                _check("no_outside_file", not (output_dir / "outside.txt").exists()),
                _check("no_blocked_write", not (workspace / "blocked" / "out.txt").exists()),
                _check("secret_redacted", "gencode-secret-value-123" not in trace_text),
            ],
        )
    finally:
        if old_secret is None:
            os.environ.pop("GENCODE_ACCEPTANCE_SECRET", None)
        else:
            os.environ["GENCODE_ACCEPTANCE_SECRET"] = old_secret


def _scenario_context_pressure(output_dir, workspace):
    src_dir = workspace / "src"
    src_dir.mkdir()
    (workspace / "README.md").write_text(("Context pressure fixture.\n" + "noise " * 900) + "\n", encoding="utf-8")
    (src_dir / "target.py").write_text("VALUE = 'old'\n", encoding="utf-8")
    for index in range(12):
        (src_dir / f"noise_{index}.py").write_text((f"# noise {index}\n" + "x = 'padding'\n" * 80), encoding="utf-8")
    agent = _build_agent(
        workspace,
        [
            '<tool>{"name":"read_file","args":{"path":"src/target.py","start":1,"end":5}}</tool>',
            '<tool name="patch_file" path="src/target.py"><old_text>VALUE = \'old\'</old_text><new_text>VALUE = \'new\'</new_text></tool>',
            "<final>Context pressure handled.</final>",
        ],
        max_steps=4,
    )
    for index in range(6):
        agent.record({"role": "user", "content": f"Historical request {index} " + ("padding " * 120), "created_at": f"history-{index}-u"})
        agent.record({"role": "assistant", "content": f"Historical answer {index} " + ("padding " * 120), "created_at": f"history-{index}-a"})
    agent.compact_history(trigger="acceptance_context_pressure", keep_recent_turns=2)
    agent.context_manager.total_budget = 12000
    answer = agent.ask("Find and update the target constant while keeping context under budget.")
    return _finalize(
        output_dir,
        workspace,
        agent,
        "context_pressure",
        checks=[
            _check("answer", answer == "Context pressure handled.", answer),
            _check("target_updated", (src_dir / "target.py").read_text(encoding="utf-8") == "VALUE = 'new'\n"),
            _check("compaction_recorded", bool(agent.session.get("compactions"))),
            _check("prompt_under_budget", not bool(agent.last_prompt_metadata.get("prompt_over_budget"))),
        ],
    )


def _scenario_provider_error_recovery(output_dir, workspace):
    _write_readme(workspace, "Gate9 provider reliability fixture.\n")
    agent = GenCode(
        model_client=ScriptedModelClient(
            [
                ProviderError(
                    "rate limited",
                    provider="openai",
                    model="gate9-test",
                    base_url="https://example.test/v1",
                    code="rate_limited",
                    http_status=429,
                    retryable=True,
                    attempts=3,
                    retry_count=2,
                    body_excerpt='{"error":"busy"}',
                )
            ]
        ),
        workspace=_scenario_workspace(workspace),
        session_store=SessionStore(workspace / ".gencode" / "sessions"),
        approval_policy="auto",
        max_steps=2,
    )
    answer = agent.ask("Trigger provider failure evidence")
    events = _read_events(agent)
    return _finalize(
        output_dir,
        workspace,
        agent,
        "provider_error_recovery",
        checks=[
            _check("answer", "rate_limited" in answer and answer.startswith("模型错误"), answer),
            _check("task_failed", agent.current_task_state.status == "failed"),
            _check("stop_reason", agent.current_task_state.stop_reason == "model_error"),
            _check("provider_error_metadata", agent.last_prompt_metadata["provider_error"]["code"] == "rate_limited"),
            _check("provider_retry_count", agent.last_prompt_metadata["provider_error"]["retry_count"] == 2),
            _check("model_error_event", any(event["event"] == "model_error" for event in events)),
        ],
    )


def _scenario_live_provider_smoke(output_dir, workspace, include_live=None):
    live_enabled = include_live is True or os.environ.get(LIVE_ENV_FLAG, "").strip().lower() in {"1", "true", "yes"}
    if not live_enabled:
        return _skipped_record(output_dir, workspace, "live_provider_smoke", f"set {LIVE_ENV_FLAG}=1 to enable live provider smoke")
    config = resolve_provider_config(start=workspace)
    if not config.api_key:
        return _skipped_record(output_dir, workspace, "live_provider_smoke", f"provider {config.name} has no api key")
    _write_readme(workspace, "Gate8 live provider fixture.\n")
    if config.protocol == "openai":
        model_client = OpenAICompatibleModelClient(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=0,
            timeout=60,
        )
    elif config.protocol == "anthropic":
        model_client = AnthropicCompatibleModelClient(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=0,
            timeout=60,
        )
    else:
        return _skipped_record(output_dir, workspace, "live_provider_smoke", f"unsupported protocol {config.protocol}")
    agent = GenCode(
        model_client=model_client,
        workspace=_scenario_workspace(workspace),
        session_store=SessionStore(workspace / ".gencode" / "sessions"),
        approval_policy="never",
        max_steps=1,
        max_new_tokens=64,
        secret_env_names=["GENCODE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"],
    )
    answer = agent.ask("Return exactly this final answer and do not use tools: <final>live provider smoke ok</final>")
    return _finalize(
        output_dir,
        workspace,
        agent,
        "live_provider_smoke",
        checks=[
            _check("answer", "live provider smoke ok" in answer.lower(), answer),
            _check("provider_name", bool(config.name), config.name),
        ],
    )


def _build_agent(workspace, outputs, max_steps=6):
    workspace_context = _scenario_workspace(workspace)
    return GenCode(
        model_client=ScriptedModelClient(outputs),
        workspace=workspace_context,
        session_store=SessionStore(workspace / ".gencode" / "sessions"),
        approval_policy="auto",
        max_steps=max_steps,
    )


def _scenario_workspace(workspace):
    return WorkspaceContext.build(workspace, repo_root_override=workspace)


def _finalize(output_dir, workspace, agent, scenario_id, checks):
    run_dir = agent.current_run_dir
    report_path = run_dir / "report.json"
    trace_path = run_dir / "trace.jsonl"
    task_state_path = run_dir / "task_state.json"
    session_event_path = agent.session_event_bus.path
    checks.extend(
        [
            _check("report_exists", report_path.is_file()),
            _check("trace_exists", trace_path.is_file()),
            _check("task_state_exists", task_state_path.is_file()),
            _check("session_events_exists", session_event_path.is_file()),
        ]
    )
    return {
        "id": scenario_id,
        "workspace_relpath": _relpath(workspace, output_dir),
        "session_path": _relpath(agent.session_path, output_dir),
        "session_event_path": _relpath(session_event_path, output_dir),
        "run_dir": _relpath(run_dir, output_dir),
        "report_path": _relpath(report_path, output_dir),
        "trace_path": _relpath(trace_path, output_dir),
        "task_state_path": _relpath(task_state_path, output_dir),
        "checks": checks,
    }


def _write_readme(workspace, text):
    (workspace / "README.md").write_text(text, encoding="utf-8")


def _read_events(agent):
    return [
        json.loads(line)
        for line in agent.session_event_bus.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _check(name, condition, detail=""):
    return {"name": name, "status": "passed" if condition else "failed", "detail": str(detail)}


def _skipped_record(output_dir, workspace, scenario_id, reason):
    return {
        "id": scenario_id,
        "status": "skipped",
        "optional": True,
        "skip_reason": str(reason),
        "workspace_relpath": _relpath(workspace, output_dir),
        "checks": [{"name": "skipped", "status": "passed", "detail": str(reason)}],
    }


def _relpath(path, root):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _remove_tree(path):
    for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if child.is_dir():
            child.rmdir()
        else:
            child.unlink()
    path.rmdir()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run GenCode Gate8 deterministic real-session acceptance scenarios.")
    parser.add_argument("--output-dir", default="artifacts/gate8-real-session-acceptance", help="Directory for workspaces and summary artifacts.")
    parser.add_argument("--live-provider", action="store_true", help=f"Enable optional live provider smoke. Also enabled by {LIVE_ENV_FLAG}=1.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    summary = run_acceptance(Path(args.output_dir), include_live=True if args.live_provider else None)
    print(json.dumps({"status": summary["status"], "scenario_count": summary["scenario_count"]}, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
