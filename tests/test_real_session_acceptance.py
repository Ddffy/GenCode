import json
import importlib.util
from pathlib import Path


def _load_run_acceptance():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_real_session_acceptance.py"
    spec = importlib.util.spec_from_file_location("run_real_session_acceptance", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_acceptance


def test_gate8_acceptance_harness_writes_real_session_evidence_bundle(tmp_path):
    run_acceptance = _load_run_acceptance()
    output_dir = tmp_path / "gate8-evidence"

    summary = run_acceptance(output_dir)

    assert summary["status"] == "passed"
    assert summary["scenario_count"] >= 9

    scenario_ids = {scenario["id"] for scenario in summary["scenarios"]}
    assert {
        "bugfix_pytest",
        "plan_todo_explore",
        "skill_inline",
        "worker_write_scope",
        "resume_continuation",
        "security_rejection",
        "context_pressure",
        "provider_error_recovery",
        "live_provider_smoke",
    }.issubset(scenario_ids)

    for scenario in summary["scenarios"]:
        assert scenario["status"] in {"passed", "skipped"}
        if scenario["status"] == "skipped":
            assert scenario.get("optional") is True
            assert scenario.get("skip_reason")
            continue
        assert scenario["report_path"]
        assert scenario["trace_path"]
        assert scenario["session_event_path"]
        report = json.loads((output_dir / scenario["report_path"]).read_text(encoding="utf-8"))
        events = [
            json.loads(line)
            for line in (output_dir / scenario["session_event_path"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if scenario["id"] == "provider_error_recovery":
            assert report["status"] == "failed"
            assert report["stop_reason"] == "model_error"
        else:
            assert report["status"] == "completed"
            assert report["stop_reason"] == "final_answer_returned"
        assert any(event["event"] == "turn_finished" for event in events)
        assert (output_dir / scenario["trace_path"]).exists()

    bugfix = next(scenario for scenario in summary["scenarios"] if scenario["id"] == "bugfix_pytest")
    bugfix_report = json.loads((output_dir / bugfix["report_path"]).read_text(encoding="utf-8"))
    assert "src/calculator.py" in bugfix_report["task_state"]["changed_paths"]
    assert (output_dir / bugfix["workspace_relpath"] / "src" / "calculator.py").read_text(encoding="utf-8") == (
        "def add(a, b):\n"
        "    return a + b\n"
    )

    plan = next(scenario for scenario in summary["scenarios"] if scenario["id"] == "plan_todo_explore")
    plan_report = json.loads((output_dir / plan["report_path"]).read_text(encoding="utf-8"))
    assert plan_report["todos"]["items"][0]["status"] == "done"
    assert plan_report["workers"]["items"][0]["subagent_type"] == "Explore"

    worker = next(scenario for scenario in summary["scenarios"] if scenario["id"] == "worker_write_scope")
    worker_report = json.loads((output_dir / worker["report_path"]).read_text(encoding="utf-8"))
    assert worker_report["workers"]["items"][0]["write_scope"] == ["notes"]
    assert (output_dir / worker["workspace_relpath"] / "notes" / "first.txt").read_text(encoding="utf-8") == "first\n"
    assert (output_dir / worker["workspace_relpath"] / "notes" / "second.txt").read_text(encoding="utf-8") == "second\n"

    resume = next(scenario for scenario in summary["scenarios"] if scenario["id"] == "resume_continuation")
    resume_report = json.loads((output_dir / resume["report_path"]).read_text(encoding="utf-8"))
    assert resume_report["todos"]["items"][0]["status"] == "done"
    assert resume_report["resume_status"] in {"full-valid", "workspace-mismatch"}
    assert (output_dir / resume["workspace_relpath"] / "notes" / "resume.txt").read_text(encoding="utf-8") == "continued\n"

    security = next(scenario for scenario in summary["scenarios"] if scenario["id"] == "security_rejection")
    security_events = [
        json.loads(line)
        for line in (output_dir / security["session_event_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event.get("tool_error_code") == "invalid_arguments" for event in security_events)
    security_report = json.loads((output_dir / security["report_path"]).read_text(encoding="utf-8"))
    assert security_report["redacted_env"]["secret_env_count"] >= 1
    assert "write_scope_mismatch" in security_report["workers"]["items"][0]["tool_error_codes"]
    assert security_report["workers"]["items"][0]["trace_path"]

    context = next(scenario for scenario in summary["scenarios"] if scenario["id"] == "context_pressure")
    context_report = json.loads((output_dir / context["report_path"]).read_text(encoding="utf-8"))
    assert context_report["prompt_metadata"]["prompt_over_budget"] is False
    assert "src/target.py" in context_report["task_state"]["changed_paths"]
    assert context_report["compactions"]

    provider_error = next(scenario for scenario in summary["scenarios"] if scenario["id"] == "provider_error_recovery")
    provider_error_report = json.loads((output_dir / provider_error["report_path"]).read_text(encoding="utf-8"))
    assert provider_error_report["status"] == "failed"
    assert provider_error_report["stop_reason"] == "model_error"
    assert provider_error_report["prompt_metadata"]["provider_error"]["code"] == "rate_limited"
    assert provider_error_report["prompt_metadata"]["provider_error"]["retry_count"] == 2

    summary_path = output_dir / "gate8-real-session-acceptance.json"
    markdown_path = output_dir / "gate8-real-session-acceptance.md"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["status"] == "passed"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Gate8 Real Session Acceptance" in markdown
    assert "plan_todo_explore" in markdown
