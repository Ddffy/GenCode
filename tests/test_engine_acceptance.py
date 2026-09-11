"""End-to-end engine acceptance tests for user-visible turn behavior."""

import json
import subprocess
import sys

from gencode.testing import ScriptedModelClient
from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.providers import ProviderError


def build_agent(tmp_path, outputs, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".gencode" / "sessions")
    return GenCode(
        model_client=ScriptedModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        **kwargs,
    )


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_engine_streams_a_real_session_with_tool_artifacts(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="notes/result.txt"><content>ok\n</content></tool>',
            "<final>Wrote it.</final>",
        ],
    )

    events = list(agent.engine.run_turn("create the result file"))

    assert [event["type"] for event in events] == [
        "turn_started",
        "model_requested",
        "model_parsed",
        "tool_call",
        "tool_result",
        "model_requested",
        "model_parsed",
        "final",
        "turn_finished",
    ]
    assert events[-2]["content"] == "Wrote it."
    assert (tmp_path / "notes" / "result.txt").read_text(encoding="utf-8") == "ok\n"

    persisted_events = read_jsonl(agent.session_event_bus.path)
    assert [event["event"] for event in persisted_events][-7:] == [
        "tool_finished",
        "context_orchestrator_decision",
        "context_usage_recorded",
        "model_requested",
        "model_parsed",
        "assistant_message",
        "turn_finished",
    ]

    report_path = agent.current_run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["final_answer"] == "Wrote it."


def test_engine_reports_context_budget_summary_from_prompt_metadata(tmp_path):
    agent = build_agent(tmp_path, ["<final>Done.</final>"])

    list(agent.engine.run_turn("summarize context usage"))

    report = json.loads(
        (agent.current_run_dir / "report.json").read_text(encoding="utf-8")
    )
    summary = report["evidence_summaries"]["context_budget_summary"]
    usage = report["prompt_metadata"]["context_usage"]
    assert summary["schema_version"] == "gencode.context_budget_summary.v1"
    assert summary["budget_unit"] == "tokens_estimated"
    assert summary["token_estimator"] == "context_usage_analyzer"
    assert summary["estimated_tokens"] == usage["total_estimated_tokens"]
    assert summary["effective_window"] == (
        usage["context_window"] - usage["reserved_output_tokens"]
    )
    assert summary["prompt_changed_by_phase_3"] is False
    assert summary["reductions"] == []
    assert "pressure_tier" in summary
    assert "usage_source" in summary
    assert summary["snip_count"] == 0
    assert summary["prune_count"] == 0
    assert summary["summary_called"] is False
    assert summary["summary_delta_event_count"] == 0
    assert summary["replacement_cache_hits"] == 0
    assert summary["replacement_records_created"] == 0
    assert summary["replacement_ledger_enabled"] is True
    assert summary["provider_usage_available"] is False
    assert summary["saved_chars"] == 0
    assert summary["cached_tokens"] == 0


def test_engine_records_provider_error_as_failed_run(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            ProviderError(
                "rate limited",
                provider="openai",
                model="gpt-test",
                base_url="https://example.test/v1",
                code="rate_limited",
                http_status=429,
                retryable=True,
                attempts=3,
                retry_count=2,
            )
        ],
    )

    events = list(agent.engine.run_turn("call a rate limited provider"))

    assert events[-2]["type"] == "stop"
    assert "rate_limited" in events[-2]["content"]
    assert events[-2]["content"].startswith("模型错误")
    report = json.loads(
        (agent.current_run_dir / "report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "failed"
    assert report["stop_reason"] == "model_error"
    assert report["prompt_metadata"]["provider_error"]["code"] == "rate_limited"
    assert report["prompt_metadata"]["provider_error"]["retry_count"] == 2

    trace_events = read_jsonl(agent.current_run_dir / "trace.jsonl")
    model_error = next(
        event for event in trace_events if event["event"] == "model_error"
    )
    assert model_error["error"]["http_status"] == 429

    persisted_events = read_jsonl(agent.session_event_bus.path)
    assert any(
        event["event"] == "model_error" and event["code"] == "rate_limited"
        for event in persisted_events
    )


def test_worker_notification_drained_during_turn_is_streamed(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"agent","args":{"description":"Inspect","prompt":"Read README","subagent_type":"Explore"}}</tool>',
            "<final>Child done.</final>",
            "<final>Parent done.</final>",
        ],
        max_steps=3,
    )

    events = list(agent.engine.run_turn("delegate and continue"))

    notifications = [
        event for event in events if event["type"] == "worker_notification"
    ]
    assert len(notifications) == 1
    assert "<task-id>agent_1</task-id>" in notifications[0]["content"]


def test_verification_signal_passes_after_workspace_verification(tmp_path):
    command = subprocess.list2cmdline([sys.executable, "-m", "compileall", "notes"])
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="notes/result.py"><content>VALUE = 1\n</content></tool>',
            f'<tool>{{"name":"run_shell","args":{{"command":{json.dumps(command)},"timeout":20}}}}</tool>',
            "<final>Verified.</final>",
        ],
        max_steps=3,
    )

    events = list(agent.engine.run_turn("write and verify python code"))

    assert events[-2]["content"] == "Verified."
    report = json.loads(
        (agent.current_run_dir / "report.json").read_text(encoding="utf-8")
    )
    signal = report["evidence_summaries"]["verification_signal"]
    assert signal["schema_version"] == "gencode.verification_signal.v1"
    assert signal["state"] == "passed"
    assert signal["command"] == command
    assert signal["command_class"] == "compile"
    assert signal["after_last_workspace_change"] is True
    assert signal["changed_paths_present"] is True
    assert signal["covers_changed_paths"] is False
    assert signal["coverage_confidence"] == "unknown"
    assert "notes/result.py" in signal["changed_paths"]
