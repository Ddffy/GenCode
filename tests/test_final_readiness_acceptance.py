"""Acceptance tests for final-readiness behavior inside real engine turns."""

import json
import subprocess
import sys

from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.testing import ScriptedModelClient


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


def test_soft_final_readiness_does_not_warn_for_low_pressure_missing_provider_usage(tmp_path):
    agent = build_agent(
        tmp_path,
        ["<final>done</final>"],
        final_readiness_mode="soft",
    )

    events = list(agent.engine.run_turn("answer directly"))

    assert [event["type"] for event in events] == [
        "turn_started",
        "model_requested",
        "model_parsed",
        "final",
        "turn_finished",
    ]


def test_soft_final_readiness_reminds_once_then_allows_unchanged_final(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="notes/result.txt"><content>ok\n</content></tool>',
            "<final>Done without verification.</final>",
            "<final>Done without verification.</final>",
        ],
        final_readiness_mode="soft",
        max_steps=3,
    )

    events = list(agent.engine.run_turn("write the result"))

    assert [event["type"] for event in events if event["type"] == "runtime_notice"] == [
        "runtime_notice"
    ]
    assert events[-2]["type"] == "final"
    assert events[-2]["content"] == "Done without verification."

    trace = read_jsonl(agent.current_run_dir / "trace.jsonl")
    readiness = [event for event in trace if event["event"] == "final_readiness_decision"]
    assert [(event["decision"], event["reminder_already_sent"]) for event in readiness] == [
        ("remind", False),
        ("warn", True),
    ]
    report = json.loads(
        (agent.current_run_dir / "report.json").read_text(encoding="utf-8")
    )
    assert report["evidence_summaries"]["final_readiness_summary"]["remind_count"] == 1
    assert report["evidence_summaries"]["final_readiness_summary"]["warn_count"] == 1
    assert (
        report["evidence_summaries"]["final_readiness_summary"]["schema_version"]
        == "gencode.final_readiness_summary.v1"
    )


def test_strict_final_readiness_blocks_unverified_workspace_changes(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="notes/result.txt"><content>ok\n</content></tool>',
            "<final>Done without verification.</final>",
        ],
        final_readiness_mode="strict",
        max_steps=2,
    )

    events = list(agent.engine.run_turn("write the result"))

    stop_event = next(event for event in events if event["type"] == "stop")
    assert "Files changed" in stop_event["content"]
    assert events[-1]["stop_reason"] == "final_gate_blocked"

    trace = read_jsonl(agent.current_run_dir / "trace.jsonl")
    readiness = [event for event in trace if event["event"] == "final_readiness_decision"]
    assert [(event["decision"], event["action"]) for event in readiness] == [
        ("block", "block")
    ]

    report = json.loads(
        (agent.current_run_dir / "report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "stopped"
    assert report["stop_reason"] == "final_gate_blocked"
    assert report["evidence_summaries"]["final_readiness_summary"]["block_count"] == 1


def test_strict_final_readiness_blocks_partial_success_workspace_changes(tmp_path):
    script = (
        "from pathlib import Path; Path('notes/result.txt').parent.mkdir(exist_ok=True); "
        "Path('notes/result.txt').write_text('partial\\n'); raise SystemExit(1)"
    )
    command = subprocess.list2cmdline([sys.executable, "-c", script])
    agent = build_agent(
        tmp_path,
        [
            f'<tool>{{"name":"run_shell","args":{{"command":{json.dumps(command)},"timeout":20}}}}</tool>',
            "<final>Partial write is fine.</final>",
        ],
        final_readiness_mode="strict",
        max_steps=2,
    )

    events = list(agent.engine.run_turn("write the result with shell"))

    stop_event = next(event for event in events if event["type"] == "stop")
    assert "partially succeeded" in stop_event["content"]
    assert events[-1]["stop_reason"] == "final_gate_blocked"

    trace = read_jsonl(agent.current_run_dir / "trace.jsonl")
    tool_event = next(event for event in trace if event["event"] == "tool_executed")
    assert tool_event["status"] == "partial_success"
    assert tool_event["workspace_changed"] is True

    readiness = [event for event in trace if event["event"] == "final_readiness_decision"]
    assert [(event["decision"], event["action"]) for event in readiness] == [
        ("block", "block")
    ]


def test_soft_final_readiness_warns_for_net_negative_llm_compaction(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            """## Goal
Continue the large task.

## Files Read
- README.md

## Next Steps
- Finish the task.
""",
            "<final>Done after compact.</final>",
            "<final>Done after compact.</final>",
        ],
        final_readiness_mode="soft",
        max_steps=2,
    )
    agent.model_client.context_window = 1000
    agent.model_client.last_completion_metadata = {
        "input_tokens": 500,
        "output_tokens": 500,
        "total_tokens": 1000,
    }
    for index in range(5):
        agent.record({"role": "user", "content": f"request {index} " + ("x" * 900)})
        agent.record({"role": "assistant", "content": f"answer {index} " + ("y" * 900)})

    events = list(agent.engine.run_turn("finish"))

    assert any(event["type"] == "runtime_notice" for event in events)
    trace = read_jsonl(agent.current_run_dir / "trace.jsonl")
    readiness = [event for event in trace if event["event"] == "final_readiness_decision"]
    assert any(event["decision"] == "remind" for event in readiness)
    assert any("compact_net_negative" in event["reasons"] for event in readiness)
