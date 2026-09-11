"""Tests for artifact-backed retention of long tool results."""

import hashlib
import json
import subprocess
import sys

from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.core.context_manager import ContextManager
from gencode.core.run_store import RunStore
from gencode.testing import ScriptedModelClient
from gencode.tools.base import RegisteredTool


def build_agent(tmp_path, outputs=None, **kwargs):
    (tmp_path / "README.md").write_text("hello world\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".gencode" / "sessions")
    return GenCode(
        model_client=ScriptedModelClient(outputs or []),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        **kwargs,
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_long_shell_output_is_clipped_and_full_output_is_saved_as_run_artifact(tmp_path):
    script = "print('x'*6000)"
    command = subprocess.list2cmdline([sys.executable, "-c", script])
    agent = build_agent(
        tmp_path,
        [
            f'<tool>{{"name":"run_shell","args":{{"command":{json.dumps(command)},"timeout":20}}}}</tool>',
            "<final>captured</final>",
        ],
    )

    assert agent.ask("produce long shell output") == "captured"

    tool_item = next(item for item in agent.session["history"] if item["role"] == "tool" and item["name"] == "run_shell")
    assert len(tool_item["content"]) < 1200
    assert "full output saved:" in tool_item["content"]
    assert "full output saved:" in agent.model_client.prompts[1]

    report = json.loads((agent.current_run_dir / "report.json").read_text(encoding="utf-8"))
    artifact_path = report["runtime_reminders"][0]["artifact_path"] if report["runtime_reminders"] else agent._last_tool_result_metadata["full_output_artifact"]
    full_output = (tmp_path / artifact_path).read_text(encoding="utf-8")
    assert "x" * 6000 in full_output
    assert tool_item["content_sha256"] == hashlib.sha256(full_output.encode("utf-8")).hexdigest()

    trace_events = read_jsonl(agent.current_run_dir / "trace.jsonl")
    tool_event = next(event for event in trace_events if event["event"] == "tool_executed")
    assert tool_event["full_output_artifact"] == artifact_path
    assert tool_event["content_sha256"] == tool_item["content_sha256"]


def test_run_shell_status_is_parsed_from_full_result_before_artifact_rendering(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"run_shell","args":{"command":"synthetic long failure","timeout":20}}</tool>',
            "<final>captured</final>",
        ],
    )
    long_stdout = "x" * 3000
    agent.tools["run_shell"] = RegisteredTool(
        name="run_shell",
        schema={"command": "str", "timeout": "int=20"},
        description="Synthetic shell command.",
        risky=True,
        runner=lambda args: f"stdout:\n{long_stdout}\nexit_code: 1\nstderr:\nboom",
    )

    assert agent.ask("run synthetic shell") == "captured"

    trace_events = read_jsonl(agent.current_run_dir / "trace.jsonl")
    tool_event = next(event for event in trace_events if event["event"] == "tool_executed")
    assert tool_event["status"] == "error"
    assert tool_event["tool_error_code"] == "tool_failed"
    assert tool_event["full_output_artifact"]


def test_long_tool_output_artifact_ref_survives_external_run_store(tmp_path):
    external_runs = tmp_path.parent / f"{tmp_path.name}-external-runs"
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"run_shell","args":{"command":"synthetic long output","timeout":20}}</tool>',
            "<final>captured</final>",
        ],
        run_store=RunStore(external_runs),
    )
    agent.tools["run_shell"] = RegisteredTool(
        name="run_shell",
        schema={"command": "str", "timeout": "int=20"},
        description="Synthetic shell command.",
        risky=True,
        runner=lambda args: "exit_code: 0\nstdout:\n" + ("x" * 3000),
    )

    assert agent.ask("run synthetic shell") == "captured"

    tool_item = next(
        item
        for item in agent.session["history"]
        if item["role"] == "tool" and item["name"] == "run_shell"
    )
    artifact_ref = tool_item["artifact_ref"]
    assert artifact_ref
    assert (external_runs.parent / artifact_ref).exists()


def test_long_read_file_result_is_artifact_backed_when_history_is_microcompacted(tmp_path):
    large_text = "\n".join(f"line-{index} " + ("x" * 40) for index in range(120))
    (tmp_path / "large.txt").write_text(large_text, encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"large.txt","start":1,"end":120}}</tool>',
            "<final>read</final>",
            "<final>committed</final>",
        ],
    )

    assert agent.ask("read the large file") == "read"
    tool_item = next(item for item in agent.session["history"] if item["role"] == "tool")
    original_history_content = tool_item["content"]
    artifact_ref = tool_item["artifact_ref"]

    assert artifact_ref.endswith(".txt")
    assert tool_item["original_chars"] > len(original_history_content)
    assert "line-119" in (tmp_path / artifact_ref).read_text(encoding="utf-8")

    for index in range(4):
        agent.record({"role": "user", "content": f"later user {index}"})
        agent.record({"role": "assistant", "content": f"later answer {index}"})

    before_history = json.dumps(agent.session["history"], sort_keys=True)
    prompt, metadata = ContextManager(agent).build("continue")

    persisted_tool_item = next(
        item for item in agent.session["history"] if item.get("artifact_ref") == artifact_ref
    )
    assert json.dumps(agent.session["history"], sort_keys=True) == before_history
    assert "context_replacements" not in agent.session
    assert persisted_tool_item["content"] == original_history_content
    assert artifact_ref in prompt
    assert "line-119" not in prompt
    assert metadata["history"]["microcompact_artifact_refs"] == [artifact_ref]
    assert metadata["history"]["microcompact_saved_chars"] > 0
    assert metadata["history"]["proposed_replacements"]

    assert agent.ask("continue") == "committed"

    event_id = persisted_tool_item["event_id"]
    assert agent.session["context_replacements"][event_id]["content_sha256"] == persisted_tool_item["content_sha256"]
    assert agent.session["context_replacements"][event_id]["artifact_ref"] == artifact_ref


def test_recent_long_tool_result_is_not_microcompact_stubbed(tmp_path):
    large_text = "\n".join(f"line-{index} " + ("x" * 40) for index in range(120))
    (tmp_path / "large.txt").write_text(large_text, encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"large.txt","start":1,"end":120}}</tool>',
            "<final>read</final>",
        ],
    )

    assert agent.ask("read the large file") == "read"
    prompt, metadata = ContextManager(agent).build("continue")

    assert "read_file output saved:" not in prompt
    assert "full output saved:" in prompt
    assert metadata["history"]["microcompact_artifact_refs"] == []


def test_microcompact_keeps_old_tool_result_tied_to_current_changed_path(tmp_path):
    large_text = "\n".join(f"line-{index} " + ("x" * 40) for index in range(120))
    (tmp_path / "large.txt").write_text(large_text, encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"large.txt","start":1,"end":120}}</tool>',
            "<final>read</final>",
        ],
    )

    assert agent.ask("read the large file") == "read"
    for index in range(4):
        agent.record({"role": "user", "content": f"later user {index}"})
        agent.record({"role": "assistant", "content": f"later answer {index}"})
    agent.current_task_state.changed_paths = ["large.txt"]

    prompt, metadata = ContextManager(agent).build("continue")

    assert "read_file output saved:" not in prompt
    assert "full output saved:" in prompt
    assert metadata["history"]["microcompact_artifact_refs"] == []


def test_microcompact_keeps_latest_failed_tool_result_visible(tmp_path):
    script = "import sys; [print(f'FAIL-{i}') for i in range(140)]; raise SystemExit(1)"
    command = subprocess.list2cmdline([sys.executable, "-c", script])
    agent = build_agent(
        tmp_path,
        [
            f'<tool>{{"name":"run_shell","args":{{"command":{json.dumps(command)},"timeout":20}}}}</tool>',
            "<final>captured failure</final>",
        ],
    )

    assert agent.ask("capture a long failure") == "captured failure"
    for index in range(4):
        agent.record({"role": "user", "content": f"later-{index}"})
        agent.record({"role": "assistant", "content": f"done-{index}"})

    prompt, metadata = ContextManager(agent).build("continue")

    assert "FAIL-0" in prompt
    assert "run_shell output saved:" not in prompt
    assert metadata["history"]["microcompact_artifact_refs"] == []


def test_microcompact_keeps_latest_workspace_changing_tool_result_visible(tmp_path):
    script = "from pathlib import Path; Path('notes').mkdir(exist_ok=True); Path('notes/out.txt').write_text('ok\\n'); [print(f'CHANGED-{i}') for i in range(140)]"
    command = subprocess.list2cmdline([sys.executable, "-c", script])
    agent = build_agent(
        tmp_path,
        [
            f'<tool>{{"name":"run_shell","args":{{"command":{json.dumps(command)},"timeout":20}}}}</tool>',
            "<final>captured change</final>",
        ],
    )

    assert agent.ask("capture a long workspace change") == "captured change"
    for index in range(4):
        agent.record({"role": "user", "content": f"later-{index}"})
        agent.record({"role": "assistant", "content": f"done-{index}"})

    prompt, metadata = ContextManager(agent).build("continue")

    assert "CHANGED-0" in prompt
    assert "run_shell output saved:" not in prompt
    assert metadata["history"]["microcompact_artifact_refs"] == []
