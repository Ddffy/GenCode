import json

from gencode.testing import ScriptedModelClient
from gencode import GenCode, SessionStore, WorkspaceContext


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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_runtime_evidence_graph_and_verifier_are_derived_from_real_tool_run(tmp_path):
    (tmp_path / "package.json").write_text(
        '{"scripts":{"test":"vitest run","build":"vite build"}}\n',
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_smoke.py").write_text("def test_smoke():\n    assert True\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="src/api.py"><content>@app.get("/api/items")\ndef list_items():\n    return fetch("/api/users")\n</content></tool>',
            "<final>Wrote API file.</final>",
        ],
    )

    assert agent.ask("add an api file") == "Wrote API file."

    report = json.loads((agent.current_run_dir / "report.json").read_text(encoding="utf-8"))
    task_state = json.loads((agent.current_run_dir / "task_state.json").read_text(encoding="utf-8"))
    graph = report["artifact_graph"]

    assert graph["changed_paths"] == ["src/api.py"]
    assert graph["categories"]["backend"] == ["src/api.py"]
    assert "/api/items" in graph["route_refs"]
    assert "/api/users" in graph["api_refs"]
    assert task_state["artifact_graph"] == graph

    commands = [item["command"] for item in report["verifier_suggestions"]]
    assert "npm test" in commands
    assert "npm run build" in commands
    assert "uv run python -m pytest -q" in commands
    assert task_state["verifier_suggestions"] == report["verifier_suggestions"]

    trace_events = read_jsonl(agent.current_run_dir / "trace.jsonl")
    tool_event = next(event for event in trace_events if event["event"] == "tool_executed")
    assert tool_event["phase"] == "tool"
    assert tool_event["status"] == "ok"
    assert tool_event["turn_id"] == agent.current_task_state.task_id
    assert tool_event["artifact_paths"] == ["src/api.py"]
    assert tool_event["span_id"]


def test_runtime_reminder_records_failed_tool_without_breaking_the_turn(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="patch_file" path="missing.py"><old_text>x</old_text><new_text>y</new_text></tool>',
            "<final>Could not patch missing file.</final>",
        ],
    )

    assert agent.ask("patch missing file") == "Could not patch missing file."

    report = json.loads((agent.current_run_dir / "report.json").read_text(encoding="utf-8"))
    reminders = report["runtime_reminders"]

    assert reminders
    assert reminders[-1]["event"] == "tool_executed"
    assert reminders[-1]["tool"] == "patch_file"
    assert reminders[-1]["status"] == "rejected"
    assert reminders[-1]["message"]
    assert json.loads((agent.current_run_dir / "task_state.json").read_text(encoding="utf-8"))["runtime_reminders"] == reminders
