"""Acceptance tests for todo ledger evidence in run reports."""

import json

from gencode.testing import ScriptedModelClient
from gencode import GenCode, SessionStore, WorkspaceContext


def build_agent(tmp_path, outputs=None, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    return GenCode(
        model_client=ScriptedModelClient(outputs or []),
        workspace=workspace,
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
        **kwargs,
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_todo_tools_persist_state_and_emit_session_events(tmp_path):
    agent = build_agent(tmp_path)

    added = agent.run_tool("todo_add", {"content": "Draft worker manager", "priority": "high"})
    updated = agent.run_tool("todo_update", {"todo_id": "todo_1", "status": "in_progress", "note": "started"})
    listed = agent.run_tool("todo_list", {})

    assert "todo_1" in added
    assert "in_progress" in updated
    assert "todo_1 [in_progress] high - Draft worker manager" in listed
    assert agent.session["todos"]["items"][0]["status"] == "in_progress"

    events = read_jsonl(agent.session_event_bus.path)
    todo_events = [event for event in events if event["event"] == "todo_changed"]
    assert [event["action"] for event in todo_events] == ["add", "update"]
    assert todo_events[-1]["todo"]["note"] == "started"


def test_todo_tools_are_available_in_plan_mode_and_prompt_context(tmp_path):
    agent = build_agent(tmp_path, ["<final>Plan ready.</final>"])
    agent.enter_plan_mode("subagent")

    assert "todo_add" in agent.available_tools()
    assert "todo_update" in agent.available_tools()
    assert "todo_list" in agent.available_tools()

    agent.run_tool("todo_add", {"content": "Write active plan", "status": "in_progress"})
    prompt = agent.prompt("continue")

    assert "Task ledger:" in prompt
    assert "todo_1 [in_progress]" in prompt
    assert "Use todo tools to keep the task ledger current." in prompt


def test_todo_changes_are_written_to_task_state_and_report(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"todo_add","args":{"content":"Implement Gate6","priority":"high"}}</tool>',
            '<tool>{"name":"todo_update","args":{"todo_id":"todo_1","status":"done","note":"verified"}}</tool>',
            "<final>Gate6 tracked.</final>",
        ],
        max_steps=4,
    )

    assert agent.ask("track Gate6") == "Gate6 tracked."

    report = json.loads((agent.current_run_dir / "report.json").read_text(encoding="utf-8"))
    task_state = json.loads((agent.current_run_dir / "task_state.json").read_text(encoding="utf-8"))

    assert report["todos"]["items"][0]["status"] == "done"
    assert [change["action"] for change in report["todo_changes"]] == ["add", "update"]
    assert task_state["todo_changes"] == report["todo_changes"]


def test_soft_final_readiness_reminds_for_current_run_high_priority_todo(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"todo_add","args":{"content":"Ship Gate6","priority":"high"}}</tool>',
            "<final>Gate6 is ready.</final>",
            "<final>Gate6 is ready.</final>",
        ],
        final_readiness_mode="soft",
        max_steps=3,
    )

    events = list(agent.engine.run_turn("track Gate6"))

    assert [event["type"] for event in events if event["type"] == "runtime_notice"] == [
        "runtime_notice"
    ]
    assert events[-2]["content"] == "Gate6 is ready."

    trace = read_jsonl(agent.current_run_dir / "trace.jsonl")
    readiness = [event for event in trace if event["event"] == "final_readiness_decision"]
    assert readiness[0]["reasons"] == ["unresolved_high_priority_todo"]
