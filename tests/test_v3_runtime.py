import json

from gencode.testing import ScriptedModelClient
from gencode import Engine, GenCode, SessionEventBus, SessionStore, WorkspaceContext


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


def read_session_events(agent):
    path = agent.session_event_bus.path
    assert path.exists()
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def event_names(agent):
    return [event["event"] for event in read_session_events(agent)]


def test_engine_drives_real_session_and_persists_event_timeline(tmp_path):
    agent = build_agent(tmp_path, ["<final>Done.</final>"])

    assert isinstance(agent.engine, Engine)
    assert isinstance(agent.session_event_bus, SessionEventBus)

    answer = agent.ask("ship v3")

    assert answer == "Done."
    assert agent.session_path.exists()
    assert (
        agent.session_event_bus.path
        == tmp_path / ".gencode" / "sessions" / f"{agent.session['id']}.events.jsonl"
    )
    assert event_names(agent) == [
        "session_started",
        "turn_started",
        "user_message",
        "context_orchestrator_decision",
        "context_usage_recorded",
        "model_requested",
        "model_parsed",
        "assistant_message",
        "turn_finished",
    ]


def test_engine_wraps_real_tools_with_session_events(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="notes/result.txt"><content>ok\n</content></tool>',
            "<final>Wrote it.</final>",
        ],
    )

    answer = agent.ask("write a file")

    assert answer == "Wrote it."
    assert (tmp_path / "notes" / "result.txt").read_text(encoding="utf-8") == "ok\n"
    events = read_session_events(agent)
    names = [event["event"] for event in events]
    assert "tool_started" in names
    assert "tool_finished" in names
    tool_finished = next(event for event in events if event["event"] == "tool_finished")
    assert tool_finished["tool_name"] == "write_file"
    assert tool_finished["status"] == "ok"
    assert tool_finished["workspace_changed"] is True


def test_plan_mode_allows_only_the_active_plan_artifact_until_plan_is_written(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path=".gencode/plans/v3-plan.md"><content># Plan\n- Build Engine\n</content></tool>',
            "<final>Plan ready.</final>",
        ],
        max_steps=3,
    )

    plan_path = agent.enter_plan_mode("v3")

    assert plan_path == ".gencode/plans/v3-plan.md"
    assert agent.runtime_mode == "plan"
    rejected = agent.run_tool(
        "write_file", {"path": "src.py", "content": "print('no')\n"}
    )
    assert "plan mode" in rejected
    assert not (tmp_path / "src.py").exists()

    answer = agent.ask("draft the v3 plan")

    assert answer == "Plan ready."
    assert agent.runtime_mode == "default"
    assert (
        (tmp_path / ".gencode" / "plans" / "v3-plan.md")
        .read_text(encoding="utf-8")
        .startswith("# Plan")
    )
    names = event_names(agent)
    assert names.count("runtime_mode_changed") == 2


def test_plan_mode_rejects_final_before_the_plan_artifact_exists(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "<final>Looks done.</final>",
            '<tool name="write_file" path=".gencode/plans/v3-plan.md"><content># Plan\n</content></tool>',
            "<final>Now done.</final>",
        ],
        max_steps=3,
    )

    agent.enter_plan_mode("v3")

    assert agent.ask("make a plan") == "Now done."
    assert any(
        "Plan mode requires writing" in item["content"]
        for item in agent.session["history"]
    )


def test_plan_mode_tools_enter_and_exit_runtime_mode(tmp_path):
    agent = build_agent(tmp_path, [])

    entered = agent.run_tool("enter_plan_mode", {"topic": "Refactor Auth"})

    assert "mode: plan" in entered
    assert ".gencode/plans/refactor-auth-plan.md" in entered
    assert agent.runtime_mode == "plan"
    assert agent.active_tool_profile.name == "plan"

    exited = agent.run_tool("exit_plan_mode", {})

    assert exited == "mode: default"
    assert agent.runtime_mode == "default"
    assert agent.active_tool_profile.name == "default"


def test_plan_path_accepts_absolute_path_inside_workspace(tmp_path):
    """模型偶尔给绝对路径，如 /Users/u/repo/.gencode/plans/foo —— 自动相对化，
    不应该让 agent 多走一次重试。"""
    from gencode.core.plan_mode import _plan_path

    assert (
        _plan_path("Student Mgmt", "/Users/u/repo/.gencode/plans/student-mgmt.md")
        == ".gencode/plans/student-mgmt.md"
    )
    assert (
        _plan_path("X", "./.gencode/plans/x-plan.md") == ".gencode/plans/x-plan.md"
    )
    # 真正越界的还是要拒
    import pytest

    with pytest.raises(ValueError, match="plan path must stay"):
        _plan_path("X", "/etc/passwd")
    with pytest.raises(ValueError, match="plan path must stay"):
        _plan_path("X", ".gencode/plans/../escape.md")


def test_provider_surface_allows_profiles_without_reintroducing_ollama_client():
    import gencode

    parser = gencode.build_arg_parser()
    provider_action = next(
        action for action in parser._actions if action.dest == "provider"
    )

    assert provider_action.choices is None
    assert not hasattr(gencode, "OllamaModelClient")
    assert parser.parse_args(["--provider", "deepseek"]).provider == "deepseek"
