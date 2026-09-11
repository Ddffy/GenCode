from gencode.testing import ScriptedModelClient
from gencode import GenCode, SessionStore, WorkspaceContext


def build_agent(tmp_path, outputs, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    return GenCode(
        model_client=ScriptedModelClient(outputs),
        workspace=workspace,
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
        **kwargs,
    )


def test_ask_user_tool_returns_callback_choice(tmp_path):
    agent = build_agent(
        tmp_path, [], ask_user_callback=lambda question, choices: choices[1]
    )

    result = agent.run_tool("ask_user", {"question": "Ship?", "choices": ["no", "yes"]})

    assert result == "yes"


def test_ask_user_tool_fails_closed_without_interactive_callback(tmp_path):
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("ask_user", {"question": "Ship?", "choices": ["yes"]})

    assert result == "error: ask_user requires interactive mode"


def test_plan_mode_allows_ask_user_tool(tmp_path):
    agent = build_agent(
        tmp_path, [], ask_user_callback=lambda question, choices: choices[0]
    )
    agent.enter_plan_mode("release")

    result = agent.run_tool(
        "ask_user", {"question": "Which release?", "choices": ["staging", "prod"]}
    )

    assert result == "staging"
