import json

from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.testing import ScriptedModelClient


VALID_HANDOFF = """## Goal
Continue the task.

## Files Read
- README.md

## Next Steps
- Continue from the compact summary.
"""


def build_agent(tmp_path, outputs=None):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return GenCode(
        model_client=ScriptedModelClient(outputs or []),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
    )


def add_turn(agent, index):
    agent.record({"role": "user", "content": f"request {index} " + ("x" * 80)})
    agent.record({"role": "assistant", "content": f"answer {index} " + ("y" * 80)})


def compact_output(agent, command):
    from gencode.cli import handle_repl_command

    handled, should_exit, output = handle_repl_command(agent, command)

    assert handled is True
    assert should_exit is False
    return json.loads(output)


def test_compact_cli_defaults_to_deterministic(tmp_path):
    agent = build_agent(tmp_path)
    for index in range(5):
        add_turn(agent, index)

    output = compact_output(agent, "/compact")

    assert output["summary_mode"] == "deterministic"
    assert output["summary_called"] is True
    assert "compact_call_usage" not in output


def test_compact_cli_llm_uses_handoff_mode(tmp_path):
    agent = build_agent(tmp_path, [VALID_HANDOFF])
    agent.model_client.last_completion_metadata = {
        "input_tokens": 40,
        "output_tokens": 10,
        "total_tokens": 50,
    }
    for index in range(5):
        add_turn(agent, index)

    output = compact_output(agent, "/compact --llm")

    assert output["summary_mode"] == "llm"
    assert output["compact_call_usage"]["total_tokens"] == 50
    assert "net_benefit_tokens" in output


def test_compact_cli_auto_low_pressure_uses_deterministic(tmp_path):
    agent = build_agent(tmp_path, [VALID_HANDOFF])
    agent.last_prompt_metadata = {
        "context_usage": {"pressure_tier": "tier1_warn"},
    }
    for index in range(5):
        add_turn(agent, index)

    output = compact_output(agent, "/compact --auto")

    assert output["summary_mode"] == "deterministic"
    assert agent.model_client.prompts == []


def test_compact_cli_auto_tier3_uses_llm(tmp_path):
    agent = build_agent(tmp_path, [VALID_HANDOFF])
    agent.last_prompt_metadata = {
        "context_usage": {"pressure_tier": "tier3_summary"},
    }
    for index in range(5):
        add_turn(agent, index)

    output = compact_output(agent, "/compact --auto")

    assert output["summary_mode"] == "llm"
    assert agent.model_client.prompts


def test_context_command_reports_llm_handoff_status(tmp_path):
    from gencode.cli import handle_repl_command

    agent = build_agent(tmp_path)
    agent.last_prompt_metadata = {
        "context_usage": {"pressure_tier": "tier3_summary"},
        "context_orchestrator": {
            "summary_mode": "llm",
            "pressure_tier": "tier3_summary",
            "pre_compact_estimated_tokens": 1200,
            "post_compact_estimated_tokens": 800,
            "compact_call_usage": {"total_tokens": 50},
        },
    }

    handled, should_exit, output = handle_repl_command(agent, "/context")

    payload = json.loads(output)
    assert handled is True
    assert should_exit is False
    assert payload["llm_handoff_status"] == {
        "last_compact_mode": "llm",
        "compact_call_tokens": 50,
        "net_benefit_tokens": 350,
        "handoff_armed": True,
    }
