from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.core.compact import CompactPlan
from gencode.core.context_manager import ContextManager
from gencode.testing import ScriptedModelClient


def build_agent(tmp_path, outputs=None):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return GenCode(
        model_client=ScriptedModelClient(outputs or []),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
    )


def add_turn(agent, index, content_size=24):
    agent.record({"role": "user", "content": f"request {index} " + ("x" * content_size)})
    agent.record({"role": "assistant", "content": f"answer {index} " + ("y" * content_size)})


def test_plan_is_deterministic_and_does_not_mutate_session(tmp_path):
    agent = build_agent(tmp_path)
    for index in range(4):
        add_turn(agent, index)
    history = list(agent.session["history"])

    plan = agent.compact_manager.plan(trigger="manual", keep_recent_turns=2)

    assert isinstance(plan, CompactPlan)
    assert plan.trigger == "manual"
    assert plan.keep_recent_turns == 2
    assert len(plan.delta_event_ids) == 4
    assert len(plan.protected_event_ids) == 4
    assert plan.no_op_reason is None
    assert agent.session["history"] == history
    assert "context_summary" not in agent.session


def test_first_compact_creates_boundary_before_protected_recent_turns(tmp_path):
    agent = build_agent(tmp_path)
    for index in range(5):
        add_turn(agent, index)
    last_delta_event_id = agent.session["history"][5]["event_id"]
    protected_event_ids = {item["event_id"] for item in agent.session["history"][6:]}

    summary = agent.compact_history(trigger="manual", keep_recent_turns=2)

    assert summary["summary_called"] is True
    assert summary["delta_event_count"] == 6
    assert summary["last_included_event_id"] == last_delta_event_id
    assert agent.session["context_summary"]["last_included_event_id"] == last_delta_event_id
    assert agent.session["context_summary"]["source_event_count"] == 6
    assert agent.session["history"][0]["kind"] == "compact_summary"
    assert {item["event_id"] for item in agent.session["history"][1:]} == protected_event_ids


def test_llm_compact_creates_handoff_summary_and_records_usage(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            """## Goal
Ship LLM handoff compaction.

## Files Read
- gencode/core/compact.py

## Key Decisions
- Use complete_model.

## Next Steps
- Run focused tests.
"""
        ],
    )
    agent.model_client.last_completion_metadata = {
        "input_tokens": 80,
        "output_tokens": 20,
        "total_tokens": 100,
        "cached_tokens": 5,
        "provider_protocol": "openai",
        "provider_model": "test-model",
    }
    for index in range(5):
        add_turn(agent, index)

    summary = agent.compact_history(trigger="manual", keep_recent_turns=2, summary_mode="llm")

    assert summary["summary_called"] is True
    assert summary["summary_mode"] == "llm"
    assert summary["compact_call_usage"] == {
        "input_tokens": 80,
        "output_tokens": 20,
        "total_tokens": 100,
        "cached_tokens": 5,
        "model": "test-model",
        "provider": "openai",
    }
    assert "## Goal\nShip LLM handoff compaction." in agent.session["history"][0]["content"]
    assert "- gencode/core/compact.py" in agent.session["history"][0]["content"]
    assert "compact_call_usage" not in agent.session["compactions"][-1]


def test_llm_compact_parse_failure_falls_back_to_deterministic(tmp_path):
    agent = build_agent(tmp_path, ["not a handoff summary"])
    for index in range(5):
        add_turn(agent, index)

    summary = agent.compact_history(trigger="manual", keep_recent_turns=2, summary_mode="llm")

    assert summary["summary_called"] is True
    assert summary["summary_mode"] == "deterministic_fallback"
    assert summary["compact_call_usage"]["input_tokens"] == 0
    assert agent.session["history"][0]["content"].startswith("Compacted session summary:")


def test_second_compact_only_summarizes_delta_since_previous_boundary(tmp_path):
    agent = build_agent(tmp_path)
    for index in range(5):
        add_turn(agent, index)
    first = agent.compact_history(trigger="manual", keep_recent_turns=2)
    first_summary_event_id = agent.session["context_summary"]["summary_event_id"]

    for index in range(5, 7):
        add_turn(agent, index)
    plan = agent.compact_manager.plan(trigger="manual", keep_recent_turns=2)
    second = agent.compact_history(trigger="manual", keep_recent_turns=2)

    assert plan.prior_summary_event_id == first_summary_event_id
    assert len(plan.delta_event_ids) == 4
    assert first["last_included_event_id"] != second["last_included_event_id"]
    assert second["source_event_count"] == first["source_event_count"] + 4
    assert "Incremental compacted delta:" in agent.session["history"][0]["content"]
    assert len(agent.session["history"]) == 5


def test_no_delta_compact_returns_summary_without_history_mutation(tmp_path):
    agent = build_agent(tmp_path)
    for index in range(2):
        add_turn(agent, index)
    before = list(agent.session["history"])

    summary = agent.compact_history(trigger="manual", keep_recent_turns=2)

    assert summary["summary_called"] is False
    assert summary["no_op_reason"]
    assert summary["pre_items"] == summary["post_items"] == len(before)
    assert summary["pre_tokens"] == summary["post_tokens"]
    assert agent.session["history"] == before
    assert "context_summary" not in agent.session


def test_existing_session_without_context_summary_compacts_successfully(tmp_path):
    agent = build_agent(tmp_path)
    for index in range(4):
        add_turn(agent, index)
    agent.session.pop("context_summary", None)

    summary = agent.compact_history(trigger="manual", keep_recent_turns=1)

    assert summary["summary_called"] is True
    assert agent.session["context_summary"]["summary_event_id"] == agent.session["history"][0]["event_id"]
    assert agent.session["context_summary"]["source_event_count"] == 6


def test_auto_over_budget_no_delta_records_no_summary_call(tmp_path):
    agent = build_agent(tmp_path, ["<final>done</final>"])
    for index in range(3):
        agent.record(
            {
                "role": "system",
                "kind": "compact_summary",
                "turn_id": f"compact_{index}",
                "content": "Compacted session summary:\n" + ("s" * 300),
            }
        )
    add_turn(agent, 99, content_size=300)
    before = list(agent.session["history"])
    agent.context_manager = ContextManager(
        agent,
        total_budget=100,
        section_budgets={"prefix": 40, "memory": 40, "relevant_memory": 40, "history": 40},
        section_floors={"prefix": 40, "memory": 40, "relevant_memory": 40, "history": 40},
    )

    assert agent.ask("finish") == "done"

    summary = agent.last_prompt_metadata["auto_compaction_summary"]
    assert summary["summary_called"] is False
    assert summary["no_op_reason"]
    assert agent.last_prompt_metadata["auto_compacted"] is False
    assert agent.session["history"][: len(before)] == before


def test_deterministic_compact_preserves_early_user_constraints(tmp_path):
    agent = build_agent(tmp_path)
    agent.record({"role": "user", "content": "修 compact，但是不要改公共 API。"})
    agent.record({"role": "assistant", "content": "收到。"})
    for index in range(4):
        add_turn(agent, index)

    agent.compact_history(trigger="manual", keep_recent_turns=1)

    summary = agent.session["history"][0]["content"]
    assert "- User constraints:" in summary
    assert "不要改公共 API" in summary


def test_deterministic_compact_empty_evidence_fields_render_dash(tmp_path):
    agent = build_agent(tmp_path)
    for index in range(4):
        add_turn(agent, index)

    agent.compact_history(trigger="manual", keep_recent_turns=1)

    summary = agent.session["history"][0]["content"]
    assert "- User constraints: -" in summary
    assert "- Key decisions: -" in summary
    assert "- Rejected paths: -" in summary
    assert "- Last error context: -" in summary
    assert "- Critical artifacts: -" in summary


def test_deterministic_compact_extracts_mixed_language_evidence(tmp_path):
    agent = build_agent(tmp_path)
    agent.record(
        {
            "role": "user",
            "content": "Please keep CLI stable. 不要改公共 API。Only change compact.py.",
        }
    )
    agent.record(
        {
            "role": "assistant",
            "content": "Decided to use rules because deterministic. Tried LLM but doesn't work.",
        }
    )
    agent.record(
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "gencode/core/compact.py"},
            "content": "class CompactManager: ...",
            "artifact_ref": "artifact://read/compact",
        }
    )
    agent.record(
        {
            "role": "tool",
            "name": "run_shell",
            "args": {"command": "uv run pytest tests/test_compact.py"},
            "content": "FAILED tests/test_compact.py::test_x - KeyError: input_tokens",
        }
    )
    for index in range(4):
        add_turn(agent, index)

    agent.compact_history(trigger="manual", keep_recent_turns=1)

    summary = agent.session["history"][0]["content"]
    assert "Please keep CLI stable" in summary
    assert "不要改公共 API" in summary
    assert "Only change compact.py" in summary
    assert "Decided to use rules because deterministic" in summary
    assert "Tried LLM but doesn't work" in summary
    assert "KeyError: input_tokens" in summary
    assert "gencode/core/compact.py" in summary
    assert "artifact://read/compact" in summary
