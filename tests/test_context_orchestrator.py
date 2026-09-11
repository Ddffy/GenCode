from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.core.context_budget_summary import context_budget_summary
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


def test_orchestrator_build_wraps_context_manager_without_low_pressure_prompt_drift(tmp_path):
    agent = build_agent(tmp_path)
    agent.memory.append_note("deploy key is red", tags=("deploy",), created_at="2026-04-07T10:00:00+00:00")
    expected_prompt, _ = ContextManager(agent).build("Where is the deploy key?")

    snapshot = agent.context_orchestrator.snapshot(
        "Where is the deploy key?",
        prefix_refresh={"workspace_changed": False, "prefix_changed": False},
    )
    result = agent.context_orchestrator.build(snapshot)

    assert result.prompt == expected_prompt
    assert result.should_compact is False
    assert result.compact_trigger is None
    assert result.metadata["context_orchestrator"]["version"] == "local-v1"
    assert result.metadata["context_orchestrator"]["summary_called"] is False
    assert result.metadata["context_orchestrator"]["replacement_cache_hits"] == 0
    assert result.metadata["prefix_hash"] == agent.prefix_state.hash


def test_orchestrator_build_records_no_op_auto_compaction_decision(tmp_path):
    agent = build_agent(tmp_path)
    for index in range(3):
        agent.record(
            {
                "role": "system",
                "kind": "compact_summary",
                "turn_id": f"compact_{index}",
                "content": "Compacted session summary:\n" + ("s" * 300),
            }
        )
    for role in ("user", "assistant"):
        agent.record({"role": role, "content": role + ("x" * 300), "turn_id": "recent"})
    agent.context_manager = ContextManager(
        agent,
        total_budget=100,
        section_budgets={"prefix": 10_000, "memory": 10_000, "relevant_memory": 10_000, "history": 40_000},
        section_floors={"prefix": 10_000, "memory": 10_000, "relevant_memory": 10_000, "history": 40_000},
    )

    snapshot = agent.context_orchestrator.snapshot("finish")
    result = agent.context_orchestrator.build(snapshot)

    assert result.should_compact is False
    assert result.metadata["auto_compacted"] is False
    assert result.metadata["auto_compaction_summary"]["summary_called"] is False
    assert result.metadata["context_orchestrator"]["compact_trigger"] == "auto_prompt_over_budget"


def test_orchestrator_tier3_triggers_llm_compaction_before_prompt_over_budget(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            """## Goal
Continue the large task.

## Next Steps
- Use the compacted handoff.
"""
        ],
    )
    agent.model_client.context_window = 1000
    agent.model_client.last_completion_metadata = {
        "input_tokens": 90,
        "output_tokens": 10,
        "total_tokens": 100,
        "provider_protocol": "openai",
        "provider_model": "test-model",
    }
    for index in range(5):
        agent.record({"role": "user", "content": f"request {index} " + ("x" * 900)})
        agent.record({"role": "assistant", "content": f"answer {index} " + ("y" * 900)})

    snapshot = agent.context_orchestrator.snapshot("finish")
    result = agent.context_orchestrator.build(snapshot)

    assert result.metadata["prompt_over_budget"] is False
    assert result.metadata["context_usage"]["pressure_tier"] == "tier3_summary"
    assert result.should_compact is True
    assert result.compact_trigger == "auto_pressure_compact"
    assert result.metadata["auto_compaction_summary"]["summary_mode"] == "llm"
    assert result.metadata["context_orchestrator"]["compact_call_usage"]["total_tokens"] == 100
    assert context_budget_summary(result.metadata)["compact_call_usage"]["provider"] == "openai"


def test_orchestrator_tier3_insufficient_delta_does_not_compact(tmp_path):
    agent = build_agent(tmp_path, ["unused"])
    agent.model_client.context_window = 200
    agent.record({"role": "user", "content": "request " + ("x" * 900)})
    agent.record({"role": "assistant", "content": "answer " + ("y" * 900)})

    snapshot = agent.context_orchestrator.snapshot("finish")
    result = agent.context_orchestrator.build(snapshot)

    assert result.metadata["context_usage"]["pressure_tier"] == "tier3_summary"
    assert result.should_compact is False
    assert result.compact_trigger is None
    assert agent.model_client.prompts == []


def test_orchestrator_explains_over_budget_short_history_without_compaction(tmp_path):
    agent = build_agent(tmp_path)
    agent.context_manager = ContextManager(
        agent,
        total_budget=80,
        section_budgets={"prefix": 40, "memory": 40, "relevant_memory": 40, "history": 40},
        section_floors={"prefix": 40, "memory": 40, "relevant_memory": 40, "history": 40},
    )

    snapshot = agent.context_orchestrator.snapshot("finish")
    result = agent.context_orchestrator.build(snapshot)

    assert result.metadata["prompt_over_budget"] is True
    assert result.metadata["auto_compacted"] is False
    assert result.metadata["auto_compaction_skip_reason"] == "history_too_short_for_auto_compaction"
    assert result.metadata["context_orchestrator"]["skip_reason"] == "history_too_short_for_auto_compaction"


def test_orchestrator_prompt_over_budget_uses_deterministic_compaction(tmp_path):
    agent = build_agent(tmp_path, ["unused"])
    for index in range(5):
        agent.record({"role": "user", "content": f"request {index} " + ("x" * 300)})
        agent.record({"role": "assistant", "content": f"answer {index} " + ("y" * 300)})
    agent.context_manager = ContextManager(
        agent,
        total_budget=100,
        section_budgets={"prefix": 10_000, "memory": 10_000, "relevant_memory": 10_000, "history": 40_000},
        section_floors={"prefix": 10_000, "memory": 10_000, "relevant_memory": 10_000, "history": 40_000},
    )

    result = agent.context_orchestrator.build(agent.context_orchestrator.snapshot("finish"))

    assert result.should_compact is True
    assert result.compact_trigger == "auto_prompt_over_budget"
    assert result.metadata["auto_compaction_summary"]["summary_mode"] == "deterministic"
    assert agent.model_client.prompts == []


def test_orchestrator_prompt_over_budget_wins_over_tier3_llm(tmp_path):
    agent = build_agent(tmp_path, ["unused"])
    agent.model_client.context_window = 1000
    for index in range(5):
        agent.record({"role": "user", "content": f"request {index} " + ("x" * 900)})
        agent.record({"role": "assistant", "content": f"answer {index} " + ("y" * 900)})
    agent.context_manager = ContextManager(
        agent,
        total_budget=100,
        section_budgets={"prefix": 10_000, "memory": 10_000, "relevant_memory": 10_000, "history": 40_000},
        section_floors={"prefix": 10_000, "memory": 10_000, "relevant_memory": 10_000, "history": 40_000},
    )

    result = agent.context_orchestrator.build(agent.context_orchestrator.snapshot("finish"))

    assert result.metadata["prompt_over_budget"] is True
    assert result.metadata["context_usage"]["pressure_tier"] == "tier3_summary"
    assert result.compact_trigger == "auto_prompt_over_budget"
    assert result.metadata["auto_compaction_summary"]["summary_mode"] == "deterministic"
    assert agent.model_client.prompts == []
