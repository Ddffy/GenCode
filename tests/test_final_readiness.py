"""Unit tests for final-readiness gate decisions and notices."""

from gencode.core.final_readiness import (
    evaluate_final_readiness,
    extract_required_artifact_paths,
    readiness_notice,
)
from gencode.core.task_state import TaskState


def task_state():
    return TaskState.create(task_id="task_1", run_id="run_1", user_request="demo")


def test_final_readiness_detects_unresolved_current_run_high_priority_todo():
    state = task_state()
    state.todo_changes = [
        {
            "action": "add",
            "todo": {
                "id": "todo_1",
                "priority": "high",
                "status": "pending",
            },
        }
    ]

    decision = evaluate_final_readiness(state, "strict")

    assert decision["decision"] == "warn"
    assert decision["action"] == "none"
    assert decision["reasons"] == ["unresolved_high_priority_todo"]


def test_final_readiness_uses_latest_current_run_todo_state():
    state = task_state()
    state.todo_changes = [
        {
            "action": "add",
            "todo": {"id": "todo_1", "priority": "high", "status": "pending"},
        },
        {
            "action": "update",
            "todo": {"id": "todo_1", "priority": "high", "status": "done"},
        },
    ]

    decision = evaluate_final_readiness(state, "strict")

    assert decision["decision"] == "allow"
    assert decision["reasons"] == []


def test_final_readiness_detects_unreduced_context_pressure():
    state = task_state()
    state.evidence_summaries = {
        "context_budget_summary": {
            "pressure_ratio": 0.98,
            "reductions": [],
        }
    }

    decision = evaluate_final_readiness(state, "strict")

    assert decision["decision"] == "warn"
    assert decision["action"] == "none"
    assert decision["reasons"] == ["context_pressure_without_reduction"]


def test_final_readiness_allows_context_pressure_after_successful_reduction():
    state = task_state()
    state.evidence_summaries = {
        "context_budget_summary": {
            "pressure_ratio": 0.98,
            "reductions": [{"source": "microcompact", "saved_chars": 100}],
        }
    }

    decision = evaluate_final_readiness(state, "strict")

    assert decision["decision"] == "allow"
    assert decision["reasons"] == []


def test_final_readiness_reports_context_observability_gaps():
    state = task_state()
    state.evidence_summaries = {
        "context_budget_summary": {
            "pressure_tier": "tier3_summary",
            "pressure_ratio": 0.96,
            "reductions": [{"source": "microcompact", "saved_chars": 100}],
            "summary_called": True,
            "summary_delta_event_count": 0,
            "replacement_ledger_enabled": False,
            "provider_usage_available": False,
        }
    }

    decision = evaluate_final_readiness(state, "strict")

    assert decision["decision"] == "warn"
    assert decision["reasons"] == [
        "tier3_summary_without_delta",
        "replacement_ledger_disabled_under_pressure",
        "provider_real_token_usage_unavailable",
    ]


def test_final_readiness_allows_missing_provider_usage_at_low_pressure():
    state = task_state()
    state.evidence_summaries = {
        "context_budget_summary": {
            "pressure_tier": "tier0_observe",
            "pressure_ratio": 0.2,
            "provider_usage_available": False,
        }
    }

    decision = evaluate_final_readiness(state, "strict")

    assert decision["decision"] == "allow"
    assert decision["reasons"] == []


def test_final_readiness_warns_on_negative_llm_compact_net_benefit():
    state = task_state()
    state.evidence_summaries = {
        "context_budget_summary": {
            "summary_mode": "llm",
            "compact_net_benefit_tokens": -25,
        }
    }

    decision = evaluate_final_readiness(state, "strict")

    assert decision["decision"] == "warn"
    assert decision["reasons"] == ["compact_net_negative"]


def test_final_readiness_allows_non_negative_or_unknown_compact_net_benefit():
    for net in (0, 50, None):
        state = task_state()
        state.evidence_summaries = {
            "context_budget_summary": {
                "summary_mode": "llm",
                "compact_net_benefit_tokens": net,
            }
        }

        decision = evaluate_final_readiness(state, "strict")

        assert decision["decision"] == "allow"
        assert decision["reasons"] == []


def test_final_readiness_warns_on_low_quality_llm_compact_summary():
    state = task_state()
    state.evidence_summaries = {
        "context_budget_summary": {
            "summary_mode": "llm",
            "compact_summary_has_next_steps": True,
            "compact_summary_has_file_references": False,
        }
    }

    decision = evaluate_final_readiness(state, "strict")

    assert decision["decision"] == "warn"
    assert decision["reasons"] == ["compact_summary_quality_low"]


def test_final_readiness_ignores_deterministic_compact_summary_quality():
    state = task_state()
    state.evidence_summaries = {
        "context_budget_summary": {
            "summary_mode": "deterministic",
            "compact_summary_has_next_steps": False,
            "compact_summary_has_file_references": False,
        }
    }

    decision = evaluate_final_readiness(state, "strict")

    assert decision["decision"] == "allow"
    assert decision["reasons"] == []


def test_final_readiness_blocks_tier3_compaction_without_token_savings():
    state = task_state()
    state.evidence_summaries = {
        "context_budget_summary": {
            "pressure_tier": "tier3_summary",
            "pressure_ratio": 0.90,
            "pre_compact_estimated_tokens": 1200,
            "post_compact_estimated_tokens": 1200,
            "reductions": [{"source": "microcompact", "saved_chars": 1}],
        }
    }

    decision = evaluate_final_readiness(state, "strict")

    assert decision["decision"] == "block"
    assert decision["action"] == "block"
    assert decision["reasons"] == ["context_pressure_compaction_failed"]


def test_final_readiness_blocks_partial_success_workspace_change():
    state = task_state()
    state.runtime_reminders = [
        {
            "event": "tool_executed",
            "tool": "run_shell",
            "status": "partial_success",
            "workspace_changed": True,
            "affected_paths": ["notes/result.txt"],
        }
    ]

    decision = evaluate_final_readiness(state, "strict")

    assert decision["decision"] == "block"
    assert decision["action"] == "block"
    assert decision["reasons"] == ["partial_success_workspace_changed"]


def test_required_artifact_extraction_tracks_output_directory(tmp_path):
    prompt = f"""
输入文件：
- `provider_capabilities.json`

请完成以下产物，全部写入 `{tmp_path}/out/`：
1. `provider_scorecard.json`
2. `openclaw_config_patch.json`
3. `failover_playbook.md`

## 执行约束
- 不要修改 `test_config.py`
"""

    paths = extract_required_artifact_paths(prompt, tmp_path)

    assert paths == [
        "out/provider_scorecard.json",
        "out/openclaw_config_patch.json",
        "out/failover_playbook.md",
    ]


def test_required_artifact_extraction_ignores_negated_output_requests(tmp_path):
    prompt = "Do not create `forbidden.py`.\n请生成 `final_report.md`。"

    paths = extract_required_artifact_paths(prompt, tmp_path)

    assert paths == ["final_report.md"]


def test_required_artifact_extraction_keeps_mixed_input_output_line_scoped(tmp_path):
    prompt = "Use input file `source.json` and produce `result.json`."

    paths = extract_required_artifact_paths(prompt, tmp_path)

    assert paths == ["result.json"]


def test_required_artifact_extraction_clears_output_context_at_plain_constraints(tmp_path):
    prompt = f"""
Create `final_report.md` under `{tmp_path}/out/`.

Constraints:
- keep `config.yaml` unchanged
"""

    paths = extract_required_artifact_paths(prompt, tmp_path)

    assert paths == ["out/final_report.md"]


def test_required_artifact_extraction_ignores_do_not_modify_after_output(tmp_path):
    prompt = "Please create `final_report.md`.\nDo not modify `test_config.py`."

    paths = extract_required_artifact_paths(prompt, tmp_path)

    assert paths == ["final_report.md"]


def test_final_readiness_detects_missing_required_artifacts(tmp_path):
    state = task_state()
    state.user_request = "请生成 `final_report.md` 和 `progress.md`。"
    (tmp_path / "progress.md").write_text("done\n", encoding="utf-8")

    decision = evaluate_final_readiness(state, "soft", workspace_root=tmp_path)

    assert decision["decision"] == "remind"
    assert decision["action"] == "runtime_notice"
    assert decision["reasons"] == ["missing_required_artifact"]
    summary = decision["required_artifact_summary"]
    assert summary["declared_paths"] == ["final_report.md", "progress.md"]
    assert summary["missing_paths"] == ["final_report.md"]


def test_readiness_notice_uses_catalog_messages_not_raw_codes():
    notice = readiness_notice(
        {
            "action": "runtime_notice",
            "reasons": ["changed_paths_without_verification"],
        }
    )

    assert "changed_paths_without_verification" not in notice
    assert "Files changed" in notice
    assert "successful verification" in notice


def test_final_readiness_summary_has_schema_version():
    from gencode.core.final_readiness import reduce_final_readiness_summary

    summary = reduce_final_readiness_summary({}, {"decision": "warn", "reasons": []})

    assert summary["schema_version"] == "gencode.final_readiness_summary.v1"
