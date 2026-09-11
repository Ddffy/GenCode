"""End-to-end validation for LLM handoff context compaction Phase 1."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_llm_handoff import (
    scenario_delta_too_small_skips_compaction,
    scenario_llm_failure_falls_back_to_deterministic,
    scenario_low_pressure_no_compaction,
    scenario_net_benefit_calculation,
    scenario_over_budget_prefers_deterministic,
    scenario_replacement_ledger_survives_llm_compact,
    scenario_tier3_triggers_llm_handoff,
)


def test_tier3_triggers_llm_handoff(tmp_path):
    result = scenario_tier3_triggers_llm_handoff(tmp_path)

    assert result["compact_trigger"] == "auto_pressure_compact"
    assert result["summary_mode"] == "llm"
    assert result["history_summary_kind"] == "compact_summary"
    assert result["structured_summary"] is True
    assert result["compact_call_usage"]["input_tokens"] > 0
    assert result["compact_usage_persisted"] is False
    assert result["final_answer"] == "auth middleware is ready"


def test_llm_failure_falls_back_to_deterministic(tmp_path):
    result = scenario_llm_failure_falls_back_to_deterministic(tmp_path)

    assert result["compact_trigger"] == "auto_pressure_compact"
    assert result["summary_mode"] == "deterministic_fallback"
    assert result["summary_text"].startswith("Compacted session summary:")
    assert result["final_answer"] == "fallback completed"


def test_low_pressure_no_compaction(tmp_path):
    result = scenario_low_pressure_no_compaction(tmp_path)

    assert result["compact_trigger"] is None
    assert result["summary_mode"] == ""
    assert result["compact_summary_count"] == 0
    assert result["compaction_event_count"] == 0


def test_over_budget_prefers_deterministic(tmp_path):
    result = scenario_over_budget_prefers_deterministic(tmp_path)

    assert result["compact_trigger"] == "auto_prompt_over_budget"
    assert result["summary_mode"] == "deterministic"
    assert result["compact_call_usage"] is None
    assert result["llm_prompt_count"] == 0


def test_delta_too_small_skips_compaction(tmp_path):
    result = scenario_delta_too_small_skips_compaction(tmp_path)

    assert result["compact_trigger"] is None
    assert result["should_compact"] is False
    assert result["compact_summary_count"] == 1
    assert result["model_prompt_count"] == 0


def test_replacement_ledger_survives_llm_compact(tmp_path):
    result = scenario_replacement_ledger_survives_llm_compact(tmp_path)

    assert result["ledger_before"] == result["ledger_after"]
    assert "event_abc" in result["ledger_after"]


def test_net_benefit_calculation():
    result = scenario_net_benefit_calculation()

    assert result["positive"] == 2000
    assert result["negative"] == -400
