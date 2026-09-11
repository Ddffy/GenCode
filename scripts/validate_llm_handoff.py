#!/usr/bin/env python3
"""End-to-end validation of LLM handoff context compaction Phase 1."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.core.context_budget_summary import context_budget_summary
from gencode.core.context_manager import ContextManager
from gencode.testing import ScriptedModelClient


HANDOFF_LLM_OUTPUT = """## Goal
Implement authentication middleware

## Files Read
- src/auth.js
- src/config/jwt.js

## Files Modified
- src/auth.js

## Key Decisions
- Use RS256 for JWT

## Next Steps
- Add token refresh endpoint
- Write integration tests for auth flow
"""


def scenario_tier3_triggers_llm_handoff(tmp_path=None):
    tmp_path = _tmp_path(tmp_path)
    agent = _build_agent(
        tmp_path,
        [HANDOFF_LLM_OUTPUT, "<final>auth middleware is ready</final>"],
        context_window=1000,
    )
    _fill_history(agent, rounds=6, chars_per_message=900)

    final_answer = agent.ask("finish the auth work")
    decision = _last_orchestrator_decision(agent)
    summary_item = _first_compact_summary(agent)
    compactions = agent.session.get("compactions", [])

    return {
        "compact_trigger": decision["context_orchestrator"].get("compact_trigger"),
        "summary_mode": decision["context_orchestrator"].get("summary_mode"),
        "history_summary_kind": summary_item.get("kind"),
        "structured_summary": "## Goal" in summary_item.get("content", "")
        and "## Next Steps" in summary_item.get("content", ""),
        "compact_call_usage": decision["context_orchestrator"].get("compact_call_usage"),
        "compact_usage_persisted": bool(compactions and "compact_call_usage" in compactions[-1]),
        "final_answer": final_answer,
    }


def scenario_llm_failure_falls_back_to_deterministic(tmp_path=None):
    tmp_path = _tmp_path(tmp_path)
    agent = _build_agent(
        tmp_path,
        [
            "I apologize but I cannot produce a summary in that format.",
            "<final>fallback completed</final>",
        ],
        context_window=1000,
    )
    _fill_history(agent, rounds=6, chars_per_message=900)

    final_answer = agent.ask("finish despite compact failure")
    decision = _last_orchestrator_decision(agent)
    summary_item = _first_compact_summary(agent)

    return {
        "compact_trigger": decision["context_orchestrator"].get("compact_trigger"),
        "summary_mode": decision["context_orchestrator"].get("summary_mode"),
        "summary_text": summary_item.get("content", ""),
        "final_answer": final_answer,
    }


def scenario_low_pressure_no_compaction(tmp_path=None):
    tmp_path = _tmp_path(tmp_path)
    agent = _build_agent(tmp_path, ["<final>low pressure response</final>"], context_window=200_000)
    _fill_history(agent, rounds=2, chars_per_message=100)

    agent.ask("handle a small request")
    decision = _last_orchestrator_decision(agent)

    return {
        "compact_trigger": decision["context_orchestrator"].get("compact_trigger"),
        "summary_mode": decision["context_orchestrator"].get("summary_mode"),
        "compact_summary_count": len(_compact_summaries(agent)),
        "compaction_event_count": len(_events(agent, "compaction_created")),
    }


def scenario_over_budget_prefers_deterministic(tmp_path=None):
    tmp_path = _tmp_path(tmp_path)
    agent = _build_agent(tmp_path, ["<final>over budget done</final>"], context_window=1000)
    _fill_history(agent, rounds=6, chars_per_message=900)
    _set_context_budget(
        agent,
        total_budget=5000,
        section_budget=40_000,
    )

    agent.ask("finish over budget")
    decision = _last_orchestrator_decision(agent)

    return {
        "compact_trigger": decision["context_orchestrator"].get("compact_trigger"),
        "summary_mode": decision["context_orchestrator"].get("summary_mode"),
        "compact_call_usage": decision["context_orchestrator"].get("compact_call_usage"),
        "llm_prompt_count": len(agent.model_client.prompts) - 1,
    }


def scenario_delta_too_small_skips_compaction(tmp_path=None):
    tmp_path = _tmp_path(tmp_path)
    agent = _build_agent(tmp_path, ["unused"], context_window=200)
    agent.record(
        {
            "role": "system",
            "kind": "compact_summary",
            "content": "Compacted session summary:\n- Goal: keep going",
            "turn_id": "compact",
        }
    )
    agent.record({"role": "user", "content": "request " + ("x" * 900), "turn_id": "recent"})
    agent.record({"role": "assistant", "content": "answer " + ("y" * 900), "turn_id": "recent"})

    result = agent.context_orchestrator.build(agent.context_orchestrator.snapshot("finish"))

    return {
        "compact_trigger": result.compact_trigger,
        "should_compact": result.should_compact,
        "compact_summary_count": len(_compact_summaries(agent)),
        "model_prompt_count": len(agent.model_client.prompts),
    }


def scenario_replacement_ledger_survives_llm_compact(tmp_path=None):
    tmp_path = _tmp_path(tmp_path)
    agent = _build_agent(
        tmp_path,
        [HANDOFF_LLM_OUTPUT, "<final>ledger ok</final>"],
        context_window=1000,
    )
    ledger = {
        "event_abc": {
            "content_sha256": "deadbeef",
            "replacement_text": "stub",
            "saved_chars": 100,
        }
    }
    agent.session["context_replacements"] = dict(ledger)
    _fill_history(agent, rounds=6, chars_per_message=900)

    agent.ask("finish with ledger")

    return {
        "ledger_before": ledger,
        "ledger_after": agent.session.get("context_replacements", {}),
    }


def scenario_net_benefit_calculation():
    positive = context_budget_summary(
        {
            "context_usage": {"context_window": 4000, "total_estimated_tokens": 500},
            "context_orchestrator": {
                "compact_call_usage": {"total_tokens": 200},
                "pre_compact_estimated_tokens": 2700,
                "post_compact_estimated_tokens": 500,
            },
        }
    )["compact_net_benefit_tokens"]
    negative = context_budget_summary(
        {
            "context_usage": {"context_window": 4000, "total_estimated_tokens": 900},
            "context_orchestrator": {
                "compact_call_usage": {"total_tokens": 500},
                "pre_compact_estimated_tokens": 1000,
                "post_compact_estimated_tokens": 900,
            },
        }
    )["compact_net_benefit_tokens"]
    return {"positive": positive, "negative": negative}


def main():
    scenarios = [
        scenario_tier3_triggers_llm_handoff,
        scenario_llm_failure_falls_back_to_deterministic,
        scenario_low_pressure_no_compaction,
        scenario_over_budget_prefers_deterministic,
        scenario_delta_too_small_skips_compaction,
        scenario_replacement_ledger_survives_llm_compact,
        scenario_net_benefit_calculation,
    ]
    failures = []
    for scenario in scenarios:
        try:
            result = scenario()
            _validate_result(scenario.__name__, result)
            print(f"PASS: {scenario.__name__}")
        except AssertionError as exc:
            failures.append(f"{scenario.__name__}: {exc}")
            print(f"FAIL: {scenario.__name__}: {exc}")
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures.append(f"{scenario.__name__}: {type(exc).__name__}: {exc}")
            print(f"ERROR: {scenario.__name__}: {type(exc).__name__}: {exc}")
    print("\n" + "=" * 60)
    print(f"Results: {len(scenarios) - len(failures)}/{len(scenarios)} passed")
    if failures:
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All scenarios passed.")
    return 0


def _validate_result(name, result):
    checks = {
        "scenario_tier3_triggers_llm_handoff": lambda r: (
            r["compact_trigger"] == "auto_pressure_compact"
            and r["summary_mode"] == "llm"
            and r["structured_summary"]
            and r["compact_call_usage"]["input_tokens"] > 0
            and not r["compact_usage_persisted"]
            and r["final_answer"] == "auth middleware is ready"
        ),
        "scenario_llm_failure_falls_back_to_deterministic": lambda r: (
            r["compact_trigger"] == "auto_pressure_compact"
            and r["summary_mode"] == "deterministic_fallback"
            and r["summary_text"].startswith("Compacted session summary:")
            and r["final_answer"] == "fallback completed"
        ),
        "scenario_low_pressure_no_compaction": lambda r: (
            r["compact_trigger"] is None
            and r["summary_mode"] == ""
            and r["compact_summary_count"] == 0
            and r["compaction_event_count"] == 0
        ),
        "scenario_over_budget_prefers_deterministic": lambda r: (
            r["compact_trigger"] == "auto_prompt_over_budget"
            and r["summary_mode"] == "deterministic"
            and r["compact_call_usage"] is None
            and r["llm_prompt_count"] == 0
        ),
        "scenario_delta_too_small_skips_compaction": lambda r: (
            r["compact_trigger"] is None
            and not r["should_compact"]
            and r["compact_summary_count"] == 1
            and r["model_prompt_count"] == 0
        ),
        "scenario_replacement_ledger_survives_llm_compact": lambda r: r["ledger_before"] == r["ledger_after"],
        "scenario_net_benefit_calculation": lambda r: r["positive"] == 2000 and r["negative"] == -400,
    }
    assert checks[name](result), f"unexpected result: {result}"


def _build_agent(tmp_path, responses, *, context_window):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    client = ScriptedModelClient(responses)
    client.context_window = context_window
    client.last_completion_metadata = {
        "input_tokens": 950,
        "output_tokens": 50,
        "total_tokens": 1000,
        "cached_tokens": 0,
        "provider_protocol": "openai",
        "provider_model": "test-model",
        "provider_base_url": "http://localhost",
    }
    return GenCode(
        model_client=client,
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
    )


def _fill_history(agent, *, rounds, chars_per_message):
    for index in range(rounds):
        agent.record({"role": "user", "content": f"request {index} " + ("x" * chars_per_message)})
        agent.record({"role": "assistant", "content": f"answer {index} " + ("y" * chars_per_message)})


def _set_context_budget(agent, *, total_budget, section_budget):
    agent.context_manager = ContextManager(
        agent,
        total_budget=total_budget,
        section_budgets={
            "prefix": section_budget,
            "memory": section_budget,
            "relevant_memory": section_budget,
            "history": section_budget,
        },
        section_floors={
            "prefix": section_budget,
            "memory": section_budget,
            "relevant_memory": section_budget,
            "history": section_budget,
        },
    )


def _first_compact_summary(agent):
    summaries = _compact_summaries(agent)
    assert summaries, "expected compact_summary in session history"
    return summaries[0]


def _compact_summaries(agent):
    return [item for item in agent.session.get("history", []) if item.get("kind") == "compact_summary"]


def _last_orchestrator_decision(agent):
    decisions = _events(agent, "context_orchestrator_decision")
    assert decisions, "expected context_orchestrator_decision event"
    return decisions[-1]


def _events(agent, event_name):
    path = agent.session_event_bus.path
    if not path.exists():
        return []
    return [
        event
        for event in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if event.get("event") == event_name
    ]


def _tmp_path(tmp_path):
    if tmp_path is not None:
        return Path(tmp_path)
    return Path(tempfile.mkdtemp(prefix="gencode-llm-handoff-"))


if __name__ == "__main__":
    raise SystemExit(main())
