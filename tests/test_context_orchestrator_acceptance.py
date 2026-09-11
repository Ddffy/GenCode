import json

from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.testing import ScriptedModelClient


def build_agent(tmp_path, outputs=None):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return GenCode(
        model_client=ScriptedModelClient(outputs or []),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_real_turn_emits_context_orchestrator_decision_and_report_metadata(tmp_path):
    agent = build_agent(tmp_path, ["<final>done</final>"])

    list(agent.engine.run_turn("summarize context"))

    trace = read_jsonl(agent.current_run_dir / "trace.jsonl")
    trace_events = [event["event"] for event in trace]
    assert "context_orchestrator_decision" in trace_events
    assert trace_events.index("context_orchestrator_decision") < trace_events.index("prompt_built")

    decision = next(event for event in trace if event["event"] == "context_orchestrator_decision")
    assert decision["phase"] == "prompt"
    assert decision["context_orchestrator"]["version"] == "local-v1"

    session_events = read_jsonl(agent.session_event_bus.path)
    assert any(event["event"] == "context_orchestrator_decision" for event in session_events)

    report = json.loads((agent.current_run_dir / "report.json").read_text(encoding="utf-8"))
    orchestrator = report["prompt_metadata"]["context_orchestrator"]
    summary = report["evidence_summaries"]["context_budget_summary"]
    assert orchestrator["version"] == "local-v1"
    assert summary["pressure_tier"] == orchestrator["pressure_tier"]
    assert summary["usage_source"] == orchestrator["usage_source"]
    assert summary["summary_called"] is False
    assert "replacement_cache_hits" in summary


def test_real_turn_reports_tier3_llm_compaction_usage(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            """## Goal
Continue after compaction.

## Next Steps
- Finish the request.
""",
            "<final>done</final>",
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

    list(agent.engine.run_turn("finish"))

    report = json.loads((agent.current_run_dir / "report.json").read_text(encoding="utf-8"))
    orchestrator = report["prompt_metadata"]["context_orchestrator"]
    summary = report["evidence_summaries"]["context_budget_summary"]

    assert orchestrator["compact_trigger"] == "auto_pressure_compact"
    assert orchestrator["summary_mode"] == "llm"
    assert orchestrator["compact_call_usage"]["total_tokens"] == 100
    assert summary["compact_call_usage"]["provider"] == "openai"
    assert isinstance(summary["compact_net_benefit_tokens"], int)
