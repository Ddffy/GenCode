import json
from unittest.mock import patch

from gencode.providers.clients import AnthropicCompatibleModelClient


def test_anthropic_client_records_usage_metadata():
    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "content": [{"type": "text", "text": "<final>ok</final>"}],
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "input_tokens_details": {"cached_tokens": 30},
                    },
                }
            ).encode("utf-8")

    client = AnthropicCompatibleModelClient(
        model="claude-test",
        base_url="https://example.com/v1",
        api_key="test-key",
        temperature=0.0,
        timeout=30,
    )

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        assert client.complete("hello", 42) == "<final>ok</final>"

    assert client.last_completion_metadata["input_tokens"] == 100
    assert client.last_completion_metadata["output_tokens"] == 20
    assert client.last_completion_metadata["cached_tokens"] == 30
    assert client.last_completion_metadata["cache_hit"] is True


def test_deterministic_prompt_experiment_pairs_full_and_no_reduction(tmp_path):
    from gencode.evaluation.context_cost import run_deterministic_prompt_experiment

    payload = run_deterministic_prompt_experiment(
        output_dir=tmp_path / "context-cost",
        repetitions=1,
    )

    rows = payload["rows"]
    assert {row["variant"] for row in rows} == {"full_orchestrator", "no_context_reduction"}
    assert payload["summary"]["estimated_proxy_only"]["paired_task_count"] == 1
    by_variant = {row["variant"]: row for row in rows}
    assert (
        by_variant["full_orchestrator"]["usage"]["input_tokens"]
        < by_variant["no_context_reduction"]["usage"]["input_tokens"]
    )


def test_summarize_paired_rows_rejects_quality_regression():
    from gencode.evaluation.context_cost import CostUsage, ExperimentRow, summarize_paired_rows

    def row(task_id, variant, verification_status, tokens):
        return ExperimentRow(
            task_id=task_id,
            layer="deterministic",
            variant=variant,
            repeat=0,
            status="completed",
            verification_status=verification_status,
            tool_steps=1,
            attempts=1,
            prompt_estimated_tokens=tokens,
            usage=CostUsage(
                input_tokens=tokens,
                cached_tokens=0,
                output_tokens=0,
                usage_source="estimated_proxy",
                model_call_count=0,
            ),
            cost_usd=tokens / 1_000_000,
            saved_chars=0,
            replacement_cache_hits=0,
            summary_called=False,
            summary_delta_event_count=0,
            report_path="report.json",
            trace_path="trace.jsonl",
        )

    summary = summarize_paired_rows(
        [
            row("task", "full_orchestrator", "failed", 500),
            row("task", "no_context_reduction", "passed", 1000),
        ],
        treatment="full_orchestrator",
        control="no_context_reduction",
    )

    proxy = summary["estimated_proxy_only"]
    assert proxy["median_cost_delta_pct"] < 0
    assert proxy["quality_regression_count"] == 1
    assert proxy["claimable_cost_win"] is False


def test_summarize_paired_rows_splits_actual_and_proxy():
    from gencode.evaluation.context_cost import CostUsage, ExperimentRow, summarize_paired_rows

    def row(task_id, variant, tokens, source):
        return ExperimentRow(
            task_id=task_id,
            layer="layer",
            variant=variant,
            repeat=0,
            status="completed",
            verification_status="passed",
            tool_steps=1,
            attempts=1,
            prompt_estimated_tokens=tokens,
            usage=CostUsage(
                input_tokens=tokens,
                cached_tokens=0,
                output_tokens=1,
                usage_source=source,
                model_call_count=1,
            ),
            cost_usd=tokens / 1_000_000,
            saved_chars=0,
            replacement_cache_hits=0,
            summary_called=False,
            summary_delta_event_count=0,
            report_path="report.json",
            trace_path="trace.jsonl",
        )

    summary = summarize_paired_rows(
        [
            row("actual", "full_orchestrator", 700, "actual"),
            row("actual", "no_context_reduction", 1000, "actual"),
            row("proxy", "full_orchestrator", 800, "estimated_proxy"),
            row("proxy", "no_context_reduction", 1000, "estimated_proxy"),
        ],
        treatment="full_orchestrator",
        control="no_context_reduction",
    )

    assert summary["actual_only"]["paired_task_count"] == 1
    assert summary["estimated_proxy_only"]["paired_task_count"] == 1
    assert summary["mixed_or_invalid"]["paired_task_count"] == 0
    assert summary["real_usage_row_count"] == 2
    assert summary["estimated_proxy_row_count"] == 2


def test_claimable_cost_win_rejects_unknown_verification_and_negative_compact_net():
    from gencode.evaluation.context_cost import CostUsage, ExperimentRow, summarize_paired_rows

    def row(variant, verification_status, compact_net=None):
        return ExperimentRow(
            task_id="task",
            layer="deterministic",
            variant=variant,
            repeat=0,
            status="completed",
            verification_status=verification_status,
            tool_steps=1,
            attempts=1,
            prompt_estimated_tokens=500 if variant == "full_orchestrator" else 1000,
            usage=CostUsage(
                input_tokens=500 if variant == "full_orchestrator" else 1000,
                cached_tokens=0,
                output_tokens=0,
                usage_source="estimated_proxy",
                model_call_count=0,
            ),
            cost_usd=(500 if variant == "full_orchestrator" else 1000) / 1_000_000,
            saved_chars=0,
            replacement_cache_hits=0,
            summary_called=False,
            summary_delta_event_count=0,
            compact_net_benefit_tokens=compact_net,
            report_path="report.json",
            trace_path="trace.jsonl",
        )

    unknown = summarize_paired_rows(
        [
            row("full_orchestrator", "unknown"),
            row("no_context_reduction", "passed"),
        ],
        treatment="full_orchestrator",
        control="no_context_reduction",
    )
    negative_net = summarize_paired_rows(
        [
            row("full_orchestrator", "passed", compact_net=-1),
            row("no_context_reduction", "passed"),
        ],
        treatment="full_orchestrator",
        control="no_context_reduction",
    )
    clean = summarize_paired_rows(
        [
            row("full_orchestrator", "passed", compact_net=10),
            row("no_context_reduction", "passed"),
        ],
        treatment="full_orchestrator",
        control="no_context_reduction",
    )

    assert unknown["estimated_proxy_only"]["claimable_cost_win"] is False
    assert negative_net["estimated_proxy_only"]["claimable_cost_win"] is False
    assert clean["estimated_proxy_only"]["claimable_cost_win"] is True


def test_markdown_report_includes_net_benefit_section(tmp_path):
    from gencode.evaluation.context_cost import render_markdown_report, run_deterministic_prompt_experiment

    payload = run_deterministic_prompt_experiment(
        output_dir=tmp_path / "context-cost",
        repetitions=1,
    )

    report = render_markdown_report(payload)

    assert "## Net Benefit" in report
    assert "net_saved = baseline_input_tokens - optimized_input_tokens - compact_call_tokens" in report
    assert "compact_call_tokens: 0" in report
    assert "Claimable cost win: True" in report


def test_scripted_e2e_experiment_records_tool_use(tmp_path):
    from gencode.evaluation.context_cost import run_scripted_e2e_experiment

    payload = run_scripted_e2e_experiment(tmp_path / "context-cost", repetitions=1)

    assert payload["summary"]["estimated_proxy_only"]["paired_task_count"] == 1
    assert all(row["tool_steps"] > 0 for row in payload["rows"])
    assert all(row["verification_status"] == "passed" for row in payload["rows"])


def test_write_experiment_artifacts_has_no_secret_markers(tmp_path):
    from gencode.evaluation.context_cost import run_deterministic_prompt_experiment, write_experiment_artifacts

    payload = run_deterministic_prompt_experiment(tmp_path / "context-cost", repetitions=1)
    written = write_experiment_artifacts(payload, tmp_path / "artifacts")

    assert (tmp_path / "artifacts" / "results.json").is_file()
    assert (tmp_path / "artifacts" / "paired_rows.csv").is_file()
    assert (tmp_path / "artifacts" / "report.md").is_file()
    combined = "\n".join(
        (tmp_path / "artifacts" / name).read_text(encoding="utf-8")
        for name in ("results.json", "paired_rows.csv", "report.md")
    )
    assert written["json"].endswith("results.json")
    assert "sk-" not in combined
    assert "api_key" not in combined
    assert "api-key" not in combined
