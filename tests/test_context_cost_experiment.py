import json
import subprocess
import sys

from gencode.evaluation.context_cost import (
    CostUsage,
    ExperimentRow,
    ProviderPricing,
    collect_rows_from_run_manifest,
    compute_cost_usd,
    extract_usage_from_artifacts,
    render_markdown_report,
    run_deterministic_prompt_experiment,
    run_scripted_e2e_experiment,
    summarize_paired_rows,
    write_experiment_artifacts,
)


def test_compute_cost_usd_uses_cached_input_discount():
    pricing = ProviderPricing(
        input_per_1m=2.00,
        cached_input_per_1m=0.20,
        output_per_1m=8.00,
    )
    usage = CostUsage(
        input_tokens=1000,
        cached_tokens=400,
        output_tokens=100,
        usage_source="actual",
        model_call_count=1,
    )

    cost = compute_cost_usd(usage, pricing)

    assert round(cost, 6) == 0.00208
    assert usage.uncached_input_tokens == 600


def test_extract_usage_sums_all_model_calls_from_trace(tmp_path):
    report_path = tmp_path / "report.json"
    trace_path = tmp_path / "trace.jsonl"
    report_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "tool_steps": 2,
                "attempts": 1,
                "evidence_summaries": {
                    "verification_signal": {"state": "passed"},
                    "context_budget_summary": {
                        "saved_chars": 2000,
                        "replacement_cache_hits": 3,
                        "summary_called": True,
                        "summary_delta_event_count": 4,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "prompt_built",
                        "prompt_metadata": {
                            "context_usage": {"total_estimated_tokens": 900}
                        },
                    }
                ),
                json.dumps(
                    {
                        "event": "model_parsed",
                        "completion_metadata": {
                            "input_tokens": 1000,
                            "cached_tokens": 400,
                            "output_tokens": 70,
                            "provider_protocol": "openai",
                            "provider_model": "gpt-5.4-example",
                        },
                    }
                ),
                json.dumps(
                    {
                        "event": "prompt_built",
                        "prompt_metadata": {
                            "context_usage": {"total_estimated_tokens": 300}
                        },
                    }
                ),
                json.dumps(
                    {
                        "event": "model_parsed",
                        "completion_metadata": {
                            "input_tokens": 200,
                            "cached_tokens": 100,
                            "output_tokens": 10,
                            "provider_protocol": "openai",
                            "provider_model": "gpt-5.4-example",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    row = extract_usage_from_artifacts(
        report_path,
        trace_path,
        task_id="task-1",
        layer="live",
        variant="full_orchestrator",
        repeat=0,
        pricing=None,
    )

    assert row.usage.input_tokens == 1200
    assert row.usage.cached_tokens == 500
    assert row.usage.output_tokens == 80
    assert row.usage.usage_source == "actual"
    assert row.usage.model_call_count == 2
    assert row.prompt_estimated_tokens == 1200
    assert row.saved_chars == 2000
    assert row.replacement_cache_hits == 3
    assert row.verification_status == "passed"


def test_extract_usage_downgrades_partial_or_synthetic_metadata_to_proxy(tmp_path):
    report_path = tmp_path / "report.json"
    trace_path = tmp_path / "trace.jsonl"
    report_path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "prompt_built",
                        "prompt_metadata": {
                            "context_usage": {"total_estimated_tokens": 400}
                        },
                    }
                ),
                json.dumps(
                    {
                        "event": "model_parsed",
                        "completion_metadata": {
                            "input_tokens": 100,
                            "output_tokens": 10,
                            "synthetic": True,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    row = extract_usage_from_artifacts(
        report_path,
        trace_path,
        task_id="task-1",
        layer="scripted",
        variant="full_orchestrator",
        repeat=0,
        pricing=ProviderPricing(2.0, 0.2, 8.0),
        verification_status="passed",
        allow_verification_override=True,
    )

    assert row.usage.usage_source == "estimated_proxy"
    assert row.usage.input_tokens == 400
    assert row.usage.output_tokens == 0
    assert row.verification_status == "passed"


def test_extract_usage_requires_current_provider_identity_for_actual_bucket(tmp_path):
    report_path = tmp_path / "report.json"
    trace_path = tmp_path / "trace.jsonl"
    report_path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "prompt_built",
                        "prompt_metadata": {
                            "context_usage": {"total_estimated_tokens": 300}
                        },
                    }
                ),
                json.dumps(
                    {
                        "event": "model_parsed",
                        "completion_metadata": {
                            "input_tokens": 100,
                            "output_tokens": 10,
                            "usage_source": "actual",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    row = extract_usage_from_artifacts(
        report_path,
        trace_path,
        task_id="task-1",
        layer="scripted",
        variant="full_orchestrator",
        repeat=0,
        pricing=ProviderPricing(2.0, 0.2, 8.0),
    )

    assert row.usage.usage_source == "estimated_proxy"
    assert row.usage.input_tokens == 300


def test_extract_usage_requires_complete_provider_token_metadata(tmp_path):
    report_path = tmp_path / "report.json"
    trace_path = tmp_path / "trace.jsonl"
    report_path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "prompt_built",
                        "prompt_metadata": {
                            "context_usage": {"total_estimated_tokens": 500}
                        },
                    }
                ),
                json.dumps(
                    {
                        "event": "model_parsed",
                        "completion_metadata": {
                            "input_tokens": 400,
                            "provider_protocol": "openai",
                            "provider_model": "gpt-5.4-example",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    row = extract_usage_from_artifacts(
        report_path,
        trace_path,
        task_id="task-1",
        layer="live",
        variant="full_orchestrator",
        repeat=0,
        pricing=ProviderPricing(2.0, 0.2, 8.0),
    )

    assert row.usage.usage_source == "estimated_proxy"
    assert row.usage.input_tokens == 500


def _row(
    task_id,
    variant,
    input_tokens,
    *,
    status="completed",
    verification_status="passed",
    tool_steps=1,
    attempts=1,
    cost_usd=None,
    usage_source="actual",
):
    return ExperimentRow(
        task_id=task_id,
        layer="scripted",
        variant=variant,
        repeat=0,
        status=status,
        verification_status=verification_status,
        tool_steps=tool_steps,
        attempts=attempts,
        prompt_estimated_tokens=input_tokens,
        usage=CostUsage(
            input_tokens=input_tokens,
            cached_tokens=0,
            output_tokens=10,
            usage_source=usage_source,
            model_call_count=1,
        ),
        cost_usd=input_tokens / 1_000_000 if cost_usd is None else cost_usd,
        saved_chars=0,
        replacement_cache_hits=0,
        summary_called=False,
        summary_delta_event_count=0,
        report_path="report.json",
        trace_path="trace.jsonl",
    )


def test_summarize_paired_rows_splits_actual_proxy_and_reports_quality_regressions():
    summary = summarize_paired_rows(
        [
            _row("a", "full_orchestrator", 700, cost_usd=0.7),
            _row("a", "no_context_reduction", 1000, cost_usd=1.0),
            _row("b", "full_orchestrator", 900, tool_steps=4, cost_usd=0.9),
            _row("b", "no_context_reduction", 1000, tool_steps=1, cost_usd=1.0),
        ],
        treatment="full_orchestrator",
        control="no_context_reduction",
    )

    actual = summary["actual_only"]
    assert actual["paired_task_count"] == 2
    assert actual["median_uncached_input_delta_pct"] == -0.2
    assert actual["quality_regression_count"] == 1
    assert actual["success_rate_treatment"] == 1.0
    assert actual["verifier_pass_rate_control"] == 1.0
    assert actual["avg_tool_steps_treatment"] == 2.5
    assert actual["cost_per_successful_task_treatment"] == 0.8
    assert actual["billable_input_tokens_per_task_treatment"] == 800
    assert actual["total_input_tokens_per_task_control"] == 1000
    assert actual["output_tokens_per_task_treatment"] == 10
    assert actual["claimable_cost_win"] is False
    assert summary["real_usage_row_count"] == 4
    assert summary["estimated_proxy_only"]["paired_task_count"] == 0
    assert summary["mixed_or_invalid"]["paired_task_count"] == 0


def test_lower_cost_with_unknown_or_failed_verification_is_not_claimable():
    unknown_summary = summarize_paired_rows(
        [
            _row("a", "full_orchestrator", 500, verification_status="unknown"),
            _row("a", "no_context_reduction", 1000, verification_status="passed"),
        ],
        treatment="full_orchestrator",
        control="no_context_reduction",
    )
    failed_summary = summarize_paired_rows(
        [
            _row("a", "full_orchestrator", 500, verification_status="failed"),
            _row("a", "no_context_reduction", 1000, verification_status="passed"),
        ],
        treatment="full_orchestrator",
        control="no_context_reduction",
    )

    assert unknown_summary["actual_only"]["median_cost_delta_pct"] < 0
    assert unknown_summary["actual_only"]["quality_regression_count"] == 1
    assert unknown_summary["actual_only"]["unknown_verification_count"] == 1
    assert unknown_summary["actual_only"]["claimable_cost_win"] is False
    assert failed_summary["actual_only"]["quality_regression_count"] == 1
    assert failed_summary["actual_only"]["claimable_cost_win"] is False


def test_deterministic_prompt_experiment_pairs_full_and_no_reduction(tmp_path):
    payload = run_deterministic_prompt_experiment(
        output_dir=tmp_path / "context-cost",
        repetitions=1,
    )

    assert payload["summary"]["estimated_proxy_only"]["paired_task_count"] >= 1
    assert payload["summary"]["actual_only"]["paired_task_count"] == 0
    assert all(row["usage"]["usage_source"] == "estimated_proxy" for row in payload["rows"])
    assert {row["variant"] for row in payload["rows"]} == {
        "full_orchestrator",
        "no_context_reduction",
    }


def test_scripted_e2e_experiment_records_quality_and_report_paths(tmp_path):
    payload = run_scripted_e2e_experiment(
        output_dir=tmp_path / "context-cost",
        repetitions=1,
    )

    assert payload["summary"]["actual_only"]["paired_task_count"] == 0
    assert payload["summary"]["estimated_proxy_only"]["paired_task_count"] >= 1
    assert payload["summary"]["estimated_proxy_only"]["quality_regression_count"] == 0
    assert all(row["status"] == "completed" for row in payload["rows"])
    assert all(row["report_path"] for row in payload["rows"])


def test_collect_rows_from_run_manifest_reads_existing_reports(tmp_path):
    report = tmp_path / "full" / "report.json"
    trace = tmp_path / "full" / "trace.jsonl"
    report.parent.mkdir()
    report.write_text(
        json.dumps(
            {
                "status": "completed",
                "tool_steps": 1,
                "attempts": 1,
                "evidence_summaries": {
                    "verification_signal": {"state": "passed"},
                    "context_budget_summary": {"saved_chars": 50},
                },
            }
        ),
        encoding="utf-8",
    )
    trace.write_text(
        json.dumps(
            {
                "event": "model_parsed",
                "completion_metadata": {
                    "input_tokens": 100,
                    "cached_tokens": 20,
                    "output_tokens": 10,
                    "provider_protocol": "openai",
                    "provider_model": "gpt-5.4-example",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "runs": [
            {
                "task_id": "live-1",
                "layer": "live",
                "variant": "full_orchestrator",
                "repeat": 0,
                "report_path": report.as_posix(),
                "trace_path": trace.as_posix(),
                "verification_status": "failed",
            }
        ]
    }

    rows = collect_rows_from_run_manifest(manifest, pricing=ProviderPricing(1, 0.1, 2))

    assert rows[0].task_id == "live-1"
    assert rows[0].usage.usage_source == "actual"
    assert rows[0].verification_status == "passed"


def test_render_markdown_report_separates_actual_and_proxy_usage():
    payload = {
        "pricing": {
            "input_per_1m": 2.0,
            "cached_input_per_1m": 0.2,
            "output_per_1m": 8.0,
        },
        "summary": {
            "actual_only": {
                "paired_task_count": 1,
                "quality_regression_count": 0,
                "unknown_verification_count": 0,
                "median_cost_delta_pct": -0.2,
                "claimable_cost_win": True,
                "cost_per_successful_task_treatment": 0.01,
                "cost_per_successful_task_control": 0.012,
                "success_rate_treatment": 1.0,
                "success_rate_control": 1.0,
                "verifier_pass_rate_treatment": 1.0,
                "verifier_pass_rate_control": 1.0,
                "avg_tool_steps_treatment": 2.0,
                "avg_tool_steps_control": 3.0,
                "avg_attempts_treatment": 1.0,
                "avg_attempts_control": 1.0,
                "billable_input_tokens_per_task_treatment": 800.0,
                "billable_input_tokens_per_task_control": 1000.0,
                "total_input_tokens_per_task_treatment": 1000.0,
                "total_input_tokens_per_task_control": 1200.0,
                "output_tokens_per_task_treatment": 90.0,
                "output_tokens_per_task_control": 100.0,
            },
            "estimated_proxy_only": {
                "paired_task_count": 1,
                "quality_regression_count": 0,
                "unknown_verification_count": 0,
                "median_cost_delta_pct": -0.25,
                "claimable_cost_win": True,
                "billable_input_tokens_per_task_treatment": 700.0,
                "billable_input_tokens_per_task_control": 1000.0,
                "total_input_tokens_per_task_treatment": 700.0,
                "total_input_tokens_per_task_control": 1000.0,
                "output_tokens_per_task_treatment": 0.0,
                "output_tokens_per_task_control": 0.0,
            },
            "mixed_or_invalid": {"paired_task_count": 0},
            "real_usage_row_count": 2,
            "estimated_proxy_row_count": 2,
        },
        "rows": [],
    }

    report = render_markdown_report(payload)

    assert "# Context Cost Experiment" in report
    assert "Real provider rows: 2" in report
    assert "Estimated proxy rows: 2" in report
    assert "Actual-only quality regressions: 0" in report
    assert "Actual-only configured-price win: True" in report
    assert "Actual-only cost per successful task: 0.01 vs 0.012" in report
    assert "Actual-only billable input tokens/task: 800.0 vs 1000.0" in report
    assert "Actual-only total input tokens/task: 1000.0 vs 1200.0" in report
    assert "Actual-only output tokens/task: 90.0 vs 100.0" in report
    assert "Estimated-proxy billable input tokens/task: 700.0 vs 1000.0" in report
    assert "Pricing basis: configured, not provider-authenticated" in report
    assert "Input $/1M: 2.0" in report


def test_write_experiment_artifacts_writes_json_csv_and_markdown(tmp_path):
    payload = run_deterministic_prompt_experiment(
        output_dir=tmp_path / "work",
        repetitions=1,
    )
    output_dir = tmp_path / "artifacts"

    written = write_experiment_artifacts(payload, output_dir)

    assert written["json"] == str(output_dir / "results.json")
    assert (output_dir / "results.json").is_file()
    assert (output_dir / "paired_rows.csv").is_file()
    assert (output_dir / "report.md").is_file()


def test_context_cost_cli_deterministic_smoke(tmp_path):
    output_dir = tmp_path / "cli-output"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_context_cost_experiment.py",
            "--mode",
            "deterministic",
            "--output-dir",
            str(output_dir),
            "--repetitions",
            "1",
        ],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["json"] == str(output_dir / "results.json")
    assert (output_dir / "paired_rows.csv").is_file()
