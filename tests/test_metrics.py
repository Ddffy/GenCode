import os
from unittest.mock import patch

from gencode.evaluation.metrics import (
    _provider_profile,
    main as metrics_main,
    run_context_ablation_v2,
    run_memory_fidelity_v1,
    run_memory_ablation_v2,
    run_recovery_ablation_v2,
    write_benchmark_core_report,
)


def test_run_context_ablation_v2_writes_expected_artifact(tmp_path):
    artifact_path = tmp_path / "artifacts" / "context-ablation-v2.json"

    artifact = run_context_ablation_v2(
        artifact_path=artifact_path,
        repetitions=1,
    )

    assert artifact_path.exists()
    assert artifact["artifact_type"] == "context-ablation-v2"
    assert artifact["config_count"] == 12
    assert len(artifact["configs"]) == 12
    assert "current_request_preserved_rate" in artifact["summary"]


def test_metrics_cli_context_ab_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert metrics_main(["--run", "context_ab"]) == 0

    assert (tmp_path / "artifacts" / "context-ab-v1" / "results.json").is_file()
    assert (tmp_path / "artifacts" / "context-ab-v1" / "report.md").is_file()


def test_provider_profile_uses_project_toml_before_legacy_gencode_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".gencode.toml").write_text(
        "\n".join(
            [
                "[providers.deepseek]",
                'protocol = "anthropic"',
                'api_key = "sk-project-deepseek"',
                'model = "deepseek-v4-flash"',
                'base_url = "https://api.deepseek.com/anthropic"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with patch.dict(
        os.environ,
        {
            "GENCODE_DEEPSEEK_API_KEY": "sk-legacy-deepseek",
            "GENCODE_DEEPSEEK_MODEL": "legacy-deepseek-model",
            "GENCODE_DEEPSEEK_API_BASE": "https://legacy.deepseek.example/anthropic",
        },
        clear=True,
    ):
        profile = _provider_profile("deepseek")

    assert profile["status"] == "ready"
    assert profile["api_key"] == "sk-project-deepseek"
    assert profile["model"] == "deepseek-v4-flash"
    assert profile["base_url"] == "https://api.deepseek.com/anthropic"


def test_run_memory_ablation_v2_writes_expected_artifact(tmp_path):
    artifact_path = tmp_path / "artifacts" / "memory-ablation-v2.json"

    artifact = run_memory_ablation_v2(
        artifact_path=artifact_path,
        repetitions=1,
    )

    assert artifact_path.exists()
    assert artifact["artifact_type"] == "memory-ablation-v2"
    assert artifact["task_count"] == 12
    assert set(artifact["variants"]) == {"memory_on", "memory_off", "memory_irrelevant"}
    assert "memory_hit_rate" in artifact["variants"]["memory_on"]


def test_memory_fidelity_irrelevant_memory_present_category(tmp_path):
    artifact = run_memory_fidelity_v1(tmp_path / "artifacts" / "memory-fidelity-v1.json")
    row = next(row for row in artifact["rows"] if row["category"] == "irrelevant_memory_present")

    assert row["passed"]
    assert not row["distractor_selected"]


def test_memory_fidelity_superseded_fact_category(tmp_path):
    artifact = run_memory_fidelity_v1(tmp_path / "artifacts" / "memory-fidelity-v1.json")
    row = next(row for row in artifact["rows"] if row["category"] == "superseded_fact")

    assert row["passed"]
    assert row["new_fact_selected"]
    assert row["old_fact_superseded"]


def test_memory_fidelity_secret_shaped_category(tmp_path):
    artifact = run_memory_fidelity_v1(tmp_path / "artifacts" / "memory-fidelity-v1.json")
    row = next(row for row in artifact["rows"] if row["category"] == "secret_shaped")

    assert row["passed"]
    assert not row["secret_selected"]


def test_run_memory_fidelity_v1_writes_expected_artifact(tmp_path):
    artifact_path = tmp_path / "artifacts" / "memory-fidelity-v1.json"

    artifact = run_memory_fidelity_v1(artifact_path)

    assert artifact_path.exists()
    assert artifact["artifact_type"] == "memory-fidelity-v1"
    assert artifact["summary"]["irrelevant_injection_rate"] == 0
    assert artifact["summary"]["supersede_success_rate"] == 1
    assert artifact["summary"]["secret_exposure_rate"] == 0
    assert artifact["summary"]["stale_detection_rate"] == 1
    assert artifact["summary"]["stale_use_rate"] == 0
    assert artifact["summary"]["poison_quarantine_rate"] == 1
    assert artifact["summary"]["benign_recall_retention_rate"] == 1
    assert artifact["schema_version"] == 1
    assert {row["category"] for row in artifact["rows"]} == {
        "irrelevant_memory_present",
        "superseded_fact",
        "secret_shaped",
        "stale_evidence",
        "prompt_injection",
    }


def test_memory_fidelity_stale_and_prompt_injection_categories(tmp_path):
    artifact = run_memory_fidelity_v1(tmp_path / "artifacts" / "memory-fidelity-v1.json")
    stale = next(row for row in artifact["rows"] if row["category"] == "stale_evidence")
    poison = next(row for row in artifact["rows"] if row["category"] == "prompt_injection")

    assert stale["passed"]
    assert stale["stale_detected"]
    assert not stale["stale_selected"]
    assert poison["passed"]
    assert poison["attack_quarantined"]
    assert poison["benign_selected"]


def test_run_recovery_ablation_v2_writes_expected_artifact(tmp_path):
    artifact_path = tmp_path / "artifacts" / "recovery-ablation-v2.json"

    artifact = run_recovery_ablation_v2(
        artifact_path=artifact_path,
        repetitions=1,
    )

    assert artifact_path.exists()
    assert artifact["artifact_type"] == "recovery-ablation-v2"
    assert artifact["schema_version"] == 2
    assert artifact["task_count"] == 11
    assert set(artifact["variants"]) == {"resume_enabled", "resume_disabled"}
    assert set(artifact["variants"]["resume_enabled"]["summary"]) >= {
        "resume_success_rate",
        "stale_reanchor_rate",
        "workspace_drift_detection_rate",
        "resume_false_accept_rate",
        "resumption_success_rate",
        "first_action_correctness",
        "todo_continuity_rate",
    }
    assert artifact["variants"]["resume_enabled"]["summary"]["resumption_success_rate"] >= 0.8
    assert artifact["variants"]["resume_enabled"]["summary"]["first_action_correctness"] >= 0.8
    assert artifact["variants"]["resume_enabled"]["summary"]["todo_continuity_rate"] >= 0.8


def test_write_benchmark_core_report_marks_resume_safe_metrics(tmp_path):
    run_context_ablation_v2(tmp_path / "artifacts" / "context-ablation-v2.json", repetitions=1)
    run_memory_ablation_v2(tmp_path / "artifacts" / "memory-ablation-v2.json", repetitions=1)
    run_memory_fidelity_v1(tmp_path / "artifacts" / "memory-fidelity-v1.json")
    run_recovery_ablation_v2(tmp_path / "artifacts" / "recovery-ablation-v2.json", repetitions=1)
    harness_artifact_path = tmp_path / "artifacts" / "harness-regression-v2.json"
    harness_artifact_path.write_text(
        '{"summary":{"total_tasks":12,"pass_rate":1.0,"within_budget_rate":1.0,"verifier_pass_rate":1.0},"failure_category_counts":{}}',
        encoding="utf-8",
    )

    report_path = tmp_path / "docs" / "metrics" / "gencode-benchmark-core-report.md"
    report_text = write_benchmark_core_report(
        report_path=report_path,
        harness_artifact_path=harness_artifact_path,
        context_artifact_path=tmp_path / "artifacts" / "context-ablation-v2.json",
        memory_artifact_path=tmp_path / "artifacts" / "memory-ablation-v2.json",
        recovery_artifact_path=tmp_path / "artifacts" / "recovery-ablation-v2.json",
        fidelity_artifact_path=tmp_path / "artifacts" / "memory-fidelity-v1.json",
    )

    assert report_path.exists()
    assert "可以安全写进简历的指标" in report_text
    assert "只适合放文档/面试展开的指标" in report_text
    assert "resume_success_rate" in report_text
    assert "resumption_success_rate" in report_text
    assert "first_action_correctness" in report_text
    assert "todo_continuity_rate" in report_text
    assert "memory_hit_rate" in report_text
    assert "Context Efficiency Under Follow-up" in report_text
    assert "Memory Fidelity" in report_text


def test_write_benchmark_core_report_includes_optional_context_ab(tmp_path):
    from gencode.evaluation.context_cost import run_deterministic_prompt_experiment, write_experiment_artifacts

    run_context_ablation_v2(tmp_path / "artifacts" / "context-ablation-v2.json", repetitions=1)
    run_memory_ablation_v2(tmp_path / "artifacts" / "memory-ablation-v2.json", repetitions=1)
    run_memory_fidelity_v1(tmp_path / "artifacts" / "memory-fidelity-v1.json")
    run_recovery_ablation_v2(tmp_path / "artifacts" / "recovery-ablation-v2.json", repetitions=1)
    harness_artifact_path = tmp_path / "artifacts" / "harness-regression-v2.json"
    harness_artifact_path.write_text(
        '{"summary":{"total_tasks":12,"pass_rate":1.0,"within_budget_rate":1.0,"verifier_pass_rate":1.0},"failure_category_counts":{}}',
        encoding="utf-8",
    )
    context_ab_dir = tmp_path / "artifacts" / "context-ab-v1"
    write_experiment_artifacts(
        run_deterministic_prompt_experiment(context_ab_dir, repetitions=1),
        context_ab_dir,
    )

    report_text = write_benchmark_core_report(
        report_path=tmp_path / "docs" / "metrics" / "gencode-benchmark-core-report.md",
        harness_artifact_path=harness_artifact_path,
        context_artifact_path=tmp_path / "artifacts" / "context-ablation-v2.json",
        memory_artifact_path=tmp_path / "artifacts" / "memory-ablation-v2.json",
        recovery_artifact_path=tmp_path / "artifacts" / "recovery-ablation-v2.json",
        fidelity_artifact_path=tmp_path / "artifacts" / "memory-fidelity-v1.json",
        context_ab_artifact_path=context_ab_dir / "results.json",
    )

    assert "Context A/B (Scripted)" in report_text
    assert "claimable_cost_win：True" in report_text


def test_write_benchmark_core_report_falls_back_to_local_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    local_artifacts = tmp_path / "_local" / "benchmark" / "artifacts"
    local_artifacts.mkdir(parents=True)
    run_context_ablation_v2(local_artifacts / "context-ablation-v2.json", repetitions=1)
    run_memory_ablation_v2(local_artifacts / "memory-ablation-v2.json", repetitions=1)
    run_memory_fidelity_v1(local_artifacts / "memory-fidelity-v1.json")
    run_recovery_ablation_v2(local_artifacts / "recovery-ablation-v2.json", repetitions=1)
    (local_artifacts / "harness-regression-v2.json").write_text(
        '{"summary":{"total_tasks":12,"pass_rate":1.0,"within_budget_rate":1.0,"verifier_pass_rate":1.0},"failure_category_counts":{}}',
        encoding="utf-8",
    )

    report_text = write_benchmark_core_report()

    assert "Harness Regression" in report_text
    assert "Context Efficiency Under Follow-up" in report_text
    assert "Memory Fidelity" in report_text
