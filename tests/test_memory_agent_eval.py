import json

from gencode.evaluation.memory_agent_eval import run_memory_agent_eval_v1
from gencode.evaluation.metrics import main as metrics_main, write_benchmark_core_report


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_input_artifacts(tmp_path):
    artifacts = tmp_path / "_local" / "benchmark" / "artifacts"
    _write_json(
        artifacts / "memory-ablation-v2.json",
        {
            "artifact_type": "memory-ablation-v2",
            "variants": {
                "memory_on": {"repeated_reads": 0, "avg_tool_steps": 0.0, "correct_rate": 1.0, "memory_hit_rate": 1.0},
                "memory_off": {"repeated_reads": 60, "avg_tool_steps": 1.0, "correct_rate": 1.0, "memory_hit_rate": 0.0},
            },
        },
    )
    _write_json(
        artifacts / "memory-fidelity-v1.json",
        {
            "artifact_type": "memory-fidelity-v1",
            "summary": {
                "pass_rate": 1.0,
                "irrelevant_injection_rate": 0.0,
                "secret_exposure_rate": 0.0,
                "stale_detection_rate": 1.0,
                "stale_use_rate": 0.0,
                "poison_quarantine_rate": 1.0,
                "benign_recall_retention_rate": 1.0,
                "supersede_success_rate": 1.0,
            },
        },
    )
    _write_json(
        artifacts / "dream-quality-v1.json",
        {
            "artifact_type": "dream-quality-v1",
            "summary": {
                "signal_retention_rate": 1.0,
                "noise_rejection_rate": 1.0,
                "secret_rejection_rate": 1.0,
                "dedupe_rate": 1.0,
                "relative_date_absolutization_rate": 1.0,
            },
        },
    )
    _write_json(
        artifacts / "recovery-ablation-v2.json",
        {
            "artifact_type": "recovery-ablation-v2",
            "variants": {
                "resume_enabled": {
                    "summary": {
                        "resume_false_accept_rate": 0.0,
                        "resume_success_rate": 0.9,
                        "stale_reanchor_rate": 1.0,
                        "workspace_drift_detection_rate": 1.0,
                        "resumption_success_rate": 1.0,
                        "first_action_correctness": 1.0,
                        "todo_continuity_rate": 1.0,
                    }
                }
            },
        },
    )
    _write_json(
        artifacts / "harness-regression-v2.json",
        {
            "summary": {
                "total_tasks": 12,
                "pass_rate": 1.0,
                "within_budget_rate": 1.0,
                "verifier_pass_rate": 1.0,
            },
            "failure_category_counts": {},
        },
    )
    _write_json(
        artifacts / "context-ablation-v2.json",
        {
            "artifact_type": "context-ablation-v2",
            "config_count": 12,
            "summary": {
                "avg_full_prompt_chars": 100.0,
                "avg_raw_prompt_chars": 120.0,
                "avg_prompt_compression_ratio": 0.1,
                "max_prompt_compression_ratio": 0.2,
                "current_request_preserved_rate": 1.0,
            },
        },
    )
    return artifacts


def test_memory_agent_eval_separates_contract_from_challenge(tmp_path):
    artifacts = _write_input_artifacts(tmp_path)
    artifact = run_memory_agent_eval_v1(
        artifact_path=artifacts / "memory-agent-eval-v1.json",
        report_path=tmp_path / "report.md",
        challenge_artifact_path=artifacts / "memory-challenge-v1.json",
        memory_ablation_path=artifacts / "memory-ablation-v2.json",
        memory_fidelity_path=artifacts / "memory-fidelity-v1.json",
        dream_quality_path=artifacts / "dream-quality-v1.json",
        recovery_ablation_path=artifacts / "recovery-ablation-v2.json",
    )

    assert artifact["contract"]["summary"]["total_cases"] == 8
    assert artifact["contract"]["summary"]["case_pass_rate"] == 1
    assert artifact["challenge"]["case_count"] >= 50
    assert set(artifact["challenge"]["variants"]) == {"memory_on", "memory_off", "naive_recent", "unsafe_memory"}
    assert artifact["summary"]["contract_case_pass_rate"] == 1
    assert artifact["summary"]["task_correctness_rate"] < 1


def test_memory_challenge_covers_longmemeval_abilities(tmp_path):
    artifact = run_memory_agent_eval_v1(
        artifact_path=tmp_path / "memory-agent-eval-v1.json",
        report_path=tmp_path / "report.md",
        challenge_artifact_path=tmp_path / "memory-challenge-v1.json",
    )

    categories = set(artifact["challenge"]["case_categories"])
    assert categories >= {
        "information_extraction",
        "multi_session_reasoning",
        "temporal_reasoning",
        "knowledge_updates",
        "abstention",
        "agentic_efficiency",
    }
    assert artifact["challenge"]["case_count"] >= 50


def test_memory_challenge_reports_comparative_metrics(tmp_path):
    artifact = run_memory_agent_eval_v1(
        artifact_path=tmp_path / "memory-agent-eval-v1.json",
        report_path=tmp_path / "report.md",
        challenge_artifact_path=tmp_path / "memory-challenge-v1.json",
    )

    variants = artifact["challenge"]["variants"]
    memory_on = variants["memory_on"]["summary"]
    memory_off = variants["memory_off"]["summary"]
    naive = variants["naive_recent"]["summary"]
    unsafe = variants["unsafe_memory"]["summary"]
    comparisons = artifact["challenge"]["comparisons"]

    assert memory_on["answer_accuracy"] > memory_off["answer_accuracy"]
    assert memory_on["evidence_recall_at_k"] > memory_off["evidence_recall_at_k"]
    assert memory_on["answer_accuracy"] > naive["answer_accuracy"]
    assert memory_on["stale_use_rate"] < unsafe["stale_use_rate"]
    assert memory_on["secret_exposure_rate"] < unsafe["secret_exposure_rate"]
    assert memory_on["false_resume_accept_rate"] < unsafe["false_resume_accept_rate"]
    assert comparisons["memory_on_vs_memory_off"]["evidence_recall_delta"] > 0
    assert comparisons["memory_on_vs_unsafe_memory"]["secret_exposure_reduction"] > 0
    assert memory_on["failed"] > 0


def test_memory_report_does_not_present_contract_as_benchmark(tmp_path):
    report_path = tmp_path / "report.md"
    run_memory_agent_eval_v1(
        artifact_path=tmp_path / "memory-agent-eval-v1.json",
        report_path=report_path,
        challenge_artifact_path=tmp_path / "memory-challenge-v1.json",
    )
    text = report_path.read_text(encoding="utf-8")

    assert "Contract Verification" in text
    assert "Challenge Benchmark" in text
    assert "memory_on_vs_memory_off" in text
    assert "8/8" in text
    assert "不能把 contract pass rate 当作长期记忆能力得分" in text
    assert "GenCode 长期记忆能力达到 100%" not in text


def test_metrics_cli_memory_challenge_and_core_report(tmp_path, monkeypatch):
    _write_input_artifacts(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert metrics_main(["--run", "memory_challenge"]) == 0
    challenge_path = tmp_path / "_local" / "benchmark" / "artifacts" / "memory-challenge-v1.json"
    assert challenge_path.is_file()
    challenge = json.loads(challenge_path.read_text(encoding="utf-8"))
    assert challenge["artifact_type"] == "memory-challenge-v1"
    assert set(challenge["variants"]) == {"memory_on", "memory_off", "naive_recent", "unsafe_memory"}

    assert metrics_main(["--run", "memory_agent_eval"]) == 0
    report_text = write_benchmark_core_report()
    assert "Memory Challenge Benchmark" in report_text
    assert "memory_on evidence_recall_at_k" in report_text
    assert "memory_on_vs_unsafe_memory secret_exposure_reduction" in report_text
