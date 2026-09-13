import json

from gencode.evaluation.knowledge_eval import main, run_knowledge_evaluation


def test_typed_knowledge_evaluation_meets_all_gates(tmp_path):
    output = tmp_path / "knowledge-eval.json"

    artifact = run_knowledge_evaluation(output)

    assert artifact["passed"] is True
    assert all(artifact["gates"].values())
    assert artifact["metrics"]["wiki_recall_at_3"] >= 0.90
    assert artifact["metrics"]["skill_activation_accuracy"] >= 0.95
    assert artifact["metrics"]["spec_binding_accuracy"] == 1.0
    assert artifact["metrics"]["poison_or_invalid_exposure_rate"] == 0.0
    assert artifact["metrics"]["unified_baseline_cross_kind_hits"] > 0
    assert artifact["ablation"]["baseline_metrics"]["wiki_recall_at_3"] == 0.0
    assert (
        artifact["ablation"]["delta_enabled_minus_baseline"]["wiki_recall_at_3"] == 1.0
    )
    assert (
        artifact["ablation"]["delta_enabled_minus_baseline"][
            "skill_activation_accuracy"
        ]
        == 0.75
    )
    assert (
        artifact["ablation"]["delta_enabled_minus_baseline"]["spec_binding_accuracy"]
        == 0.75
    )
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True


def test_knowledge_eval_cli_asserts_gates(tmp_path):
    output = tmp_path / "cli-knowledge-eval.json"
    assert main(["--output", str(output), "--assert-gates"]) == 0
