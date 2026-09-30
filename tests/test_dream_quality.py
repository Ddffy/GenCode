from gencode.evaluation.dream_quality import run_dream_quality_v1


def test_dream_quality_checks_candidate_provenance_and_quarantine(tmp_path):
    artifact = run_dream_quality_v1(artifact_path=tmp_path / "dream-quality.json")

    assert artifact["summary"]["total_cases"] == 4
    assert artifact["summary"]["failed"] == 0
    assert artifact["summary"]["signal_retention_rate"] == 1.0
    assert artifact["summary"]["noise_rejection_rate"] == 1.0
    assert artifact["summary"]["secret_rejection_rate"] == 1.0
    assert artifact["summary"]["dedupe_rate"] == 1.0
