from gencode.evaluation.native_tool_calling import run_native_tool_calling_evaluation


def test_native_tool_calling_evaluation_covers_roundtrip_and_recovery(tmp_path):
    result = run_native_tool_calling_evaluation(tmp_path / "native-eval")

    assert result["passed"] == result["total"] == 2
    assert result["native_roundtrip_rate"] == 1.0
    assert result["tool_schema_count"] > 0
    recovery = next(row for row in result["rows"] if row["case_id"] == "native_invalid_args_recovery")
    assert recovery["recovered_after_invalid_arguments"] is True
    assert "invalid_arguments" in recovery["tool_statuses"] or recovery["invalid_argument_rejected"]
