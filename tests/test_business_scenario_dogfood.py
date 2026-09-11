import importlib.util
import inspect
import json
import os
from pathlib import Path

import pytest


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_business_scenario_dogfood.py"
    spec = importlib.util.spec_from_file_location("run_business_scenario_dogfood", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_business_scenario_dogfood_uses_real_provider_only():
    module = _load_module()
    source = inspect.getsource(module)

    assert "ScriptedModelClient" not in source
    assert "resolve_provider_config" in source
    assert "OpenAICompatibleModelClient" in source
    assert "AnthropicCompatibleModelClient" in source


@pytest.mark.skipif(
    os.environ.get("GENCODE_RUN_LIVE_BUSINESS_DOGFOOD") != "1",
    reason="live provider dogfood is opt-in",
)
def test_business_scenario_dogfood_covers_three_user_workflows_live(tmp_path):
    module = _load_module()
    output_dir = tmp_path / "business-dogfood"

    summary = module.run_dogfood(output_dir)

    assert summary["status"] == "passed"
    assert {scenario["id"] for scenario in summary["scenarios"]} == {
        "order_pricing_bugfix",
        "release_readiness_review",
        "incident_resume_fix",
    }
    assert "api_key" not in summary["provider"]

    for scenario in summary["scenarios"]:
        assert scenario["status"] == "passed"
        report = json.loads((output_dir / scenario["report_path"]).read_text(encoding="utf-8"))
        assert report["status"] == "completed"
        assert (output_dir / scenario["trace_path"]).exists()
        assert (output_dir / scenario["session_event_path"]).exists()

    order = next(scenario for scenario in summary["scenarios"] if scenario["id"] == "order_pricing_bugfix")
    assert "subtotal - discount + tax" in (
        output_dir / order["workspace_relpath"] / "src" / "order_pricing.py"
    ).read_text(encoding="utf-8")

    release = next(scenario for scenario in summary["scenarios"] if scenario["id"] == "release_readiness_review")
    assert (output_dir / release["workspace_relpath"] / "reports" / "release-readiness.md").exists()

    incident = next(scenario for scenario in summary["scenarios"] if scenario["id"] == "incident_resume_fix")
    incident_report = json.loads((output_dir / incident["report_path"]).read_text(encoding="utf-8"))
    assert any(item["status"] == "done" for item in incident_report["todos"]["items"])
