from gencode.core.context_pressure import ContextPressure, ContextPressureController


def identity(**overrides):
    value = {
        "provider": "openai",
        "provider_base_url": "https://api.example.test/v1",
        "model": "gpt-test",
        "context_window": 1000,
        "prompt_cache_key": "cache-a",
        "prompt_hash": "prompt-a",
    }
    value.update(overrides)
    return value


def test_matching_identity_uses_real_input_tokens_for_pressure():
    pressure = ContextPressureController().evaluate(
        estimated_input_tokens=400,
        context_window=1000,
        current_identity=identity(),
        last_completion_metadata={
            **identity(),
            "input_tokens": 650,
        },
    )

    assert pressure.input_tokens == 650
    assert pressure.actual_input_tokens == 650
    assert pressure.last_actual_input_tokens == 650
    assert pressure.usage_source == "actual"
    assert pressure.calibration_source == "current_identity_match"
    assert pressure.pressure_ratio == 0.65


def test_stale_provider_model_or_context_window_falls_back_to_estimate():
    controller = ContextPressureController()

    for stale in (
        identity(provider="anthropic"),
        identity(provider_base_url="https://other.example.test/v1"),
        identity(model="gpt-other"),
        identity(context_window=2000),
    ):
        pressure = controller.evaluate(
            estimated_input_tokens=400,
            context_window=1000,
            current_identity=identity(),
            last_completion_metadata={**stale, "input_tokens": 650},
        )

        assert pressure.input_tokens == 400
        assert pressure.actual_input_tokens is None
        assert pressure.last_actual_input_tokens == 650
        assert pressure.usage_source == "estimated"
        assert pressure.calibration_source == "last_completion_identity_mismatch"


def test_prompt_hash_mismatch_falls_back_to_estimate():
    pressure = ContextPressureController().evaluate(
        estimated_input_tokens=400,
        context_window=1000,
        current_identity=identity(),
        last_completion_metadata={**identity(prompt_hash="old"), "input_tokens": 650},
    )

    assert pressure.input_tokens == 400
    assert pressure.actual_input_tokens is None
    assert pressure.last_actual_input_tokens == 650
    assert pressure.usage_source == "estimated"
    assert pressure.calibration_source == "last_completion_identity_mismatch"


def test_missing_metadata_falls_back_to_estimate():
    pressure = ContextPressureController().evaluate(
        estimated_input_tokens=400,
        context_window=1000,
        current_identity=identity(),
        last_completion_metadata={},
    )

    assert pressure.input_tokens == 400
    assert pressure.actual_input_tokens is None
    assert pressure.last_actual_input_tokens is None
    assert pressure.usage_source == "estimated"
    assert pressure.calibration_source == "missing_last_completion_metadata"


def test_pressure_tier_boundaries():
    assert ContextPressure(599, 1000, 1000).pressure_tier == "tier0_observe"
    assert ContextPressure(600, 1000, 1000).pressure_tier == "tier1_snip"
    assert ContextPressure(800, 1000, 1000).pressure_tier == "tier2_prune"
    assert ContextPressure(950, 1000, 1000).pressure_tier == "tier3_summary"


def test_cache_tokens_are_passed_through_when_present():
    pressure = ContextPressureController().evaluate(
        estimated_input_tokens=400,
        context_window=1000,
        current_identity=identity(),
        last_completion_metadata={
            **identity(),
            "input_tokens": 650,
            "cached_tokens": 128,
        },
    )

    assert pressure.cached_tokens == 128
    assert pressure.to_context_usage_fields()["cached_tokens"] == 128


def test_cache_tokens_are_not_current_when_identity_mismatches():
    pressure = ContextPressureController().evaluate(
        estimated_input_tokens=400,
        context_window=1000,
        current_identity=identity(),
        last_completion_metadata={
            **identity(provider_base_url="https://other.example.test/v1"),
            "input_tokens": 650,
            "cached_tokens": 128,
        },
    )

    assert pressure.usage_source == "estimated"
    assert pressure.cached_tokens is None
    assert pressure.to_context_usage_fields()["cached_tokens"] is None


def test_sanitized_provider_base_url_matches_without_secret_leak(tmp_path):
    from gencode import GenCode, SessionStore, WorkspaceContext
    from gencode.core.context_manager import ContextManager
    from gencode.testing import ScriptedModelClient

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    client = ScriptedModelClient([])
    client.provider = "openai"
    client.model = "gpt-test"
    client.base_url = "https://user:secret@example.test:8443/v1?api_key=sk-real-secret"
    agent = GenCode(
        model_client=client,
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
    )
    _, first_metadata = ContextManager(agent).build("first")
    identity = first_metadata["context_usage"]["current_identity"]
    agent.last_prompt_metadata = first_metadata
    agent.last_completion_metadata = {
        **identity,
        "provider_base_url": "https://example.test:8443/v1",
        "input_tokens": 123,
        "cached_tokens": 45,
    }

    _, second_metadata = ContextManager(agent).build("first")
    usage = second_metadata["context_usage"]

    assert usage["current_identity"]["provider_base_url"] == "https://example.test:8443/v1"
    assert "secret" not in usage["current_identity"]["provider_base_url"]
    assert "api_key" not in usage["current_identity"]["provider_base_url"]
    assert usage["usage_source"] == "actual"
    assert usage["actual_input_tokens"] == 123
    assert usage["cached_tokens"] == 45
