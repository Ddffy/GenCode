from gencode.core.context_handoff import HandoffAdapter, HandoffParser, HandoffSummary, render_handoff_summary
from gencode.testing import ScriptedModelClient


VALID_HANDOFF = """## Goal
Implement context handoff compaction.

## Constraints
- Keep deterministic fallback
- Do not change low-pressure prompts

## Files Read
- gencode/core/compact.py

## Files Modified
- gencode/core/context_handoff.py

## Key Decisions
- Use complete_model boundary

## Blockers
- None

## Next Steps
- Wire CompactManager
"""


def test_parser_valid_output_extracts_sections():
    summary = HandoffParser().parse(VALID_HANDOFF)

    assert summary.goal == "Implement context handoff compaction."
    assert summary.constraints == ("Keep deterministic fallback", "Do not change low-pressure prompts")
    assert summary.files_read == ("gencode/core/compact.py",)
    assert summary.files_modified == ("gencode/core/context_handoff.py",)
    assert summary.key_decisions == ("Use complete_model boundary",)
    assert summary.blockers == ("None",)
    assert summary.next_steps == ("Wire CompactManager",)


def test_parser_missing_optional_sections_keeps_required_fields():
    summary = HandoffParser().parse(
        """## Goal
Keep going.

## Next Steps
- Run tests
"""
    )

    assert summary.goal == "Keep going."
    assert summary.constraints == ()
    assert summary.next_steps == ("Run tests",)


def test_parser_missing_goal_signals_parse_failure():
    summary = HandoffParser().parse(
        """## Next Steps
- Continue
"""
    )

    assert summary.goal == ""
    assert summary.next_steps == ("Continue",)
    assert summary.raw_text


def test_render_handoff_summary_preserves_critical_sections():
    text = render_handoff_summary(HandoffParser().parse(VALID_HANDOFF))

    assert "## Goal\nImplement context handoff compaction." in text
    assert "- gencode/core/compact.py" in text
    assert "- Use complete_model boundary" in text
    assert "- Wire CompactManager" in text


def test_adapter_success_records_usage_and_prompt_context():
    client = ScriptedModelClient([VALID_HANDOFF])
    client.last_completion_metadata = {
        "input_tokens": 120,
        "output_tokens": 40,
        "total_tokens": 160,
        "cached_tokens": 10,
        "provider_model": "model-a",
        "provider_protocol": "openai",
    }

    adapter = HandoffAdapter(client)
    summary = adapter.generate(
        delta_text="[User]: implement it",
        prior_summary_text="Prior context",
    )

    assert isinstance(summary, HandoffSummary)
    assert summary.goal == "Implement context handoff compaction."
    assert client.prompts
    assert "Prior Summary" in client.prompts[0]
    assert "Conversation Delta" in client.prompts[0]
    assert adapter.last_usage["total_tokens"] == 160
    assert summary.raw_text == VALID_HANDOFF


def test_adapter_model_failure_returns_none_and_clears_usage():
    adapter = HandoffAdapter(ScriptedModelClient([RuntimeError("boom")]))

    assert adapter.generate("delta") is None
    assert adapter.last_usage is None


def test_adapter_parse_failure_returns_none_but_records_usage():
    client = ScriptedModelClient(["garbage"])
    client.last_completion_metadata = {"input_tokens": 12, "output_tokens": 3}
    adapter = HandoffAdapter(client)

    assert adapter.generate("delta") is None
    assert adapter.last_usage["input_tokens"] == 12
    assert adapter.last_usage["output_tokens"] == 3
    assert adapter.last_usage["total_tokens"] == 15


def test_adapter_missing_next_steps_returns_none():
    adapter = HandoffAdapter(ScriptedModelClient(["## Goal\nKeep going.\n"]))

    assert adapter.generate("delta") is None


def test_adapter_bad_usage_metadata_does_not_break_handoff():
    client = ScriptedModelClient([VALID_HANDOFF])
    client.last_completion_metadata = {
        "input_tokens": "not-a-number",
        "output_tokens": None,
        "total_tokens": "bad",
        "cached_tokens": object(),
    }
    adapter = HandoffAdapter(client)

    summary = adapter.generate("delta")

    assert summary.goal == "Implement context handoff compaction."
    assert adapter.last_usage["input_tokens"] == 0
    assert adapter.last_usage["output_tokens"] == 0
    assert adapter.last_usage["total_tokens"] == 0
    assert adapter.last_usage["cached_tokens"] == 0


def test_adapter_negative_usage_metadata_is_clamped_to_zero():
    client = ScriptedModelClient([VALID_HANDOFF])
    client.last_completion_metadata = {
        "input_tokens": -10,
        "output_tokens": -2,
        "total_tokens": -12,
        "cached_tokens": -1,
    }
    adapter = HandoffAdapter(client)

    assert adapter.generate("delta") is not None
    assert adapter.last_usage["input_tokens"] == 0
    assert adapter.last_usage["output_tokens"] == 0
    assert adapter.last_usage["total_tokens"] == 0
    assert adapter.last_usage["cached_tokens"] == 0
