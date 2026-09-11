"""Unit tests for loop transition payloads and summary reduction."""

import pytest

from gencode.core.turn_transitions import (
    CONTINUE_TOOL_BATCH_EXECUTED,
    TERMINAL_FINAL_ANSWER_RETURNED,
    build_transition,
    emit_continue_transition,
    emit_terminal_transition,
    reduce_transition_summary,
)


def test_build_transition_uses_string_kind_values():
    event = build_transition(
        kind="continue",
        reason=CONTINUE_TOOL_BATCH_EXECUTED,
        attempt_index=3,
        tool_call_count=2,
        tool_requested_count=3,
        tool_executed_count=2,
    )

    assert event == {
        "kind": "continue",
        "reason": "tool_batch_executed",
        "attempt_index": 3,
        "tool_call_count": 2,
        "tool_requested_count": 3,
        "tool_executed_count": 2,
    }


def test_reduce_transition_summary_allows_only_one_terminal_transition():
    summary = reduce_transition_summary(
        {},
        build_transition(
            kind="terminal",
            reason=TERMINAL_FINAL_ANSWER_RETURNED,
            attempt_index=1,
            stop_reason=TERMINAL_FINAL_ANSWER_RETURNED,
        ),
    )
    assert summary["schema_version"] == "gencode.transition_summary.v1"

    with pytest.raises(ValueError, match="terminal transition"):
        reduce_transition_summary(
            summary,
            build_transition(
                kind="terminal",
                reason=TERMINAL_FINAL_ANSWER_RETURNED,
                attempt_index=2,
                stop_reason=TERMINAL_FINAL_ANSWER_RETURNED,
            ),
        )


def test_reduce_transition_summary_tracks_tool_request_and_execution_counts():
    summary = reduce_transition_summary(
        {},
        build_transition(
            kind="continue",
            reason=CONTINUE_TOOL_BATCH_EXECUTED,
            attempt_index=1,
            tool_call_count=1,
            tool_requested_count=2,
            tool_executed_count=1,
        ),
    )

    assert summary["tool_requested_count"] == 2
    assert summary["tool_executed_count"] == 1


def test_transition_wrappers_emit_continue_and_terminal_events():
    events = []

    class Agent:
        def emit_trace(self, task_state, event, payload):
            events.append((event, payload))
            return payload

    class TaskState:
        attempts = 4

    emit_continue_transition(Agent(), TaskState(), CONTINUE_TOOL_BATCH_EXECUTED)
    emit_terminal_transition(
        Agent(), TaskState(), TERMINAL_FINAL_ANSWER_RETURNED,
        stop_reason=TERMINAL_FINAL_ANSWER_RETURNED,
    )

    assert events[0][1]["kind"] == "continue"
    assert events[0][1]["reason"] == CONTINUE_TOOL_BATCH_EXECUTED
    assert events[1][1]["kind"] == "terminal"
    assert events[1][1]["stop_reason"] == TERMINAL_FINAL_ANSWER_RETURNED
