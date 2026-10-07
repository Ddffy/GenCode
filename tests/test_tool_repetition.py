from gencode.core.tool_repetition import tool_call_repetition_reason


def _read(path, start, end, *, content=None, artifact_ref=""):
    result = content if content is not None else "\n".join(
        f"{line:>4}: code" for line in range(start, end + 1)
    )
    return {
        "role": "tool",
        "name": "read_file",
        "args": {"path": path, "start": start, "end": end},
        "content": result,
        "tool_status": "ok",
        "artifact_ref": artifact_ref,
    }


def test_repeated_read_of_an_already_visible_range_is_rejected():
    history = [
        {"role": "user", "content": "inspect the file"},
        _read("gencode/core/engine.py", 1, 50),
    ]

    assert tool_call_repetition_reason(
        history, "read_file", {"path": "./gencode/core/engine.py", "start": 10, "end": 20}
    ) == "read_range_already_covered"


def test_reading_a_new_range_or_a_clipped_artifact_remains_allowed():
    history = [
        {"role": "user", "content": "inspect the file"},
        _read("src/large.py", 1, 200, content="full output saved: artifact\n1: partial", artifact_ref="artifact"),
    ]

    assert not tool_call_repetition_reason(
        history, "read_file", {"path": "src/large.py", "start": 30, "end": 40}
    )
    assert not tool_call_repetition_reason(
        history, "read_file", {"path": "src/large.py", "start": 201, "end": 220}
    )


def test_same_file_read_budget_stops_unbounded_window_scanning():
    history = [{"role": "user", "content": "inspect the file"}]
    history.extend(_read("src/large.py", start, start + 2) for start in range(1, 17, 2))

    assert tool_call_repetition_reason(
        history, "read_file", {"path": "src/large.py", "start": 30, "end": 32}
    ) == "read_file_path_budget_exhausted"


def test_goal_worker_search_budget_stops_keyword_churn():
    history = [{"role": "user", "content": "implement the requested change"}]
    history.extend(
        {
            "role": "tool",
            "name": "search",
            "args": {"path": "src/cli.ts", "pattern": f"term_{index}"},
            "content": f"match {index}",
            "tool_status": "ok",
        }
        for index in range(16)
    )

    assert tool_call_repetition_reason(
        history,
        "search",
        {"path": "src/cli.ts", "pattern": "a new keyword"},
        max_search_calls=16,
    ) == "search_turn_budget_exhausted"
    assert not tool_call_repetition_reason(
        history,
        "search",
        {"path": "src/cli.ts", "pattern": "a new keyword"},
    )


def test_file_write_resets_prior_read_evidence_for_that_file():
    history = [
        {"role": "user", "content": "inspect then update"},
        _read("src/file.py", 1, 20),
        {
            "role": "tool",
            "name": "write_file",
            "args": {"path": "src/file.py"},
            "content": "updated",
            "tool_status": "ok",
        },
    ]

    assert not tool_call_repetition_reason(
        history, "read_file", {"path": "src/file.py", "start": 1, "end": 20}
    )
