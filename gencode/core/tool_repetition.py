"""Repeated tool-call guardrails."""

import os
import re

FILE_MUTATION_TOOLS = {"write_file", "patch_file"}
MEDIA_INSPECTION_TOOLS = {"inspect_image"}
MAX_MEDIA_INSPECTIONS_PER_PATH = 2
MAX_READ_FILE_CALLS_PER_PATH = 5
_READ_LINE_RE = re.compile(r"^\s*(\d+):")


def is_repeated_tool_call(history, name, args):
    return bool(tool_call_repetition_reason(history, name, args))


def tool_call_repetition_reason(history, name, args, *, max_search_calls=None):
    current_turn = _current_turn_history(history)
    tool_events = [
        (index, item)
        for index, item in enumerate(current_turn)
        if item.get("role") == "tool"
    ]
    if name == "search" and max_search_calls is not None:
        search_calls = sum(item.get("name") == "search" for _, item in tool_events)
        if search_calls >= int(max_search_calls):
            return "search_turn_budget_exhausted"
    if name in MEDIA_INSPECTION_TOOLS and _media_path_inspection_count(tool_events, args) >= MAX_MEDIA_INSPECTIONS_PER_PATH:
        return "repeated_identical_call"
    if name == "read_file":
        return _read_file_repetition_reason(current_turn, args)
    matches = [
        (index, item)
        for index, item in tool_events
        if item.get("name") == name and item.get("args") == args
    ]
    if name in FILE_MUTATION_TOOLS:
        if not matches:
            return ""
        last_index, last_match = matches[-1]
        if not _failed_file_write_retry_is_now_informed(
            current_turn, last_index, last_match
        ):
            return "repeated_identical_call"
        return ""
    return "repeated_identical_call" if len(matches) >= 2 else ""


def _read_file_repetition_reason(current_turn, args):
    path = _normalized_path((args or {}).get("path", ""))
    if not path:
        return ""
    start = _range_value((args or {}).get("start"), 1)
    end = _range_value((args or {}).get("end"), 200)
    reads = []
    last_write_index = -1
    for index, item in enumerate(current_turn):
        if item.get("role") != "tool":
            continue
        item_args = item.get("args") or {}
        if item.get("name") == "run_shell":
            last_write_index = index
            continue
        if _normalized_path(item_args.get("path", "")) != path:
            continue
        if item.get("name") in FILE_MUTATION_TOOLS:
            last_write_index = index
        elif item.get("name") == "read_file":
            reads.append((index, item))

    reads = [
        (index, item)
        for index, item in reads
        if index > last_write_index and _successful_read(item)
    ]
    if any(
        _range_value((item.get("args") or {}).get("start"), 1) == start
        and _range_value((item.get("args") or {}).get("end"), 200) == end
        for _, item in reads
    ):
        return "repeated_identical_call"
    if len(reads) >= MAX_READ_FILE_CALLS_PER_PATH:
        return "read_file_path_budget_exhausted"
    for _, item in reads:
        if item.get("artifact_ref"):
            continue
        lines = _returned_line_numbers(item.get("content", ""))
        if lines and min(lines) <= start and max(lines) >= end:
            return "read_range_already_covered"
    return ""


def _successful_read(item):
    if str(item.get("tool_status", "ok")) not in {"", "ok", "success"}:
        return False
    if str(item.get("tool_error_code", "")):
        return False
    return not str(item.get("content", "")).startswith("error:")


def _returned_line_numbers(content):
    return {
        int(match.group(1))
        for line in str(content or "").splitlines()
        if (match := _READ_LINE_RE.match(line)) is not None
    }


def _normalized_path(path):
    value = str(path or "").strip().replace("\\", "/")
    if not value:
        return ""
    return os.path.normpath(value).replace("\\", "/").casefold()


def _range_value(value, default):
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _media_path_inspection_count(tool_events, args):
    path = str((args or {}).get("path", ""))
    if not path:
        return 0
    return sum(
        1
        for _, item in tool_events
        if item.get("name") in MEDIA_INSPECTION_TOOLS
        and str((item.get("args") or {}).get("path", "")) == path
        and not str(item.get("content", "")).startswith("error:")
    )


def repeated_tool_call_metadata(tool, error_code="repeated_identical_call"):
    return {
        "tool_status": "rejected",
        "tool_error_code": str(error_code),
        "security_event_type": "",
        "risk_level": "high" if tool.risky else "low",
        "read_only": tool.read_only,
        "affected_paths": [],
        "workspace_changed": False,
        "diff_summary": [],
    }


def _failed_file_write_retry_is_now_informed(current_turn, last_index, last_match):
    content = str(last_match.get("content", ""))
    if not content.startswith("error:"):
        return False
    path = str((last_match.get("args") or {}).get("path", ""))
    if not path:
        return False
    for item in current_turn[last_index + 1 :]:
        if item.get("role") != "tool" or item.get("name") != "read_file":
            continue
        args = item.get("args") or {}
        if (
            str(args.get("path", "")) == path
            and not str(item.get("content", "")).startswith("error:")
        ):
            return True
    return False


def _current_turn_history(history):
    history = list(history)
    for index in range(len(history) - 1, -1, -1):
        if history[index].get("role") == "user":
            return history[index + 1 :]
    return history
