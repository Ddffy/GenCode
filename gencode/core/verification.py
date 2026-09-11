"""Verification evidence reducer for shell-command tool traces."""

import re
import shlex

VERIFICATION_SIGNAL_SCHEMA = "gencode.verification_signal.v1"


def reduce_verification_signal(previous, event, changed_paths):
    signal = dict(previous or {})
    if event.get("event") != "tool_executed":
        return signal
    if event.get("workspace_changed"):
        signal = {
            "schema_version": VERIFICATION_SIGNAL_SCHEMA,
            "state": "missing",
            "last_workspace_change_span_id": str(event.get("span_id", "")),
            "changed_paths": list(changed_paths or []),
        }
    command = str((event.get("args", {}) or {}).get("command", "")).strip()
    command_class = classify_verification_command(command)
    if event.get("name") != "run_shell" or not command_class:
        return signal
    passed = str(event.get("status", "")) in {"", "ok"}
    signal.update(
        {
            "schema_version": VERIFICATION_SIGNAL_SCHEMA,
            "state": "passed" if passed else "failed",
            "source_span_id": str(event.get("span_id", "")),
            "command": command,
            "command_class": command_class,
            "after_last_workspace_change": bool(signal.get("last_workspace_change_span_id") or changed_paths),
            "changed_paths_present": bool(changed_paths),
            "covers_changed_paths": False,
            "coverage_confidence": "unknown",
            "changed_paths": list(changed_paths or []),
        }
    )
    return signal
def classify_verification_command(command):
    raw_command = str(command).strip()
    # shlex strips backslashes from unquoted Windows paths; inspect raw text first.
    windows_python = re.match(
        r"^[\'\"]?([A-Za-z]:[\\/].*?python(?:3(?:\.\d+)?)?\.exe)[\'\"]?\s+-m\s+(pytest|compileall)\b",
        raw_command,
        flags=re.IGNORECASE,
    )
    if windows_python:
        return {"pytest": "test", "compileall": "compile"}[windows_python.group(2).lower()]
    try:
        tokens = shlex.split(raw_command)
    except ValueError:
        tokens = str(command).split()
    tokens = [token.lower() for token in tokens]
    if not tokens or tokens[0] in {"echo", "printf", "grep", "rg", "cat"}:
        return ""
    if tokens[0] == "uv" and len(tokens) > 2 and tokens[1] == "run":
        while len(tokens) > 2 and tokens[2].startswith("-"):
            tokens = tokens[:2] + tokens[3:]
        tokens = tokens[2:]
    python_cmd = tokens[0].replace("\\", "/").rsplit("/", 1)[-1]
    if len(tokens) > 2 and _is_python_command(python_cmd) and tokens[1] == "-m":
        return {"pytest": "test", "compileall": "compile"}.get(tokens[2], "")
    if tokens[0] in {"pytest", "tox"}:
        return "test"
    if tokens[0] == "ruff" and len(tokens) > 1 and tokens[1] == "check":
        return "lint"
    if tokens[0] in {"mypy", "pyright"}:
        return "typecheck"
    if tokens[0] in {"npm", "pnpm"}:
        return _js_command_class(tokens)
    if tokens[:2] in (["yarn", "test"], ["go", "test"], ["cargo", "test"], ["make", "test"]):
        return "test"
    return ""
def _js_command_class(tokens):
    if len(tokens) < 2:
        return ""
    if tokens[1] == "test":
        return "test"
    if len(tokens) > 2 and tokens[1:3] in (["run", "test"], ["run", "build"]):
        return "test" if tokens[2] == "test" else "build"
    return "build" if tokens[1] == "build" else ""
def _is_python_command(command):
    basename = str(command).removesuffix(".exe")
    suffix = basename.removeprefix("python3.")
    return basename in {"python", "python3"} or (
        suffix != basename and suffix.replace(".", "").isdigit()
    )
