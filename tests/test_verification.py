"""Unit tests for verification signal extraction from tool traces."""

from gencode.core.verification import reduce_verification_signal


def tool_event(command, status="ok"):
    return {
        "event": "tool_executed",
        "name": "run_shell",
        "args": {"command": command},
        "status": status,
        "span_id": "span_verify",
    }


def test_verification_classifier_rejects_marker_only_shell_commands():
    for command in (
        "echo pytest",
        "printf pytest",
        "python -c \"print('pytest')\"",
        "grep pytest README.md",
    ):
        assert reduce_verification_signal({}, tool_event(command), ["src/app.py"]) == {}


def test_verification_classifier_accepts_common_test_commands():
    cases = {
        "pytest -q": "test",
        "uv run pytest tests -q": "test",
        "python -m pytest -q": "test",
        "python3.11 -m compileall gencode": "compile",
        "ruff check gencode tests": "lint",
        "mypy gencode": "typecheck",
        "pyright": "typecheck",
        "python -m compileall gencode": "compile",
        r"C:\Python312\python.exe -m compileall gencode": "compile",
        "npm test": "test",
        "npm run test": "test",
        "npm run build": "build",
        "pnpm test": "test",
        "pnpm run build": "build",
        "yarn test": "test",
        "tox": "test",
        "go test ./...": "test",
        "cargo test": "test",
        "make test": "test",
    }

    for command, command_class in cases.items():
        signal = reduce_verification_signal({}, tool_event(command), ["src/app.py"])
        assert signal["schema_version"] == "gencode.verification_signal.v1"
        assert signal["state"] == "passed", command
        assert signal["command_class"] == command_class
        assert signal["after_last_workspace_change"] is True
        assert signal["changed_paths_present"] is True
        assert signal["covers_changed_paths"] is False
        assert signal["coverage_confidence"] == "unknown"


def test_verification_signal_marks_workspace_change_as_missing_until_verified():
    changed = {
        "event": "tool_executed",
        "name": "write_file",
        "workspace_changed": True,
        "span_id": "span_change",
    }

    signal = reduce_verification_signal({}, changed, ["src/app.py"])

    assert signal == {
        "schema_version": "gencode.verification_signal.v1",
        "state": "missing",
        "last_workspace_change_span_id": "span_change",
        "changed_paths": ["src/app.py"],
    }

    verified = reduce_verification_signal(signal, tool_event("pytest -q"), ["src/app.py"])

    assert verified["state"] == "passed"
    assert verified["last_workspace_change_span_id"] == "span_change"
    assert verified["after_last_workspace_change"] is True
