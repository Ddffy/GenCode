import os
import subprocess
import sys
from pathlib import Path

import pytest

from gencode.features.sandbox.config import SandboxConfig
from gencode.features.sandbox.runner import SandboxRunner


def test_required_sandbox_rejects_when_backend_is_unavailable(tmp_path):
    runner = SandboxRunner(
        SandboxConfig(mode="required", backend="bubblewrap"), which=lambda name: None
    )

    with pytest.raises(RuntimeError, match="sandbox required but unavailable"):
        runner.run("echo hi", cwd=tmp_path, env={}, timeout=5)


def test_required_sandbox_does_not_honor_excluded_commands(tmp_path):
    runner = SandboxRunner(
        SandboxConfig(mode="required", backend="bubblewrap", excluded_commands=("*",)),
        which=lambda name: None,
    )

    with pytest.raises(RuntimeError, match="sandbox required but unavailable"):
        runner.run("echo hi", cwd=tmp_path, env={}, timeout=5)


def test_best_effort_sandbox_records_degrade_and_runs_without_backend(tmp_path):
    events = []
    runner = SandboxRunner(
        SandboxConfig(mode="best_effort", backend="bubblewrap"),
        which=lambda name: None,
        emit_event=lambda event, payload: events.append((event, payload)),
    )

    result = runner.run(
        subprocess.list2cmdline([sys.executable, "-c", "print(42)"]),
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout=5,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "42"
    assert events[0][0] == "sandbox_unavailable"


def test_off_sandbox_keeps_plain_subprocess_behavior(tmp_path):
    runner = SandboxRunner(SandboxConfig(mode="off"), run=subprocess.run)

    command = subprocess.list2cmdline([sys.executable, "-c", "import os; print(os.getcwd())"])
    result = runner.run(command, cwd=tmp_path, env=os.environ.copy(), timeout=5)

    assert Path(result.stdout.strip()) == tmp_path
