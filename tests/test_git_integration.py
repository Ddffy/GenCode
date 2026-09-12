import json
import subprocess

from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.testing import ScriptedModelClient


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )


def _build_git_agent(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "gencode-tests@example.com")
    _git(tmp_path, "config", "user.name", "GenCode Tests")
    (tmp_path / "sample.txt").write_text("before\n", encoding="utf-8")
    _git(tmp_path, "add", "sample.txt")
    _git(tmp_path, "commit", "-qm", "fixture baseline")
    return GenCode(
        model_client=ScriptedModelClient([]),
        workspace=WorkspaceContext.build(tmp_path, repo_root_override=tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
    )


def test_workspace_changes_are_committed_per_tool_and_failed_verification_is_undone(tmp_path):
    agent = _build_git_agent(tmp_path)

    assert agent.git.enabled is True
    agent.run_tool("read_file", {"path": "sample.txt", "start": 1, "end": 1})
    result = agent.run_tool(
        "write_file", {"path": "sample.txt", "content": "after\n"}
    )

    metadata = agent._last_tool_result_metadata
    assert result == "wrote sample.txt (6 chars)"
    assert metadata["git_auto_commit"] is True
    assert metadata["git_commit_paths"] == ["sample.txt"]
    commit_sha = metadata["git_commit_sha"]
    assert commit_sha
    assert _git(tmp_path, "rev-parse", "HEAD").stdout.strip() == commit_sha

    failed = agent.run_tool(
        "run_shell", {"command": "python -m pytest -q", "timeout": 20}
    )

    metadata = agent._last_tool_result_metadata
    assert "git] reset" in failed
    assert metadata["git_undo_performed"] is True
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "before\n"
    assert _git(tmp_path, "log", "-1", "--pretty=%s").stdout.strip() == "fixture baseline"

    events = [
        json.loads(line)
        for line in agent.session_event_bus.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(item["event"] == "git_commit_created" for item in events)
    assert any(item["event"] == "git_undo" for item in events)


def test_non_verification_shell_failure_does_not_undo_agent_commit(tmp_path):
    agent = _build_git_agent(tmp_path)
    agent.run_tool("read_file", {"path": "sample.txt", "start": 1, "end": 1})
    agent.run_tool("write_file", {"path": "sample.txt", "content": "after\n"})

    result = agent.run_tool(
        "run_shell",
        {"command": 'python -c "import sys; sys.exit(1)"', "timeout": 20},
    )

    metadata = agent._last_tool_result_metadata
    assert "git] reset" not in result
    assert metadata["git_verification_command"] is False
    assert metadata["git_undo_performed"] is False
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "after\n"
    assert _git(tmp_path, "log", "-1", "--pretty=%s").stdout.startswith("gencode: write_file")


def test_dirty_workspace_is_not_auto_reset(tmp_path):
    agent = _build_git_agent(tmp_path)
    (tmp_path / "sample.txt").write_text("user edit\n", encoding="utf-8")

    agent.run_tool("read_file", {"path": "sample.txt", "start": 1, "end": 1})
    result = agent.run_tool(
        "write_file", {"path": "sample.txt", "content": "agent edit\n"}
    )

    metadata = agent._last_tool_result_metadata
    assert result == "wrote sample.txt (11 chars)"
    assert metadata["git_auto_commit"] is False
    assert metadata["git_commit_skipped_paths"] == ["sample.txt"]
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "agent edit\n"


def test_manual_undo_can_find_the_latest_gencode_commit_after_runtime_restart(tmp_path):
    first = _build_git_agent(tmp_path)
    first.run_tool("read_file", {"path": "sample.txt", "start": 1, "end": 1})
    first.run_tool("write_file", {"path": "sample.txt", "content": "after\n"})

    second = GenCode(
        model_client=ScriptedModelClient([]),
        workspace=WorkspaceContext.build(tmp_path, repo_root_override=tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
    )
    message = second.undo_last_git_commit()

    assert "git] reset" in message
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "before\n"
    assert _git(tmp_path, "log", "-1", "--pretty=%s").stdout.strip() == "fixture baseline"
