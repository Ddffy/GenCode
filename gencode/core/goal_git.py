"""Isolated Git worktrees and serialized integration for Goal mode."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from .workspace import IGNORED_PATH_NAMES

_INTERNAL_PATH_NAMES = IGNORED_PATH_NAMES | {".pico"}


class GoalGitError(RuntimeError):
    pass


class GoalGit:
    def __init__(self, workspace_root, goal_id):
        self.root = Path(workspace_root).resolve()
        self.goal_id = _safe_component(goal_id)
        self.goal_dir = self.root / ".gencode" / "goals" / self.goal_id
        self.integration_path = self.goal_dir / "integration"
        self.integration_branch = f"gencode/goal/{self.goal_id}/integration"
        self._workers = self.goal_dir / "worktrees"

    def preflight(self):
        root = self.git(["rev-parse", "--show-toplevel"]).strip()
        if not root or os.path.normcase(str(Path(root).resolve())) != os.path.normcase(str(self.root)):
            raise GoalGitError("/goal requires the workspace root to be the Git repository root")
        status = self.git(["status", "--porcelain=v1", "--untracked-files=all"])
        dirty = _non_internal_status_lines(status)
        if dirty:
            raise GoalGitError("/goal refused to start: commit, stash, or discard workspace changes first")
        base = self.git(["rev-parse", "--verify", "HEAD"]).strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", base):
            raise GoalGitError("/goal requires an existing base commit")
        return base

    def create_integration(self, base_commit):
        if self.integration_path.exists():
            raise GoalGitError(f"Goal integration worktree already exists: {self.integration_path}")
        self.goal_dir.mkdir(parents=True, exist_ok=True)
        self.git(
            ["worktree", "add", "-b", self.integration_branch, str(self.integration_path), base_commit]
        )
        return self.head(self.integration_path)

    def reopen_integration(self, branch):
        if branch != self.integration_branch:
            raise GoalGitError("saved integration branch does not match this Goal")
        if self.integration_path.exists():
            if not (self.integration_path / ".git").exists():
                raise GoalGitError("saved integration path is not a Git worktree")
            current = self.git(["branch", "--show-current"], cwd=self.integration_path).strip()
            if current != branch:
                raise GoalGitError("saved integration worktree is on the wrong branch")
            if _non_internal_status_lines(self.git(["status", "--porcelain=v1", "--untracked-files=all"], cwd=self.integration_path)):
                raise GoalGitError("saved integration worktree is dirty; preserve it and inspect before resume")
            return self.head(self.integration_path)
        self.git(["worktree", "add", str(self.integration_path), branch])
        return self.head(self.integration_path)

    def create_worker(self, node_id, attempt, base_commit):
        node = _safe_component(node_id)
        attempt_id = f"{node}-attempt-{int(attempt):02d}"
        path = self._workers / attempt_id
        branch = f"gencode/goal/{self.goal_id}/{node}/attempt-{int(attempt):02d}"
        if path.exists():
            raise GoalGitError(f"worker worktree path already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.git(["worktree", "add", "-b", branch, str(path), base_commit])
        return {"path": str(path), "branch": branch, "attempt_id": attempt_id}

    def commit_worker_changes(self, worktree, message):
        status = self.git(["status", "--porcelain=v1", "--untracked-files=all"], cwd=worktree)
        if not _non_internal_status_lines(status):
            return self.head(worktree)
        self.git(["add", "-A", "--", ".", ":(exclude).gencode", ":(exclude).pico"], cwd=worktree)
        self.git(["commit", "-m", message], cwd=worktree)
        if _non_internal_status_lines(self.git(["status", "--porcelain=v1", "--untracked-files=all"], cwd=worktree)):
            raise GoalGitError("worker worktree remained dirty after commit")
        return self.head(worktree)

    def merge_worker(self, commit, node_id):
        result = self._run(["merge", "--no-ff", "--no-edit", commit], cwd=self.integration_path)
        if result.returncode != 0:
            self._run(["merge", "--abort"], cwd=self.integration_path)
            raise GoalGitError(
                f"serial integration failed for {node_id}: "
                f"{(result.stderr or result.stdout).strip()[:1200]}"
            )
        return self.head(self.integration_path)

    def integration_head(self):
        return self.head(self.integration_path)

    def diff(self, base_commit, head_commit, limit=40000):
        text = self.git(
            ["diff", "--no-ext-diff", "--no-renames", f"{base_commit}...{head_commit}"],
            cwd=self.integration_path,
        )
        if len(text) > limit:
            return text[:limit] + f"\n...[diff truncated at {limit} characters]"
        return text

    def changed_files(self, base_commit, head_commit):
        text = self.git(
            ["diff", "--name-only", f"{base_commit}...{head_commit}"],
            cwd=self.integration_path,
        )
        return [line.strip() for line in text.splitlines() if line.strip()]

    def remove_worker(self, path, branch):
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self._workers.resolve())
        except ValueError as exc:
            raise GoalGitError("refusing to remove a worktree outside this Goal") from exc
        if not branch.startswith(f"gencode/goal/{self.goal_id}/") or not re.fullmatch(r"gencode/goal/[A-Za-z0-9_-]+/[A-Za-z0-9_/-]+", branch):
            raise GoalGitError("refusing to delete an unexpected worker branch")
        self.git(["worktree", "remove", "--force", str(resolved)])
        self.git(["branch", "-D", "--", str(branch)])

    def cleanup_workers(self, attempts):
        for attempt in attempts:
            if attempt.get("worktree") and attempt.get("branch"):
                self.remove_worker(attempt["worktree"], attempt["branch"])

    @staticmethod
    def head(cwd):
        return GoalGit._checked(["rev-parse", "--verify", "HEAD"], cwd).strip()

    def git(self, args, cwd=None):
        return self._checked(args, cwd or self.root)

    @staticmethod
    def _checked(args, cwd):
        result = GoalGit._run(args, cwd)
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "git command failed").strip()
            raise GoalGitError(message[:1200])
        return result.stdout

    @staticmethod
    def _run(args, cwd):
        try:
            return subprocess.run(
                ["git", *map(str, args)],
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return subprocess.CompletedProcess(["git", *map(str, args)], 1, "", str(exc))


def _safe_component(value):
    text = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", text):
        raise GoalGitError(f"invalid Goal Git identifier: {text!r}")
    return text


def _non_internal_status_lines(status):
    return [
        line
        for line in str(status or "").splitlines()
        if len(line) < 4
        or not _internal_path_status(line[3:])
    ]


def _internal_path_status(value):
    path = str(value or "").split(" -> ", 1)[0].strip('"')
    parts = path.replace("\\", "/").split("/")
    return any(part in _INTERNAL_PATH_NAMES for part in parts)
