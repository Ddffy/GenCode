"""Git-backed checkpoints for workspace-changing agent actions.

The integration deliberately keeps Git at the edge of the runtime.  Tools still
perform the actual file changes; this module only records a safe baseline,
commits files changed by the agent, and can reset the latest agent commit after
a failed verification command.  Git failures are reported as metadata and do
not turn a successful tool call into a runtime failure.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
from pathlib import Path

_REPO_LOCKS: dict[str, threading.RLock] = {}
_REPO_LOCKS_GUARD = threading.Lock()


def _repo_lock(root: Path) -> threading.RLock:
    key = os.path.normcase(str(root.resolve()))
    with _REPO_LOCKS_GUARD:
        return _REPO_LOCKS.setdefault(key, threading.RLock())


_VERIFICATION_PATTERNS = (
    r"(?:^|[\s;&|])pytest(?:\s|$)",
    r"python(?:\d+(?:\.\d+)?)?\s+-m\s+pytest(?:\s|$)",
    r"python(?:\d+(?:\.\d+)?)?\s+-m\s+unittest(?:\s|$)",
    r"(?:^|[\s;&|])unittest(?:\s|$)",
    r"(?:^|[\s;&|])tox(?:\s|$)",
    r"(?:^|[\s;&|])nose2?(?:\s|$)",
    r"(?:^|[\s;&|])ruff\s+(?:check|format)(?:\s|$)",
    r"(?:^|[\s;&|])(?:mypy|pyright)(?:\s|$)",
    r"(?:^|[\s;&|])(?:npm|pnpm|yarn)\s+(?:run\s+)?(?:test|lint|typecheck)(?:\s|$)",
    r"(?:^|[\s;&|])cargo\s+test(?:\s|$)",
    r"(?:^|[\s;&|])go\s+test(?:\s|$)",
    r"(?:^|[\s;&|])dotnet\s+test(?:\s|$)",
    r"(?:^|[\s;&|])(?:mvn|gradle|make)\s+(?:[^;&|]*\s+)?test(?:\s|$)",
)
_VERIFICATION_RE = re.compile("|".join(_VERIFICATION_PATTERNS), re.IGNORECASE)


class GitIntegration:
    """Manage agent-owned Git commits without touching unrelated user work.

    A runtime instance owns its list of commits.  A process-wide repository
    lock prevents concurrent parent/worker runtimes from interleaving Git
    index operations.  The implementation is a no-op for a non-Git workspace
    or when the workspace root is not the Git root, which keeps fixture tests
    and read-only directories compatible.
    """

    MODIFYING_TOOLS = frozenset({"write_file", "patch_file", "run_shell"})

    def __init__(
        self,
        root,
        *,
        auto_commit=True,
        auto_undo=True,
        event_sink=None,
    ):
        self.root = Path(root).resolve()
        self.auto_commit = bool(auto_commit)
        self.auto_undo = bool(auto_undo)
        self.event_sink = event_sink
        self._lock = _repo_lock(self.root)
        self._turn = None
        self._commits: list[dict] = []
        self._undos: list[dict] = []
        self.enabled = False
        self.disabled_reason = ""
        self.repo_root = self.root
        self._discover_repository()

    # -- public lifecycle -------------------------------------------------

    def begin_turn(self, run_id=""):
        """Capture the HEAD and dirty paths before a turn starts."""
        if not self.enabled:
            self._turn = {
                "run_id": str(run_id or ""),
                "head": "",
                "dirty_paths": set(),
                "commits": [],
                "baseline_clean": False,
            }
            return self._turn_payload()

        with self._lock:
            self._turn = {
                "run_id": str(run_id or ""),
                "head": self.head(),
                "dirty_paths": self.status_paths(),
                "commits": [],
            }
            self._turn["baseline_clean"] = not self._turn["dirty_paths"]
            return self._turn_payload()

    def on_tool_result(self, name, args, metadata, *, run_id="", exit_code=None):
        """Commit an observed workspace change and undo failed verification.

        ``metadata`` is copied so callers can safely retain their base tool
        evidence.  Auto-undo is intentionally limited to commands that look
        like tests, lint, or type checks; commands such as ``grep`` can return
        exit code 1 as a normal result and must not erase code changes.
        """
        result = dict(metadata or {})
        result.setdefault("git_enabled", bool(self.enabled))
        result.setdefault("git_auto_commit", False)
        result.setdefault("git_auto_undo", False)
        result.setdefault("git_commit_sha", "")
        result.setdefault("git_commit_parent", "")
        result.setdefault("git_commit_paths", [])
        result.setdefault("git_undo_performed", False)
        result.setdefault("git_undo_reason", "")

        if not self.enabled:
            result.setdefault("git_reason", self.disabled_reason or "not_a_git_workspace")
            return result

        run_id = str(run_id or (self._turn or {}).get("run_id", "") or "")
        self._ensure_turn(run_id)
        changed_paths = list(result.get("affected_paths", []) or [])
        if (
            self.auto_commit
            and name in self.MODIFYING_TOOLS
            and bool(result.get("workspace_changed"))
            and changed_paths
        ):
            commit_metadata = self.commit_changes(
                changed_paths, tool_name=name, run_id=run_id
            )
            result.update(commit_metadata)

        if name == "run_shell" and exit_code not in (None, 0):
            command = str((args or {}).get("command", ""))
            is_verification = self.is_verification_command(command)
            result["git_verification_command"] = bool(is_verification)
            if self.auto_undo and is_verification:
                undo = self.undo_last_commit(run_id=run_id, automatic=True)
                result.update(undo)
        return result

    def undo_last_commit(self, *, run_id="", automatic=False):
        """Reset the latest safe agent commit, preserving the commit's parent.

        Automatic reset is refused when the turn began dirty or when any new
        worktree change is present.  This is the important difference from a
        blind ``git reset --hard HEAD~1``: user edits and other worker changes
        are never silently discarded.
        """
        base = {
            "git_undo_performed": False,
            "git_undo_reason": "",
            "git_undo_commit_sha": "",
            "git_undo_commit_parent": "",
            "git_undo_paths": [],
            "git_undo_message": "",
        }
        if not self.enabled:
            base["git_undo_reason"] = self.disabled_reason or "not_a_git_workspace"
            return base

        with self._lock:
            candidates = self._commits
            if automatic and self._turn is not None:
                turn_commits = {id(item) for item in self._turn.get("commits", [])}
                candidates = [item for item in candidates if id(item) in turn_commits]
            if not candidates and not automatic:
                discovered = self._head_agent_commit()
                if discovered is not None:
                    self._commits.append(discovered)
                    candidates = [discovered]
            if not candidates:
                base["git_undo_reason"] = "no_agent_commit"
                return base

            commit = candidates[-1]
            sha = str(commit.get("sha", ""))
            parent = str(commit.get("parent", ""))
            if self.head() != sha:
                base["git_undo_reason"] = "commit_is_not_head"
                return base
            if automatic and not bool(commit.get("baseline_clean", False)):
                base["git_undo_reason"] = "dirty_workspace_at_turn_start"
                return base
            if self.status_paths():
                base["git_undo_reason"] = "workspace_dirty_after_commit"
                return base

            added_paths = list(commit.get("added_paths", []) or [])
            if parent:
                reset = self._run(["reset", "--hard", parent])
            else:
                # An unborn repository has no parent commit.  Delete only the
                # temporary agent ref; committed files are cleaned below.
                reset = self._run(["update-ref", "-d", "HEAD"])
            if reset.returncode != 0:
                base["git_undo_reason"] = "reset_failed"
                base["git_undo_error"] = self._command_error(reset)
                return base

            # reset --hard leaves files that were introduced by the commit as
            # untracked.  Remove only those exact paths, never a broad clean.
            removed_untracked = []
            for path in added_paths:
                clean = self._run(["clean", "-f", "--", path])
                if clean.returncode == 0 and not self._path_exists(path):
                    removed_untracked.append(path)

            if parent and self.head() != parent:
                base["git_undo_reason"] = "reset_verification_failed"
                return base
            self._commits.remove(commit)
            if self._turn is not None:
                self._turn["commits"] = [
                    item for item in self._turn.get("commits", []) if item is not commit
                ]
            base.update(
                {
                    "git_undo_performed": True,
                    "git_undo_reason": "verification_failed" if automatic else "manual",
                    "git_undo_commit_sha": sha,
                    "git_undo_commit_parent": parent,
                    "git_undo_paths": list(commit.get("paths", [])),
                    "git_undo_removed_paths": removed_untracked,
                    "git_undo_message": (
                        f"[git] reset {sha[:8]} {'after failed verification' if automatic else 'for manual /undo'}; "
                        f"restored {parent[:8] if parent else 'empty repository'}"
                    ),
                }
            )
            self._undos.append(
                {
                    "sha": sha,
                    "parent": parent,
                    "paths": list(commit.get("paths", [])),
                    "automatic": bool(automatic),
                    "reason": base["git_undo_reason"],
                }
            )
            self._emit(
                "git_undo",
                {
                    "run_id": str(run_id or ""),
                    "automatic": bool(automatic),
                    "commit_sha": sha,
                    "parent": parent,
                    "paths": list(commit.get("paths", [])),
                },
            )
            return base

    def report(self):
        """Return compact report data suitable for a run artifact."""
        turn = self._turn or {}
        return {
            "enabled": bool(self.enabled),
            "auto_commit": bool(self.auto_commit),
            "auto_undo": bool(self.auto_undo),
            "repo_root": str(self.repo_root),
            "disabled_reason": self.disabled_reason,
            "turn_baseline": str(turn.get("head", "") or ""),
            "turn_baseline_clean": bool(turn.get("baseline_clean", False)),
            "commits": [
                {
                    "sha": item.get("sha", ""),
                    "parent": item.get("parent", ""),
                    "paths": list(item.get("paths", [])),
                    "tool_name": item.get("tool_name", ""),
                    "run_id": item.get("run_id", ""),
                }
                for item in self._commits
            ],
            "undos": [dict(item) for item in self._undos],
            "head": self.head() if self.enabled else "",
            "dirty_paths": sorted(self.status_paths()) if self.enabled else [],
        }

    def status_text(self):
        if not self.enabled:
            return self.disabled_reason or "not a git workspace"
        status = self._run(["status", "--short"])
        return status.stdout.strip() or "clean"

    @staticmethod
    def is_verification_command(command):
        return bool(_VERIFICATION_RE.search(str(command or "").strip()))

    # -- commit implementation ------------------------------------------

    def commit_changes(self, paths, *, tool_name, run_id):
        payload = {
            "git_auto_commit": False,
            "git_commit_sha": "",
            "git_commit_parent": "",
            "git_commit_paths": [],
            "git_commit_message": "",
            "git_commit_error": "",
            "git_commit_skipped_paths": [],
        }
        if not self.enabled or not self.auto_commit:
            return payload
        with self._lock:
            self._ensure_turn(run_id)
            baseline_dirty = set(self._turn.get("dirty_paths", set()))
            normalized = []
            skipped = []
            for path in paths:
                relative = self._relative_path(path)
                if not relative or self._is_internal(relative):
                    continue
                if relative in baseline_dirty:
                    skipped.append(relative)
                    continue
                if relative not in normalized:
                    normalized.append(relative)
            payload["git_commit_skipped_paths"] = skipped
            if not normalized:
                payload["git_commit_error"] = "all_changed_paths_were_dirty_at_turn_start"
                return payload

            before = self.head()
            add = self._run(["add", "-A", "--", *normalized])
            if add.returncode != 0:
                payload["git_commit_error"] = self._command_error(add)
                return payload
            staged = self._staged_paths(normalized)
            if not staged:
                self._unstage(normalized)
                return payload
            message = f"gencode: {str(tool_name or 'workspace change').strip()}"
            if run_id:
                message += f" ({str(run_id).strip()[:64]})"
            commit_result = self._run(["commit", "--only", "-m", message, "--", *staged])
            if commit_result.returncode != 0:
                payload["git_commit_error"] = self._command_error(commit_result)
                self._unstage(staged)
                return payload
            sha = self.head()
            if not sha or sha == before:
                payload["git_commit_error"] = "commit_head_did_not_advance"
                return payload
            parent = before
            record = {
                "sha": sha,
                "parent": parent,
                "paths": list(staged),
                "added_paths": self._added_paths(sha),
                "tool_name": str(tool_name or ""),
                "run_id": str(run_id or ""),
                "baseline_clean": bool(self._turn.get("baseline_clean", False)),
            }
            self._commits.append(record)
            self._turn.setdefault("commits", []).append(record)
            payload.update(
                {
                    "git_auto_commit": True,
                    "git_commit_sha": sha,
                    "git_commit_parent": parent,
                    "git_commit_paths": list(staged),
                    "git_commit_message": message,
                }
            )
            self._emit(
                "git_commit_created",
                {
                    "run_id": str(run_id or ""),
                    "tool_name": str(tool_name or ""),
                    "sha": sha,
                    "parent": parent,
                    "paths": list(staged),
                    "message": message,
                },
            )
            return payload

    # -- Git plumbing ----------------------------------------------------

    def head(self):
        result = self._run(["rev-parse", "HEAD"])
        return result.stdout.strip() if result.returncode == 0 else ""

    def status_paths(self):
        result = self._run(
            ["status", "--porcelain=v1", "--untracked-files=all", "--no-renames"]
        )
        if result.returncode != 0:
            return set()
        paths = set()
        for line in result.stdout.splitlines():
            if len(line) < 4:
                continue
            raw = line[3:]
            if " -> " in raw:
                raw = raw.rsplit(" -> ", 1)[-1]
            relative = self._relative_path(raw)
            if relative and not self._is_internal(relative):
                paths.add(relative)
        return paths

    def _discover_repository(self):
        result = self._run(["rev-parse", "--show-toplevel"])
        if result.returncode != 0:
            self.disabled_reason = "not_a_git_workspace"
            return
        try:
            discovered = Path(result.stdout.strip()).resolve()
        except (OSError, TypeError, ValueError):
            self.disabled_reason = "git_root_unreadable"
            return
        # A workspace rooted below a larger repository must not stage files
        # outside the workspace.  Require the same root for safe auto commits.
        if os.path.normcase(str(discovered)) != os.path.normcase(str(self.root)):
            self.disabled_reason = "workspace_root_is_not_git_root"
            self.repo_root = discovered
            return
        self.repo_root = discovered
        self.enabled = True

    def _run(self, args):
        try:
            return subprocess.run(
                ["git", *[str(arg) for arg in args]],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return subprocess.CompletedProcess(
                ["git", *[str(arg) for arg in args]],
                returncode=1,
                stdout="",
                stderr=str(exc),
            )

    @staticmethod
    def _command_error(result):
        return (str(result.stderr or "").strip() or str(result.stdout or "").strip())[:500]

    def _staged_paths(self, candidates):
        result = self._run(["diff", "--cached", "--name-only", "--", *candidates])
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def _unstage(self, paths):
        if paths:
            self._run(["restore", "--staged", "--", *paths])

    def _added_paths(self, sha):
        result = self._run(["diff-tree", "--root", "--no-commit-id", "--name-status", "-r", sha])
        if result.returncode != 0:
            return []
        added = []
        for line in result.stdout.splitlines():
            status, _, path = line.partition("\t")
            if status == "A" and path:
                added.append(path.strip())
        return added

    def _head_agent_commit(self):
        """Reconstruct the latest agent commit after a runtime restart."""
        sha = self.head()
        if not sha:
            return None
        subject = self._run(["log", "-1", "--format=%s"]).stdout.strip()
        if not subject.startswith("gencode:"):
            return None
        parent_result = self._run(["rev-parse", f"{sha}^"])
        parent = parent_result.stdout.strip() if parent_result.returncode == 0 else ""
        return {
            "sha": sha,
            "parent": parent,
            "paths": self._commit_paths(sha),
            "added_paths": self._added_paths(sha),
            "tool_name": subject[len("gencode:") :].strip(),
            "run_id": "",
            "baseline_clean": not bool(self.status_paths()),
        }

    def _commit_paths(self, sha):
        result = self._run(["diff-tree", "--root", "--no-commit-id", "--name-only", "-r", sha])
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def _ensure_turn(self, run_id):
        if self._turn is None:
            self.begin_turn(run_id)

    def _turn_payload(self):
        turn = self._turn or {}
        return {
            "enabled": bool(self.enabled),
            "head": str(turn.get("head", "") or ""),
            "dirty_paths": sorted(turn.get("dirty_paths", set())),
            "baseline_clean": bool(turn.get("baseline_clean", False)),
        }

    def _relative_path(self, path):
        try:
            candidate = Path(str(path))
            if candidate.is_absolute():
                candidate = candidate.resolve().relative_to(self.repo_root)
            else:
                candidate = Path(os.path.normpath(str(candidate)))
            relative = candidate.as_posix()
            if relative in {"", "."} or relative == ".." or relative.startswith("../"):
                return ""
            return relative
        except (OSError, ValueError):
            return ""

    @staticmethod
    def _is_internal(path):
        first = str(path).replace("\\", "/").split("/", 1)[0]
        return first in {
            ".git", ".gencode", ".pico", "__pycache__", ".pytest_cache",
            ".ruff_cache", ".venv", "venv",
        }

    def _path_exists(self, path):
        try:
            return (self.repo_root / path).exists()
        except OSError:
            return False

    def _emit(self, event, payload):
        if not callable(self.event_sink):
            return
        try:
            self.event_sink(event, payload)
        except Exception:  # noqa: BLE001,S110 - event observers must not break Git
            # Observability must never break a commit or undo operation.
            pass


def attach_tool_metadata(agent, name, args, metadata, *, exit_code=None):
    """Apply Git evidence without making Git an execution dependency."""
    integration = getattr(agent, "git", None)
    if integration is None:
        return metadata
    try:
        updated = integration.on_tool_result(
            name, args, metadata, run_id=getattr(agent, "current_run_id", ""),
            exit_code=exit_code,
        )
        if updated.get("git_undo_performed"):
            invalidate = getattr(agent, "invalidate_stale_memory", None)
            if callable(invalidate):
                invalidate()
        return updated
    except Exception as exc:  # noqa: BLE001 - Git must remain best effort
        updated = dict(metadata)
        updated["git_error"] = str(exc)[:500]
        return updated
