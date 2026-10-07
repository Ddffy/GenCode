"""Durable JSON snapshots and append-only events for Goal runs."""

from __future__ import annotations

import errno
import json
import os
import re
import tempfile
import threading
from pathlib import Path


class GoalStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def goal_dir(self, goal_id):
        return self.root / _safe_id(goal_id)

    def path(self, goal_id):
        return self.goal_dir(goal_id) / "goal.json"

    def events_path(self, goal_id):
        return self.goal_dir(goal_id) / "events.jsonl"

    def save(self, goal):
        path = self.path(goal["goal_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", delete=False, dir=path.parent,
                prefix=path.name + ".", suffix=".tmp",
            ) as handle:
                json.dump(goal, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                temporary = Path(handle.name)
            os.replace(temporary, path)
        return path

    def load(self, goal_id):
        return json.loads(self.path(goal_id).read_text(encoding="utf-8"))

    def append_event(self, goal_id, event):
        path = self.events_path(goal_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def acquire_lease(self, goal_id):
        path = self.goal_dir(goal_id) / "run.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()).encode("ascii"))
            handle.flush()
            return GoalLease(handle)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            if getattr(exc, "errno", None) in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise RuntimeError(f"Goal {goal_id} is already running in another process") from exc
            raise

    def list(self):
        rows = []
        for path in self.root.glob("*/goal.json"):
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda item: item.get("created_at", ""), reverse=True)


def _safe_id(value):
    text = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", text):
        raise ValueError("invalid Goal id")
    return text


class GoalLease:
    def __init__(self, handle):
        self.handle = handle

    def release(self):
        if self.handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None
