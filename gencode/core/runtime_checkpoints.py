"""Runtime workspace snapshot and checkpoint helpers."""

import hashlib
import uuid

from ..features import memory as memorylib
from .workspace import IGNORED_PATH_NAMES, clip, now

CHECKPOINT_SCHEMA_VERSION = "phase1-v1"


class RuntimeCheckpointsMixin:
    def capture_workspace_snapshot(self):
        snapshot = {}
        for path in self.root.rglob("*"):
            try:
                relative_parts = path.relative_to(self.root).parts
            except ValueError:
                continue
            if any(part in IGNORED_PATH_NAMES for part in relative_parts) or not path.is_file():
                continue
            try:
                snapshot[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                continue
        return snapshot

    @staticmethod
    def diff_workspace_snapshots(before, after):
        changed_paths = []
        summaries = []
        for path in sorted(set(before) | set(after)):
            if before.get(path) == after.get(path):
                continue
            changed_paths.append(path)
            if path not in before:
                summaries.append(f"created:{path}")
            elif path not in after:
                summaries.append(f"deleted:{path}")
            else:
                summaries.append(f"modified:{path}")
        return changed_paths, summaries

    def create_checkpoint(self, task_state, user_message, trigger):
        state = self.checkpoint_state()
        current = self.current_checkpoint()
        checkpoint_id = "ckpt_" + uuid.uuid4().hex[:8]
        key_files_by_path = {}
        freshness = {}
        for path in self.memory.to_dict()["working"]["recent_files"]:
            file_freshness = memorylib.file_freshness(path, self.root)
            freshness[path] = file_freshness
            key_files_by_path[str(path)] = {
                "path": str(path),
                "freshness": file_freshness,
                "line_ranges": [],
            }
        for path, ranges in self._current_turn_read_ranges().items():
            file_freshness = memorylib.file_freshness(path, self.root)
            freshness[path] = file_freshness
            item = key_files_by_path.setdefault(
                path,
                {"path": path, "freshness": file_freshness, "line_ranges": []},
            )
            item["line_ranges"] = ranges
        git = getattr(self, "git", None)
        git_state = {
            "enabled": bool(git is not None and git.enabled),
            "head": git.head() if git is not None and git.enabled else "",
            "dirty_paths": sorted(git.status_paths())
            if git is not None and git.enabled
            else [],
        }
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": current.get("checkpoint_id", "") if current else "",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "created_at": now(),
            "current_goal": str(user_message),
            "completed": [task_state.final_answer] if task_state.final_answer else [],
            "excluded": [],
            "current_blocker": "" if str(task_state.stop_reason or "") in ("", "final_answer_returned") else str(task_state.stop_reason),
            "next_step": self.infer_next_step(task_state),
            "key_files": list(key_files_by_path.values()),
            "freshness": freshness,
            "summary": f"{trigger}: {clip(str(user_message), 120)}",
            "runtime_identity": self.current_runtime_identity(),
            "git": git_state,
            "active_spec_ids": list(getattr(task_state, "active_spec_ids", [])),
        }
        state["items"][checkpoint_id] = checkpoint
        state["current_id"] = checkpoint_id
        task_state.checkpoint_id = checkpoint_id
        self.session["runtime_identity"] = checkpoint["runtime_identity"]
        self.session_path = self.session_store.save(self.session)
        return checkpoint

    def _current_turn_read_ranges(self):
        history = list(self.session.get("history", []))
        turn_start = next(
            (
                index
                for index in range(len(history) - 1, -1, -1)
                if history[index].get("role") == "user"
            ),
            0,
        )
        result = {}
        for item in history[turn_start + 1 :]:
            if item.get("role") != "tool":
                continue
            if str(item.get("tool_status", "ok")) not in {"", "ok", "success"}:
                continue
            args = item.get("args") or {}
            raw_path = str(args.get("path", "")).strip()
            if not raw_path:
                continue
            if item.get("name") in {"write_file", "patch_file"}:
                result.pop(raw_path, None)
                continue
            if item.get("name") != "read_file":
                continue
            start = int(args.get("start", 1) or 1)
            end = int(args.get("end", 200) or 200)
            ranges = result.setdefault(raw_path, [])
            current = (start, end)
            if current not in ranges:
                ranges.append(current)
        return {
            path: [f"{start}-{end}" for start, end in sorted(ranges)]
            for path, ranges in result.items()
        }
