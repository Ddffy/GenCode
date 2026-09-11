"""Read-only helpers for GenCode run/session evidence.

These helpers intentionally do not import or instantiate the GenCode runtime. They
only interpret files that a completed CLI/REPL/TUI run left under `.gencode/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunEvidence:
    workspace: Path
    run_dir: Path | None
    report_path: Path | None
    trace_path: Path | None
    task_state_path: Path | None
    session_path: Path | None
    session_event_path: Path | None
    report: dict
    task_state: dict
    trace_events: list[dict]
    session_events: list[dict]

    @classmethod
    def latest(cls, workspace: Path) -> "RunEvidence":
        workspace = Path(workspace).resolve()
        run_dir = _latest_dir(workspace / ".gencode" / "runs")
        report_path = _existing(run_dir / "report.json") if run_dir else None
        trace_path = _existing(run_dir / "trace.jsonl") if run_dir else None
        task_state_path = _existing(run_dir / "task_state.json") if run_dir else None
        session_path = _latest_file(workspace / ".gencode" / "sessions", "*.json")
        session_event_path = _latest_file(workspace / ".gencode" / "sessions", "*.events.jsonl")
        return cls(
            workspace=workspace,
            run_dir=run_dir,
            report_path=report_path,
            trace_path=trace_path,
            task_state_path=task_state_path,
            session_path=session_path,
            session_event_path=session_event_path,
            report=_read_json(report_path),
            task_state=_read_json(task_state_path),
            trace_events=_read_jsonl(trace_path),
            session_events=_read_jsonl(session_event_path),
        )

    @property
    def session_id(self) -> str:
        return self.session_path.stem if self.session_path else ""

    def status(self) -> str:
        return str(self.report.get("status") or self.task_state.get("status") or "")

    def stop_reason(self) -> str:
        return str(self.report.get("stop_reason") or self.task_state.get("stop_reason") or "")

    def changed_paths(self) -> list[str]:
        task_changed = self.task_state.get("changed_paths") or []
        graph_changed = ((self.report.get("artifact_graph") or {}).get("changed_paths")) or []
        return list(dict.fromkeys([*task_changed, *graph_changed]))

    def tool_events(self, name: str | None = None) -> list[dict]:
        events = [event for event in self.trace_events if event.get("event") == "tool_executed"]
        if name is None:
            return events
        return [event for event in events if self.tool_name(event) == name]

    def tool_names(self) -> list[str]:
        return [self.tool_name(event) for event in self.tool_events()]

    def has_tools(self, *names: str) -> bool:
        seen = set(self.tool_names())
        return all(name in seen for name in names)

    def tool_error_codes(self, name: str | None = None) -> list[str]:
        return [str(event.get("tool_error_code") or "") for event in self.tool_events(name)]

    def full_output_artifacts(self) -> list[str]:
        artifacts = [
            str(event.get("full_output_artifact") or "")
            for event in self.tool_events()
            if event.get("full_output_artifact")
        ]
        for reminder in self.report.get("runtime_reminders") or []:
            artifact = str(reminder.get("artifact_path") or "")
            if artifact:
                artifacts.append(artifact)
        return list(dict.fromkeys(artifacts))

    def runtime_reminder_contains(self, text: str) -> bool:
        haystack = json.dumps(self.report.get("runtime_reminders") or [], ensure_ascii=False)
        return str(text) in haystack

    def has_session_event(self, event_name: str, **fields) -> bool:
        for event in self.session_events:
            if event.get("event") != event_name:
                continue
            if all(event.get(key) == value for key, value in fields.items()):
                return True
        return False

    @staticmethod
    def tool_name(event: dict) -> str:
        return str(event.get("name") or event.get("tool_name") or event.get("tool") or "")


def _latest_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = [path for path in root.iterdir() if path.is_dir()]
    return max(dirs, key=lambda path: path.stat().st_mtime) if dirs else None


def _latest_file(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    files = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def _existing(path: Path) -> Path | None:
    return path if path.exists() else None


def _read_json(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path | None) -> list[dict]:
    if not path or not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
