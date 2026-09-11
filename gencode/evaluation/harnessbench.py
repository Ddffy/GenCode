"""Harness-Bench evidence metadata helpers.

GenCode already persists run trace, report, task state, session JSON, and session
events under `.gencode/`. Harness-Bench adapters need a stable manifest that points
to those files, otherwise process grading only sees a shell return code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_adapter_metadata(
    workspace: str | Path,
    *,
    session_id: str = "",
    returncode: int = 0,
) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    latest_run = _latest_dir(workspace / ".gencode" / "runs")
    sessions_root = workspace / ".gencode" / "sessions"
    session_path = sessions_root / f"{session_id}.json" if session_id else _latest_file(sessions_root, "*.json")
    event_path = (
        sessions_root / f"{session_id}.events.jsonl"
        if session_id
        else _latest_file(sessions_root, "*.events.jsonl")
    )

    metadata: dict[str, Any] = {
        "returncode": int(returncode),
        "workspace": str(workspace),
        "gencode_evidence_available": bool(latest_run),
        "gencode_evidence_missing": [],
    }
    _add_path(metadata, "gencode_run_dir", latest_run)
    if latest_run:
        _add_path(metadata, "gencode_trace_path", latest_run / "trace.jsonl")
        _add_path(metadata, "gencode_report_path", latest_run / "report.json")
        _add_path(metadata, "gencode_task_state_path", latest_run / "task_state.json")
    _add_path(metadata, "gencode_session_path", session_path)
    _add_path(metadata, "gencode_session_event_path", event_path)
    transcript_path = _write_process_transcript(session_path)
    _add_path(metadata, "gencode_process_transcript_path", transcript_path)

    required = (
        "gencode_run_dir",
        "gencode_trace_path",
        "gencode_report_path",
        "gencode_task_state_path",
        "gencode_session_path",
        "gencode_session_event_path",
        "gencode_process_transcript_path",
    )
    metadata["gencode_evidence_missing"] = [
        key for key in required if not metadata.get(key)
    ]
    metadata["gencode_evidence_available"] = not metadata["gencode_evidence_missing"]
    return metadata


def write_adapter_metadata(
    workspace: str | Path,
    output: str | Path,
    *,
    session_id: str = "",
    returncode: int = 0,
) -> dict[str, Any]:
    metadata = build_adapter_metadata(
        workspace, session_id=session_id, returncode=returncode
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit GenCode Harness-Bench adapter metadata.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--session-id", default="")
    parser.add_argument("--returncode", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    metadata = build_adapter_metadata(
        args.workspace,
        session_id=args.session_id,
        returncode=args.returncode,
    )
    text = json.dumps(metadata, ensure_ascii=True, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


def _add_path(metadata: dict[str, Any], key: str, path: Path | None) -> None:
    if path and path.exists():
        metadata[key] = str(path)


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


def _write_process_transcript(session_path: Path | None) -> Path | None:
    if not session_path or not session_path.is_file():
        return None
    try:
        payload = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    history = payload.get("history") if isinstance(payload, dict) else None
    if not isinstance(history, list):
        return None

    transcript_path = session_path.with_suffix(".process.jsonl")
    rows = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"user", "assistant", "tool", "toolResult"}:
            continue
        message = {
            key: item[key]
            for key in ("role", "content", "name", "tool_call_id", "toolCallId")
            if key in item
        }
        rows.append(json.dumps({"message": message}, ensure_ascii=True))
    if not rows:
        return None
    transcript_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return transcript_path


if __name__ == "__main__":
    raise SystemExit(main())
