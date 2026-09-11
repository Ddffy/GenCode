"""Deterministic evaluation for the native Tool Calling path.

The experiment keeps the model scripted, so a result change is attributable to
the Runtime contract rather than provider sampling.  It exercises the same
GenCode Engine, tool validation, history persistence and run evidence used by a
real turn, but supplies structured ``ModelResult.tool_calls`` instead of XML
tool tags.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import GenCode, SessionStore, WorkspaceContext
from ..providers.base import ModelResult
from ..testing import NativeScriptedModelClient, ScriptedModelClient


def _tool_call(call_id, name, args):
    return ModelResult(
        text=f"Calling {name}.",
        metadata={"native_tool_calling": True},
        tool_calls=({"id": call_id, "name": name, "args": args},),
        stop_reason="tool_use",
    )


def _final(text):
    return ModelResult(
        text=text,
        metadata={"native_tool_calling": True},
        stop_reason="stop",
    )


def _run_legacy_case(root: Path, case_id: str, outputs: list[str]) -> dict:
    workspace = root / f"{case_id}_legacy"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "README.md").write_text("native tool calling works\n", encoding="utf-8")
    client = ScriptedModelClient(outputs)
    agent = GenCode(
        model_client=client,
        workspace=WorkspaceContext.build(workspace),
        session_store=SessionStore(workspace / ".gencode" / "sessions"),
        approval_policy="auto",
    )
    agent.ask("Inspect README.md with native tools")
    input_chars = sum(len(str(prompt)) for prompt in client.prompts)
    return {
        "model_calls": len(client.prompts),
        "estimated_input_tokens": (input_chars + 3) // 4,
    }


def _run_case(root: Path, case_id: str, outputs: list[ModelResult], legacy_outputs: list[str]) -> dict:
    workspace = root / case_id
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "README.md").write_text("native tool calling works\n", encoding="utf-8")
    client = NativeScriptedModelClient(outputs)
    agent = GenCode(
        model_client=client,
        workspace=WorkspaceContext.build(workspace),
        session_store=SessionStore(workspace / ".gencode" / "sessions"),
        approval_policy="auto",
    )
    answer = agent.ask("Inspect README.md with native tools")
    history = agent.session.get("history", [])
    assistant_calls = [
        item for item in history if item.get("role") == "assistant" and item.get("tool_calls")
    ]
    tool_results = [item for item in history if item.get("role") == "tool"]
    native_message_roundtrip = any(
        any(message.get("role") == "assistant" and message.get("tool_calls") for message in turn)
        and any(message.get("role") == "tool" and message.get("tool_call_id") for message in turn)
        for turn in client.messages
    )
    passed = bool(answer.strip()) and bool(assistant_calls) and bool(tool_results)
    message_chars = sum(len(json.dumps(turn, ensure_ascii=False)) for turn in client.messages)
    schema_chars = sum(
        len(json.dumps(tools, ensure_ascii=False)) for tools in client.tools_seen
    )
    legacy = _run_legacy_case(root, case_id, legacy_outputs)
    native_tokens = (message_chars + schema_chars + 3) // 4
    return {
        "case_id": case_id,
        "passed": passed,
        "answer": answer,
        "model_calls": len(client.messages),
        "tool_schema_count": len(client.tools_seen[0]) if client.tools_seen else 0,
        "assistant_tool_call_count": len(assistant_calls),
        "tool_result_count": len(tool_results),
        "tool_statuses": [str(item.get("tool_status", "")) for item in tool_results],
        "invalid_argument_rejected": any(
            str(item.get("tool_error_code", "")) == "invalid_arguments" for item in tool_results
        ),
        "native_message_roundtrip": native_message_roundtrip,
        "tool_call_ids": [
            str(item.get("tool_call_id", "")) for item in tool_results if item.get("tool_call_id")
        ],
        "prompt_mode": "native_messages",
        "message_chars": message_chars,
        "tool_schema_chars": schema_chars,
        "estimated_input_tokens": native_tokens,
        "legacy_estimated_input_tokens": legacy["estimated_input_tokens"],
        "legacy_model_calls": legacy["model_calls"],
        "estimated_token_delta_pct": round(
            100 * (legacy["estimated_input_tokens"] - native_tokens)
            / max(1, legacy["estimated_input_tokens"]),
            2,
        ),
    }


def run_native_tool_calling_evaluation(output_dir, repetitions: int = 1) -> dict:
    """Run success and invalid-argument recovery cases and persist evidence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for repeat in range(max(1, int(repetitions))):
        rows.append(
            _run_case(
                output_dir / f"repeat-{repeat + 1}",
                "native_success",
                [
                    _tool_call("native-read-1", "read_file", {"path": "README.md", "start": 1, "end": 1}),
                    _final("README inspected with a native tool call."),
                ],
                [
                    '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":1}}</tool>',
                    "<final>README inspected with a native tool call.</final>",
                ],
            )
        )
        recovery = _run_case(
            output_dir / f"repeat-{repeat + 1}",
            "native_invalid_args_recovery",
                [
                _tool_call("native-bad-1", "read_file", {}),
                _tool_call("native-read-2", "read_file", {"path": "README.md", "start": 1, "end": 1}),
                    _final("Recovered after native argument validation."),
                ],
                [
                    '<tool>{"name":"read_file","args":{}}</tool>',
                    '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":1}}</tool>',
                    "<final>Recovered after native argument validation.</final>",
                ],
        )
        # The first call must be rejected, but the second call must still run.
        recovery["recovered_after_invalid_arguments"] = (
            recovery["passed"]
            and recovery["model_calls"] >= 3
            and recovery["invalid_argument_rejected"]
        )
        rows.append(recovery)

    summary = {
        "experiment": "native_tool_calling",
        "protocol": "provider-native structured messages",
        "rows": rows,
        "passed": sum(1 for row in rows if row["passed"]),
        "total": len(rows),
        "native_roundtrip_rate": sum(1 for row in rows if row["native_message_roundtrip"]) / max(1, len(rows)),
        "tool_schema_count": rows[0]["tool_schema_count"] if rows else 0,
        "avg_estimated_input_tokens": round(
            sum(row["estimated_input_tokens"] for row in rows) / max(1, len(rows)), 2
        ),
        "avg_legacy_estimated_input_tokens": round(
            sum(row["legacy_estimated_input_tokens"] for row in rows) / max(1, len(rows)), 2
        ),
        "avg_estimated_token_delta_pct": round(
            sum(row["estimated_token_delta_pct"] for row in rows) / max(1, len(rows)), 2
        ),
    }
    (output_dir / "native_tool_calling.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
