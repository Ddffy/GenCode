"""Phased tool execution shared by serial and bounded-parallel callers.

Only the raw ``RegisteredTool.execute`` call is safe to move to another
thread. Validation, policy decisions, evidence updates, memory writes, and Git
integration stay on the engine thread so concurrent reads cannot race shared
runtime state.
"""

import asyncio
import re
from dataclasses import dataclass

from gencode.core.actions.git_integration import attach_tool_metadata
from gencode.core.actions.governance import record_governance_decision
from gencode.core.actions.tool_policy import ToolPolicyChecker
from gencode.core.actions.tool_repetition import repeated_tool_call_metadata
from gencode.core.actions.tool_result_artifacts import prepare_tool_result_observation


@dataclass(frozen=True)
class PreparedToolCall:
    name: str
    args: dict
    tool: object | None
    rejected_result: str = ""
    rejected_metadata: dict | None = None
    record_rejection_note: bool = False

    @property
    def allowed(self):
        return self.rejected_metadata is None


@dataclass(frozen=True)
class RawToolResult:
    content: str = ""
    error: Exception | None = None


@dataclass(frozen=True)
class ToolCallOutcome:
    content: str
    metadata: dict


class ToolCancelledError(RuntimeError):
    pass


async def prepare_tool_call(agent, name, args):
    """Run lookup, validation, repetition, permission, and policy in order."""
    tool = agent.tools.get(name)
    if tool is None:
        metadata = tool_result_metadata(
            None, status="rejected", error_code="unknown_tool",
            risk_level="high", read_only=False,
        )
        record_governance_decision(
            agent, name, args, decision="deny", reason_code="unknown_tool",
            decision_type="tool_lookup",
        )
        return PreparedToolCall(
            name, args, None, f"error: unknown tool '{name}'", metadata
        )
    try:
        agent.validate_tool(name, args)
    except Exception as exc:  # noqa: BLE001 - validation errors are model feedback
        example = agent.tool_example(name)
        message = f"error: invalid arguments for {name}: {exc}"
        if example:
            message += f"\nexample: {example}"
        security_event_type = (
            "path_escape" if "path escapes workspace" in str(exc) else ""
        )
        metadata = tool_result_metadata(
            tool, status="rejected", error_code="invalid_arguments",
            security_event_type=security_event_type,
        )
        record_governance_decision(
            agent, name, args, decision="deny", reason_code="invalid_arguments",
            decision_type="tool_validation", original_reason=str(exc),
            security_event_type=security_event_type,
        )
        return PreparedToolCall(name, args, tool, message, metadata)
    repetition_reason = (
        agent.repeated_tool_call_reason(name, args)
        if hasattr(agent, "repeated_tool_call_reason")
        else (
            "repeated_identical_call"
            if agent.repeated_tool_call(name, args)
            else ""
        )
    )
    if repetition_reason:
        metadata = repeated_tool_call_metadata(tool, repetition_reason)
        record_governance_decision(
            agent, name, args, decision="deny",
            reason_code=repetition_reason, decision_type="tool_repetition",
        )
        message = (
            "error: Goal Worker search budget exhausted; use the evidence already collected to make the scoped edit, "
            "or return a concise blocker"
            if repetition_reason == "search_turn_budget_exhausted"
            else f"error: {repetition_reason} for {name}; use the existing evidence, search for an unseen detail, or return a final answer"
        )
        return PreparedToolCall(name, args, tool, message, metadata)
    policy = ToolPolicyChecker(agent).check(tool, args)
    emit_tool_policy_decision(agent, tool, args, policy)
    if not policy.allowed:
        record_governance_decision(
            agent, name, args, decision=policy.decision, reason_code=policy.reason,
            decision_type="tool_policy", original_reason=policy.reason,
            security_event_type="tool_policy",
        )
        metadata = tool_result_metadata(
            tool, status="rejected", error_code=policy.reason,
            security_event_type="tool_policy",
        )
        return PreparedToolCall(
            name, args, tool, policy.message, metadata,
            record_rejection_note=True,
        )

    decision = await agent.permission_checker.check_async(tool, args)
    emit_permission_decision(agent, tool, args, decision)
    permission_reason = (
        "read_only_violation"
        if not decision.allowed and getattr(agent, "read_only", False)
        else decision.reason
    )
    record_governance_decision(
        agent, name, args, decision=decision.decision,
        reason_code=permission_reason, decision_type="permission",
        original_reason=decision.reason,
        security_event_type=decision.security_event_type or (
            "read_only_block" if permission_reason == "read_only_violation" else ""
        ),
    )
    if not decision.allowed:
        metadata = tool_result_metadata(
            tool, status="rejected", error_code=decision.reason,
            security_event_type=decision.security_event_type,
        )
        return PreparedToolCall(
            name, args, tool, permission_error(agent, tool, decision), metadata
        )
    record_governance_decision(
        agent, name, args, decision=policy.decision, reason_code=policy.reason,
        decision_type="tool_policy", original_reason=policy.reason,
        security_event_type="tool_policy" if not policy.allowed else "",
    )
    if not policy.allowed:
        metadata = tool_result_metadata(
            tool, status="rejected", error_code=policy.reason,
            security_event_type="tool_policy",
        )
        return PreparedToolCall(
            name, args, tool, policy.message, metadata,
            record_rejection_note=True,
        )
    return PreparedToolCall(name, args, tool)


def begin_tool_execution(agent, prepared):
    return agent.capture_workspace_snapshot() if prepared.tool.risky else {}


async def execute_tool_raw(prepared, timeout=None, cancel_event=None):
    """Execute a prepared tool without mutating runtime bookkeeping."""
    execution = asyncio.create_task(prepared.tool.execute(prepared.args))
    cancel_waiter = (
        asyncio.create_task(cancel_event.wait()) if cancel_event is not None else None
    )
    timeout_waiter = (
        asyncio.create_task(asyncio.sleep(float(timeout))) if timeout else None
    )
    waiting = {execution}
    if cancel_waiter is not None:
        waiting.add(cancel_waiter)
    if timeout_waiter is not None:
        waiting.add(timeout_waiter)
    try:
        done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        if execution in done:
            result = execution.result()
            return RawToolResult(content=result.content)
        if cancel_waiter is not None and cancel_waiter in done:
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
            return RawToolResult(error=ToolCancelledError("tool cancelled"))
        execution.cancel()
        await asyncio.gather(execution, return_exceptions=True)
        return RawToolResult(error=asyncio.TimeoutError("tool execution timed out"))
    except asyncio.CancelledError:
        execution.cancel()
        await asyncio.gather(execution, return_exceptions=True)
        raise
    except Exception as exc:  # noqa: BLE001 - tool failures are returned as observations
        return RawToolResult(error=exc)
    finally:
        for waiter in (cancel_waiter, timeout_waiter):
            if waiter is not None and not waiter.done():
                waiter.cancel()
        await asyncio.gather(
            *(waiter for waiter in (cancel_waiter, timeout_waiter) if waiter is not None),
            return_exceptions=True,
        )


def finalize_tool_call(agent, prepared, before_snapshot, raw):
    """Commit one outcome to runtime state; callers must invoke serially."""
    if not prepared.allowed:
        metadata = dict(prepared.rejected_metadata or {})
        agent._last_tool_result_metadata = metadata
        if prepared.record_rejection_note:
            agent.record_process_note_for_tool(prepared.name, metadata)
        return ToolCallOutcome(prepared.rejected_result, metadata)

    tool = prepared.tool
    name = prepared.name
    args = prepared.args
    try:
        if raw.error is not None:
            raise raw.error
        full_result = raw.content
        pending_metadata = dict(
            getattr(agent, "_pending_tool_result_metadata", {}) or {}
        )
        agent._pending_tool_result_metadata = {}
        exit_code = run_shell_exit_code(full_result) if name == "run_shell" else 0
        result, artifact_metadata = prepare_tool_result_observation(
            agent, name, full_result
        )
        after_snapshot = (
            agent.capture_workspace_snapshot() if tool.risky else before_snapshot
        )
        affected_paths, diff_summary = agent.diff_workspace_snapshots(
            before_snapshot, after_snapshot
        )
        workspace_changed = bool(affected_paths)
        tool_status = "ok"
        tool_error_code = ""
        if name == "run_shell" and exit_code != 0:
            tool_status = "partial_success" if workspace_changed else "error"
            tool_error_code = (
                "tool_partial_success" if workspace_changed else "tool_failed"
            )
        agent.update_memory_after_tool(name, args, result)
        metadata = attach_tool_metadata(
            agent, name, args,
            tool_result_metadata(
                tool, status=tool_status, error_code=tool_error_code,
                affected_paths=affected_paths, workspace_changed=workspace_changed,
                workspace_fingerprint=agent.workspace.fingerprint(),
                diff_summary=diff_summary, **artifact_metadata, **pending_metadata,
            ),
            exit_code=exit_code if name == "run_shell" else None,
        )
        agent._last_tool_result_metadata = metadata
        agent.record_process_note_for_tool(name, metadata)
        return ToolCallOutcome(result + git_undo_suffix(metadata), metadata)
    except Exception as exc:  # noqa: BLE001 - preserve partial-success evidence
        after_snapshot = (
            agent.capture_workspace_snapshot() if tool.risky else before_snapshot
        )
        affected_paths, diff_summary = agent.diff_workspace_snapshots(
            before_snapshot, after_snapshot
        )
        workspace_changed = bool(affected_paths)
        security_event_type = (
            "path_escape" if "path escapes workspace" in str(exc) else ""
        )
        if name == "run_shell" and "sandbox required but unavailable" in str(exc):
            record_governance_decision(
                agent, name, args, decision="deny",
                reason_code="sandbox_rejected_command", decision_type="sandbox",
                original_reason=str(exc), security_event_type="sandbox",
            )
        metadata = attach_tool_metadata(
            agent, name, args,
            tool_result_metadata(
                tool,
                status="partial_success" if workspace_changed else "error",
                error_code=(
                    "tool_cancelled"
                    if isinstance(exc, ToolCancelledError)
                    else "tool_timeout" if isinstance(exc, asyncio.TimeoutError)
                    else "tool_partial_success" if workspace_changed else "tool_failed"
                ),
                security_event_type=security_event_type,
                affected_paths=affected_paths, workspace_changed=workspace_changed,
                workspace_fingerprint=agent.workspace.fingerprint(),
                diff_summary=diff_summary,
            ),
            exit_code=-1 if name == "run_shell" else None,
        )
        agent._last_tool_result_metadata = metadata
        agent.record_process_note_for_tool(name, metadata)
        message = f"error: tool {name} failed: {exc}"
        return ToolCallOutcome(message + git_undo_suffix(metadata), metadata)


async def execute_prepared_tool(agent, prepared):
    if not prepared.allowed:
        return finalize_tool_call(agent, prepared, {}, RawToolResult())
    before_snapshot = begin_tool_execution(agent, prepared)
    raw = await execute_tool_raw(
        prepared,
        timeout=getattr(agent, "tool_timeout_seconds", None),
        cancel_event=getattr(agent, "turn_cancel_event", None),
    )
    return finalize_tool_call(agent, prepared, before_snapshot, raw)


def run_shell_exit_code(result):
    match = re.search(r"exit_code:\s*(-?\d+)", str(result))
    return int(match.group(1)) if match else 0


def tool_result_metadata(
    tool, *, status, error_code="", security_event_type="", risk_level=None,
    read_only=None, affected_paths=None, workspace_changed=False,
    workspace_fingerprint=None, diff_summary=None, **extra
):
    metadata = {
        "tool_status": status,
        "tool_error_code": error_code,
        "security_event_type": security_event_type,
        "risk_level": risk_level if risk_level is not None else (
            "high" if tool.risky else "low"
        ),
        "read_only": read_only if read_only is not None else tool.read_only,
        "affected_paths": list(affected_paths or []),
        "workspace_changed": bool(workspace_changed),
        "diff_summary": list(diff_summary or []),
        **extra,
    }
    if workspace_fingerprint is not None:
        metadata["workspace_fingerprint"] = workspace_fingerprint
    return metadata


def emit_permission_decision(agent, tool, args, decision):
    agent.session_event_bus.emit(
        "permission_decision",
        {
            "tool_name": tool.name,
            "decision": decision.decision,
            "reason": decision.reason,
            "security_event_type": decision.security_event_type,
            "tool_profile": agent.active_tool_profile.name,
            "args": args or {},
        },
    )


def emit_tool_policy_decision(agent, tool, args, decision):
    agent.session_event_bus.emit(
        "tool_policy_decision",
        {
            "tool_name": tool.name,
            "decision": decision.decision,
            "reason": decision.reason,
            "args": args or {},
        },
    )


def permission_error(agent, tool, decision):
    if decision.reason == "plan_mode_path_mismatch":
        return f"error: plan mode can only write the active plan artifact ({agent.plan_mode.plan_path})"
    if decision.reason == "plan_mode_tool_not_allowed":
        return f"error: plan mode only allows read-only tools or writing the active plan artifact ({agent.plan_mode.plan_path})"
    if decision.reason == "write_scope_mismatch":
        return f"error: worker write_scope does not allow {tool.name} on this path"
    if decision.reason in {"approval_denied", "tool_not_allowed"}:
        return f"error: approval denied for {tool.name}"
    return f"error: permission denied for {tool.name}: {decision.reason}"


def git_undo_suffix(metadata):
    if metadata.get("git_undo_performed") and metadata.get("git_undo_message"):
        return "\n\n" + str(metadata["git_undo_message"])
    return ""
