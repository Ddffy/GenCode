"""Minimal bounded concurrency for independent built-in read tools."""

import asyncio
import json
import time
from dataclasses import dataclass

from ...tools.base import ToolEffect
from gencode.core.actions.tool_execution import (
    RawToolResult,
    ToolCancelledError,
    begin_tool_execution,
    execute_tool_raw,
    prepare_tool_call,
)

DEFAULT_TOOL_CONCURRENCY = 4


@dataclass(frozen=True)
class ParallelRawCall:
    prepared: object
    before_snapshot: dict
    raw: RawToolResult
    duration_ms: int
    started: bool = True
    skipped_reason: str = ""


def can_parallelize_tool_batch(agent, payloads, *, remaining_steps):
    """Accept only a complete, unique, side-effect-free batch.

    Falling back to the existing serial loop is deliberate: mixed read/write
    batches, partial-budget batches, duplicate calls, and callers already
    inside an asyncio loop keep their old semantics.
    """
    if (
        agent.abort_requested
        or len(payloads) < 2
        or len(payloads) > remaining_steps
    ):
        return False
    identities = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            return False
        name = str(payload.get("name", ""))
        tool = agent.tools.get(name)
        if tool is None:
            return False
        capability = tool.capability
        if (
            not tool.read_only
            or capability.effect is not ToolEffect.READ
            or not capability.concurrency_safe
        ):
            return False
        identity = _call_identity(name, payload.get("args", {}) or {})
        if identity in identities:
            return False
        identities.add(identity)
    return True


async def execute_parallel_safe_tools(agent, payloads, *, max_concurrency=DEFAULT_TOOL_CONCURRENCY):
    """Preflight serially and execute only raw reads concurrently.

    The returned calls are not committed to runtime state yet. Engine code
    finalizes them one at a time in source order so memory, history, trace, and
    checkpoints retain deterministic ordering.
    """
    prepared_calls = []
    preflight_ms = []
    before_snapshots = []
    for payload in payloads:
        started_at = time.monotonic()
        prepared = await prepare_tool_call(
            agent, str(payload.get("name", "")), payload.get("args", {}) or {}
        )
        prepared_calls.append(prepared)
        preflight_ms.append(int((time.monotonic() - started_at) * 1000))
        before_snapshots.append(
            begin_tool_execution(agent, prepared) if prepared.allowed else {}
        )

    raw_results = await _execute_raw_batch(
        agent,
        prepared_calls,
        max_concurrency=max_concurrency,
        timeout=getattr(agent, "tool_timeout_seconds", None),
    )
    return [
        ParallelRawCall(
            prepared=prepared,
            before_snapshot=before_snapshot,
            raw=raw,
            duration_ms=preflight_duration + raw_duration,
            started=started,
            skipped_reason=skipped_reason,
        )
        for prepared, before_snapshot, (
            raw,
            raw_duration,
            started,
            skipped_reason,
        ), preflight_duration in zip(
            prepared_calls, before_snapshots, raw_results, preflight_ms
        )
    ]


async def _execute_raw_batch(agent, prepared_calls, *, max_concurrency, timeout=None):
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
    started = [False] * len(prepared_calls)
    interruption = ""

    async def run_one(index, prepared):
        if not prepared.allowed:
            return RawToolResult(), 0, False, ""
        started_at = time.monotonic()
        try:
            async with semaphore:
                if interruption:
                    return (
                        RawToolResult(
                            error=ToolCancelledError(
                                "tool skipped after control request"
                            )
                        ),
                        0,
                        False,
                        interruption,
                    )
                started[index] = True
                task_state = getattr(agent, "current_task_state", None)
                agent.session_event_bus.emit(
                    "tool_started",
                    {
                        "run_id": getattr(task_state, "run_id", ""),
                        "tool_name": prepared.name,
                        "args": prepared.args,
                    },
                )
                raw = await execute_tool_raw(
                    prepared,
                    timeout=timeout,
                    cancel_event=getattr(agent, "turn_cancel_event", None),
                )
        except asyncio.CancelledError:
            if not interruption:
                raise
            reason = "" if started[index] else interruption
            return (
                RawToolResult(
                    error=ToolCancelledError("tool cancelled by control request")
                ),
                int((time.monotonic() - started_at) * 1000),
                started[index],
                reason,
            )
        except Exception as exc:  # noqa: BLE001 - executor failure becomes tool evidence
            raw = RawToolResult(error=exc)
        return raw, int((time.monotonic() - started_at) * 1000), started[index], ""

    tasks = [
        asyncio.create_task(run_one(index, prepared))
        for index, prepared in enumerate(prepared_calls)
    ]
    pending = set(tasks)
    control_queue = getattr(agent, "turn_control_queue", None)
    control_task = (
        asyncio.create_task(control_queue.get()) if control_queue is not None else None
    )
    try:
        while pending:
            waiting = set(pending)
            if control_task is not None:
                waiting.add(control_task)
            done, _ = await asyncio.wait(
                waiting, return_when=asyncio.FIRST_COMPLETED
            )
            if control_task is not None and control_task in done:
                control = control_task.result()
                kind = control.get("type")
                if kind == "queue":
                    agent.queued_turns.append(str(control.get("content", "")))
                    control_task = asyncio.create_task(control_queue.get())
                elif kind == "steer":
                    agent.pending_steer_message = str(control.get("content", ""))
                    interruption = "steered"
                    control_task = None
                elif kind == "cancel":
                    agent.abort_requested = True
                    agent.cancel_requested = True
                    cancel_event = getattr(agent, "turn_cancel_event", None)
                    if cancel_event is not None:
                        cancel_event.set()
                    interruption = "aborted"
                    control_task = None
            pending.difference_update(task for task in done if task in pending)
            if interruption:
                for task in pending:
                    task.cancel()
                break

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        normalized = []
        for index, result in enumerate(results):
            if isinstance(result, BaseException):
                if isinstance(result, asyncio.CancelledError):
                    reason = interruption if not started[index] else ""
                    raw = RawToolResult(
                        error=ToolCancelledError("tool cancelled by control request")
                    )
                    normalized.append((raw, 0, started[index], reason))
                else:
                    normalized.append(
                        (RawToolResult(error=result), 0, started[index], "")
                    )
            else:
                normalized.append(result)
        return normalized
    finally:
        unfinished = [task for task in tasks if not task.done()]
        for task in unfinished:
            task.cancel()
        if unfinished:
            await asyncio.gather(*unfinished, return_exceptions=True)
        if control_task is not None and not control_task.done():
            control_task.cancel()
            await asyncio.gather(control_task, return_exceptions=True)


def _call_identity(name, args):
    try:
        encoded = json.dumps(args, sort_keys=True, ensure_ascii=True)
    except (TypeError, ValueError):
        encoded = repr(args)
    return name, encoded
