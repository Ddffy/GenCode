"""Minimal bounded concurrency for independent built-in read tools."""

import asyncio
import json
import time
from dataclasses import dataclass

from ..tools.base import ToolEffect
from .tool_execution import (
    RawToolResult,
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
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
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


def execute_parallel_safe_tools(agent, payloads, *, max_concurrency=DEFAULT_TOOL_CONCURRENCY):
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
        prepared = prepare_tool_call(
            agent, str(payload.get("name", "")), payload.get("args", {}) or {}
        )
        prepared_calls.append(prepared)
        preflight_ms.append(int((time.monotonic() - started_at) * 1000))
        before_snapshots.append(
            begin_tool_execution(agent, prepared) if prepared.allowed else {}
        )

    raw_results = asyncio.run(
        _execute_raw_batch(prepared_calls, max_concurrency=max_concurrency)
    )
    return [
        ParallelRawCall(
            prepared=prepared,
            before_snapshot=before_snapshot,
            raw=raw,
            duration_ms=preflight_duration + raw_duration,
        )
        for prepared, before_snapshot, (raw, raw_duration), preflight_duration in zip(
            prepared_calls, before_snapshots, raw_results, preflight_ms
        )
    ]


async def _execute_raw_batch(prepared_calls, *, max_concurrency):
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def run_one(prepared):
        if not prepared.allowed:
            return RawToolResult(), 0
        started_at = time.monotonic()
        try:
            async with semaphore:
                raw = await asyncio.to_thread(execute_tool_raw, prepared)
        except Exception as exc:  # noqa: BLE001 - executor failure becomes tool evidence
            raw = RawToolResult(error=exc)
        return raw, int((time.monotonic() - started_at) * 1000)

    return await asyncio.gather(*(run_one(item) for item in prepared_calls))


def _call_identity(name, args):
    try:
        encoded = json.dumps(args, sort_keys=True, ensure_ascii=True)
    except (TypeError, ValueError):
        encoded = repr(args)
    return name, encoded
