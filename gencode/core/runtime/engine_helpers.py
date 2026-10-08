"""Control-loop tool and retry helpers shared by Engine.

These helpers execute tool payloads and handle retry summaries while Engine
keeps the turn loop shape visible. Terminal-state policy lives in
completion_governance.
"""

import asyncio
import time

from ...providers.base import complete_model_async, stream_model
from ...providers.errors import ProviderError
from gencode.core.runtime.completion_governance import finish_stopped_run
from gencode.core.context.context_usage import estimate_tokens
from gencode.core.runtime.engine_read_only_qa import READ_ONLY_QA_FINAL_MAX_NEW_TOKENS
from gencode.core.runtime.model_io.native_messages import build_native_messages
from gencode.core.actions.parallel_tools import (
    can_parallelize_tool_batch,
    execute_parallel_safe_tools,
)
from gencode.core.runtime.task_state import STOP_REASON_GOAL_TOKEN_BUDGET_EXHAUSTED
from gencode.core.actions.tool_execution import finalize_tool_call
from gencode.core.runtime.workspace_context import clip, now


class GoalTokenCallBudget:
    def __init__(self, engine, agent, task_state, run_started_at):
        self.engine = engine
        self.agent = agent
        self.task_state = task_state
        self.run_started_at = run_started_at
        self.budget = getattr(agent, "goal_token_budget", None)
        self.reservation = None

    @property
    def enabled(self):
        return self.budget is not None

    async def reserve(self, user_message, prompt, prompt_metadata, qa_final_only):
        max_output_tokens = (
            min(self.agent.max_new_tokens, READ_ONLY_QA_FINAL_MAX_NEW_TOKENS)
            if qa_final_only
            else self.agent.max_new_tokens
        )
        if self.budget is None:
            return max_output_tokens, []
        context_usage = dict(prompt_metadata.get("context_usage", {}) or {})
        estimated_input_tokens = max(
            int(context_usage.get("total_estimated_tokens", 0) or 0),
            estimate_tokens(len(prompt)),
        )
        role = str(getattr(self.agent, "goal_role", "unknown") or "unknown")
        self.reservation = await self.budget.reserve(
            estimated_input_tokens, max_output_tokens, role
        )
        if self.reservation is not None:
            return max_output_tokens, []
        usage = self.budget.snapshot()
        self.task_state.evidence_summaries["goal_token_usage"] = usage
        final = (
            "Goal token budget exhausted before the next model call "
            f"({usage['total_tokens']}/{usage['max_total_tokens']} tokens; role={role})."
        )
        events = list(
            finish_stopped_run(
                self.engine,
                self.task_state,
                user_message,
                final,
                STOP_REASON_GOAL_TOKEN_BUDGET_EXHAUSTED,
                self.run_started_at,
            )
        )
        return max_output_tokens, events

    async def record(self, result):
        metadata = dict(
            result.metadata
            or getattr(self.agent.model_client, "last_completion_metadata", {})
            or {}
        )
        if self.budget is not None and self.reservation is not None:
            usage = await self.budget.record(
                self.reservation,
                getattr(self.agent, "goal_role", "unknown"),
                metadata,
                result.text,
            )
            self.task_state.evidence_summaries["goal_token_usage"] = usage
        self.reservation = None
        return metadata

    async def release(self):
        if self.budget is not None and self.reservation is not None:
            await self.budget.release(self.reservation)
            self.reservation = None

    def stop_before_tool(self, user_message):
        if self.budget is None or not self.budget.blocked:
            return []
        usage = self.budget.snapshot()
        self.task_state.evidence_summaries["goal_token_usage"] = usage
        role = str(usage.get("budget_exhausted_role", "") or "")
        role_text = f"; role={role}" if role else ""
        final = (
            "Goal token budget exhausted after this model call; "
            "the returned tool request was not executed. "
            f"({usage['total_tokens']}/{usage['max_total_tokens']} tokens{role_text})."
        )
        return list(
            finish_stopped_run(
                self.engine,
                self.task_state,
                user_message,
                final,
                STOP_REASON_GOAL_TOKEN_BUDGET_EXHAUSTED,
                self.run_started_at,
            )
        )


def _set_fast_read_only_qa(agent, enabled):
    previous = getattr(agent, "_fast_read_only_qa_previous_tool_profile", None)
    agent.fast_read_only_qa = bool(enabled)
    if enabled and previous is None:
        agent._fast_read_only_qa_previous_tool_profile = agent._active_tool_profile_name
        agent.set_tool_profile("readonly")
    elif not enabled and previous is not None:
        agent.set_tool_profile(previous)
        del agent._fast_read_only_qa_previous_tool_profile


def complete_runtime_model(agent, prompt, user_message, max_new_tokens, tools=None, **kwargs):
    """Dispatch native-capable clients through structured messages."""
    messages = None
    if (
        getattr(agent.model_client, "supports_native_tool_calling", False)
        and hasattr(agent.model_client, "stream_messages")
    ):
        messages = build_native_messages(agent, user_message, prompt=prompt)
    return stream_model(
        agent.model_client,
        prompt,
        max_new_tokens,
        tools=tools,
        messages=messages,
        **kwargs,
    )


def handle_prompt_checkpoints(engine, task_state, user_message, prompt_metadata):
    """Persist recovery checkpoints implied by the freshly built prompt."""
    agent = engine.runtime

    def create(trigger):
        checkpoint = agent.create_checkpoint(task_state, user_message, trigger=trigger)
        agent.run_store.write_task_state(task_state)
        agent.emit_trace(
            task_state,
            "checkpoint_created",
            {"checkpoint_id": checkpoint["checkpoint_id"], "trigger": trigger},
        )

    status = prompt_metadata.get("resume_status")
    if status == "partial-stale":
        create("freshness_mismatch")
    elif status == "workspace-mismatch":
        agent.emit_trace(
            task_state,
            "runtime_identity_mismatch",
            {"fields": list(prompt_metadata.get("runtime_identity_mismatch_fields", []))},
        )
        create("workspace_mismatch")
    pressure = prompt_metadata.get("pressure", {}).get("tier", "tier0_observe")
    if prompt_metadata.get("budget_reductions") or pressure != "tier0_observe":
        create("context_reduction")


async def execute_tool_payload(engine, task_state, user_message, payload):
    agent = engine.runtime
    tool_started_at, event = _start_tool_payload(agent, task_state, payload)
    yield event

    tool_result = await agent.run_tool(event["name"], event["args"])
    tool_metadata = dict(agent._last_tool_result_metadata or {})
    tool_duration_ms = int((time.monotonic() - tool_started_at) * 1000)
    for finished in _finish_tool_payload(
        engine, task_state, user_message, payload,
        tool_result, tool_metadata, tool_duration_ms,
    ):
        yield finished


async def execute_parallel_tool_payloads(
    engine, task_state, user_message, payloads, attempted
):
    """Run raw safe reads concurrently and commit outcomes in source order."""
    agent = engine.runtime
    for payload in payloads:
        _tool_started_at, event = _start_tool_payload(
            agent, task_state, payload, record_state=False, emit_started=False
        )
        yield event

    raw_calls = await execute_parallel_safe_tools(agent, payloads)
    attempted[0] = sum(not item.skipped_reason for item in raw_calls)
    for payload, raw_call in zip(payloads, raw_calls):
        if raw_call.skipped_reason:
            for event in record_skipped_tool_call(
                engine, task_state, payload, reason=raw_call.skipped_reason
            ):
                yield event
            continue
        task_state.record_tool(payload.get("name", ""))
        finalize_started_at = time.monotonic()
        outcome = finalize_tool_call(
            agent,
            raw_call.prepared,
            raw_call.before_snapshot,
            raw_call.raw,
        )
        finalize_ms = int((time.monotonic() - finalize_started_at) * 1000)
        # Keep duration per call rather than charging later results for earlier
        # results' serialized history/checkpoint work.
        tool_duration_ms = raw_call.duration_ms + finalize_ms
        for finished in _finish_tool_payload(
            engine, task_state, user_message, payload,
            outcome.content, outcome.metadata, tool_duration_ms,
        ):
            yield finished


async def execute_tool_batch(
    engine, task_state, user_message, payloads, *, tool_steps, step_budget, attempted
):
    agent = engine.runtime
    remaining_steps = max(0, step_budget - tool_steps)
    if can_parallelize_tool_batch(
        agent, payloads, remaining_steps=remaining_steps
    ):
        async for event in execute_parallel_tool_payloads(
            engine, task_state, user_message, payloads, attempted
        ):
            yield event
        return

    executed_tools = 0
    for index, payload in enumerate(payloads):
        await engine._process_control_events()
        if agent.pending_steer_message:
            for skipped in payloads[index:]:
                for event in record_skipped_tool_call(
                    engine, task_state, skipped, reason="steered"
                ):
                    yield event
            break
        if tool_steps + executed_tools >= step_budget or agent.abort_requested:
            reason = (
                "step_budget_exhausted"
                if tool_steps + executed_tools >= step_budget
                else "aborted"
            )
            for event in record_skipped_tool_call(
                engine, task_state, payload, reason=reason
            ):
                yield event
            continue
        async for event in execute_tool_payload(
            engine, task_state, user_message, payload
        ):
            yield event
        executed_tools += 1
        await engine._process_control_events()
    attempted[0] = executed_tools


def _start_tool_payload(
    agent, task_state, payload, *, record_state=True, emit_started=True
):
    name = payload.get("name", "")
    args = payload.get("args", {})
    if record_state:
        task_state.record_tool(name)
    tool_started_at = time.monotonic()
    if emit_started:
        agent.session_event_bus.emit(
            "tool_started",
            {"run_id": task_state.run_id, "tool_name": name, "args": args},
        )
    return tool_started_at, {
        "type": "tool_call",
        "run_id": task_state.run_id,
        "name": name,
        "args": args,
    }


def _finish_tool_payload(
    engine, task_state, user_message, payload,
    tool_result, tool_metadata, tool_duration_ms,
):
    agent = engine.runtime
    name = payload.get("name", "")
    args = payload.get("args", {})
    agent.session_event_bus.emit(
        "tool_finished",
        {
            "run_id": task_state.run_id,
            "tool_name": name,
            "status": tool_metadata.get("tool_status", ""),
            "tool_error_code": tool_metadata.get("tool_error_code", ""),
            "workspace_changed": bool(tool_metadata.get("workspace_changed", False)),
            "affected_paths": list(tool_metadata.get("affected_paths", [])),
            "duration_ms": tool_duration_ms,
        },
    )
    history_item = {
        "role": "tool",
        "name": name,
        "args": args,
        "content": tool_result,
        "created_at": now(),
        "tool_status": str(tool_metadata.get("tool_status", "")),
        "tool_error_code": str(tool_metadata.get("tool_error_code", "")),
        "workspace_changed": bool(tool_metadata.get("workspace_changed", False)),
        "affected_paths": list(tool_metadata.get("affected_paths", []) or []),
    }
    # Native providers require the assistant call and its tool result to share
    # the provider-issued ID.  Text-protocol calls simply omit this field and
    # keep their legacy history shape.
    if payload.get("id"):
        history_item["tool_call_id"] = str(payload["id"])
    if tool_metadata.get("full_output_artifact"):
        history_item.update(
            {
                "artifact_ref": tool_metadata["full_output_artifact"],
                "original_chars": int(tool_metadata.get("original_chars", 0) or 0),
                "content_sha256": str(tool_metadata.get("content_sha256", "")),
            }
        )
    if tool_metadata.get("media_refs"):
        history_item["media_refs"] = list(tool_metadata.get("media_refs", []) or [])
    agent.record(history_item)
    agent.run_store.write_task_state(task_state)
    agent.emit_trace(
        task_state,
        "tool_executed",
        {
            "name": name,
            "args": args,
            "result": clip(tool_result, 500),
            "duration_ms": tool_duration_ms,
            **tool_metadata,
        },
    )
    checkpoint = agent.create_checkpoint(
        task_state, user_message, trigger="tool_executed"
    )
    agent.run_store.write_task_state(task_state)
    agent.emit_trace(
        task_state,
        "checkpoint_created",
        {"checkpoint_id": checkpoint["checkpoint_id"], "trigger": "tool_executed"},
    )
    yield {
        "type": "tool_result",
        "run_id": task_state.run_id,
        "name": name,
        "content": tool_result,
        "metadata": tool_metadata,
    }


SKIPPED_TOOL_MESSAGE = "error: skipped, this call did not run (per-turn tool budget or abort)"


def record_skipped_tool_call(engine, task_state, payload, *, reason):
    """Record a result for a call that was never executed.

    A budget or abort check happens per call, so a batch of tool calls can be cut
    short in the middle. Every tool_use block must be followed by a matching
    tool_result, or providers reject the next request outright ("tool_use ids were
    found without tool_result blocks immediately after"), which leaves the whole
    session unusable rather than failing just the current turn. So the remaining
    calls still get a recorded result: it keeps the history pairable and tells the
    model why the results it asked for are missing.
    """
    agent = engine.runtime
    name = str(payload.get("name", ""))
    args = payload.get("args", {}) or {}
    metadata = {
        "tool_status": "rejected",
        "tool_error_code": reason,
        "workspace_changed": False,
        "affected_paths": [],
        "read_only": False,
        "risk_level": "low",
        "security_event_type": "",
        "diff_summary": [],
    }
    history_item = {
        "role": "tool",
        "name": name,
        "args": args,
        "content": SKIPPED_TOOL_MESSAGE,
        "created_at": now(),
        **metadata,
    }
    if payload.get("id"):
        history_item["tool_call_id"] = str(payload["id"])
    agent.record(history_item)
    agent.emit_trace(
        task_state, "tool_skipped", {"name": name, "args": args, "reason": reason}
    )
    yield {
        "type": "tool_result",
        "run_id": task_state.run_id,
        "name": name,
        "content": SKIPPED_TOOL_MESSAGE,
        "metadata": metadata,
    }


def should_retry_model_error(exc, provider_retries):
    if not isinstance(exc, ProviderError):
        return False
    code = str(getattr(exc, "code", "") or "")
    if code not in {"empty_response"}:
        return False
    return provider_retries.get(code, 0) < 1


_STEP_LIMIT_SUMMARY_NOTICE = (
    "The read-only investigation budget for this turn is exhausted. Do not request or "
    "call tools. Answer the original user request below now, in the user's language, "
    "using the evidence already in the conversation. Keep the complete answer under "
    "700 Chinese characters. Use five concise bullets covering the architecture, "
    "creation/scheduling, runtime/state, result return, and continue/stop/failure; "
    "include key files/functions. Do not repeat exploration history or narrate progress. "
    "Mention only material uncertainty instead of guessing.\n\n"
    "Original user request:\n"
)
STEP_LIMIT_SUMMARY_TIMEOUT_SECONDS = 6


async def request_step_limit_summary(engine, task_state, user_message):
    """Ask the model to finish from collected evidence without further tools.

    Returns final text, or None if the model requests another tool or fails.
    Side effects: emits a trace event but does NOT mutate session history —
    the caller decides whether to record the resulting final.
    """
    agent = engine.runtime
    started_at = time.monotonic()
    native_mode = bool(
        getattr(agent.model_client, "supports_native_tool_calling", False)
        and hasattr(agent.model_client, "stream_messages")
    )
    try:
        protocol_notice = (
            "For the legacy text protocol, wrap the answer in <final>...</final>.\n\n"
            if not native_mode
            else "For native tool calling, return ordinary answer text; do not use XML tags.\n\n"
        )
        summary_request = (
            _STEP_LIMIT_SUMMARY_NOTICE + protocol_notice + str(user_message)
        )
        rendered_sections = dict(
            getattr(
                getattr(agent, "context_manager", None), "last_rendered_sections", {}
            )
            or {}
        )
        if not rendered_sections:
            rendered_sections = {"prefix": str(getattr(agent, "prefix", ""))}
        prompt = "\n\n".join(
            str(rendered_sections.get(section, "") or "").strip()
            for section in (
                "prefix",
                "memory",
                "skills",
                "relevant_memory",
                "repo_map",
                "history",
            )
            if str(rendered_sections.get(section, "") or "").strip()
        )
        prompt = f"{prompt}\n\n{summary_request}".strip()
        messages = None
        if native_mode:
            messages = build_native_messages(
                agent,
                summary_request,
                prompt=prompt,
                max_history_items=40
                if getattr(agent, "fast_read_only_qa", False)
                else 16,
            )
        result = await asyncio.wait_for(
            complete_model_async(
                agent.model_client,
                prompt,
                min(int(agent.max_new_tokens), 700),
                tools=None if native_mode else [],
                messages=messages,
            ),
            timeout=STEP_LIMIT_SUMMARY_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # Summary generation is best-effort; preserve the task state.  # noqa: BLE001
        agent.emit_trace(
            task_state,
            "step_limit_summary_failed",
            {"error": clip(str(exc), 200)},
        )
        return None
    raw = (result.text or "").strip() if result else ""
    if native_mode and result and result.tool_calls:
        kind, payload = "unexpected_native_tool_calls", None
    elif native_mode and raw:
        kind, payload = "native_final", raw
    else:
        kind, payload = agent.parse(raw)
    duration_ms = int((time.monotonic() - started_at) * 1000)
    completion_metadata = dict(getattr(result, "metadata", {}) or {}) if result else {}
    agent.emit_trace(
        task_state,
        "step_limit_summary",
        {
            "kind": kind,
            "duration_ms": duration_ms,
            "produced": bool(kind in {"final", "native_final"} and payload),
            "output_tokens": completion_metadata.get("output_tokens"),
            "stop_reason": str(getattr(result, "stop_reason", "") or "")
            if result
            else "",
        },
    )
    if kind in {"final", "native_final"} and payload:
        return str(payload).strip()
    return None
