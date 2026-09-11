"""Turn-level runtime engine.

The runtime owns state and persistence. Engine owns the control loop that turns
one user request into model calls, tool executions, and user-visible events.
"""

import time

from ..features import memory as memorylib
from .completion_governance import (
    final_readiness_action,
    finish_limited_run,
    finish_stopped_run,
    finish_successful_run,
)
from .context_replacements import commit_proposed_replacements
from .model_errors import finish_model_error
from .engine_helpers import (
    execute_tool_payload,
    complete_runtime_model,
    handle_prompt_checkpoints,
    request_step_limit_summary,
    should_retry_model_error,
)
from .task_state import STOP_REASON_FINAL_GATE_BLOCKED, TaskState
from .turn_transitions import (
    CONTINUE_FINAL_READINESS_NOTICE,
    CONTINUE_PARSE_RETRY,
    CONTINUE_PLAN_NOTICE,
    CONTINUE_PROVIDER_RETRY,
    CONTINUE_TOOL_BATCH_EXECUTED,
    emit_continue_transition,
)
from .workspace import clip, now

CHECKPOINT_NONE_STATUS = "no-checkpoint"


class Engine:
    def __init__(self, runtime):
        self.runtime = runtime

    def ask(self, user_message):
        final_answer = ""
        for event in self.run_turn(user_message):
            if event["type"] in {"final", "stop"}:
                final_answer = event["content"]
        return final_answer

    def drain_worker_notifications(self):
        agent = self.runtime
        notifications = agent.worker_manager.drain_notifications()
        for notification in notifications:
            agent.record({"role": "user", "content": notification, "created_at": now()})
            agent.session_event_bus.emit(
                "worker_notification_drained",
                {
                    "run_id": getattr(agent, "current_run_id", ""),
                    "content": clip(notification, 500),
                },
            )
        return notifications

    def _drain_worker_notification_events(self):
        for notification in self.drain_worker_notifications():
            yield {
                "type": "worker_notification",
                "run_id": getattr(self.runtime, "current_run_id", ""),
                "content": notification,
            }

    def run_turn(self, user_message):
        agent = self.runtime
        run_started_at = time.monotonic()
        task_state = TaskState.create(
            run_id=agent.new_run_id(),
            task_id=agent.new_task_id(),
            user_request=user_message,
        )
        task_state.resume_status = agent.resume_state.get(
            "status", CHECKPOINT_NONE_STATUS
        )
        agent.current_task_state = task_state
        agent.current_turn_id = task_state.task_id
        agent.current_run_id = task_state.run_id
        agent.current_run_dir = agent.run_store.start_run(task_state)
        agent.session_event_bus.emit(
            "turn_started",
            {
                "run_id": task_state.run_id,
                "task_id": task_state.task_id,
                "runtime_mode": agent.runtime_mode,
            },
        )
        yield {
            "type": "turn_started",
            "run_id": task_state.run_id,
            "task_id": task_state.task_id,
        }

        agent.memory.set_task_summary(user_message)
        agent.record({"role": "user", "content": user_message, "created_at": now()})
        agent.session_event_bus.emit(
            "user_message",
            {"run_id": task_state.run_id, "content": clip(user_message, 300)},
        )
        agent.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(user_message, 300),
            },
        )

        tool_steps = 0
        attempts = 0
        provider_retries = {}
        # 不放大 attempts，避免出现"看不见的隐形重试"——失败必须被用户察觉。
        max_attempts = agent.max_steps + 2

        while tool_steps < agent.max_steps and attempts < max_attempts:
            if agent.abort_requested:
                yield from finish_stopped_run(
                    self,
                    task_state,
                    user_message,
                    "Stopped after abort request.",
                    "aborted",
                    run_started_at,
                )
                return
            yield from self._drain_worker_notification_events()
            attempts += 1
            task_state.record_attempt()
            agent.run_store.write_task_state(task_state)
            prompt_started_at = time.monotonic()
            prompt, prompt_metadata = agent._build_prompt_and_metadata(user_message)
            if commit_proposed_replacements(agent.session, prompt_metadata):
                agent.session_path = agent.session_store.save(agent.session)
            agent.emit_trace(
                task_state,
                "prompt_built",
                {
                    "prompt_metadata": prompt_metadata,
                    "duration_ms": int((time.monotonic() - prompt_started_at) * 1000),
                },
            )
            structured_memory = getattr(getattr(agent, "memory", None), "last_retrieval", None)
            if structured_memory is not None:
                agent.emit_trace(
                    task_state,
                    "memory.retrieval",
                    {
                        "query_hash": structured_memory.get("query_hash", ""),
                        "selected": list(structured_memory.get("selected", [])),
                        "rejected": list(structured_memory.get("rejected", [])),
                        "workspace_fingerprint": memorylib.workspace_fingerprint(agent.root),
                    },
                )
            for file_read in memorylib.memory_file_read_payloads(agent.memory_dir, agent.root, reason="retrieval"):
                agent.emit_trace(task_state, "memory.file_read", file_read)
            handle_prompt_checkpoints(self, task_state, user_message, prompt_metadata)
            agent.emit_trace(
                task_state,
                "model_requested",
                {
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                },
            )
            agent.session_event_bus.emit(
                "model_requested",
                {
                    "run_id": task_state.run_id,
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                },
            )
            yield {
                "type": "model_requested",
                "run_id": task_state.run_id,
                "attempts": task_state.attempts,
                "tool_steps": task_state.tool_steps,
            }

            prompt_cache_key = None
            prompt_cache_retention = None
            if getattr(agent.model_client, "supports_prompt_cache", False):
                prompt_cache_key = prompt_metadata.get("prompt_cache_key")
                prompt_cache_retention = "in_memory"

            model_started_at = time.monotonic()
            native_tools = (
                agent.native_tool_definitions()
                if hasattr(agent, "native_tool_definitions")
                else []
            )
            try:
                result = complete_runtime_model(
                    agent,
                    prompt,
                    user_message,
                    agent.max_new_tokens,
                    tools=native_tools,
                    prompt_cache_key=prompt_cache_key,
                    prompt_cache_retention=prompt_cache_retention,
                )
            except Exception as exc:
                if agent.abort_requested:
                    yield from finish_stopped_run(
                        self,
                        task_state,
                        user_message,
                        "Stopped after abort request.",
                        "aborted",
                        run_started_at,
                    )
                    return
                if should_retry_model_error(exc, provider_retries):
                    code = getattr(exc, "code", type(exc).__name__)
                    provider_retries[code] = provider_retries.get(code, 0) + 1
                    agent.session_event_bus.emit(
                        "model_retry_scheduled",
                        {
                            "run_id": task_state.run_id,
                            "code": code,
                            "attempts": task_state.attempts,
                            "retry_count": provider_retries[code],
                        },
                    )
                    agent.emit_trace(
                        task_state,
                        "model_retry_scheduled",
                        {
                            "code": code,
                            "duration_ms": int(
                                (time.monotonic() - model_started_at) * 1000
                            ),
                            "retry_count": provider_retries[code],
                        },
                    )
                    emit_continue_transition(agent, task_state, CONTINUE_PROVIDER_RETRY)
                    continue
                yield from finish_model_error(
                    self,
                    task_state,
                    user_message,
                    prompt_metadata,
                    exc,
                    int((time.monotonic() - model_started_at) * 1000),
                    int((time.monotonic() - run_started_at) * 1000),
                )
                return
            if agent.abort_requested:
                yield from finish_stopped_run(
                    self,
                    task_state,
                    user_message,
                    "Stopped after abort request.",
                    "aborted",
                    run_started_at,
                )
                return
            raw = result.text
            completion_metadata = dict(
                result.metadata
                or getattr(agent.model_client, "last_completion_metadata", {})
                or {}
            )
            if completion_metadata:
                prompt_metadata.update(completion_metadata)
            agent.last_completion_metadata = completion_metadata
            agent.last_prompt_metadata = prompt_metadata
            native_tool_calls = tuple(getattr(result, "tool_calls", ()) or ())
            if native_tool_calls:
                kind = "tool" if len(native_tool_calls) == 1 else "tools"
                payload = (
                    native_tool_calls[0]
                    if len(native_tool_calls) == 1
                    else list(native_tool_calls)
                )
                # Preserve the structured assistant turn so resume/compact can
                # explain which native calls produced the following results.
                agent.record(
                    {
                        "role": "assistant",
                        "content": raw,
                        "tool_calls": list(native_tool_calls),
                        "created_at": now(),
                    }
                )
            elif completion_metadata.get("native_tool_calling") and raw.strip():
                # Native providers return ordinary assistant text for a final
                # answer; the legacy parser must not turn that into a retry.
                payload = (
                    agent.extract(raw, "final")
                    if "<final>" in raw
                    else raw.strip()
                )
                kind = "final"
            else:
                kind, payload = agent.parse(raw)
            duration_ms = int((time.monotonic() - model_started_at) * 1000)
            agent.emit_trace(
                task_state,
                "model_parsed",
                {
                    "kind": kind,
                    "completion_metadata": completion_metadata,
                    "duration_ms": duration_ms,
                },
            )
            agent.session_event_bus.emit(
                "model_parsed",
                {"run_id": task_state.run_id, "kind": kind, "duration_ms": duration_ms},
            )
            yield {
                "type": "model_parsed",
                "run_id": task_state.run_id,
                "kind": kind,
                "duration_ms": duration_ms,
            }

            if kind in {"tool", "tools"}:
                tools = [payload] if kind == "tool" else list(payload)
                executed_tools = 0
                for tool_payload in tools:
                    if tool_steps >= agent.max_steps:
                        break
                    yield from execute_tool_payload(
                        self, task_state, user_message, tool_payload
                    )
                    tool_steps += 1
                    executed_tools += 1
                    if agent.abort_requested:
                        break
                if agent.abort_requested:
                    yield from finish_stopped_run(
                        self,
                        task_state,
                        user_message,
                        "Stopped after abort request.",
                        "aborted",
                        run_started_at,
                    )
                    return
                emit_continue_transition(
                    agent, task_state, CONTINUE_TOOL_BATCH_EXECUTED,
                    tool_call_count=executed_tools,
                    tool_requested_count=len(tools),
                    tool_executed_count=executed_tools,
                )
                continue

            if kind == "retry":
                agent.record(
                    {"role": "assistant", "content": payload, "created_at": now()}
                )
                agent.session_event_bus.emit(
                    "assistant_message",
                    {
                        "run_id": task_state.run_id,
                        "kind": "retry",
                        "content": clip(payload, 500),
                    },
                )
                agent.run_store.write_task_state(task_state)
                yield {"type": "retry", "run_id": task_state.run_id, "content": payload}
                emit_continue_transition(agent, task_state, CONTINUE_PARSE_RETRY)
                continue

            final = (payload or raw).strip()
            yield from self._drain_worker_notification_events()
            if agent.runtime_mode == "plan" and not agent.plan_mode.can_finish():
                notice = agent.plan_mode.final_notice()
                agent.record(
                    {"role": "assistant", "content": notice, "created_at": now()}
                )
                agent.session_event_bus.emit(
                    "assistant_message",
                    {
                        "run_id": task_state.run_id,
                        "kind": "runtime_notice",
                        "content": notice,
                    },
                )
                agent.run_store.write_task_state(task_state)
                yield {
                    "type": "runtime_notice",
                    "run_id": task_state.run_id,
                    "content": notice,
                }
                emit_continue_transition(agent, task_state, CONTINUE_PLAN_NOTICE)
                continue

            readiness_action, notice = final_readiness_action(self, task_state, final)
            if readiness_action == "runtime_notice":
                yield {
                    "type": "runtime_notice",
                    "run_id": task_state.run_id,
                    "content": notice,
                }
                emit_continue_transition(
                    agent,
                    task_state,
                    CONTINUE_FINAL_READINESS_NOTICE,
                )
                continue
            if readiness_action == "block":
                yield from finish_stopped_run(
                    self,
                    task_state,
                    user_message,
                    notice,
                    STOP_REASON_FINAL_GATE_BLOCKED,
                    run_started_at,
                )
                return

            yield from finish_successful_run(
                self, task_state, user_message, final, run_started_at
            )
            return

        if attempts >= max_attempts and tool_steps < agent.max_steps:
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
            task_state.stop_retry_limit(final)
        else:
            summary = None
            if tool_steps > 0:
                summary = request_step_limit_summary(self, task_state, user_message)
            if summary:
                final = (
                    summary
                    + "\n\n— 已达本轮 step 预算上限（max_steps）。以上是当前进展总结。"
                    "继续工作：在 REPL 输入 /resume 续接本会话，或直接说"
                    "「继续」让我接着干。"
                )
            else:
                final = "Stopped after reaching the step limit without a final answer."
            task_state.stop_step_limit(final)
        yield from finish_limited_run(
            self, task_state, user_message, final, run_started_at
        )
