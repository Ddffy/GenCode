"""Turn-level runtime engine.

The runtime owns state and persistence. Engine owns the control loop that turns
one user request into model calls, tool executions, and user-visible events.
"""

import asyncio
import contextvars
import time
from contextlib import AsyncExitStack

from ..features import memory as memorylib
from .completion_governance import (
    final_readiness_action,
    finish_limited_run,
    finish_stopped_run,
    finish_successful_run,
)
from .context_replacements import commit_proposed_replacements
from .engine_failures import finish_unexpected_run_error
from .engine_helpers import (
    GoalTokenCallBudget,
    _set_fast_read_only_qa,
    complete_runtime_model,
    execute_tool_batch,
    handle_prompt_checkpoints,
    request_step_limit_summary,
    should_retry_model_error,
)
from .engine_read_only_qa import (
    READ_ONLY_QA_ATTEMPT_BUDGET,
    READ_ONLY_QA_FINAL_NOTICE,
    _is_read_only_qa_request,
    _read_only_qa_final_only,
    _read_only_qa_prompt,
    _turn_tool_step_budget,
)
from .engine_stream import TURN_STREAM_EVENTS, VisibleTextStream
from .model_errors import finish_model_error
from .task_state import (
    STOP_REASON_FINAL_GATE_BLOCKED,
    TaskState,
)
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
        self._background_turn_tasks = set()
        self._run_tasks = {}
        self.runtime.session_event_bus.set_degradation_handler(
            self._schedule_event_log_degradation
        )
        self._run_id_queue = contextvars.ContextVar(
            f"gencode_run_id_queue_{id(self)}", default=None
        )
        self._announced_run_ids = contextvars.ContextVar(
            f"gencode_announced_runs_{id(self)}", default=None
        )

    def ask(self, user_message):
        return self.runtime.ask(user_message)

    async def ask_async(self, user_message):
        final_answer = ""
        async for event in self.run_turn(user_message):
            if event["type"] in {"final", "stop"}:
                final_answer = event["content"]
        return final_answer

    def _spawn_turn(self, user_message):
        run_ids = asyncio.Queue()
        task = asyncio.create_task(self._run_turn_producer(user_message, run_ids))
        self._background_turn_tasks.add(task)
        task.add_done_callback(self._observe_background_turn)
        return run_ids, task

    def _observe_background_turn(self, task):
        self._background_turn_tasks.discard(task)
        for run_id, producer in tuple(self._run_tasks.items()):
            if producer is task:
                self._run_tasks.pop(run_id, None)
        if not task.cancelled():
            task.exception()

    def _schedule_event_log_degradation(self, run_id, origin_loop=None):
        loop = origin_loop or self.runtime._bound_async_loop
        if loop is None or loop.is_closed() or not loop.is_running():
            return

        def schedule():
            task = loop.create_task(self._record_event_log_degradation(run_id))
            self._background_turn_tasks.add(task)
            task.add_done_callback(self._observe_background_turn)

        try:
            loop.call_soon_threadsafe(schedule)
        except RuntimeError:
            return

    async def start_turn(self, user_message):
        """Start a detached turn and return its first Run ID for later replay."""
        run_ids, task = self._spawn_turn(user_message)
        run_id = await run_ids.get()
        if run_id is None:
            await task
            return ""
        return str(run_id)

    async def subscribe_run(self, run_id, after_seq=0):
        """Subscribe or resume a Run event stream from its last Run sequence."""
        async for record in self.runtime.session_event_bus.subscribe(run_id, after_seq):
            event = dict(record)
            event["type"] = event.pop("event", "event")
            yield event

    async def subscribe_turn(self, run_id, after_seq=0):
        """Subscribe to the user-facing subset of a Run, resumable by cursor."""
        async for event in self.subscribe_run(run_id, after_seq):
            if event.get("type") in TURN_STREAM_EVENTS:
                if event.get("type") == "turn_finished":
                    await self.wait_for_run(run_id)
                yield event

    async def wait_for_run(self, run_id):
        """Wait for producer cleanup after a terminal event has been persisted."""
        task = self._run_tasks.get(str(run_id))
        if task is not None and task is not asyncio.current_task():
            await asyncio.gather(task, return_exceptions=True)

    async def run_turn(self, user_message):
        """Start one detached producer and stream its persisted events to this consumer.

        Closing this generator only detaches the consumer; it does not cancel the
        producer. Call ``subscribe_run(run_id, after_seq)`` to resume a stream.
        """
        run_ids, task = self._spawn_turn(user_message)
        while True:
            run_id = await run_ids.get()
            if run_id is None:
                break
            async for event in self.subscribe_turn(run_id):
                yield event
        await task

    async def _run_turn_producer(self, user_message, run_ids):
        queue_token = self._run_id_queue.set(run_ids)
        announced_token = self._announced_run_ids.set(set())
        try:
            try:
                await self._drive_turn(user_message)
            except Exception as exc:  # noqa: BLE001 - detached Runs must publish a terminal result
                await finish_unexpected_run_error(self, user_message, exc)
        finally:
            self._announced_run_ids.reset(announced_token)
            self._run_id_queue.reset(queue_token)
            await run_ids.put(None)

    async def _drive_turn(self, user_message):
        agent = self.runtime
        agent._bind_async_loop(asyncio.get_running_loop())
        inherited_session_lease = bool(agent.inherited_session_lease)
        inherited_lease = bool(agent.inherited_workspace_lease)
        async with AsyncExitStack() as lock_stack:
            if not inherited_session_lease:
                session_lease = await agent.lock_manager.acquire_session(agent.session["id"])
                await lock_stack.enter_async_context(session_lease)
                agent.session_lock_lease = session_lease
            workspace_lease = None
            if not inherited_lease:
                mode = "read" if agent.read_only else "write"
                workspace_lease = await agent.lock_manager.acquire_workspace(
                    agent.root, mode
                )
                await lock_stack.enter_async_context(workspace_lease)
                agent.workspace_lock_lease = workspace_lease
            try:
                current_message = user_message
                while True:
                    async for _ in self._run_turn_with_timeout(current_message):
                        pass
                    if not agent.queued_turns:
                        break
                    current_message = agent.queued_turns.pop(0)
            finally:
                if not inherited_session_lease:
                    agent.session_lock_lease = None
                if not inherited_lease:
                    agent.workspace_lock_lease = None

    async def _run_turn_with_timeout(self, user_message):
        agent = self.runtime
        agent.turn_control_loop = asyncio.get_running_loop()
        agent.turn_control_queue = asyncio.Queue()
        agent.turn_cancel_event = asyncio.Event()
        agent.abort_requested = False
        agent.cancel_requested = False
        self._turn_started_at = time.monotonic()
        self._steered_message = ""
        try:
            async with asyncio.timeout(agent.max_turn_seconds):
                async for event in self._publish_run_events(
                    self._run_turn_events(user_message)
                ):
                    yield event
        except asyncio.TimeoutError:
            task_state = agent.current_task_state
            if task_state is not None and task_state.status == "running":
                async for event in self._publish_run_events(
                    finish_stopped_run(
                        self,
                        task_state,
                        user_message,
                        "Stopped after reaching the turn timeout.",
                        "timeout",
                        self._turn_started_at,
                    )
                ):
                    yield event
        except asyncio.CancelledError:
            task_state = agent.current_task_state
            if task_state is not None and task_state.status == "running":
                async for event in self._publish_run_events(finish_stopped_run(
                    self,
                    task_state,
                    task_state.user_request,
                    "Stopped after caller cancellation.",
                    "aborted",
                    self._turn_started_at,
                )):
                    yield event
            raise
        finally:
            await self._process_control_events()
            agent.turn_control_queue = None
            agent.turn_control_loop = None
            agent.turn_cancel_event = None
            _set_fast_read_only_qa(agent, False)

    async def _consume_model_stream(
        self,
        agent,
        task_state,
        prompt,
        user_message,
        native_tools,
        max_new_tokens,
        prompt_cache_key,
        prompt_cache_retention,
    ):
        stream = complete_runtime_model(
            agent,
            prompt,
            user_message,
            max_new_tokens,
            tools=native_tools,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
        )
        iterator = stream.__aiter__()
        next_event = asyncio.create_task(iterator.__anext__())
        control_task = None
        visible = VisibleTextStream()
        pending_text = ""
        last_text_flush = time.monotonic()
        try:
            while True:
                control_queue = agent.turn_control_queue
                if control_task is None and control_queue is not None:
                    control_task = asyncio.create_task(control_queue.get())
                waiting = {next_event}
                if control_task is not None:
                    waiting.add(control_task)
                wait_timeout = agent.model_idle_timeout
                if pending_text:
                    wait_timeout = min(
                        wait_timeout,
                        max(0.0, 0.05 - (time.monotonic() - last_text_flush)),
                    )
                done, _ = await asyncio.wait(
                    waiting,
                    timeout=wait_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    if pending_text:
                        yield {
                            "type": "text_delta",
                            "run_id": task_state.run_id,
                            "content": pending_text,
                        }
                        pending_text = ""
                        last_text_flush = time.monotonic()
                        continue
                    next_event.cancel()
                    await asyncio.gather(next_event, return_exceptions=True)
                    await iterator.aclose()
                    raise asyncio.TimeoutError("model stream idle timeout")
                if control_task is not None and control_task in done:
                    control = control_task.result()
                    control_task = None
                    kind = control.get("type")
                    if kind == "queue":
                        agent.queued_turns.append(str(control.get("content", "")))
                    elif kind == "steer":
                        agent.pending_steer_message = str(control.get("content", ""))
                        await self._abort_model_request(agent)
                        next_event.cancel()
                        await asyncio.gather(next_event, return_exceptions=True)
                        await iterator.aclose()
                        yield {"type": "_model_steered", "run_id": task_state.run_id}
                        return
                    elif kind == "cancel":
                        agent.abort_requested = True
                        agent.cancel_requested = True
                        if agent.turn_cancel_event is not None:
                            agent.turn_cancel_event.set()
                        await self._abort_model_request(agent)
                        next_event.cancel()
                        await asyncio.gather(next_event, return_exceptions=True)
                        await iterator.aclose()
                        yield {"type": "_model_cancelled", "run_id": task_state.run_id}
                        return
                    if next_event not in done:
                        continue
                if next_event not in done:
                    continue
                try:
                    item = next_event.result()
                except StopAsyncIteration:
                    break
                if item.type == "text_delta":
                    pending_text += "".join(visible.feed(item.text))
                    if len(pending_text) >= 256:
                        yield {
                            "type": "text_delta",
                            "run_id": task_state.run_id,
                            "content": pending_text,
                        }
                        pending_text = ""
                        last_text_flush = time.monotonic()
                elif item.type == "completed":
                    pending_text += "".join(visible.finish())
                    if pending_text:
                        yield {
                            "type": "text_delta",
                            "run_id": task_state.run_id,
                            "content": pending_text,
                        }
                        pending_text = ""
                    yield {
                        "type": "_model_completed",
                        "run_id": task_state.run_id,
                        "result": item.result,
                    }
                    return
                next_event = asyncio.create_task(iterator.__anext__())
        finally:
            if not next_event.done():
                next_event.cancel()
                await asyncio.gather(next_event, return_exceptions=True)
            if control_task is not None and not control_task.done():
                control_task.cancel()
                await asyncio.gather(control_task, return_exceptions=True)
            await iterator.aclose()

    async def _abort_model_request(self, agent):
        abort = getattr(agent.model_client, "abort", None)
        if callable(abort):
            result = abort()
            if hasattr(result, "__await__"):
                await result

    async def _publish_run_events(self, events):
        if hasattr(events, "__aiter__"):
            async for event in events:
                async for published in self._publish_event(event):
                    yield published
        else:
            for event in events:
                async for published in self._publish_event(event):
                    yield published

    async def _publish_event(self, event):
        event = dict(event)
        run_id = str(event.get("run_id", "") or self.runtime.current_run_id)
        event_type = str(event.get("type", "event"))
        announced_run_ids = self._announced_run_ids.get()
        if run_id and announced_run_ids is not None and run_id not in announced_run_ids:
            producer = asyncio.current_task()
            if producer in self._background_turn_tasks:
                self._run_tasks[run_id] = producer
            run_id_queue = self._run_id_queue.get()
            if run_id_queue is not None:
                run_id_queue.put_nowait(run_id)
            announced_run_ids.add(run_id)
        if event_type == "turn_started":
            self.runtime.session_event_bus.set_active_run(run_id)
        record = await self.runtime.session_event_bus.publish(
            event_type,
            {**event, "run_id": run_id, "source": event.get("source", "runtime")},
        )
        event["run_id"] = run_id
        event["run_seq"] = int(record.get("run_seq", 0))
        event["source"] = record.get("source", "runtime")
        if record.get("event_log_degraded") or self.runtime.session_event_bus.degraded(run_id):
            await self._record_event_log_degradation(run_id)
        yield event

    def _publish_sync_events(self, events):
        yield from events

    async def _record_event_log_degradation(self, run_id):
        agent = self.runtime
        task_state = agent.current_task_state
        if task_state is None or task_state.run_id != run_id:
            return
        gap = agent.session_event_bus.degraded(run_id)
        already_recorded = (
            task_state.event_log_degraded and task_state.event_log_gap == gap
        )
        task_state.event_log_degraded = True
        task_state.event_log_gap = gap
        agent.event_log_degraded[run_id] = dict(gap)
        for attempt in range(3):
            try:
                if not already_recorded:
                    await asyncio.to_thread(
                        agent.run_store.write_task_state, task_state
                    )
                if task_state.status != "running":
                    report = agent.redact_artifact(agent.build_report(task_state))
                    await asyncio.to_thread(
                        agent.run_store.write_report, task_state, report
                    )
                return
            except OSError as exc:
                if attempt == 2:
                    agent.emit_trace(
                        task_state,
                        "event_log_degradation_persist_failed",
                        {"error_type": type(exc).__name__},
                    )
                    return
                await asyncio.sleep(0.01 * (attempt + 1))

    async def _process_control_events(self):
        agent = self.runtime
        queue = agent.turn_control_queue
        if queue is None:
            return
        while True:
            try:
                control = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            kind = control.get("type")
            if kind == "queue":
                agent.queued_turns.append(str(control.get("content", "")))
            elif kind == "steer":
                task_state = agent.current_task_state
                if task_state is not None and task_state.status != "running":
                    agent.queued_turns.append(str(control.get("content", "")))
                else:
                    agent.pending_steer_message = str(control.get("content", ""))
            elif kind == "cancel":
                agent.abort_requested = True
                agent.cancel_requested = True
                if agent.turn_cancel_event is not None:
                    agent.turn_cancel_event.set()

    async def _run_turn_events(self, user_message):
        agent = self.runtime
        run_started_at = time.monotonic()
        task_state = TaskState.create(
            run_id=agent.new_run_id(),
            task_id=agent.new_task_id(),
            user_request=user_message,
        )
        goal_token_calls = GoalTokenCallBudget(self, agent, task_state, run_started_at)
        task_state.resume_status = agent.resume_state.get(
            "status", CHECKPOINT_NONE_STATUS
        )
        if hasattr(agent, "active_spec_ids"):
            task_state.active_spec_ids = list(agent.active_spec_ids())
        agent.current_task_state = task_state
        agent.current_turn_id = task_state.task_id
        agent.current_run_id = task_state.run_id
        agent.last_knowledge_maintenance = {
            "candidates": [], "quarantined": [], "errors": [], "auto_dream": {}
        }
        agent._retrieval_section_cache = {}
        agent.current_run_dir = agent.run_store.start_run(task_state)
        # Git 基线必须在本轮第一个工具动作前捕获；后续每个写操作都只
        # 提交本轮新增的安全路径，验收失败时才能精确回到上一个 Agent 提交。
        if hasattr(agent, "git"):
            agent.git.begin_turn(task_state.run_id)
        yield {
            "type": "turn_started",
            "run_id": task_state.run_id,
            "task_id": task_state.task_id,
            "runtime_mode": agent.runtime_mode,
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
        read_only_qa = (
            not getattr(agent, "disable_fast_read_only_qa", False)
            and _is_read_only_qa_request(user_message)
        )
        step_budget = _turn_tool_step_budget(
            user_message,
            agent.max_steps,
            allow_read_only_qa=not getattr(agent, "disable_fast_read_only_qa", False),
        )
        max_attempts = min(
            step_budget + 2,
            READ_ONLY_QA_ATTEMPT_BUDGET
            if read_only_qa
            else step_budget + 2,
        )
        task_state.evidence_summaries["tool_step_budget"] = {
            "limit": step_budget,
            "kind": "read_only_qa" if read_only_qa else "default",
        }
        _set_fast_read_only_qa(agent, read_only_qa)

        while tool_steps < step_budget and attempts < max_attempts:
            await self._process_control_events()
            if agent.pending_steer_message:
                user_message = agent.pending_steer_message
                agent.pending_steer_message = ""
                agent.memory.set_task_summary(user_message)
                agent.record({"role": "user", "content": user_message, "created_at": now()})
                agent.session_event_bus.emit(
                    "user_message",
                    {"run_id": task_state.run_id, "content": clip(user_message, 300), "source": "steer"},
                )
            if agent.abort_requested:
                for event in finish_stopped_run(
                    self,
                    task_state,
                    user_message,
                    "Stopped after cancel request." if agent.cancel_requested else "Stopped after abort request.",
                    "aborted",
                    run_started_at,
                ):
                    yield event
                return
            attempts += 1
            task_state.record_attempt()
            agent.run_store.write_task_state(task_state)
            prompt_started_at = time.monotonic()
            yield {
                "type": "context_building",
                "run_id": task_state.run_id,
                "attempts": task_state.attempts,
                "tool_steps": task_state.tool_steps,
            }
            prompt_request = (
                _read_only_qa_prompt(user_message)
                if agent.fast_read_only_qa
                else user_message
            )
            native_mode = bool(
                getattr(agent.model_client, "supports_native_tool_calling", False)
                and hasattr(agent.model_client, "stream_messages")
            )
            qa_final_only = _read_only_qa_final_only(
                read_only_qa=read_only_qa,
                tool_steps=tool_steps,
                native_mode=native_mode,
            )
            prompt, prompt_metadata = await agent._build_prompt_and_metadata_async(
                prompt_request
            )
            model_user_message = (
                f"{user_message}\n\n{READ_ONLY_QA_FINAL_NOTICE}"
                if qa_final_only
                else user_message
            )
            if qa_final_only and not native_mode:
                prompt = f"{prompt}\n\n{READ_ONLY_QA_FINAL_NOTICE}"
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
            structured_knowledge = getattr(agent, "last_knowledge_retrieval", None)
            if structured_knowledge is not None and hasattr(agent, "knowledge_store"):
                knowledge_trace = agent.knowledge_store.trace_retrieval(structured_knowledge)
                task_state.knowledge_selections = knowledge_trace
                agent.emit_trace(task_state, "knowledge.retrieval", knowledge_trace)
            handle_prompt_checkpoints(self, task_state, user_message, prompt_metadata)
            request_max_new_tokens, budget_events = await goal_token_calls.reserve(
                user_message, prompt, prompt_metadata, qa_final_only
            )
            if budget_events:
                for event in budget_events:
                    yield event
                return
            agent.emit_trace(
                task_state,
                "model_requested",
                {
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                    "qa_final_only": qa_final_only,
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
            if qa_final_only:
                native_tools = []
            try:
                result = None
                completion_metadata = {}
                steered = False
                async for stream_event in self._consume_model_stream(
                    agent,
                    task_state,
                    prompt,
                    model_user_message,
                    native_tools,
                    request_max_new_tokens,
                    prompt_cache_key,
                    prompt_cache_retention,
                ):
                    if stream_event["type"] == "_model_completed":
                        result = stream_event["result"]
                    elif stream_event["type"] == "_model_steered":
                        steered = True
                    elif stream_event["type"] == "_model_cancelled":
                        break
                    else:
                        yield stream_event
                if result is not None:
                    completion_metadata = await goal_token_calls.record(result)
                await self._process_control_events()
                steered = steered or bool(agent.pending_steer_message)
                if steered:
                    await goal_token_calls.release()
                    user_message = self._steered_message or agent.pending_steer_message
                    self._steered_message = ""
                    agent.pending_steer_message = ""
                    agent.memory.set_task_summary(user_message)
                    agent.record({"role": "user", "content": user_message, "created_at": now()})
                    agent.session_event_bus.emit(
                        "user_message",
                        {"run_id": task_state.run_id, "content": clip(user_message, 300), "source": "steer"},
                    )
                    continue
                if agent.abort_requested:
                    await goal_token_calls.release()
                    for event in finish_stopped_run(
                        self,
                        task_state,
                        user_message,
                        "Stopped after cancel request.",
                        "aborted",
                        run_started_at,
                    ):
                        yield event
                    return
                if result is None:
                    raise RuntimeError("model stream ended without a completion")
            except asyncio.CancelledError:
                await goal_token_calls.release()
                raise
            except Exception as exc:  # Normalize provider/runtime failures into a stopped run.  # noqa: BLE001
                await goal_token_calls.release()
                if agent.abort_requested:
                    for event in finish_stopped_run(
                        self,
                        task_state,
                        user_message,
                        "Stopped after cancel request." if agent.cancel_requested else "Stopped after abort request.",
                        "aborted",
                        run_started_at,
                    ):
                        yield event
                    return
                if isinstance(exc, asyncio.TimeoutError):
                    for event in finish_stopped_run(
                        self,
                        task_state,
                        user_message,
                        "Stopped after the model stream became idle.",
                        "model_idle_timeout",
                        run_started_at,
                    ):
                        yield event
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
                for event in finish_model_error(
                    self,
                    task_state,
                    user_message,
                    prompt_metadata,
                    exc,
                    int((time.monotonic() - model_started_at) * 1000),
                    int((time.monotonic() - run_started_at) * 1000),
                ):
                    yield event
                return
            if agent.abort_requested:
                for event in finish_stopped_run(
                    self,
                    task_state,
                    user_message,
                    "Stopped after cancel request." if agent.cancel_requested else "Stopped after abort request.",
                    "aborted",
                    run_started_at,
                ):
                    yield event
                return
            raw = result.text
            if not completion_metadata:
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
            elif (
                completion_metadata.get("native_tool_calling")
                or completion_metadata.get("native_message_mode")
            ) and raw.strip():
                # Native structured-message calls return ordinary assistant
                # text even when tools were deliberately withheld for the
                # final-answer turn; the legacy parser must not turn it into a retry.
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
                    "goal_token_usage": task_state.evidence_summaries.get("goal_token_usage"),
                    "duration_ms": duration_ms,
                },
            )
            yield {
                "type": "model_parsed",
                "run_id": task_state.run_id,
                "kind": kind,
                "duration_ms": duration_ms,
            }

            if kind in {"tool", "tools"}:
                budget_events = goal_token_calls.stop_before_tool(user_message)
                if budget_events:
                    for event in budget_events:
                        yield event
                    return
                tools = [payload] if kind == "tool" else list(payload)
                attempted = [0]
                async for event in execute_tool_batch(
                    self,
                    task_state,
                    user_message,
                    tools,
                    tool_steps=tool_steps,
                    step_budget=step_budget,
                    attempted=attempted,
                ):
                    yield event
                executed_tools = attempted[0]
                tool_steps += executed_tools
                if agent.abort_requested:
                    for event in finish_stopped_run(
                        self,
                        task_state,
                        user_message,
                        "Stopped after cancel request." if agent.cancel_requested else "Stopped after abort request.",
                        "aborted",
                        run_started_at,
                    ):
                        yield event
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
                for event in finish_stopped_run(
                    self,
                    task_state,
                    user_message,
                    notice,
                    STOP_REASON_FINAL_GATE_BLOCKED,
                    run_started_at,
                ):
                    yield event
                return

            await self._process_control_events()
            if agent.abort_requested:
                for event in finish_stopped_run(
                    self,
                    task_state,
                    user_message,
                    "Stopped after cancel request.",
                    "aborted",
                    run_started_at,
                ):
                    yield event
                return
            if agent.pending_steer_message:
                user_message = agent.pending_steer_message
                agent.pending_steer_message = ""
                agent.memory.set_task_summary(user_message)
                agent.record({"role": "user", "content": user_message, "created_at": now()})
                continue
            for event in finish_successful_run(
                self, task_state, user_message, final, run_started_at
            ):
                yield event
            return

        attempts_exhausted = attempts >= max_attempts and tool_steps < step_budget
        summary = None
        if (
            not goal_token_calls.enabled
            and tool_steps > 0
            and (attempts_exhausted or tool_steps >= step_budget)
        ):
            summary = await request_step_limit_summary(self, task_state, user_message)
        if summary:
            final = f"{summary}\n\nstep 预算上限已耗尽；如需继续，请使用 /resume。"
            task_state.stop_step_limit(final)
        elif attempts_exhausted:
            final = "Stopped after too many model attempts without a final answer."
            task_state.stop_retry_limit(final)
        else:
            final = "Stopped after reaching the tool-step budget without a final answer."
            task_state.stop_step_limit(final)
        for event in finish_limited_run(
            self, task_state, user_message, final, run_started_at
        ):
            yield event
