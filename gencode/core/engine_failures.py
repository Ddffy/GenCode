"""Terminal handling for unexpected failures in a detached Run producer."""

import asyncio
import time

from .completion_governance import finish_stopped_run
from .workspace import clip


async def finish_unexpected_run_error(engine, user_message, exc):
    agent = engine.runtime
    task_state = agent.current_task_state
    if (
        task_state is None
        or task_state.status != "running"
        or agent.current_run_id != task_state.run_id
    ):
        raise exc
    final = f"运行时内部错误（{type(exc).__name__}），本轮已停止。"
    agent.emit_trace(
        task_state,
        "runtime_error",
        {"error_type": type(exc).__name__, "error": clip(str(exc), 500)},
    )
    try:
        await agent.session_event_bus.publish(
            "runtime_error",
            {
                "run_id": task_state.run_id,
                "error_type": type(exc).__name__,
                "error": clip(str(exc), 500),
            },
        )
    except Exception as publish_error:  # noqa: BLE001 - terminal delivery still has to be attempted
        agent.emit_trace(
            task_state,
            "runtime_error_event_failed",
            {"error_type": type(publish_error).__name__},
        )
    try:
        async for _ in engine._publish_run_events(
            finish_stopped_run(
                engine,
                task_state,
                user_message,
                final,
                "runtime_error",
                getattr(engine, "_turn_started_at", time.monotonic()),
            )
        ):
            pass
    except Exception as terminal_error:  # noqa: BLE001 - retry with the minimal terminal contract
        if agent.session_event_bus.is_completed(task_state.run_id):
            return
        agent.emit_trace(
            task_state,
            "runtime_error_terminal_fallback",
            {"error_type": type(terminal_error).__name__},
        )
        task_state.stop("runtime_error", final_answer=final)
        agent.current_run_id = ""
        try:
            await asyncio.to_thread(agent.run_store.write_task_state, task_state)
            await asyncio.to_thread(
                agent.run_store.write_report,
                task_state,
                agent.redact_artifact(agent.build_report(task_state)),
            )
        except Exception as persistence_error:  # noqa: BLE001 - event delivery must not depend on report I/O
            agent.emit_trace(
                task_state,
                "runtime_error_report_failed",
                {"error_type": type(persistence_error).__name__},
            )
        await agent.session_event_bus.publish(
            "stop",
            {"run_id": task_state.run_id, "content": final},
        )
        await agent.session_event_bus.publish(
            "turn_finished",
            {
                "run_id": task_state.run_id,
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
            },
        )
