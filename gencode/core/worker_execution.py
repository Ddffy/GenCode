"""Asynchronous worker execution routine."""

import time

from .worker_artifacts import collect_worker_artifacts
from .worker_notifications import render_worker_notification
from .workspace import clip, now


async def run_worker(manager, task, prompt, action):
    async with manager._semaphore:
        task.runtime.inherited_workspace_lease = (
            manager.runtime.workspace_lock_lease is not None
            or manager.runtime.inherited_workspace_lease
        )
        task.runtime.inherited_session_lease = (
            manager.runtime.session_lock_lease is not None
            or manager.runtime.inherited_session_lease
        )
        task.runtime.session_lock_lease = manager.runtime.session_lock_lease
        task.runtime.workspace_lock_lease = manager.runtime.workspace_lock_lease
        task.runtime.workspace_write_lock = manager.runtime.workspace_write_lock
        item = manager._get_item(task.id)
        item["status"] = "running"
        item["updated_at"] = now()
        item["notification_drained"] = False
        manager.runtime.session_event_bus.emit(
            "worker_started",
            {
                "run_id": manager.runtime.current_run_id,
                "worker_id": task.id,
                "description": task.description,
                "subagent_type": task.subagent_type,
                "action": action,
                "source": task.id,
            },
        )
        manager._save()
        started = time.monotonic()
        result = ""
        status = "failed"
        try:
            if task.stop_requested:
                result = "Stopped before worker execution started."
                status = "stopped"
            else:
                async for event in task.runtime.engine.run_turn(str(prompt or "")):
                    manager._events.put_nowait(
                        {
                            "type": "worker_event",
                            "run_id": manager.runtime.current_run_id,
                            "worker_run_id": event.get("run_id", ""),
                            "source": task.id,
                            "event": event,
                        }
                    )
                    if event.get("type") in {"final", "stop"}:
                        result = str(event.get("content", ""))
                status = "stopped" if task.stop_requested else "completed"
        except Exception as exc:  # noqa: BLE001
            result = f"error: worker failed: {exc}"
        finally:
            task.runtime.inherited_workspace_lease = False
            task.runtime.inherited_session_lease = False
            task.runtime.session_lock_lease = None
            task.runtime.workspace_lock_lease = None
        task_state = getattr(task.runtime, "current_task_state", None)
        item.update(
            {
                "status": status,
                "result": clip(result, 2000),
                "tool_steps": int(getattr(task_state, "tool_steps", 0) or 0),
                "attempts": int(getattr(task_state, "attempts", 0) or 0),
                **collect_worker_artifacts(manager.runtime.root, task.runtime, task_state),
                "duration_ms": int((time.monotonic() - started) * 1000),
                "updated_at": now(),
            }
        )
        manager._notifications.put_nowait(
            (task.id, render_worker_notification(item))
        )
        manager.runtime.session_event_bus.emit(
            "worker_finished",
            {
                "run_id": manager.runtime.current_run_id,
                "worker_id": task.id,
                "status": status,
                "duration_ms": item["duration_ms"],
                "source": task.id,
            },
        )
        manager._save()
