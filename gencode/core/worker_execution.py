"""Worker thread execution routine."""

import time

from .worker_artifacts import collect_worker_artifacts
from .worker_notifications import render_worker_notification
from .workspace import clip, now


def run_worker(manager, task, prompt, action):
    item = manager._get_item(task.id)
    with manager._lock:
        item["status"] = "running"
        item["updated_at"] = now()
        item["notification_drained"] = False
    manager.runtime.session_event_bus.emit(
        "worker_started",
        {"worker_id": task.id, "description": task.description, "subagent_type": task.subagent_type, "action": action},
    )
    manager._save()
    started = time.monotonic()
    try:
        result = task.runtime.ask(str(prompt or ""))
        status = "stopped" if task.stop_requested else "completed"
    except Exception as exc:
        result = f"error: worker failed: {exc}"
        status = "failed"
    task_state = getattr(task.runtime, "current_task_state", None)
    with manager._lock:
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
    manager._notifications.put((task.id, render_worker_notification(item)))
    manager.runtime.session_event_bus.emit(
        "worker_finished",
        {"worker_id": task.id, "status": status, "duration_ms": item["duration_ms"]},
    )
    manager._save()
