"""Session-scoped asynchronous worker lifecycle."""

import asyncio
from dataclasses import dataclass, field

from .worker_contracts import clean_scope, clean_type
from .worker_execution import run_worker
from .worker_runtime import build_child_runtime
from .workspace import now


@dataclass
class WorkerTask:
    id: str
    description: str
    subagent_type: str
    write_scope: tuple[str, ...]
    runtime: object
    task: asyncio.Task | None = None
    stop_requested: bool = False
    state: dict = field(default_factory=dict)


class WorkerManager:
    def __init__(self, runtime):
        self.runtime = runtime
        self.runtime.session.setdefault("workers", {"next_id": 1, "items": []})
        self._tasks = {}
        self._notifications = asyncio.Queue()
        self._events = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(runtime.max_worker_concurrency)
        self._bound_loop = None

    @property
    def state(self):
        return self.runtime.session.setdefault("workers", {"next_id": 1, "items": []})

    def bind_loop(self, loop):
        if self._bound_loop is loop:
            return
        if self._bound_loop is not None:
            self._notifications = asyncio.Queue()
            self._events = asyncio.Queue()
            self._semaphore = asyncio.Semaphore(self.runtime.max_worker_concurrency)
        self._bound_loop = loop
        for task in self._tasks.values():
            if task.task is not None and task.task.done():
                task.task = None
            task.runtime._bind_async_loop(loop)
            task.runtime.workspace_write_lock = self.runtime.workspace_write_lock

    async def spawn(self, description, prompt, subagent_type="worker", write_scope=None):
        subagent_type = clean_type(subagent_type)
        if self.runtime.runtime_mode == "plan" and subagent_type != "Explore":
            raise ValueError("plan mode only allows Explore agents")
        task = self._new_task(description, subagent_type, write_scope)
        self._tasks[task.id] = task
        if getattr(self.runtime, "model_client_factory", None) is None:
            await run_worker(self, task, prompt, action="spawn")
            return self._public_payload(task)
        task.task = asyncio.create_task(run_worker(self, task, prompt, action="spawn"))
        return self._public_payload(task, status="started")

    async def continue_task(self, task_id, message):
        task = self._get_active_task(task_id)
        item = self._get_item(task_id)
        if item.get("status") in {"running", "stopping"}:
            raise ValueError(f"worker is running: {task_id}")
        if self.runtime.runtime_mode == "plan" and task.subagent_type != "Explore":
            raise ValueError("plan mode only allows Explore agents")
        if getattr(self.runtime, "model_client_factory", None) is None:
            await run_worker(self, task, message, action="continue")
            return self._public_payload(task)
        task.task = asyncio.create_task(run_worker(self, task, message, action="continue"))
        return self._public_payload(task, status="started")

    async def stop_task(self, task_id):
        item = self._get_item(task_id)
        if item["status"] == "running":
            task = self._tasks.get(str(task_id))
            if task is not None:
                self._request_stop(task)
            item["status"] = "stopping"
            item["updated_at"] = now()
            self.runtime.session_event_bus.emit(
                "worker_stop_requested",
                {"worker_id": item["id"], "status": "stopping"},
            )
            self._save()
        return {
            "task_id": item["id"],
            "status": item["status"],
            "description": item["description"],
        }

    async def wait_idle(self):
        tasks = [task.task for task in self._tasks.values() if task.task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def shutdown_async(self):
        tasks = list(self._tasks.values())
        active = []
        for task in tasks:
            item = self._get_item(task.id)
            if item.get("status") in {"running", "stopping", "idle"}:
                self._request_stop(task)
                item["status"] = "stopping"
                item["updated_at"] = now()
                if task.task is not None:
                    active.append(task.task)
                self.runtime.session_event_bus.emit(
                    "worker_stop_requested",
                    {"worker_id": item["id"], "status": "stopping"},
                )
        if tasks:
            self._save()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        return {"stopped": sum(1 for task in tasks if task.stop_requested)}

    def to_dict(self):
        return {
            "next_id": int(self.state.get("next_id", 1)),
            "items": [dict(item) for item in self.state.get("items", [])],
        }

    def _new_task(self, description, subagent_type, write_scope):
        worker_id = f"agent_{int(self.state.get('next_id', 1))}"
        self.state["next_id"] = int(self.state.get("next_id", 1)) + 1
        scope = tuple(clean_scope(write_scope))
        child = build_child_runtime(self.runtime, subagent_type, scope)
        child.workspace_write_lock = self.runtime.workspace_write_lock
        item = {
            "id": worker_id,
            "description": str(description or "").strip() or "Worker task",
            "subagent_type": subagent_type,
            "write_scope": list(scope),
            "status": "idle",
            "result": "",
            "tool_steps": 0,
            "attempts": 0,
            "duration_ms": 0,
            "notification_drained": False,
            "created_at": now(),
            "updated_at": now(),
        }
        self.state.setdefault("items", []).append(item)
        self._save()
        return WorkerTask(worker_id, item["description"], subagent_type, scope, child)

    def _request_stop(self, task):
        task.stop_requested = True
        cancel = getattr(task.runtime, "cancel_current_turn", None)
        if callable(cancel):
            cancel()

    def drain_notifications(self):
        drained = []
        while True:
            try:
                task_id, notification = self._notifications.get_nowait()
            except asyncio.QueueEmpty:
                break
            item = self._get_item(task_id)
            if item.get("notification_drained"):
                continue
            item["notification_drained"] = True
            item["updated_at"] = now()
            drained.append(notification)
        if drained:
            self._save()
        return drained

    def drain_events(self):
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except asyncio.QueueEmpty:
                return events

    async def next_event(self):
        return await self._events.get()

    def _get_active_task(self, task_id):
        task = self._tasks.get(str(task_id))
        if task is None:
            raise ValueError(f"unknown or inactive worker: {task_id}")
        return task

    def _get_item(self, task_id):
        for item in self.state.setdefault("items", []):
            if item.get("id") == str(task_id):
                return item
        raise ValueError(f"unknown worker: {task_id}")

    def _public_payload(self, task, status=None):
        item = self._get_item(task.id)
        return {
            "task_id": task.id,
            "status": status or item["status"],
            "description": task.description,
        }

    def _save(self):
        self.runtime.session_path = self.runtime.session_store.save(
            self.runtime.session
        )
