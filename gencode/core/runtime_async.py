import asyncio
from contextlib import AsyncExitStack

from ..tools.base import ToolEffect
from . import tool_executor
from .session_lifecycle import clear_runtime_session, resume_runtime_session


class RuntimeAsyncMixin:
    def ask(self, user_message):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ask_async(user_message))
        raise RuntimeError("ask() cannot run inside an active event loop; use await ask_async().")

    async def ask_async(self, user_message):
        return await self.engine.ask_async(user_message)

    def abort_current_turn(self):
        self.abort_requested = True
        event = self.turn_cancel_event
        loop = self.turn_control_loop
        if event is not None and loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(event.set)
            queue = self.turn_control_queue
            if queue is not None:
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"type": "cancel", "content": ""}
                )

    def steer(self, message):
        return self._submit_turn_control("steer", message)

    def queue_turn(self, message):
        return self._submit_turn_control("queue", message)

    def cancel_current_turn(self):
        return self._submit_turn_control("cancel", "")

    def _submit_turn_control(self, kind, content):
        queue = self.turn_control_queue
        loop = self.turn_control_loop
        if queue is None or loop is None or loop.is_closed():
            if kind == "queue":
                self.queued_turns.append(str(content))
                return True
            return False
        if kind == "cancel" and self.turn_cancel_event is not None:
            self.cancel_requested = True
            loop.call_soon_threadsafe(self.turn_cancel_event.set)
        loop.call_soon_threadsafe(queue.put_nowait, {"type": kind, "content": str(content)})
        return True

    def ask_user(self, question, choices=None):
        if self.ask_user_callback is None:
            return "error: ask_user requires interactive mode"
        choices = [str(choice) for choice in (choices or [])]
        return str(self.ask_user_callback(str(question), choices))

    async def ask_user_async(self, question, choices=None):
        if self.ask_user_callback is None:
            return "error: ask_user requires interactive mode"
        result = self.ask_user_callback(
            str(question), [str(choice) for choice in (choices or [])]
        )
        if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
            result = await result
        return str(result)

    def resume_session(self, session_id):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.resume_session_async(session_id))
        raise RuntimeError(
            "resume_session() cannot run inside an active event loop; use await resume_session_async()."
        )

    async def resume_session_async(self, session_id):
        await self.worker_manager.shutdown_async()
        return resume_runtime_session(self, session_id)

    def clear_session(self):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.clear_session_async())
        raise RuntimeError(
            "clear_session() cannot run inside an active event loop; use await clear_session_async()."
        )

    async def clear_session_async(self):
        await self.worker_manager.shutdown_async()
        return clear_runtime_session(self)

    async def run_tool(self, name, args):
        self._bind_async_loop(asyncio.get_running_loop())
        tool = self.available_tools().get(str(name))
        inherited_session = self.session_lock_lease is not None or self.inherited_session_lease
        inherited_workspace = self.workspace_lock_lease is not None or self.inherited_workspace_lease
        async with AsyncExitStack() as lock_stack:
            if not inherited_session:
                session_lease = await self.lock_manager.acquire_session(self.session["id"])
                await lock_stack.enter_async_context(session_lease)
                self.session_lock_lease = session_lease
            if not inherited_workspace:
                mode = (
                    "read"
                    if tool is not None and tool.capability.effect == ToolEffect.READ
                    else "write"
                )
                workspace_lease = await self.lock_manager.acquire_workspace(
                    self.root, mode
                )
                await lock_stack.enter_async_context(workspace_lease)
                self.workspace_lock_lease = workspace_lease
            try:
                if (
                    str(name) in {"agent", "send_message", "task_stop"}
                    or (tool is not None and tool.capability.effect == ToolEffect.READ)
                ):
                    return await tool_executor.run_tool(self, name, args)
                async with self.workspace_write_lock:
                    return await tool_executor.run_tool(self, name, args)
            finally:
                if not inherited_session:
                    self.session_lock_lease = None
                if not inherited_workspace:
                    self.workspace_lock_lease = None

    def _bind_async_loop(self, loop):
        if self._bound_async_loop is loop:
            return
        if self._bound_async_loop is not None:
            self.workspace_write_lock = asyncio.Lock()
        self._bound_async_loop = loop
        manager = getattr(self, "worker_manager", None)
        if manager is not None:
            manager.bind_loop(loop)
