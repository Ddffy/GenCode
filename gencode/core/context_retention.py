"""Retention policy for prompt history tool results."""

from __future__ import annotations

from dataclasses import dataclass, field


PROTECTED_TOOL_NAMES = {
    "ask_user",
    "agent",
    "send_message",
    "task_stop",
    "todo_add",
    "todo_update",
    "todo_list",
    "enter_plan_mode",
    "exit_plan_mode",
}

BULK_TOOL_NAMES = {"read_file", "run_shell", "search"}


@dataclass(frozen=True)
class RetentionContext:
    recent_turns: set = field(default_factory=set)
    last_failed_tool_event_id: str = ""
    last_changed_tool_event_id: str = ""
    changed_paths: set = field(default_factory=set)
    pressure_tier: str = "tier0_observe"


class ContextRetentionPolicy:
    protected_tools = frozenset(PROTECTED_TOOL_NAMES)
    bulk_tools = frozenset(BULK_TOOL_NAMES)

    def should_render_tool_inline(self, item, context):
        if item.get("turn_id") in context.recent_turns:
            return True
        if item.get("name") in self.protected_tools:
            return True
        if self._is_failed_tool(item):
            return True
        event_id = str(item.get("event_id", ""))
        if event_id and event_id == context.last_failed_tool_event_id:
            return True
        if item.get("workspace_changed"):
            return True
        if event_id and event_id == context.last_changed_tool_event_id:
            return True
        return bool(self._tool_paths(item) & set(context.changed_paths))

    def can_replace_tool(self, item, context):
        if item.get("role") != "tool":
            return False
        if self.should_render_tool_inline(item, context):
            return False
        return item.get("name") in self.bulk_tools

    def _tool_paths(self, item):
        paths = {str(path) for path in item.get("affected_paths", []) if str(path).strip()}
        path_arg = str(item.get("args", {}).get("path", "")).strip()
        if path_arg:
            paths.add(path_arg)
        return paths

    @staticmethod
    def _is_failed_tool(item):
        status = str(item.get("tool_status", ""))
        return bool(status and status != "ok") or bool(item.get("tool_error_code"))
