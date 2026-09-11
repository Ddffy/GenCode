"""Session-scoped todo ledger for coordinator planning."""

from .workspace import now

VALID_STATUS = {"pending", "in_progress", "done", "blocked"}
VALID_PRIORITY = {"low", "normal", "high"}


class TodoLedger:
    def __init__(self, runtime):
        self.runtime = runtime
        self.runtime.session.setdefault("todos", {"next_id": 1, "items": []})

    @property
    def state(self):
        return self.runtime.session.setdefault("todos", {"next_id": 1, "items": []})

    def add(self, content, status="pending", priority="normal", note=""):
        status = _clean_status(status)
        priority = _clean_priority(priority)
        todo_id = f"todo_{int(self.state.get('next_id', 1))}"
        self.state["next_id"] = int(self.state.get("next_id", 1)) + 1
        item = {
            "id": todo_id,
            "content": str(content).strip(),
            "status": status,
            "priority": priority,
            "note": str(note or "").strip(),
            "created_at": now(),
            "updated_at": now(),
        }
        self.state.setdefault("items", []).append(item)
        self._record_change("add", item)
        return item

    def update(self, todo_id, **changes):
        item = self.get(todo_id)
        for key in ("content", "note"):
            if key in changes and changes[key] is not None:
                item[key] = str(changes[key]).strip()
        if changes.get("status") is not None:
            item["status"] = _clean_status(changes["status"])
        if changes.get("priority") is not None:
            item["priority"] = _clean_priority(changes["priority"])
        item["updated_at"] = now()
        self._record_change("update", item)
        return item

    def get(self, todo_id):
        for item in self.state.setdefault("items", []):
            if item.get("id") == str(todo_id):
                return item
        raise ValueError(f"unknown todo_id: {todo_id}")

    def render_list(self):
        items = list(self.state.setdefault("items", []))
        if not items:
            return "Task ledger:\n- empty"
        lines = ["Task ledger:"]
        for item in items:
            note = f" ({item['note']})" if item.get("note") else ""
            lines.append(f"- {item['id']} [{item['status']}] {item['priority']} - {item['content']}{note}")
        return "\n".join(lines)

    def render_prompt(self):
        return self.render_list()

    def to_dict(self):
        return {"next_id": int(self.state.get("next_id", 1)), "items": [dict(item) for item in self.state.get("items", [])]}

    def _record_change(self, action, item):
        payload = {"action": action, "todo": dict(item)}
        task_state = getattr(self.runtime, "current_task_state", None)
        if task_state is not None:
            task_state.todo_changes.append(payload)
        self.runtime.session_event_bus.emit("todo_changed", payload)
        self.runtime.session_path = self.runtime.session_store.save(self.runtime.session)


def _clean_status(value):
    status = str(value or "pending").strip()
    if status not in VALID_STATUS:
        raise ValueError(f"status must be one of {', '.join(sorted(VALID_STATUS))}")
    return status


def _clean_priority(value):
    priority = str(value or "normal").strip()
    if priority not in VALID_PRIORITY:
        raise ValueError(f"priority must be one of {', '.join(sorted(VALID_PRIORITY))}")
    return priority
