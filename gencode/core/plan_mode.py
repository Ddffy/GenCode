"""Plan mode policy for sessions."""

import re


def _slug(value):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip().lower()).strip("-")
    return slug or "plan"


class PlanModeManager:
    def __init__(self, runtime):
        self.runtime = runtime

    @property
    def state(self):
        return self.runtime.session.setdefault("runtime_mode", {"mode": "default"})

    @property
    def mode(self):
        return str(self.state.get("mode", "default") or "default")

    @property
    def plan_path(self):
        return str(self.state.get("plan_path", "") or "")

    def enter(self, topic, path=None):
        plan_path = _plan_path(topic, path)
        self.runtime.session["runtime_mode"] = {
            "mode": "plan",
            "topic": str(topic or ""),
            "plan_path": plan_path,
        }
        self.runtime.set_tool_profile("plan")
        self.runtime.session_path = self.runtime.session_store.save(
            self.runtime.session
        )
        self.runtime.refresh_prefix(force=True)
        self.runtime.session_event_bus.emit(
            "runtime_mode_changed",
            {"mode": "plan", "plan_path": plan_path, "topic": str(topic or "")},
        )
        return plan_path

    def exit(self):
        previous = dict(self.state)
        self.runtime.session["runtime_mode"] = {"mode": "default"}
        self.runtime.set_tool_profile("default")
        self.runtime.session_path = self.runtime.session_store.save(
            self.runtime.session
        )
        self.runtime.refresh_prefix(force=True)
        self.runtime.session_event_bus.emit(
            "runtime_mode_changed",
            {
                "mode": "default",
                "previous_mode": previous.get("mode", "default"),
                "plan_path": previous.get("plan_path", ""),
            },
        )

    def can_finish(self):
        if self.mode != "plan":
            return True
        path = self.runtime.path(self.plan_path)
        return path.is_file() and bool(path.read_text(encoding="utf-8").strip())

    def final_notice(self):
        return f"Plan mode requires writing the active plan artifact before final answer: {self.plan_path}"

    def prompt_text(self):
        if self.mode != "plan":
            return ""
        return (
            "Runtime mode: plan\n"
            f"- Active plan artifact: {self.plan_path}\n"
            "- You may inspect files, but writes must target only the active plan artifact.\n"
            "- You may launch Explore subagents, but not write-capable worker subagents.\n"
            "- Use todo tools to keep the task ledger current.\n"
            "- Return a final answer only after the active plan artifact has been written."
        )


PlanModeController = PlanModeManager


_PLAN_DIR_MARKER = "/.gencode/plans/"


def _plan_path(topic, path=None):
    if path:
        value = str(path).strip()
        # 模型有时给绝对路径，如 /Users/u/repo/.gencode/plans/foo；自动把它相对化。
        if value.startswith("/") and _PLAN_DIR_MARKER in value:
            value = value[value.index(_PLAN_DIR_MARKER) + 1 :]
        if value.startswith("./"):
            value = value[2:]
    else:
        value = f".gencode/plans/{_slug(topic)}-plan.md"
    if (
        not value.startswith(".gencode/plans/")
        or value.endswith("/")
        or ".." in value.split("/")
    ):
        raise ValueError("plan path must stay under .gencode/plans/")
    return value
