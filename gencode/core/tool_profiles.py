"""Named tool capability surfaces for runtime modes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSetProfile:
    name: str
    allowed_tools: frozenset[str]

    def allows(self, tool_name):
        return tool_name in self.allowed_tools


def build_tool_profiles(tools):
    all_tools = frozenset(tools)
    coordinator_tools = frozenset({"agent", "send_message", "task_stop"})
    mode_tools = frozenset({"enter_plan_mode", "exit_plan_mode"})
    interactive_tools = frozenset({"ask_user"})
    candidate_write_tools = frozenset({"knowledge_propose"})
    read_only = (
        frozenset(name for name, tool in tools.items() if tool.read_only)
        - coordinator_tools
        - mode_tools
        - interactive_tools
        - candidate_write_tools
    )
    plan_tools = read_only | frozenset(
        {
            "write_file",
            "patch_file",
            "agent",
            "send_message",
            "task_stop",
            "ask_user",
            "exit_plan_mode",
        }
    )
    # Dream may inspect evidence and submit inactive knowledge candidates, but
    # cannot edit repository files or activate/delete knowledge.
    dream_tools = read_only | candidate_write_tools
    worker_tools = (
        all_tools
        - coordinator_tools
        - mode_tools
        - interactive_tools
        - frozenset({"run_shell"})
        - candidate_write_tools
    )
    return {
        "default": ToolSetProfile("default", all_tools),
        "plan": ToolSetProfile("plan", plan_tools & all_tools),
        "dream": ToolSetProfile("dream", dream_tools & all_tools),
        "readonly": ToolSetProfile("readonly", read_only),
        "worker": ToolSetProfile("worker", worker_tools),
    }
