from gencode.core.context_retention import ContextRetentionPolicy, RetentionContext
from gencode.core.turn_history import should_render_tool_inline


def retention_context(**overrides):
    values = {
        "recent_turns": {"recent"},
        "last_failed_tool_event_id": "latest-failed",
        "last_changed_tool_event_id": "latest-changed",
        "changed_paths": {"src/app.py"},
        "pressure_tier": "tier0_observe",
    }
    values.update(overrides)
    return RetentionContext(**values)


def tool_item(name="read_file", **overrides):
    item = {
        "role": "tool",
        "name": name,
        "turn_id": "old",
        "event_id": "event-old",
        "args": {},
        "content": "tool output",
    }
    item.update(overrides)
    return item


def test_retention_policy_keeps_recent_failed_workspace_changing_and_changed_path_tools_inline():
    policy = ContextRetentionPolicy()
    context = retention_context()

    assert policy.should_render_tool_inline(tool_item(turn_id="recent"), context)
    assert policy.should_render_tool_inline(tool_item(tool_status="error"), context)
    assert policy.should_render_tool_inline(tool_item(tool_error_code="tool_failed"), context)
    assert policy.should_render_tool_inline(tool_item(event_id="latest-failed"), context)
    assert policy.should_render_tool_inline(tool_item(workspace_changed=True), context)
    assert policy.should_render_tool_inline(tool_item(event_id="latest-changed"), context)
    assert policy.should_render_tool_inline(tool_item(args={"path": "src/app.py"}), context)
    assert policy.should_render_tool_inline(tool_item(affected_paths=["src/app.py"]), context)


def test_retention_policy_can_stub_old_artifact_backed_bulk_tools():
    policy = ContextRetentionPolicy()
    context = retention_context()

    for name in ("read_file", "run_shell", "search"):
        assert not policy.should_render_tool_inline(
            tool_item(
                name,
                artifact_ref=f"runs/current/{name}-output.txt",
                content_sha256=f"sha-{name}",
            ),
            context,
        )


def test_retention_policy_high_pressure_does_not_override_protected_coordination_or_todo_tools():
    policy = ContextRetentionPolicy()
    context = retention_context(pressure_tier="tier3_summary")

    for name in (
        "ask_user",
        "agent",
        "send_message",
        "task_stop",
        "todo_add",
        "todo_update",
        "todo_list",
        "enter_plan_mode",
        "exit_plan_mode",
    ):
        assert policy.should_render_tool_inline(tool_item(name), context)


def test_legacy_should_render_tool_inline_delegates_to_retention_policy():
    context = retention_context(pressure_tier="tier3_summary")

    assert should_render_tool_inline(tool_item("ask_user"), context)
    assert not should_render_tool_inline(
        tool_item("read_file", artifact_ref="runs/current/read_file-output.txt", content_sha256="sha"),
        context,
    )
