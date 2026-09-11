import pytest

from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.testing import ScriptedModelClient


def build_agent(tmp_path, outputs, approval_policy="auto"):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    return GenCode(
        model_client=ScriptedModelClient(outputs),
        workspace=workspace,
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy=approval_policy,
    )


def assistant_contents(app):
    from gencode.tui.widgets import AssistantMessage

    return [message.content for message in app.query(AssistantMessage)]


async def wait_for_assistant(app, pilot, expected="", attempts=40, delay=0.1):
    for _ in range(attempts):
        await pilot.pause(delay=delay)
        text = "\n".join(assistant_contents(app))
        if expected and expected in text:
            return text
        if not expected and text:
            return text
    return "\n".join(assistant_contents(app))


async def wait_for_widget(app, pilot, selector, attempts=40, delay=0.1):
    for _ in range(attempts):
        await pilot.pause(delay=delay)
        widgets = list(app.query(selector))
        if widgets:
            return widgets[-1]
    raise AssertionError(f"timed out waiting for {selector}")


async def wait_for_tool_card_status(app, pilot, status, attempts=40, delay=0.1):
    from gencode.tui.widgets import ToolCard

    for _ in range(attempts):
        await pilot.pause(delay=delay)
        cards = list(app.query(ToolCard))
        if cards and cards[-1].status == status:
            return cards[-1]
    raise AssertionError(f"timed out waiting for tool card status {status}")


async def wait_for_input_ready(bar, pilot, attempts=40, delay=0.1):
    for _ in range(attempts):
        await pilot.pause(delay=delay)
        if not bar.input.disabled and bar.input.has_focus:
            return
    raise AssertionError("timed out waiting for input to be ready")


async def wait_for_layout(widget, pilot, attempts=40, delay=0.1):
    for _ in range(attempts):
        await pilot.pause(delay=delay)
        if widget.region.width > 0 and widget.region.height > 0:
            return
    raise AssertionError(f"timed out waiting for layout of {widget!r}")


def rendered_text(widget) -> str:
    rendered = widget.render()
    return getattr(rendered, "plain", str(rendered))


def test_cli_defaults_interactive_tty_mode_to_tui(monkeypatch):
    from gencode.cli import build_arg_parser, interaction_mode

    monkeypatch.setattr(
        "gencode.cli.sys.stdin", type("Stdin", (), {"isatty": lambda self: True})()
    )
    args = build_arg_parser().parse_args(["--cwd", "/tmp/workspace"])

    assert interaction_mode(args) == "tui"


def test_cli_keeps_prompt_as_one_shot_mode():
    from gencode.cli import build_arg_parser, interaction_mode

    args = build_arg_parser().parse_args(["inspect", "tests"])

    assert interaction_mode(args) == "one_shot"


def test_cli_repl_flag_restores_plain_repl():
    from gencode.cli import build_arg_parser, interaction_mode

    args = build_arg_parser().parse_args(["--repl", "--cwd", "/tmp/workspace"])

    assert interaction_mode(args) == "repl"


def test_cli_uses_plain_repl_for_piped_stdin(monkeypatch):
    from gencode.cli import build_arg_parser, interaction_mode

    monkeypatch.setattr(
        "gencode.cli.sys.stdin", type("Stdin", (), {"isatty": lambda self: False})()
    )
    args = build_arg_parser().parse_args(["--cwd", "/tmp/workspace"])

    assert interaction_mode(args) == "repl"


def test_cli_accepts_explicit_tui_flag():
    from gencode.cli import build_arg_parser, interaction_mode

    args = build_arg_parser().parse_args(["--tui", "--cwd", "/tmp/workspace"])

    assert args.tui is True
    assert interaction_mode(args) == "tui"
    assert args.cwd == "/tmp/workspace"


def test_status_bar_shows_runtime_identity(tmp_path):
    from gencode.tui.widgets import StatusBar

    agent = build_agent(tmp_path, [])
    status = StatusBar()

    status.update_agent(agent)

    text = rendered_text(status)
    assert "mode default" in text
    assert "session" in text


def test_status_bar_reads_context_usage_governance_fields():
    from gencode.tui.widgets import StatusBar

    status = StatusBar()

    status.update_context_usage(
        {
            "total_estimated_tokens": 1234,
            "context_window": 200000,
            "free_tokens": 198766,
        }
    )

    assert "context 1234/200000" in rendered_text(status)


def test_status_bar_optionally_shows_context_pressure_fields():
    from gencode.tui.widgets import StatusBar

    status = StatusBar()

    status.update_context_usage(
        {
            "total_estimated_tokens": 500,
            "context_window": 1000,
            "pressure_tier": "medium",
            "usage_source": "estimated",
            "cached_tokens": 64,
        }
    )

    text = rendered_text(status)
    assert "context 500/1000" in text
    assert "tier medium" in text
    assert "source estimated" in text
    assert "cached 64" in text


def test_status_bar_omits_optional_context_pressure_fields_when_absent():
    from gencode.tui.widgets import StatusBar

    status = StatusBar()

    status.update_context_usage(
        {
            "total_estimated_tokens": 500,
            "context_window": 1000,
        }
    )

    text = rendered_text(status)
    assert "context 500/1000" in text
    assert "tier" not in text
    assert "source" not in text
    assert "cached" not in text


def test_cli_plan_mode_and_session_commands_expose_runtime_state(tmp_path):
    from gencode.cli import handle_repl_command

    agent = build_agent(tmp_path, [])

    handled, should_exit, output = handle_repl_command(agent, "/plan refactor-auth")

    assert handled is True
    assert should_exit is False
    assert "mode: plan" in output
    assert ".gencode/plans/refactor-auth-plan.md" in output
    assert agent.runtime_mode == "plan"

    handled, _, output = handle_repl_command(agent, "/mode")
    assert handled is True
    assert "runtime mode: plan" in output
    assert "plan path: .gencode/plans/refactor-auth-plan.md" in output

    handled, _, output = handle_repl_command(agent, "/session")
    assert handled is True
    assert "session id:" in output
    assert "events path:" in output
    assert "runtime mode: plan" in output
    assert "worker summary:" in output

    handled, _, output = handle_repl_command(agent, "/plan-exit")
    assert handled is True
    assert output == "mode: default"
    assert agent.runtime_mode == "default"


def test_slash_command_registry_suggests_and_parses_subagent():
    from gencode.commands.slash import (
        parse_subagent_args,
        resolve_command,
        suggest_commands,
    )

    suggestions = suggest_commands("/sub")

    assert suggestions[0].name == "subagent"
    assert resolve_command("sub").name == "subagent"

    payload, error = parse_subagent_args("worker --scope README.md,src update docs")

    assert error == ""
    assert payload["subagent_type"] == "worker"
    assert payload["write_scope"] == ["README.md", "src"]
    assert payload["prompt"] == "update docs"

    skill_suggestions = [command.name for command in suggest_commands("/sk")]
    assert "skills" in skill_suggestions
    assert "skill" in skill_suggestions


@pytest.mark.asyncio
async def test_tui_slash_suggestions_complete_partial_command(tmp_path):
    from gencode.tui.app import GenCodeTuiApp
    from gencode.tui.widgets import InputBar, SlashSuggestions

    app = GenCodeTuiApp(build_agent(tmp_path, []))

    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.input.value = "/sub"
        bar.update_slash_suggestions()

        suggestions = app.query_one(SlashSuggestions)
        assert suggestions.visible is True
        assert "/subagent" in rendered_text(suggestions)

        await pilot.press("tab")
        await pilot.pause(delay=0.1)

        assert bar.input.value == "/subagent "
        assert suggestions.visible is False


def test_agents_slash_command_shows_worker_status(tmp_path):
    from gencode.cli import handle_repl_command

    agent = build_agent(tmp_path, [])

    handled, should_exit, output = handle_repl_command(agent, "/agents")

    assert handled is True
    assert should_exit is False
    assert "worker summary:" in output


def test_subagent_slash_command_launches_explore_worker(tmp_path):
    from gencode.cli import handle_repl_command

    agent = build_agent(tmp_path, ["<final>Subagent checked README.</final>"])

    handled, should_exit, output = handle_repl_command(
        agent, "/subagent explore inspect README"
    )

    assert handled is True
    assert should_exit is False
    assert "agent_1" in output
    assert "completed" in output or "started" in output


@pytest.mark.asyncio
async def test_tui_help_command_uses_existing_repl_commands(tmp_path):
    from gencode.tui.app import GenCodeTuiApp
    from gencode.tui.widgets import InputBar

    agent = build_agent(tmp_path, [])
    app = GenCodeTuiApp(agent)

    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.input.value = "/help"
        await pilot.press("enter")
        await pilot.pause(delay=0.1)

        text = "\n".join(assistant_contents(app))
        assert "Commands:" in text
        assert "/memory" in text


@pytest.mark.asyncio
async def test_tui_runs_agent_turn_and_renders_final_answer(tmp_path):
    from gencode.tui.app import GenCodeTuiApp
    from gencode.tui.widgets import InputBar

    agent = build_agent(tmp_path, ["<final>Done from TUI.</final>"])
    app = GenCodeTuiApp(agent)

    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.input.value = "ship it"
        await pilot.press("enter")
        text = await wait_for_assistant(app, pilot, "Done from TUI.")

        assert "Done from TUI." in text


@pytest.mark.asyncio
async def test_tui_hides_welcome_after_first_turn_so_chat_stays_visible(tmp_path):
    from gencode.tui.app import GenCodeTuiApp
    from gencode.tui.widgets import ChatLog, InputBar, WelcomeBanner

    agent = build_agent(tmp_path, ["<final>Done from TUI.</final>"])
    app = GenCodeTuiApp(agent)

    async with app.run_test(size=(80, 16)) as pilot:
        bar = app.query_one(InputBar)
        bar.input.value = "ship it"
        await pilot.press("enter")
        text = await wait_for_assistant(app, pilot, "Done from TUI.")
        await wait_for_input_ready(bar, pilot)

        chat = app.query_one(ChatLog)
        welcome = app.query_one(WelcomeBanner)

        assert "Done from TUI." in text
        assert "hidden" in welcome.classes
        assert chat.region.height >= 8
        assert bar.input.disabled is False
        assert bar.input.has_focus


@pytest.mark.asyncio
async def test_tui_chat_stream_uses_terminal_transcript_layout(tmp_path):
    from gencode.tui.app import GenCodeTuiApp
    from gencode.tui.widgets import AssistantMessage, ChatLog, InputBar, UserMessage

    agent = build_agent(
        tmp_path,
        ["<final>我是 gencode。\n\n- 读代码\n- 跑命令\n- 改文件</final>"],
    )
    app = GenCodeTuiApp(agent)

    async with app.run_test(size=(100, 20)) as pilot:
        bar = app.query_one(InputBar)
        bar.input.value = "你是谁"
        await pilot.press("enter")
        text = await wait_for_assistant(app, pilot, "我是 gencode。")

        chat = app.query_one(ChatLog)
        user = app.query_one(UserMessage)
        assistant = app.query_one(AssistantMessage)
        await wait_for_layout(user, pilot)
        await wait_for_layout(assistant, pilot)

        assert "我是 gencode。" in text
        assert chat.styles.scrollbar_size_horizontal == 0
        assert chat.styles.scrollbar_size_vertical == 1
        assert chat.styles.scrollbar_background.hex.lower() == "#0f1117"
        assert user.styles.border_left[0] == ""
        assert assistant.styles.border_left[0] == ""
        assert user.styles.background.hex.lower() == "#0f1117"
        assert assistant.styles.background.hex.lower() == "#0f1117"
        assert user.region.x <= chat.region.x + 2
        assert assistant.region.x <= chat.region.x + 2
        assert user.region.width >= chat.region.width - 4
        assert assistant.region.width >= chat.region.width - 4


@pytest.mark.asyncio
async def test_tui_renders_tool_card_result(tmp_path):
    from gencode.tui.app import GenCodeTuiApp
    from gencode.tui.widgets import InputBar, ToolCard

    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="notes/result.txt"><content>ok\n</content></tool>',
            "<final>Wrote it.</final>",
        ],
    )
    app = GenCodeTuiApp(agent)

    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.input.value = "write a file"
        await pilot.press("enter")
        card = await wait_for_tool_card_status(app, pilot, "success")

        cards = list(app.query(ToolCard))
        assert cards
        assert card is cards[-1]
        assert (tmp_path / "notes" / "result.txt").read_text(encoding="utf-8") == "ok\n"


@pytest.mark.asyncio
async def test_tui_approval_prompt_controls_risky_tool(tmp_path):
    from gencode.tui.app import GenCodeTuiApp
    from gencode.tui.widgets import ConfirmPrompt, InputBar

    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="notes/result.txt"><content>ok\n</content></tool>',
            "<final>Wrote it.</final>",
        ],
        approval_policy="ask",
    )
    app = GenCodeTuiApp(agent)

    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.input.value = "write a file"
        await pilot.press("enter")
        prompt = await wait_for_widget(app, pilot, ConfirmPrompt)

        assert prompt

        await pilot.press("right")
        await pilot.press("enter")
        text = await wait_for_assistant(app, pilot, "Wrote it.")

        assert "Wrote it." in text
        assert (tmp_path / "notes" / "result.txt").read_text(encoding="utf-8") == "ok\n"


@pytest.mark.asyncio
async def test_tui_ask_user_prompt_returns_selected_choice(tmp_path):
    from gencode.tui.app import GenCodeTuiApp
    from gencode.tui.widgets import AskUserPrompt, InputBar

    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"ask_user","args":{"question":"Ship?","choices":["no","yes"]}}</tool>',
            "<final>User chose yes.</final>",
        ],
    )
    app = GenCodeTuiApp(agent)

    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.input.value = "ask before shipping"
        await pilot.press("enter")
        prompt = await wait_for_widget(app, pilot, AskUserPrompt)

        assert prompt

        await pilot.press("right")
        await pilot.press("enter")
        text = await wait_for_assistant(app, pilot, "User chose yes.")

        assert "User chose yes." in text
