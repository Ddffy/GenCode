from __future__ import annotations

import asyncio
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.events import Key

from ..cli import HELP_DETAILS, handle_repl_command_async
from .widgets import (
    AskUserPrompt,
    ChatLog,
    ConfirmPrompt,
    InputBar,
    StatusBar,
    ThinkingIndicator,
    ToolCard,
    WelcomeBanner,
    format_tool_args,
)

GENCODE_TUI_CSS = """
Screen {
    layout: vertical;
    background: #0f1117;
}
"""


class GenCodeTuiApp(App):
    """Textual shell for the existing GenCode runtime.

    The TUI is deliberately a presentation layer: CLI argument parsing and agent
    construction still live in `gencode.cli`, while turns are driven through the
    Engine's replayable Run event stream, as the plain REPL does.
    """

    CSS = GENCODE_TUI_CSS
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "submit_input", "Send", priority=True, show=False),
        Binding("ctrl+l", "clear_screen", "Clear"),
        Binding("ctrl+r", "resume_run", "Resume"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, agent, **kwargs) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self._turn_count = 0
        self._running_tool_cards: list[ToolCard] = []
        self._confirm_prompt: ConfirmPrompt | None = None
        self._confirm_future: asyncio.Future | None = None
        self._ask_user_prompt: AskUserPrompt | None = None
        self._ask_user_future: asyncio.Future | None = None
        self._active_turn_task: asyncio.Task | None = None
        self._active_run_id = ""
        self._active_run_seq = 0
        self._streaming_message = None
        self._previous_approve = getattr(agent, "approve", None)
        self._previous_ask_user = getattr(agent, "ask_user_callback", None)
        self._previous_approve_async = getattr(agent, "approve_async_callback", None)
        self.agent.approve_async_callback = self._approval_callback_async
        self.agent.ask_user_callback = self._ask_user_callback_async

    def compose(self) -> ComposeResult:
        yield WelcomeBanner(
            model_name=str(getattr(self.agent.model_client, "model", "")),
            cwd=str(getattr(self.agent, "root", "")),
            approval=str(getattr(self.agent, "approval_policy", "")),
        )
        yield ChatLog()
        yield ThinkingIndicator()
        yield StatusBar()
        yield InputBar()

    def on_mount(self) -> None:
        self.query_one(StatusBar).update_agent(self.agent)
        self.query_one(InputBar).focus_input()
        self.set_interval(0.5, self._drain_idle_worker_notifications)

    def on_unmount(self) -> None:
        if self._previous_approve is not None:
            self.agent.approve = self._previous_approve
        self.agent.approve_async_callback = self._previous_approve_async
        self.agent.ask_user_callback = self._previous_ask_user

    def action_clear_screen(self) -> None:
        self.query_one(ChatLog).clear_messages()

    def action_submit_input(self) -> None:
        if self._ask_user_prompt is not None:
            self._resolve_ask_user(self._ask_user_prompt.selected_choice)
            return
        if self._confirm_prompt is not None:
            self._resolve_confirm(self._confirm_prompt.selected)
            return
        bar = self.query_one(InputBar)
        text = bar.input.value.strip() #读取输入
        if not text:
            return
        if self._active_turn_task is not None:
            bar.input.value = ""
            if text.startswith("/queue "):
                accepted = self.agent.queue_turn(text[7:].strip())
            elif text == "/cancel":
                accepted = self.agent.cancel_current_turn()
            else:
                message = text[7:].strip() if text.startswith("/steer ") else text
                accepted = self.agent.steer(message)
            self.query_one(ChatLog).add_message("user", text)
            if not accepted:
                self.query_one(ChatLog).add_message(
                    "assistant", "No active turn accepted that control request."
                )
            bar.focus_input()
            return
        bar.history.append(text)
        bar.history_index = len(bar.history)
        bar.input.value = ""
        self._hide_welcome_banner()
        if text.startswith("/"):
            self.query_one(ChatLog).add_message("user", text)
            bar.hide_slash_suggestions()
            asyncio.create_task(self._handle_command(text)) #命令
            return
        self.query_one(ChatLog).add_message("user", text)
        self._run_agent(text) #如果是文本直接发给模型

    def on_key(self, event: Key) -> None:
        if self._ask_user_prompt is not None:
            if event.key in {"right", "down"}:
                self._ask_user_prompt.select_next()
                event.prevent_default()
            elif event.key in {"left", "up"}:
                self._ask_user_prompt.select_previous()
                event.prevent_default()
            elif event.key == "enter":
                self._resolve_ask_user(self._ask_user_prompt.selected_choice)
                event.prevent_default()
            elif event.key == "escape":
                self._resolve_ask_user("")
                event.prevent_default()
            return
        if self._confirm_prompt is not None:
            if event.key in {"y", "right"}:
                self._confirm_prompt.select_allow()
                event.prevent_default()
            elif event.key in {"n", "left"}:
                self._confirm_prompt.select_deny()
                event.prevent_default()
            elif event.key == "enter":
                self._resolve_confirm(self._confirm_prompt.selected)
                event.prevent_default()
            elif event.key == "escape":
                self._resolve_confirm(False)
                event.prevent_default()
            return
        bar = self.query_one(InputBar)
        if event.key == "tab" and bar.complete_slash_suggestion() or event.key == "up" and bar.move_slash_selection(-1) or event.key == "down" and bar.move_slash_selection(1):
            event.prevent_default()
        elif event.key == "escape":
            bar.hide_slash_suggestions()
            event.prevent_default()
        elif event.key == "up":
            bar.history_prev()
            event.prevent_default()
        elif event.key == "down":
            bar.history_next()
            event.prevent_default()

    async def _handle_command(self, text: str) -> None:
        handled, should_exit, output = await handle_repl_command_async(self.agent, text)
        if should_exit:
            self.exit()
            return
        if handled:
            self.query_one(ChatLog).add_message("assistant", output)
            self.query_one(StatusBar).update_agent(self.agent)
            return
        self.query_one(ChatLog).add_message(
            "assistant", f"Unknown command. Use /help.\n\n{HELP_DETAILS}"
        )

    def _run_agent(self, text: str, *, run_id="", after_seq=0) -> None:
        self.query_one(InputBar).set_busy(True)
        self.query_one(ThinkingIndicator).show()
        self._thinking_timer = self.set_interval(
            0.15, self.query_one(ThinkingIndicator).advance
        )
        self._active_turn_task = asyncio.create_task(
            self._agent_task(text, run_id=run_id, after_seq=after_seq)
        )

    def action_resume_run(self) -> None:
        if self._active_turn_task is not None or not self._active_run_id:
            return
        self._run_agent(
            "",
            run_id=self._active_run_id,
            after_seq=self._active_run_seq,
        )

    def _drain_idle_worker_notifications(self) -> None:
        if self._active_turn_task is not None:
            return
        notifications = self.agent.engine.drain_worker_notifications()
        if not notifications:
            return
        chat = self.query_one(ChatLog)
        for notification in notifications:
            chat.add_message("assistant", f"[worker notification]\n{notification}")
        self.query_one(StatusBar).update_agent(self.agent)

    async def _agent_task(self, text: str, *, run_id="", after_seq=0) -> None:
        completed = False
        pending_render = None
        try:
            if not run_id:
                run_id = await self.agent.engine.start_turn(text)
                after_seq = 0
            self._active_run_id = run_id
            self._active_run_seq = int(after_seq)
            retries = 0
            while True:
                try:
                    async for event in self.agent.engine.subscribe_turn(
                        run_id, self._active_run_seq
                    ):
                        self._active_run_seq = max(
                            self._active_run_seq,
                            int(event.get("run_seq", self._active_run_seq) or self._active_run_seq),
                        )
                        render = self._handle_runtime_event(dict(event))
                        if render is not None:
                            pending_render = render
                        if event.get("type") == "turn_finished":
                            completed = True
                            break
                    if completed or self.agent.session_event_bus.is_completed(run_id):
                        break
                    retries += 1
                    if retries > 3:
                        raise RuntimeError("Run event stream ended before turn_finished")
                    await asyncio.sleep(0.05 * retries)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    retries += 1
                    if retries > 3:
                        raise
                    await asyncio.sleep(0.05 * retries)
            if pending_render is not None:
                await pending_render
                self.query_one(ChatLog).scroll_end(animate=False)
            completed = True
        except Exception as exc:  # Keep an unexpected turn failure visible without killing the UI loop.  # noqa: BLE001
            if self.is_running:
                try:
                    self.query_one(ChatLog).add_message("assistant", f"[Error] {exc}")
                except NoMatches:
                    pass
        finally:
            self._finish_agent_task(completed)

    def _handle_runtime_event(self, event: dict):
        event_type = str(event.get("type", ""))
        if event_type == "text_delta":
            if self._streaming_message is None:
                self._streaming_message = self.query_one(ChatLog).add_message("assistant", "")
            content = self._streaming_message.content + str(event.get("content", ""))
            return self._streaming_message.update_content(content)
        if event_type == "context_building":
            self.query_one(ThinkingIndicator).set_detail("assembling context")
            return
        if event_type == "model_requested":
            attempts = event.get("attempts", 0)
            tool_steps = event.get("tool_steps", 0)
            self.query_one(ThinkingIndicator).set_detail(
                f"model request {attempts}, tools {tool_steps}"
            )
            return
        if event_type == "model_parsed":
            kind = event.get("kind", "")
            self.query_one(ThinkingIndicator).set_detail(f"model returned {kind}")
            if kind in {"tool", "tools"}:
                self._streaming_message = None
            return
        if event_type == "tool_call":
            name = str(event.get("name", ""))
            args = event.get("args") if isinstance(event.get("args"), dict) else {}
            self.query_one(ThinkingIndicator).set_detail(f"running {name}")
            card = self.query_one(ChatLog).add_tool_call(name, args)
            self._running_tool_cards.append(card)
            return
        if event_type == "tool_result":
            self._finish_tool_card(event)
            self.query_one(ThinkingIndicator).set_detail("thinking after tool")
            return
        if event_type == "worker_notification":
            self.query_one(ChatLog).add_message(
                "assistant", f"[worker notification]\n{event.get('content', '')}"
            )
            return
        if event_type == "worker_event":
            nested = event.get("event") if isinstance(event.get("event"), dict) else {}
            if nested.get("type") == "text_delta":
                self.query_one(ChatLog).add_message(
                    "assistant",
                    f"[{event.get('source', 'worker')}] {nested.get('content', '')}",
                )
            return
        if event_type in {"retry", "runtime_notice", "final", "stop"}:
            content = str(event.get("content", ""))
            if self._streaming_message is not None:
                render = self._streaming_message.update_content(content)
                self._streaming_message = None
                return render
            else:
                self.query_one(ChatLog).add_message("assistant", content)
            return

    def _finish_tool_card(self, event: dict) -> None:
        name = str(event.get("name", ""))
        card = None
        for candidate in reversed(self._running_tool_cards):
            if candidate.tool_name == name and candidate.status == "running":
                card = candidate
                break
        if card is None:
            card = self.query_one(ChatLog).add_tool_call(name, {})
        metadata = (
            event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        )
        content = str(event.get("content", ""))
        status = str(metadata.get("tool_status", "ok") or "ok")
        if status in {"error", "rejected", "partial_success"}:
            card.set_error(content)
        else:
            card.set_success(content)

    def _hide_welcome_banner(self) -> None:
        try:
            self.query_one(WelcomeBanner).add_class("hidden")
        except NoMatches:
            pass

    def _finish_agent_task(self, completed: bool) -> None:
        if not self.is_running:
            return
        try:
            self._stop_thinking()
            bar = self.query_one(InputBar)
            bar.set_busy(False)
            bar.focus_input()
            self._active_turn_task = None
            self._streaming_message = None
            if completed:
                self._turn_count += 1
                self._active_run_id = ""
                self._active_run_seq = 0
            status = self.query_one(StatusBar)
            status.update_turns(self._turn_count)
            status.update_agent(self.agent)
            usage = (getattr(self.agent, "last_prompt_metadata", {}) or {}).get(
                "context_usage"
            ) or {}
            status.update_context_usage(usage)
        except NoMatches:
            return

    def _stop_thinking(self) -> None:
        timer = getattr(self, "_thinking_timer", None)
        if timer is not None:
            timer.stop()
            self._thinking_timer = None
        try:
            self.query_one(ThinkingIndicator).hide()
        except NoMatches:
            pass

    async def _approval_callback_async(self, name: str, args: dict) -> bool:
        self._confirm_future = asyncio.get_running_loop().create_future()
        self._show_confirm(name, args)
        return bool(await self._confirm_future)

    def _show_confirm(self, name: str, args: dict) -> None:
        prompt = ConfirmPrompt(name, format_tool_args(name, args))
        self._confirm_prompt = prompt
        chat = self.query_one(ChatLog)
        chat.mount(prompt)
        chat.call_after_refresh(chat.scroll_end, animate=False)

    def _resolve_confirm(self, approved: bool) -> None:
        if self._confirm_future is None:
            return
        if not self._confirm_future.done():
            self._confirm_future.set_result(bool(approved))
        if self._confirm_prompt is not None:
            self._confirm_prompt.remove()
        self._confirm_prompt = None
        self._confirm_future = None

    async def _ask_user_callback_async(self, question: str, choices: list[str]) -> str:
        self._ask_user_future = asyncio.get_running_loop().create_future()
        self._show_ask_user(question, choices)
        return str(await self._ask_user_future)

    def _show_ask_user(self, question: str, choices: list[str]) -> None:
        prompt = AskUserPrompt(question, choices)
        self._ask_user_prompt = prompt
        chat = self.query_one(ChatLog)
        chat.mount(prompt)
        chat.call_after_refresh(chat.scroll_end, animate=False)

    def _resolve_ask_user(self, answer: str) -> None:
        if self._ask_user_future is None:
            return
        if not self._ask_user_future.done():
            self._ask_user_future.set_result(str(answer))
        if self._ask_user_prompt is not None:
            self._ask_user_prompt.remove()
        self._ask_user_prompt = None
        self._ask_user_future = None
