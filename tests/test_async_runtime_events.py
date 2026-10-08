import asyncio
import json
import threading

import pytest
from conftest import collect_stream_events

from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.core.runtime.session.runtime_locks import RuntimeLockManager
from gencode.core.runtime.session.session_events import SessionEventBus
from gencode.providers.base import ModelResult, ModelStreamEvent
from gencode.testing import ScriptedModelClient
from gencode.tools.base import RegisteredTool


def build_agent(tmp_path, model_client):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return GenCode(
        model_client=model_client,
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
        git_auto_commit=False,
        git_auto_undo=False,
    )


class ControlledStreamClient:
    supports_prompt_cache = False

    def __init__(self, followup):
        self.followup = followup
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.prompts = []

    async def stream_result(self, prompt, max_new_tokens, **kwargs):
        del max_new_tokens, kwargs
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            yield ModelStreamEvent("text_delta", "<final>discard this")
            self.started.set()
            await self.release.wait()
            text = "<final>discard this answer</final>"
        else:
            text = f"<final>{self.followup}</final>"
        yield ModelStreamEvent("text_delta", text)
        yield ModelStreamEvent("completed", result=ModelResult(text=text))


class IdleStreamClient:
    supports_prompt_cache = False

    async def stream_result(self, prompt, max_new_tokens, **kwargs):
        del prompt, max_new_tokens, kwargs
        await asyncio.Event().wait()
        yield ModelStreamEvent("completed", result=ModelResult(text=""))


def test_async_api_streams_visible_text_and_hides_tool_protocol(tmp_path):
    agent = build_agent(
        tmp_path,
        ScriptedModelClient(
            [
                '<tool name="write_file" path="result.txt"><content>ok</content></tool>',
                "<final>Finished.</final>",
            ]
        ),
    )

    events = collect_stream_events(agent.engine.run_turn("write result"))

    visible = "".join(event["content"] for event in events if event["type"] == "text_delta")
    assert visible == "Finished."
    assert all("<tool" not in event["content"] for event in events if event["type"] == "text_delta")
    assert [event["run_seq"] for event in events] == sorted(
        event["run_seq"] for event in events
    )
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "ok"


def test_sync_ask_rejects_running_event_loop_and_async_ask_works(tmp_path):
    agent = build_agent(tmp_path, ScriptedModelClient(["<final>Done.</final>"]))

    async def run():
        with pytest.raises(RuntimeError, match="use await ask_async"):
            agent.ask("sync call inside event loop")
        return await agent.ask_async("async call")

    assert asyncio.run(run()) == "Done."


@pytest.mark.parametrize("control,expected", [("steer", "new direction"), ("queue", "second task")])
def test_turn_controls_steer_and_queue(tmp_path, control, expected):
    client = ControlledStreamClient(expected)
    agent = build_agent(tmp_path, client)

    async def run():
        turn = asyncio.create_task(agent.ask_async("first task"))
        await asyncio.wait_for(client.started.wait(), 2)
        if control == "steer":
            assert agent.steer(expected)
        else:
            assert agent.queue_turn(expected)
        client.release.set()
        return await asyncio.wait_for(turn, 5)

    assert asyncio.run(run()) == expected
    assert len(client.prompts) == 2


def test_concurrent_turn_producers_keep_run_id_queues_isolated(tmp_path):
    client = ControlledStreamClient("second answer")
    agent = build_agent(tmp_path, client)

    async def run():
        first = asyncio.create_task(agent.ask_async("first request"))
        await asyncio.wait_for(client.started.wait(), 2)
        second = asyncio.create_task(agent.ask_async("second request"))
        await asyncio.sleep(0.05)
        client.release.set()
        return await asyncio.wait_for(asyncio.gather(first, second), 5)

    assert asyncio.run(run()) == ["discard this answer", "second answer"]


def test_cancel_stops_turn_and_persists_terminal_state(tmp_path):
    client = ControlledStreamClient("unused")
    agent = build_agent(tmp_path, client)

    async def run():
        turn = asyncio.create_task(agent.ask_async("long request"))
        await asyncio.wait_for(client.started.wait(), 2)
        assert agent.cancel_current_turn()
        return await asyncio.wait_for(turn, 5)

    result = asyncio.run(run())
    assert "Stopped after cancel request" in result
    assert agent.current_task_state.status == "stopped"
    assert agent.current_task_state.stop_reason == "aborted"


@pytest.mark.parametrize(
    "turn_timeout,idle_timeout,reason",
    [(2, 0.02, "model_idle_timeout"), (0.02, 2, "timeout")],
)
def test_model_idle_and_turn_timeouts_persist_stopped_state(
    tmp_path, turn_timeout, idle_timeout, reason
):
    agent = build_agent(tmp_path, IdleStreamClient())
    agent.max_turn_seconds = turn_timeout
    agent.model_idle_timeout = idle_timeout

    result = asyncio.run(agent.ask_async("wait for a model that never streams"))

    assert result
    assert agent.current_task_state.status == "stopped"
    assert agent.current_task_state.stop_reason == reason


def test_tool_timeout_returns_structured_tool_error(tmp_path):
    agent = build_agent(tmp_path, ScriptedModelClient([]))
    original = agent.tools["read_file"]

    async def slow_read(args):
        del args
        await asyncio.sleep(1)
        return "too late"

    agent.tools["read_file"] = RegisteredTool(
        name=original.name,
        schema=original.schema,
        description=original.description,
        risky=original.risky,
        runner=slow_read,
        capability=original.capability,
    )
    agent.tool_timeout_seconds = 0.02

    result = asyncio.run(agent.run_tool("read_file", {"path": "README.md"}))

    assert "tool read_file failed" in result
    assert agent._last_tool_result_metadata["tool_error_code"] == "tool_timeout"


def test_cancel_propagates_to_an_active_tool_and_keeps_its_evidence(tmp_path):
    agent = build_agent(
        tmp_path,
        ScriptedModelClient(
            [
                '<tool name="read_file" path="README.md"></tool>',
                "<final>Should not continue.</final>",
            ]
        ),
    )
    original = agent.tools["read_file"]
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def waiting_read(args):
        del args
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    agent.tools["read_file"] = RegisteredTool(
        name=original.name,
        schema=original.schema,
        description=original.description,
        risky=original.risky,
        runner=waiting_read,
        capability=original.capability,
    )

    async def run():
        turn = asyncio.create_task(agent.ask_async("read then cancel"))
        await asyncio.wait_for(started.wait(), 2)
        assert agent.cancel_current_turn()
        return await asyncio.wait_for(turn, 5)

    result = asyncio.run(run())
    assert "Stopped after cancel request" in result
    assert cancelled.is_set()
    assert agent._last_tool_result_metadata["tool_error_code"] == "tool_cancelled"
    assert any(
        item.get("role") == "tool" and item.get("tool_error_code") == "tool_cancelled"
        for item in agent.session["history"]
    )


def test_cancelled_sync_read_adapter_does_not_hold_up_turn(tmp_path, monkeypatch):
    from gencode.tools import registry
    from gencode.tools.base import ToolCapability, ToolEffect

    started = threading.Event()
    release = threading.Event()

    def blocked_read(agent, args):
        del agent, args
        started.set()
        release.wait(2)
        return "read finished after cancellation"

    monkeypatch.setitem(registry._TOOL_RUNNERS, "read_file", blocked_read)
    agent = type(
        "ReadOnlyAgent",
        (),
        {
            "tools": {
                "read_file": RegisteredTool(
                    name="read_file",
                    schema={},
                    description="read-only",
                    risky=False,
                    runner=lambda args: asyncio.sleep(0, result=str(args)),
                    capability=ToolCapability(ToolEffect.READ, True),
                )
            }
        },
    )()

    async def run():
        task = asyncio.create_task(
            registry._async_tool_runner("read_file", agent, {})
        )
        await asyncio.wait_for(asyncio.to_thread(started.wait), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 0.5)
        assert not release.is_set()
        release.set()

    asyncio.run(run())


def test_session_event_bus_replay_cursor_source_and_run_sequence(tmp_path):
    bus = SessionEventBus("session", tmp_path / "events.jsonl")

    async def run():
        first = await bus.publish("started", {"run_id": "run-1"})
        stream = bus.subscribe("run-1", after_seq=first["run_seq"])
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        second = await bus.publish(
            "worker_update", {"run_id": "run-1", "source": "agent_2"}
        )
        received = await asyncio.wait_for(pending, 2)
        await stream.aclose()
        return first, second, received

    first, second, received = asyncio.run(run())
    assert first["run_seq"] == 1
    assert second["run_seq"] == 2
    assert received["event"] == "worker_update"
    assert received["source"] == "agent_2"
    replay = bus.replay("run-1")
    assert [event["run_seq"] for event in replay] == [1, 2]
    assert json.loads((tmp_path / "events.jsonl").read_text().splitlines()[1])["source"] == "agent_2"


def test_event_without_explicit_run_id_inherits_active_run(tmp_path):
    bus = SessionEventBus("session", tmp_path / "events.jsonl")

    async def run():
        bus.set_active_run("run-implicit")
        bus.emit("permission_decision", {"decision": "allow"})
        await bus.publish("turn_started", {"run_id": "run-implicit"})

    asyncio.run(run())
    records = bus.replay("run-implicit")
    assert [record["event"] for record in records] == [
        "permission_decision",
        "turn_started",
    ]
    assert [record["run_seq"] for record in records] == [1, 2]


def test_cancelled_publish_still_finishes_its_queued_write(tmp_path, monkeypatch):
    bus = SessionEventBus("session", tmp_path / "events.jsonl")
    writer_started = threading.Event()
    release_writer = threading.Event()
    original_record = bus._record

    def blocked_record(event, payload):
        if event == "first":
            writer_started.set()
            release_writer.wait(2)
        return original_record(event, payload)

    monkeypatch.setattr(bus, "_record", blocked_record)

    async def run():
        first = asyncio.create_task(bus.publish("first", {"run_id": "run-cancel"}))
        await asyncio.wait_for(asyncio.to_thread(writer_started.wait), 2)
        second = asyncio.create_task(bus.publish("second", {"run_id": "run-cancel"}))
        await asyncio.sleep(0)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        release_writer.set()
        await first

    asyncio.run(run())
    assert [record["event"] for record in bus.replay("run-cancel")] == [
        "first",
        "second",
    ]


def test_detached_turn_can_resume_after_stream_consumer_disconnects(tmp_path):
    client = ControlledStreamClient("resumed answer")
    agent = build_agent(tmp_path, client)

    async def run():
        stream = agent.engine.run_turn("finish after the view disconnects")
        first_event = await anext(stream)
        run_id = first_event["run_id"]
        await asyncio.wait_for(client.started.wait(), 2)
        await stream.aclose()
        client.release.set()
        resumed = [
            event
            async for event in agent.engine.subscribe_run(
                run_id, after_seq=first_event["run_seq"]
            )
        ]
        return first_event, resumed

    first, resumed = asyncio.run(run())
    assert first["type"] == "turn_started"
    assert resumed[-1]["type"] == "turn_finished"
    assert any(
        event["type"] == "final" and event["content"] == "discard this answer"
        for event in resumed
    )
    assert all(event["run_seq"] > first["run_seq"] for event in resumed)


def test_session_event_bus_marks_and_recovers_log_gap(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    bus = SessionEventBus("session", path)
    asyncio.run(bus.publish("before", {"run_id": "run-1"}))
    original_open = type(path).open

    def fail_append(target, mode="r", *args, **kwargs):
        if target == path and "a" in mode:
            raise OSError("event log unavailable")
        return original_open(target, mode, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(type(path), "open", fail_append)
        failed = asyncio.run(bus.publish("during_gap", {"run_id": "run-1"}))
        assert failed["event_log_degraded"] is True
    recovered = asyncio.run(bus.publish("after_gap", {"run_id": "run-1"}))
    assert recovered["event_log_gap_recovered"] is True
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["before", "event_log_gap", "after_gap"]
    assert events[1]["missing_from"] == events[1]["missing_to"] == 2
    assert recovered["run_seq"] == 4

    async def replay_gap():
        stream = bus.subscribe("run-1", after_seq=1)
        replayed = [await anext(stream) for _ in range(3)]
        await stream.aclose()
        return replayed

    replayed = asyncio.run(replay_gap())
    assert [event["event"] for event in replayed] == [
        "during_gap",
        "event_log_gap",
        "after_gap",
    ]
    assert [event["run_seq"] for event in replayed] == [2, 3, 4]


def test_event_log_failure_does_not_stop_turn_and_reaches_run_report(
    tmp_path, monkeypatch
):
    agent = build_agent(tmp_path, ScriptedModelClient(["<final>Still done.</final>"]))
    path = agent.session_event_bus.path
    original_open = type(path).open

    def fail_append(target, mode="r", *args, **kwargs):
        if target == path and "a" in mode:
            raise OSError("event log unavailable")
        return original_open(target, mode, *args, **kwargs)

    monkeypatch.setattr(type(path), "open", fail_append)
    assert asyncio.run(agent.ask_async("complete despite event log failure")) == "Still done."
    report = agent.run_store.load_report(agent.current_task_state.run_id)
    assert agent.current_task_state.event_log_degraded is True
    assert report["task_state"]["event_log_degraded"] is True
    assert report["task_state"]["event_log_gap"]["missing_from"] > 0


def test_failed_terminal_event_can_be_resumed_from_volatile_run_buffer(
    tmp_path, monkeypatch
):
    path = tmp_path / "events.jsonl"
    bus = SessionEventBus("session", path)
    asyncio.run(bus.publish("turn_started", {"run_id": "run-gap"}))
    original_open = type(path).open

    def fail_append(target, mode="r", *args, **kwargs):
        if target == path and "a" in mode:
            raise OSError("event log unavailable")
        return original_open(target, mode, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(type(path), "open", fail_append)
        asyncio.run(bus.publish("turn_finished", {"run_id": "run-gap"}))

    async def resume():
        return [event async for event in bus.subscribe("run-gap", after_seq=1)]

    resumed = asyncio.run(resume())
    assert len(resumed) == 1
    assert resumed[0]["event"] == "turn_finished"
    assert resumed[0]["event_log_degraded"] is True


def test_provider_rejects_sync_only_clients_on_async_stream_path():
    from gencode.providers.base import stream_model

    class SyncOnlyClient:
        def complete(self, prompt, max_new_tokens):
            return "not a stream"

    async def run():
        with pytest.raises(TypeError, match="async stream_result"):
            async for _ in stream_model(SyncOnlyClient(), "prompt", 10):
                pass

    asyncio.run(run())


def test_unexpected_runtime_failure_still_closes_detached_run(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, ScriptedModelClient(["<final>unused</final>"]))

    async def fail_prompt_build(request):
        del request
        raise RuntimeError("prompt assembly failed")

    monkeypatch.setattr(agent, "_build_prompt_and_metadata_async", fail_prompt_build)

    async def run():
        return await asyncio.wait_for(agent.ask_async("trigger internal failure"), 3)

    answer = asyncio.run(run())
    run_id = agent.current_task_state.run_id
    replay = agent.session_event_bus.replay(run_id)
    report = agent.run_store.load_report(run_id)

    assert "RuntimeError" in answer
    assert replay[-1]["event"] == "turn_finished"
    assert replay[-2]["event"] == "stop"
    assert report["stop_reason"] == "runtime_error"
    assert any(event["event"] == "runtime_error" for event in replay)


def test_cli_turn_consumes_run_subscription_with_a_resume_cursor(tmp_path, capsys):
    from gencode.cli import _consume_cli_turn

    agent = build_agent(
        tmp_path, ScriptedModelClient(["<final>CLI resumed stream.</final>"])
    )

    run_id, cursor = asyncio.run(_consume_cli_turn(agent, "show the answer"))

    assert run_id == agent.current_task_state.run_id
    assert cursor > 0
    assert "CLI resumed stream." in capsys.readouterr().out
    assert agent.session_event_bus.is_completed(run_id)


@pytest.mark.parametrize(
    ("streamed", "final"),
    [
        ("I'll inspect the worker files.", "Complete answer."),
        ("Complete answer.", "Complete answer."),
        ("<final>Complete answer.</final>", "Complete answer."),
    ],
)
def test_cli_does_not_let_progress_text_hide_or_duplicate_final(streamed, final, capsys):
    from gencode.cli import _consume_cli_turn

    class Engine:
        async def start_turn(self, _prompt):
            return "run-cli-final"

        async def subscribe_turn(self, _run_id, _cursor):
            yield {"type": "text_delta", "content": streamed, "run_seq": 1}
            yield {"type": "stop", "content": final, "run_seq": 2}
            yield {"type": "turn_finished", "run_seq": 3}

    class EventBus:
        @staticmethod
        def is_completed(_run_id):
            return False

    agent = type(
        "Agent", (), {"engine": Engine(), "session_event_bus": EventBus()}
    )()
    asyncio.run(_consume_cli_turn(agent, "explain workers"))
    output = capsys.readouterr().out

    assert streamed in output
    assert output.count(final) == 1


def test_workspace_readers_share_lock_and_writer_waits(tmp_path):
    database = tmp_path / "runtime-locks.sqlite3"
    first = RuntimeLockManager(database)
    second = RuntimeLockManager(database)

    async def run():
        read_one = await first.acquire_workspace(tmp_path, "read")
        await read_one.__aenter__()
        read_two = await second.acquire_workspace(tmp_path, "read")
        await read_two.__aenter__()
        writer_task = asyncio.create_task(second.acquire_workspace(tmp_path, "write"))
        await asyncio.sleep(0.1)
        assert not writer_task.done()
        await read_one.__aexit__(None, None, None)
        await asyncio.sleep(0.1)
        assert not writer_task.done()
        await read_two.__aexit__(None, None, None)
        writer = await asyncio.wait_for(writer_task, 2)
        await writer.__aenter__()
        await writer.__aexit__(None, None, None)

    asyncio.run(run())
