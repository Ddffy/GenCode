import asyncio

from conftest import collect_events

from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.core.parallel_tools import can_parallelize_tool_batch
from gencode.providers import ModelResult
from gencode.testing import NativeScriptedModelClient
from gencode.tools.base import (
    RegisteredTool,
    ToolCapability,
    ToolEffect,
)


def build_agent(tmp_path, tool_calls):
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    client = NativeScriptedModelClient(
        [
            ModelResult(
                text="",
                metadata={"native_tool_calling": True},
                tool_calls=tuple(tool_calls),
                stop_reason="tool_use",
            ),
            ModelResult(text="done", metadata={"native_tool_calling": True}),
        ]
    )
    return GenCode(
        model_client=client,
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
    )


def read_calls():
    return [
        {"id": "call-a", "name": "read_file", "args": {"path": "a.txt"}},
        {"id": "call-b", "name": "read_file", "args": {"path": "b.txt"}},
    ]


def test_safe_reads_overlap_but_commit_in_source_order(tmp_path):
    agent = build_agent(tmp_path, read_calls())
    release = asyncio.Event()
    active = 0
    peak = 0

    async def overlapping_read(args):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            release.set()
        await asyncio.wait_for(release.wait(), 2)
        if args["path"] == "a.txt":
            await asyncio.sleep(0.05)
        active -= 1
        return f"result:{args['path']}"

    agent.tools["read_file"] = RegisteredTool(
        name="read_file",
        schema=agent.tools["read_file"].schema,
        description="test read",
        risky=False,
        runner=overlapping_read,
        capability=agent.tools["read_file"].capability,
    )

    events = collect_events(agent.engine.run_turn("read both files"))

    assert peak == 2
    tool_events = [
        event["type"] for event in events
        if event["type"] in {"tool_call", "tool_result"}
    ]
    assert tool_events == ["tool_call", "tool_call", "tool_result", "tool_result"]
    history = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert [item["tool_call_id"] for item in history] == ["call-a", "call-b"]
    assert [item["content"] for item in history] == ["result:a.txt", "result:b.txt"]
    assert all(item["tool_status"] == "ok" for item in history)


def test_one_parallel_read_failure_does_not_drop_sibling(tmp_path):
    agent = build_agent(tmp_path, read_calls())

    async def read_or_fail(args):
        if args["path"] == "a.txt":
            raise OSError("unreadable")
        return "result:b.txt"

    agent.tools["read_file"] = RegisteredTool(
        name="read_file",
        schema=agent.tools["read_file"].schema,
        description="test read",
        risky=False,
        runner=read_or_fail,
        capability=agent.tools["read_file"].capability,
    )

    collect_events(agent.engine.run_turn("read both files"))

    history = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert [item["tool_call_id"] for item in history] == ["call-a", "call-b"]
    assert history[0]["tool_status"] == "error"
    assert "unreadable" in history[0]["content"]
    assert history[1]["tool_status"] == "ok"
    assert history[1]["content"] == "result:b.txt"


def test_steer_cancels_active_parallel_reads_and_skips_waiting_reads(tmp_path):
    calls = [
        {"id": f"call-{index}", "name": "read_file", "args": {"path": f"{index}.txt"}}
        for index in range(5)
    ]
    model = NativeScriptedModelClient(
        [
            ModelResult(
                text="",
                metadata={"native_tool_calling": True},
                tool_calls=tuple(calls),
                stop_reason="tool_use",
            ),
            ModelResult(text="steered answer", metadata={"native_tool_calling": True}),
        ]
    )
    agent = GenCode(
        model_client=model,
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
        git_auto_commit=False,
        git_auto_undo=False,
    )
    for index in range(5):
        (tmp_path / f"{index}.txt").write_text(str(index), encoding="utf-8")

    active = 0
    started_paths = []
    all_started = asyncio.Event()
    cancelled_count = 0
    original = agent.tools["read_file"]

    async def blocked_read(args):
        nonlocal active, cancelled_count
        active += 1
        started_paths.append(args["path"])
        if active == 4:
            all_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1
            cancelled_count += 1

    agent.tools["read_file"] = RegisteredTool(
        name=original.name,
        schema=original.schema,
        description=original.description,
        risky=original.risky,
        runner=blocked_read,
        capability=original.capability,
    )

    async def run():
        turn = asyncio.create_task(agent.ask_async("read five files"))
        await asyncio.wait_for(all_started.wait(), 2)
        assert agent.steer("change direction")
        answer = await asyncio.wait_for(turn, 5)
        return answer

    assert asyncio.run(run()) == "steered answer"
    assert len(started_paths) == 4
    assert "4.txt" not in started_paths
    assert cancelled_count == 4
    history = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert len(history) == 5
    assert sum(item["tool_error_code"] == "steered" for item in history) == 1
    assert sum(item["tool_error_code"] == "tool_cancelled" for item in history) == 4


def test_parallel_gate_preserves_serial_fallbacks(tmp_path):
    agent = build_agent(tmp_path, read_calls())
    safe = read_calls()
    mixed = [safe[0], {"name": "write_file", "args": {"path": "x", "content": "x"}}]
    duplicate = [safe[0], dict(safe[0], id="call-a-duplicate")]

    assert can_parallelize_tool_batch(agent, safe, remaining_steps=2)
    assert not can_parallelize_tool_batch(agent, mixed, remaining_steps=2)
    assert not can_parallelize_tool_batch(agent, duplicate, remaining_steps=2)
    assert not can_parallelize_tool_batch(agent, safe, remaining_steps=1)

    async def inside_event_loop():
        return can_parallelize_tool_batch(agent, safe, remaining_steps=2)

    assert asyncio.run(inside_event_loop()) is True


def test_capability_contract_is_fail_closed_and_rejects_misdeclared_writes(tmp_path):
    agent = build_agent(tmp_path, read_calls())

    for name in ("read_file", "list_files", "search"):
        capability = agent.tools[name].capability
        assert capability.effect is ToolEffect.READ
        assert capability.concurrency_safe is True

    definition = next(
        item for item in agent.native_tool_definitions()
        if item["name"] == "read_file"
    )
    assert set(definition) == {"name", "description", "parameters"}

    # A read-only-looking tool is still serial unless it explicitly opts in.
    assert agent.tools["knowledge_read"].capability == ToolCapability()

    # Defense in depth: concurrency_safe alone cannot make a writer parallel.
    original = agent.tools["read_file"]
    agent.tools["read_file"] = RegisteredTool(
        name="read_file",
        schema=original.schema,
        description="misdeclared writer",
        risky=False,
        runner=lambda args: _async_value(str(args)),
        capability=ToolCapability(
            effect=ToolEffect.WRITE,
            concurrency_safe=True,
        ),
    )
    assert not can_parallelize_tool_batch(
        agent, read_calls(), remaining_steps=2
    )

    # The legacy risky/read_only contract remains an independent safety gate.
    agent.tools["read_file"] = RegisteredTool(
        name="read_file",
        schema=original.schema,
        description="risky tool incorrectly declared as a safe read",
        risky=True,
        runner=lambda args: _async_value(str(args)),
        capability=ToolCapability(
            effect=ToolEffect.READ,
            concurrency_safe=True,
        ),
    )
    assert not can_parallelize_tool_batch(
        agent, read_calls(), remaining_steps=2
    )


async def _async_value(value):
    return value
