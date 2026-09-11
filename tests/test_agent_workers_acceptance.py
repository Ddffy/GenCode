import json
import threading
import time

from gencode.testing import ScriptedModelClient
from gencode import GenCode, SessionStore, WorkspaceContext


def build_agent(tmp_path, outputs, **kwargs):
    (tmp_path / "README.md").write_text("demo readme\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".gencode" / "sessions")
    return GenCode(
        model_client=ScriptedModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        **kwargs,
    )


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class BlockingModelClient:
    def __init__(self, outputs, started, release):
        self.outputs = list(outputs)
        self.started = started
        self.release = release
        self.prompts = []
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}
        self.abort_count = 0

    def complete(self, prompt, max_new_tokens, **kwargs):
        self.prompts.append(prompt)
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("blocking test client timed out")
        if not self.outputs:
            raise RuntimeError("scripted model ran out of outputs")
        return self.outputs.pop(0)

    def abort(self):
        self.abort_count += 1
        self.release.set()


def test_delegate_is_removed_from_runtime_tool_surface(tmp_path):
    agent = build_agent(tmp_path, [])

    assert "delegate" not in agent.tools
    assert "delegate" not in agent.available_tools()
    assert '"name":"delegate"' not in agent.prefix
    assert "- delegate(" not in agent.prefix
    assert not hasattr(agent, "tool_delegate")


def test_async_worker_notification_is_drained_by_coordinator_only(tmp_path):
    started = threading.Event()
    release = threading.Event()
    child_client = BlockingModelClient(["<final>Child done.</final>"], started, release)
    agent = build_agent(
        tmp_path,
        [],
        model_client_factory=lambda: child_client,
    )

    before = time.monotonic()
    payload = json.loads(
        agent.run_tool(
            "agent",
            {
                "description": "Background read",
                "prompt": "Summarize README",
                "subagent_type": "Explore",
            },
        )
    )

    assert payload["status"] == "started"
    startup_limit = 1.0 if __import__("os").name == "nt" else 0.5
    assert time.monotonic() - before < startup_limit
    assert started.wait(timeout=1)
    assert not any(
        "<task-notification>" in item.get("content", "")
        for item in agent.session["history"]
    )

    release.set()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if agent.worker_manager.to_dict()["items"][0]["status"] == "completed":
            break
        time.sleep(0.01)

    drained = agent.engine.drain_worker_notifications()

    assert len(drained) == 1
    assert "<task-id>agent_1</task-id>" in drained[0]
    assert any(
        "<task-notification>" in item.get("content", "")
        for item in agent.session["history"]
    )
    assert agent.engine.drain_worker_notifications() == []
    assert agent.worker_manager.to_dict()["items"][0]["notification_drained"] is True


def test_send_message_rejects_running_worker(tmp_path):
    started = threading.Event()
    release = threading.Event()
    agent = build_agent(
        tmp_path,
        [],
        model_client_factory=lambda: BlockingModelClient(
            ["<final>Child done.</final>"], started, release
        ),
    )

    agent.run_tool(
        "agent",
        {
            "description": "Still running",
            "prompt": "Wait for release",
            "subagent_type": "Explore",
        },
    )
    assert started.wait(timeout=1)

    rejected = agent.run_tool(
        "send_message", {"to": "agent_1", "message": "Continue now"}
    )

    release.set()
    assert "worker is running" in rejected


def test_task_stop_requests_child_runtime_abort(tmp_path):
    started = threading.Event()
    release = threading.Event()
    child_client = BlockingModelClient(["<final>Child done.</final>"], started, release)
    agent = build_agent(
        tmp_path,
        [],
        model_client_factory=lambda: child_client,
    )

    agent.run_tool(
        "agent",
        {
            "description": "Abort me",
            "prompt": "Wait until stopped",
            "subagent_type": "Explore",
        },
    )
    assert started.wait(timeout=1)

    payload = json.loads(agent.run_tool("task_stop", {"task_id": "agent_1"}))

    assert payload["status"] == "stopping"
    assert child_client.abort_count == 1
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if agent.worker_manager.to_dict()["items"][0]["status"] == "stopped":
            break
        time.sleep(0.01)
    assert agent.worker_manager.to_dict()["items"][0]["status"] == "stopped"


def test_clear_session_stops_running_background_workers(tmp_path):
    started = threading.Event()
    release = threading.Event()
    child_client = BlockingModelClient(["<final>Child done.</final>"], started, release)
    agent = build_agent(
        tmp_path,
        [],
        model_client_factory=lambda: child_client,
    )

    agent.run_tool(
        "agent",
        {
            "description": "Clear me",
            "prompt": "Wait until clear",
            "subagent_type": "Explore",
        },
    )
    assert started.wait(timeout=1)
    old_id = agent.session["id"]

    new_id = agent.clear_session()

    assert new_id != old_id
    assert child_client.abort_count == 1
    assert agent.worker_manager.to_dict()["items"] == []
    assert agent.engine.drain_worker_notifications() == []


def test_explore_agent_runs_real_readonly_child_session_and_records_notification(
    tmp_path,
):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"agent","args":{"description":"Inspect readme","prompt":"Read README.md and summarize it","subagent_type":"Explore"}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":1}}</tool>',
            "<final>README says demo readme.</final>",
            "<final>Exploration complete.</final>",
        ],
        max_steps=4,
    )

    assert agent.ask("inspect with a subagent") == "Exploration complete."

    notifications = [
        item
        for item in agent.session["history"]
        if item["role"] == "user" and "<task-notification>" in item["content"]
    ]
    assert len(notifications) == 1
    assert "<task-id>agent_1</task-id>" in notifications[0]["content"]
    assert "<status>completed</status>" in notifications[0]["content"]
    assert "README says demo readme." in notifications[0]["content"]

    events = read_jsonl(agent.session_event_bus.path)
    assert any(
        event["event"] == "worker_started" and event["worker_id"] == "agent_1"
        for event in events
    )
    assert any(
        event["event"] == "worker_finished" and event["worker_id"] == "agent_1"
        for event in events
    )
    report = json.loads(
        (agent.current_run_dir / "report.json").read_text(encoding="utf-8")
    )
    assert report["workers"]["items"][0]["id"] == "agent_1"
    assert report["workers"]["items"][0]["subagent_type"] == "Explore"


def test_worker_agent_can_be_continued_with_same_child_context_and_write_scope(
    tmp_path,
):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"agent","args":{"description":"Write notes","prompt":"Create the first note","subagent_type":"worker","write_scope":["notes"]}}</tool>',
            '<tool name="write_file" path="notes/first.txt"><content>first\n</content></tool>',
            "<final>First note written.</final>",
            '<tool>{"name":"send_message","args":{"to":"agent_1","message":"Create the second note"}}</tool>',
            '<tool name="write_file" path="notes/second.txt"><content>second\n</content></tool>',
            "<final>Second note written.</final>",
            "<final>Both worker steps are done.</final>",
        ],
        max_steps=5,
    )

    assert agent.ask("use a worker twice") == "Both worker steps are done."

    assert (tmp_path / "notes" / "first.txt").read_text(encoding="utf-8") == "first\n"
    assert (tmp_path / "notes" / "second.txt").read_text(encoding="utf-8") == "second\n"
    assert agent.model_client.prompts[4].count("First note written.") >= 1

    notifications = [
        item
        for item in agent.session["history"]
        if item["role"] == "user" and "<task-notification>" in item["content"]
    ]
    assert len(notifications) == 2
    assert all(
        "<task-id>agent_1</task-id>" in item["content"] for item in notifications
    )
    events = read_jsonl(agent.session_event_bus.path)
    assert (
        sum(
            1
            for event in events
            if event["event"] == "worker_started" and event["worker_id"] == "agent_1"
        )
        == 2
    )


def test_worker_write_scope_blocks_child_file_modification_outside_scope(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"agent","args":{"description":"Bad write","prompt":"Write outside scope","subagent_type":"worker","write_scope":["allowed"]}}</tool>',
            '<tool name="write_file" path="forbidden/out.txt"><content>no\n</content></tool>',
            "<final>Write was blocked.</final>",
            "<final>Worker reported the blocked write.</final>",
        ],
        max_steps=4,
    )

    assert agent.ask("try a scoped worker") == "Worker reported the blocked write."

    assert not (tmp_path / "forbidden" / "out.txt").exists()
    notification = next(
        item
        for item in agent.session["history"]
        if item["role"] == "user" and "<task-notification>" in item["content"]
    )
    assert "Write was blocked." in notification["content"]


def test_worker_without_write_scope_cannot_modify_workspace(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"agent","args":{"description":"No scope","prompt":"Write without scope","subagent_type":"worker"}}</tool>',
            '<tool name="write_file" path="notes/out.txt"><content>no\n</content></tool>',
            "<final>Write was blocked.</final>",
            "<final>Worker respected missing scope.</final>",
        ],
        max_steps=4,
    )

    assert agent.ask("try an unscoped worker") == "Worker respected missing scope."

    assert not (tmp_path / "notes" / "out.txt").exists()


def test_plan_mode_cannot_continue_write_capable_worker(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"agent","args":{"description":"Worker","prompt":"Read only first","subagent_type":"worker","write_scope":["notes"]}}</tool>',
            "<final>Worker ready.</final>",
            "<final>Coordinator done.</final>",
        ],
        max_steps=3,
    )

    assert agent.ask("create a worker") == "Coordinator done."
    agent.enter_plan_mode("gate7")

    rejected = agent.run_tool(
        "send_message", {"to": "agent_1", "message": "Write notes/out.txt"}
    )

    assert "plan mode only allows Explore agents" in rejected
    assert not (tmp_path / "notes" / "out.txt").exists()


def test_plan_mode_allows_only_explore_agents(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"agent","args":{"description":"Explore plan","prompt":"Read README","subagent_type":"Explore"}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":1}}</tool>',
            "<final>Explored.</final>",
            '<tool name="write_file" path=".gencode/plans/gate7-plan.md"><content># Gate7\n</content></tool>',
            "<final>Plan ready.</final>",
        ],
        max_steps=5,
    )

    agent.enter_plan_mode("gate7")
    rejected = agent.run_tool(
        "agent",
        {
            "description": "Write from plan",
            "prompt": "change files",
            "subagent_type": "worker",
            "write_scope": ["gencode"],
        },
    )

    assert "plan mode only allows Explore agents" in rejected
    assert agent.ask("plan with explore") == "Plan ready."
    assert agent.active_tool_profile.name == "default"
