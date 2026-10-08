import json
import subprocess
from datetime import datetime, timezone

import pytest

from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.cli import handle_repl_command_async
from gencode.commands.slash import resolve_command, suggest_commands
from gencode.core.runtime.goal_graph import GoalPlanError, validate_plan
from gencode.core.runtime.goal_manager import (
    _has_only_scoped_changes,
    _verifier_process_args,
    _worker_failure_summary,
)
from gencode.providers.errors import ProviderError
from gencode.testing import ScriptedModelClient


def _git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    )


def _init_repo(root, *, require_marker=False):
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "gencode-goal-tests@example.com")
    _git(root, "config", "user.name", "GenCode Goal Tests")
    (root / "README.md").write_text("Goal fixture\n", encoding="utf-8")
    test_assertion = (
        "    assert 'goal-ready' in Path('README.md').read_text()\n"
        if require_marker
        else "    assert Path('README.md').exists()\n"
    )
    (root / "test_goal_fixture.py").write_text(
        "from pathlib import Path\n\n"
        "def test_fixture_is_valid():\n"
        + test_assertion,
        encoding="utf-8",
    )
    _git(root, "add", "README.md", "test_goal_fixture.py")
    _git(root, "commit", "-qm", "fixture baseline")


def _plan(*, dependencies=None, prompt="Update README.md to include goal-ready."):
    return {
        "nodes": [
            {
                "id": "update_readme",
                "title": "Update the README",
                "prompt": prompt,
                "depends_on": dependencies or [],
                "write_scope": ["README.md"],
                "acceptance": ["README.md contains goal-ready"],
            }
        ],
        "verifiers": [["python", "-m", "pytest", "-q"]],
    }


def _final(value):
    return f"<final>{json.dumps(value, ensure_ascii=False)}</final>"


class _ClientFactory:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if not self.outputs:
            raise AssertionError("unexpected extra model client request")
        return ScriptedModelClient(self.outputs.pop(0))


def _agent(root, factory):
    return GenCode(
        model_client=ScriptedModelClient([]),
        model_client_factory=factory,
        workspace=WorkspaceContext.build(root, repo_root_override=root),
        session_store=SessionStore(root / ".gencode" / "sessions"),
        approval_policy="auto",
        max_steps=12,
    )


def test_goal_explore_runtime_uses_parent_step_budget(tmp_path):
    _init_repo(tmp_path)
    agent = _agent(tmp_path, _ClientFactory([[]]))
    agent.max_steps = 64

    planner = agent.goal_manager._role_runtime("Explore", agent.workspace)

    assert planner.max_steps == 6
    assert planner.max_new_tokens == 2048


def test_goal_worker_runtime_has_scoped_write_tools_but_no_shell(tmp_path):
    _init_repo(tmp_path)
    agent = _agent(tmp_path, _ClientFactory([[]]))
    agent.max_steps = 100

    worker = agent.goal_manager._role_runtime(
        "worker", agent.workspace, write_scope=["README.md"]
    )

    assert worker.read_only is False
    assert {"write_file", "patch_file"} <= set(worker.available_tools())
    assert "run_shell" not in worker.available_tools()
    assert worker.write_scope == ("README.md",)
    assert worker.max_steps == 32
    assert worker.max_new_tokens == 4096
    assert worker.disable_fast_read_only_qa is True
    assert worker.goal_worker is True


def test_goal_role_runtime_caps_critic_and_resolver(tmp_path):
    _init_repo(tmp_path)
    agent = _agent(tmp_path, _ClientFactory([[], []]))
    agent.max_steps = 80
    agent.max_new_tokens = 12000

    critic = agent.goal_manager._role_runtime(
        "Explore", agent.workspace, budget_role="critic"
    )
    resolver = agent.goal_manager._role_runtime(
        "Explore", agent.workspace, budget_role="query_resolver"
    )

    assert (critic.max_steps, critic.max_new_tokens, critic.goal_role) == (
        10,
        2048,
        "critic",
    )
    assert (resolver.max_steps, resolver.max_new_tokens, resolver.goal_role) == (
        4,
        1024,
        "query_resolver",
    )


def test_goal_workspace_cache_prefix_normalizes_worktree_specific_values():
    first = WorkspaceContext(
        cwd="C:/repo/.gencode/goals/g1/workers/a1",
        repo_root="C:/repo/.gencode/goals/g1/workers/a1",
        branch="gencode-goal-g1-worker-a1",
        default_branch="main",
        status=" M src/app.py",
        recent_commits=["abc123 change app"],
        project_docs={"README.md": "Stable project instructions"},
    )
    second = WorkspaceContext(
        cwd="C:/repo/.gencode/goals/g1/workers/a2",
        repo_root="C:/repo/.gencode/goals/g1/workers/a2",
        branch="gencode-goal-g1-worker-a2",
        default_branch="main",
        status=" M src/other.py",
        recent_commits=["abc123 change app"],
        project_docs={"README.md": "Stable project instructions"},
    )

    assert first.text(cache_stable=True) == second.text(cache_stable=True)
    assert first.text() != second.text()


@pytest.mark.asyncio
async def test_goal_token_budget_rejects_model_call_before_provider_request(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    monkeypatch.setenv("GENCODE_GOAL_TOKEN_BUDGET", "1")
    clients = []

    class Factory:
        def __call__(self):
            client = ScriptedModelClient([_final(_plan())])
            clients.append(client)
            return client

    agent = _agent(tmp_path, Factory())
    _, _, started = await handle_repl_command_async(agent, "/goal Update README")
    goal_id = started.split()[1]
    status = await agent.goal_manager.wait(goal_id)
    goal = agent.goal_manager.store.load(goal_id)

    assert "status: budget_exhausted" in status
    assert len(clients) == 1
    assert len(clients[0].outputs) == 1
    assert goal["token_usage"]["total_tokens"] == 0
    assert goal["token_usage"]["budget_exhausted"] is True


@pytest.mark.asyncio
async def test_goal_token_budget_records_usage_and_recovers_pending_reservations(tmp_path):
    from gencode.core.runtime.goal_budget import GoalTokenBudget
    from gencode.core.runtime.persistence.goal_store import GoalStore

    store = GoalStore(tmp_path / "goals")
    goal = {
        "goal_id": "goal-budget-test",
        "max_total_tokens": 1000,
        "token_usage": {},
    }
    store.save(goal)
    budget = GoalTokenBudget(goal, store)
    reservation = await budget.reserve(100, 50, "worker")

    assert reservation is not None
    usage = await budget.record(
        reservation,
        "worker",
        {
            "total_input_tokens": 100,
            "input_tokens": 75,
            "cached_tokens": 25,
            "output_tokens": 10,
        },
        "ten output tokens",
    )
    assert usage["total_tokens"] == 110
    assert usage["cached_tokens"] == 25
    assert usage["by_role"]["worker"]["total_tokens"] == 110
    assert usage["pending_reservations"] == {}

    pending = await budget.reserve(80, 20, "critic")
    assert pending is not None
    recovered_goal = store.load(goal["goal_id"])
    recovered = GoalTokenBudget(recovered_goal, store).snapshot()
    assert recovered["pending_reservations"] == {}
    assert recovered["total_tokens"] == 210
    assert recovered["estimated_calls"] == 1


@pytest.mark.asyncio
async def test_goal_token_budget_enforces_worker_allocation(tmp_path):
    from gencode.core.runtime.goal_budget import GoalTokenBudget
    from gencode.core.runtime.persistence.goal_store import GoalStore

    store = GoalStore(tmp_path / "goals")
    goal = {
        "goal_id": "goal-role-budget",
        "max_total_tokens": 1000,
        "token_usage": {},
    }
    store.save(goal)
    budget = GoalTokenBudget(goal, store)
    reservation = await budget.reserve(600, 100, "worker")
    assert reservation is not None
    await budget.record(
        reservation,
        "worker",
        {"total_input_tokens": 600, "output_tokens": 100},
    )

    rejected = await budget.reserve(1, 1, "worker")

    assert rejected is None
    usage = budget.snapshot()
    assert usage["total_tokens"] == 700
    assert usage["max_role_tokens"]["worker"] == 700
    assert usage["budget_exhausted_role"] == "worker"


@pytest.mark.asyncio
async def test_goal_budget_does_not_execute_tool_after_usage_exceeds_role_cap(tmp_path):
    from gencode.core.runtime.goal_budget import GoalTokenBudget
    from gencode.core.runtime.persistence.goal_store import GoalStore
    from gencode.core.runtime.task_state import STOP_REASON_GOAL_TOKEN_BUDGET_EXHAUSTED
    from gencode.providers.base import ModelResult, ModelStreamEvent

    class ToolRequestClient:
        supports_prompt_cache = False

        async def stream_result(self, prompt, max_new_tokens, **kwargs):
            del prompt, max_new_tokens, kwargs
            yield ModelStreamEvent(
                "completed",
                result=ModelResult(
                    text=(
                        '<tool>{"name":"write_file","args":'
                        '{"path":"README.md","content":"must not write"}}</tool>'
                    ),
                    metadata={
                        "provider_protocol": "openai",
                        "total_input_tokens": 70000,
                        "output_tokens": 1,
                    },
                ),
            )

    _init_repo(tmp_path)
    agent = _agent(tmp_path, _ClientFactory([[]]))
    agent.model_client = ToolRequestClient()
    agent.max_new_tokens = 128
    agent.write_scope = ("README.md",)
    goal = {
        "goal_id": "goal-tool-budget",
        "max_total_tokens": 100000,
        "token_usage": {},
    }
    store = GoalStore(tmp_path / ".gencode" / "goals")
    store.save(goal)
    agent.goal_role = "worker"
    agent.goal_token_budget = GoalTokenBudget(goal, store)

    await agent.ask_async("Update README.md")

    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "Goal fixture\n"
    assert agent.current_task_state.stop_reason == STOP_REASON_GOAL_TOKEN_BUDGET_EXHAUSTED
    assert agent.current_task_state.tool_steps == 0
    assert goal["token_usage"]["budget_exhausted"] is True


def test_goal_worker_failure_summary_explains_nonproductive_step_limit():
    from types import SimpleNamespace

    summary = _worker_failure_summary(
        SimpleNamespace(
            status="stopped",
            stop_reason="step_limit_reached",
            tool_steps=48,
            changed_paths=[],
            last_tool="search",
        )
    )

    assert "step_limit_reached" in summary
    assert "tool_steps=48" in summary
    assert "changed_paths=none" in summary
    assert "last_tool=search" in summary


def test_step_limited_worker_candidate_must_stay_inside_write_scope():
    assert _has_only_scoped_changes(
        ["src/cli.ts", "test/sessions.test.ts"],
        ["src/cli.ts", "test/sessions.test.ts"],
    )
    assert not _has_only_scoped_changes(["src/cli.ts", "package.json"], ["src/cli.ts"])
    assert not _has_only_scoped_changes(["../outside.txt"], ["src"])


@pytest.mark.asyncio
async def test_goal_verifies_scoped_candidate_when_worker_reaches_step_limit(tmp_path):
    _init_repo(tmp_path)
    factory = _ClientFactory(
        [
            [_final(_plan())],
            [
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":20}}</tool>',
                '<tool>{"name":"write_file","args":{"path":"README.md","content":"Goal fixture\\ngoal-ready\\n"}}</tool>',
                "<final>Scoped change is ready for parent verification.</final>",
            ],
            [_final({"verdict": "pass", "findings": []})],
        ]
    )
    agent = _agent(tmp_path, factory)
    agent.max_steps = 2

    _, _, started = await handle_repl_command_async(agent, "/goal update README")
    goal_id = started.split()[1]
    status = await agent.goal_manager.wait(goal_id)
    goal = agent.goal_manager.store.load(goal_id)

    assert "status: completed" in status
    assert goal["nodes"][0]["status"] == "completed"
    assert goal["attempts"][0]["status"] == "integrated"
    assert goal["attempts"][0]["completion_mode"] == "step_limit_with_scoped_changes"
    assert goal["verifier_runs"][-1]["passed"] is True
    assert "goal-ready" in (
        tmp_path / ".gencode" / "goals" / goal_id / "integration" / "README.md"
    ).read_text(encoding="utf-8")


def test_windows_package_verifiers_use_command_processor(monkeypatch):
    from gencode.core.runtime import goal_manager

    monkeypatch.setattr(goal_manager.os, "name", "nt")
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")

    assert _verifier_process_args(["npm", "run", "test:unit"]) == [
        r"C:\Windows\System32\cmd.exe",
        "/d",
        "/s",
        "/c",
        "npm run test:unit",
    ]
    assert _verifier_process_args(["python", "-m", "pytest", "-q"]) == [
        "python",
        "-m",
        "pytest",
        "-q",
    ]


@pytest.mark.asyncio
async def test_goal_slash_runs_worker_worktree_merge_verifier_and_critic(tmp_path):
    _init_repo(tmp_path)
    factory = _ClientFactory(
        [
            [_final(_plan())],
            [
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":20}}</tool>',
                '<tool>{"name":"write_file","args":{"path":"README.md","content":"Goal fixture\\ngoal-ready\\n"}}</tool>',
                "<final>Updated README.md and confirmed the requested content.</final>",
            ],
            [_final({"verdict": "pass", "findings": []})],
        ]
    )
    agent = _agent(tmp_path, factory)

    handled, should_exit, started = await handle_repl_command_async(
        agent, "/goal Update README with goal-ready and verify it"
    )
    goal_id = started.split()[1]
    assert handled is True
    assert should_exit is False
    assert "background planning started" in started

    status = await agent.goal_manager.wait(goal_id)

    goal = agent.goal_manager.store.load(goal_id)
    assert "status: completed" in status
    assert goal["graph_revision"] == 1
    assert goal["replan_count"] == 0
    assert goal["nodes"][0]["status"] == "completed"
    assert goal["verifier_runs"][-1]["passed"] is True
    assert goal["critic_rounds"] == 1
    assert (tmp_path / "README.md").read_text(encoding="utf-8").startswith("Goal fixture\n")
    assert _git(tmp_path, "rev-parse", "HEAD").stdout.strip() == goal["base_commit"]
    integration = tmp_path / ".gencode" / "goals" / goal_id / "integration"
    assert "goal-ready" in (integration / "README.md").read_text(encoding="utf-8")
    assert goal["integration_commit"] != goal["base_commit"]
    assert goal["candidate_files"] == ["README.md"]
    assert all(item.get("worktree_cleaned") for item in goal["attempts"] if item["status"] == "integrated")
    events = [
        json.loads(line)
        for line in agent.goal_manager.store.events_path(goal_id).read_text(encoding="utf-8").splitlines()
    ]
    assert any(item["event"] == "dag_replanned" for item in events) is False
    assert any(item["event"] == "worker_integrated" for item in events)


@pytest.mark.asyncio
async def test_critic_provider_failure_preserves_candidate_without_replanning(tmp_path):
    _init_repo(tmp_path)
    factory = _ClientFactory(
        [
            [_final(_plan())],
            [
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":20}}</tool>',
                '<tool>{"name":"write_file","args":{"path":"README.md","content":"Goal fixture\\ngoal-ready\\n"}}</tool>',
                "<final>Updated README.md and confirmed the requested content.</final>",
            ],
            [
                ProviderError(
                    "billing detail must not be copied to Goal evidence",
                    provider="anthropic",
                    model="test-model",
                    code="http_error",
                    http_status=402,
                )
            ],
        ]
    )
    agent = _agent(tmp_path, factory)

    _, _, started = await handle_repl_command_async(
        agent, "/goal Update README with goal-ready and verify it"
    )
    goal_id = started.split()[1]
    status = await agent.goal_manager.wait(goal_id)
    goal = agent.goal_manager.store.load(goal_id)
    events = [
        json.loads(line)
        for line in agent.goal_manager.store.events_path(goal_id).read_text(encoding="utf-8").splitlines()
    ]

    assert "status: blocked" in status
    assert goal["failure"] == (
        "Critic execution failed (stop_reason=model_error, code=http_error, http_status=402); "
        "candidate preserved for /goal resume"
    )
    assert goal["critic_rounds"] == 0
    assert goal["replan_count"] == 0
    assert goal["verifier_runs"][-1]["passed"] is True
    assert goal["integration_commit"] != goal["base_commit"]
    assert goal["nodes"][0]["status"] == "completed"
    assert "billing detail" not in goal["failure"]
    assert any(item["event"] == "critic_unavailable" for item in events)
    assert not any(item["event"] == "dag_replanned" for item in events)

    candidate_commit = goal["integration_commit"]
    factory.outputs.append([_final({"verdict": "pass", "findings": []})])
    goal["deadline_at"] = "2000-01-01T00:00:00+00:00"
    agent.goal_manager.store.save(goal)
    resumed = await agent.goal_manager.resume(goal_id)
    assert "resumed" in resumed
    status = await agent.goal_manager.wait(goal_id)
    goal = agent.goal_manager.store.load(goal_id)

    assert "status: completed" in status
    assert goal["integration_commit"] == candidate_commit
    assert goal["critic_rounds"] == 1
    assert goal["replan_count"] == 0
    assert goal["execution_window_count"] == 2
    assert datetime.fromisoformat(goal["deadline_at"]) > datetime.now(timezone.utc)
    assert goal["attempts"][0]["worktree_cleaned"] is True


@pytest.mark.asyncio
async def test_worker_high_impact_query_waits_for_parent_then_uses_fresh_attempt(tmp_path):
    _init_repo(tmp_path)
    factory = _ClientFactory(
        [
            [_final(_plan())],
            ['<final><goal_query>{"missing":"May I edit pyproject.toml?","why":"The requested change appears to require a dependency edit.","options":["edit it","avoid it"],"impact":"permission"}</goal_query></final>'],
            [
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":20}}</tool>',
                '<tool>{"name":"write_file","args":{"path":"README.md","content":"Goal fixture\\ngoal-ready\\n"}}</tool>',
                "<final>Completed the scoped task.</final>",
            ],
            [_final({"verdict": "pass", "findings": []})],
        ]
    )
    agent = _agent(tmp_path, factory)
    _, _, started = await handle_repl_command_async(agent, "/goal update README")
    goal_id = started.split()[1]
    status = await agent.goal_manager.wait(goal_id)
    goal = agent.goal_manager.store.load(goal_id)

    assert "status: waiting_for_parent" in status
    query = next(item for item in goal["queries"] if item["status"] == "waiting_for_parent")
    assert f"parent query: {query['query_id']}" in status
    assert goal["nodes"][0]["status"] == "waiting_for_parent"
    listed = await agent.goal_manager.command("list")
    resume = await agent.goal_manager.command(f"resume {goal_id}")
    wrong_answer = await agent.goal_manager.command(f'answer {goal_id} query_missing "do something"')
    assert goal_id in listed
    assert "waiting for a parent decision" in resume
    assert "No unanswered parent query" in wrong_answer

    goal["deadline_at"] = "2000-01-01T00:00:00+00:00"
    agent.goal_manager.store.save(goal)
    _, _, answered = await handle_repl_command_async(
        agent,
        f'/goal answer {goal_id} {query["query_id"]} "Keep edits limited to README.md"',
    )
    assert "new Worker attempt" in answered
    final_status = await agent.goal_manager.wait(goal_id)
    goal = agent.goal_manager.store.load(goal_id)

    assert "status: completed" in final_status
    assert len(goal["attempts"]) == 2
    assert goal["attempts"][0]["attempt_id"] != goal["attempts"][1]["attempt_id"]
    assert goal["attempts"][0]["worktree"] != goal["attempts"][1]["worktree"]
    assert goal["queries"][0]["status"] == "answered"
    assert goal["graph_revision"] == 1
    assert goal["replan_count"] == 0
    assert datetime.fromisoformat(goal["deadline_at"]) > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_dependency_query_is_replanned_by_parent_before_dispatching_new_dag(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "prerequisite.txt").write_text("pending\n", encoding="utf-8")
    _git(tmp_path, "add", "prerequisite.txt")
    _git(tmp_path, "commit", "-qm", "add prerequisite fixture")
    revised_plan = {
        "nodes": [
            {
                "id": "prepare_dependency",
                "title": "Prepare the prerequisite",
                "prompt": "Record the prerequisite before the README update.",
                "depends_on": [],
                "write_scope": ["prerequisite.txt"],
                "acceptance": ["Prerequisite is recorded"],
            },
            {
                "id": "update_readme",
                "title": "Update the README",
                "prompt": "Add goal-ready to README.md after the prerequisite is recorded.",
                "depends_on": ["prepare_dependency"],
                "write_scope": ["README.md"],
                "acceptance": ["README.md contains goal-ready"],
            },
        ],
        "verifiers": [["python", "-m", "pytest", "-q"]],
    }
    factory = _ClientFactory(
        [
            [_final(_plan())],
            ['<final><goal_query>{"missing":"A prerequisite node is missing.","why":"The README change depends on it.","options":["add prerequisite node"],"impact":"dependency"}</goal_query></final>'],
            [_final(revised_plan)],
            [
                '<tool>{"name":"read_file","args":{"path":"prerequisite.txt","start":1,"end":20}}</tool>',
                '<tool>{"name":"write_file","args":{"path":"prerequisite.txt","content":"ready\\n"}}</tool>',
                "<final>The prerequisite is ready.</final>",
            ],
            [
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":20}}</tool>',
                '<tool>{"name":"write_file","args":{"path":"README.md","content":"Goal fixture\\ngoal-ready\\n"}}</tool>',
                "<final>Updated the README after the prerequisite.</final>",
            ],
            [_final({"verdict": "pass", "findings": []})],
        ]
    )
    agent = _agent(tmp_path, factory)
    _, _, started = await handle_repl_command_async(agent, "/goal update README")
    goal_id = started.split()[1]

    status = await agent.goal_manager.wait(goal_id)
    goal = agent.goal_manager.store.load(goal_id)
    events = [
        json.loads(line)
        for line in agent.goal_manager.store.events_path(goal_id).read_text(encoding="utf-8").splitlines()
    ]

    assert "status: completed" in status
    assert goal["replan_count"] == 1
    assert [node["id"] for node in goal["nodes"]] == ["prepare_dependency", "update_readme"]
    assert goal["nodes"][1]["depends_on"] == ["prepare_dependency"]
    assert goal["queries"][0]["status"] == "replanned"
    assert [event["event"] for event in events].count("dag_replanned") == 1
    assert not any(attempt["status"] == "running" for attempt in goal["attempts"])


@pytest.mark.asyncio
async def test_failed_verifier_replans_completed_node_with_new_spec_and_attempt(tmp_path):
    _init_repo(tmp_path, require_marker=True)
    revised_plan = _plan(prompt="Read README.md, add goal-ready, and preserve existing text.")
    factory = _ClientFactory(
        [
            [_final(_plan())],
            [
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":20}}</tool>',
                '<tool>{"name":"write_file","args":{"path":"README.md","content":"Initial implementation\\n"}}</tool>',
                "<final>The initial implementation is complete.</final>",
            ],
            [_final(revised_plan)],
            [
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":20}}</tool>',
                '<tool>{"name":"write_file","args":{"path":"README.md","content":"Goal fixture\\ngoal-ready\\n"}}</tool>',
                "<final>Added the required marker after the verifier failure.</final>",
            ],
            [_final({"verdict": "pass", "findings": []})],
        ]
    )
    agent = _agent(tmp_path, factory)
    _, _, started = await handle_repl_command_async(agent, "/goal add goal-ready to README")
    goal_id = started.split()[1]

    status = await agent.goal_manager.wait(goal_id)
    goal = agent.goal_manager.store.load(goal_id)

    assert "status: completed" in status
    assert goal["replan_count"] == 1
    assert goal["graph_revision"] == 2
    assert goal["nodes"][0]["spec_revision"] == 2
    assert goal["nodes"][0]["revision_history"][0]["status"] == "completed"
    assert [run["passed"] for run in goal["verifier_runs"]] == [False, True]
    assert len(goal["attempts"]) == 2
    assert goal["attempts"][0]["worktree"] != goal["attempts"][1]["worktree"]
    assert goal["attempts"][1]["attempt_number"] == 2


@pytest.mark.asyncio
async def test_goal_does_not_integrate_a_completed_worker_without_changes(tmp_path):
    _init_repo(tmp_path)
    plan = _plan()
    factory = _ClientFactory(
        [
            [_final(plan)],
            ["<final>Completed the requested change.</final>"],
            ["<final>Completed the requested change.</final>"],
            [_final(plan)],
        ]
    )
    agent = _agent(tmp_path, factory)
    _, _, started = await handle_repl_command_async(agent, "/goal update README")
    goal_id = started.split()[1]

    status = await agent.goal_manager.wait(goal_id)
    goal = agent.goal_manager.store.load(goal_id)
    events = [
        json.loads(line)
        for line in agent.goal_manager.store.events_path(goal_id).read_text(encoding="utf-8").splitlines()
    ]

    assert "status: blocked" in status
    assert goal["nodes"][0]["status"] == "failed"
    assert goal["nodes"][0]["last_error"] == "worker completed without producing workspace changes"
    assert [attempt["status"] for attempt in goal["attempts"]] == ["failed", "failed"]
    assert not any(event["event"] == "worker_integrated" for event in events)
    assert _git(tmp_path, "rev-parse", "HEAD").stdout.strip() == goal["base_commit"]


@pytest.mark.asyncio
async def test_goal_rejects_dirty_base_and_cyclic_planner_output(tmp_path):
    _init_repo(tmp_path)
    factory = _ClientFactory([[_final(_plan())]])
    agent = _agent(tmp_path, factory)
    (tmp_path / "README.md").write_text("user changes\n", encoding="utf-8")

    handled, _, result = await handle_repl_command_async(agent, "/goal do something")
    assert handled is True
    assert "refused to start" in result

    (tmp_path / "README.md").write_text("Goal fixture\n", encoding="utf-8")
    cyclic = {
        "nodes": [
            {"id": "a", "title": "A", "prompt": "A", "depends_on": ["b"], "write_scope": ["README.md"], "acceptance": ["A"]},
            {"id": "b", "title": "B", "prompt": "B", "depends_on": ["a"], "write_scope": ["README.md"], "acceptance": ["B"]},
        ],
        "verifiers": [["python", "-m", "pytest"]],
    }
    with pytest.raises(GoalPlanError, match="cycle"):
        validate_plan(cyclic)


@pytest.mark.asyncio
async def test_goal_interface_blocks_cyclic_plan_without_creating_integration_worktree(tmp_path):
    _init_repo(tmp_path)
    cyclic = {
        "nodes": [
            {"id": "a", "title": "A", "prompt": "A", "depends_on": ["b"], "write_scope": ["README.md"], "acceptance": ["A"]},
            {"id": "b", "title": "B", "prompt": "B", "depends_on": ["a"], "write_scope": ["README.md"], "acceptance": ["B"]},
        ],
        "verifiers": [["python", "-m", "pytest"]],
    }
    agent = _agent(tmp_path, _ClientFactory([[_final(cyclic)]]))

    handled, _, started = await handle_repl_command_async(agent, "/goal implement a cyclic plan")
    goal_id = started.split()[1]
    status = await agent.goal_manager.wait(goal_id)
    goal = agent.goal_manager.store.load(goal_id)

    assert handled is True
    assert "status: blocked" in status
    assert "dependency cycle" in goal["failure"]
    assert goal["integration_commit"] == ""
    assert not (tmp_path / ".gencode" / "goals" / goal_id / "integration").exists()


@pytest.mark.asyncio
async def test_goal_cancel_before_background_task_starts_is_persisted(tmp_path):
    _init_repo(tmp_path)
    factory = _ClientFactory([[_final(_plan())]])
    agent = _agent(tmp_path, factory)

    handled, _, started = await handle_repl_command_async(agent, "/goal do work")
    goal_id = started.split()[1]
    cancelled = await agent.goal_manager.command(f"cancel {goal_id}")
    status = agent.goal_manager.store.load(goal_id)["status"]

    assert handled is True
    assert "cancelled" in cancelled
    assert status == "cancelled"
    assert factory.calls == 0


def test_goal_contract_bounds_dag_size_scope_and_verifier_commands():
    for scope in ("..", r"C:\secrets.txt"):
        unsafe_scope = _plan()
        unsafe_scope["nodes"][0]["write_scope"] = [scope]
        with pytest.raises(GoalPlanError, match="unsafe write_scope"):
            validate_plan(unsafe_scope)

    unsafe_verifier = _plan()
    unsafe_verifier["verifiers"] = [["python", "-c", "print('unsafe')"]]
    with pytest.raises(GoalPlanError, match="python -m"):
        validate_plan(unsafe_verifier)

    shell_verifier = _plan()
    shell_verifier["verifiers"] = [["pytest", ";", "git", "reset", "--hard"]]
    with pytest.raises(GoalPlanError, match="shell operators"):
        validate_plan(shell_verifier)

    arbitrary_pnpm_script = _plan()
    arbitrary_pnpm_script["verifiers"] = [["pnpm", "run", "publish"]]
    with pytest.raises(GoalPlanError, match="unsupported verifier"):
        validate_plan(arbitrary_pnpm_script)

    unit_test_script = _plan()
    unit_test_script["verifiers"] = [["npm", "run", "test:unit"]]
    assert validate_plan(unit_test_script)["verifiers"] == [["npm", "run", "test:unit"]]

    arbitrary_npm_script = _plan()
    arbitrary_npm_script["verifiers"] = [["npm", "run", "publish"]]
    with pytest.raises(GoalPlanError, match="unsupported verifier"):
        validate_plan(arbitrary_npm_script)

    verifier_object = _plan()
    verifier_object["verifiers"] = [
        {"id": "pytest", "node_id": "update_readme", "argv": ["python", "-m", "pytest", "-q"]}
    ]
    assert validate_plan(verifier_object)["verifiers"] == [["python", "-m", "pytest", "-q"]]

    unsafe_verifier_object = _plan()
    unsafe_verifier_object["verifiers"] = [
        {"id": "shell", "argv": ["cmd.exe", "/c", "git reset --hard"]}
    ]
    with pytest.raises(GoalPlanError, match="unsupported verifier"):
        validate_plan(unsafe_verifier_object)


def test_goal_replan_budget_is_hard_capped(tmp_path):
    agent = GenCode(
        model_client=ScriptedModelClient([]),
        workspace=WorkspaceContext.build(tmp_path, repo_root_override=tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
    )
    goal = {
        "goal_id": "goal_limit_test",
        "graph_revision": 4,
        "replan_count": 3,
        "max_replans": 3,
        "nodes": [],
        "attempts": [],
        "queries": [],
        "verifiers": [],
        "failure": "",
    }

    assert asyncio_run(agent.goal_manager._replan(goal, "attempt failure")) is False
    assert goal["replan_count"] == 3
    assert "replan limit reached" in goal["failure"]


def asyncio_run(awaitable):
    import asyncio

    return asyncio.run(awaitable)


def test_goal_replaces_legacy_model_visible_worker_tools(tmp_path):
    agent = _agent(tmp_path, _ClientFactory([]))

    assert resolve_command("goal").name == "goal"
    assert resolve_command("agents") is None
    assert resolve_command("subagent") is None
    assert [item.name for item in suggest_commands("/go")] == ["goal"]
    assert {"agent", "send_message", "task_stop"}.isdisjoint(agent.tools)


def test_goal_status_rejects_path_traversal(tmp_path):
    agent = GenCode(
        model_client=ScriptedModelClient([]),
        workspace=WorkspaceContext.build(tmp_path, repo_root_override=tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
    )

    assert "could not load Goal" in agent.goal_manager.status_text("../outside")
