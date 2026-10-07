"""Durable, bounded Goal orchestration over isolated Git worktrees."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

from .goal_budget import GoalTokenBudget
from .goal_git import GoalGit, GoalGitError
from .goal_graph import (
    GoalPlanError,
    node_fingerprint,
    topological_order,
    validate_plan,
)
from .goal_store import GoalStore
from .task_state import STOP_REASON_GOAL_TOKEN_BUDGET_EXHAUSTED
from .worker_runtime import build_child_runtime
from .workspace import WorkspaceContext, clip, now

DEFAULT_GOAL_SECONDS = 30 * 60
DEFAULT_GOAL_TOKEN_BUDGET = 500_000
MAX_REPLANS = 3
MAX_ATTEMPTS_PER_NODE = 2
MAX_CRITIC_ROUNDS = 2
MAX_QUERY_ROUNDS = 3
MAX_VERIFIER_SECONDS = 600
MAX_WORKER_NODES = 16
MAX_PLANNER_STEPS = 6
MAX_WORKER_STEPS_PER_ATTEMPT = 32
MAX_CRITIC_STEPS = 10
MAX_QUERY_RESOLVER_STEPS = 4
MAX_PLANNER_OUTPUT_TOKENS = 2048
MAX_WORKER_OUTPUT_TOKENS = 4096
MAX_CRITIC_OUTPUT_TOKENS = 2048
MAX_QUERY_RESOLVER_OUTPUT_TOKENS = 1024


class GoalBudgetExceeded(RuntimeError):
    pass


def _configured_goal_token_budget():
    try:
        return max(
            1,
            int(os.environ.get("GENCODE_GOAL_TOKEN_BUDGET", DEFAULT_GOAL_TOKEN_BUDGET)),
        )
    except (TypeError, ValueError):
        return DEFAULT_GOAL_TOKEN_BUDGET


class GoalManager:
    def __init__(self, runtime):
        self.runtime = runtime
        self.store = GoalStore(Path(runtime.root) / ".gencode" / "goals")
        self._active: dict[str, asyncio.Task] = {}
        self._cancel: dict[str, asyncio.Event] = {}
        self._leases = {}
        self._token_budgets: dict[str, GoalTokenBudget] = {}
        self._semaphore = asyncio.Semaphore(max(1, int(runtime.max_worker_concurrency)))

    async def command(self, raw_args):
        try:
            tokens = shlex.split(str(raw_args or ""))
        except ValueError as exc:
            return f"error: invalid /goal arguments: {exc}"
        if not tokens:
            return "Usage: /goal <objective> | status [id] | list | wait [id] | resume <id> | cancel [id] | answer <id> <query-id> <answer>"
        action = tokens[0].lower()
        if action == "list":
            return self.list_text()
        if action == "status":
            return self.status_text(tokens[1] if len(tokens) > 1 else "")
        if action == "wait":
            return await self.wait(tokens[1] if len(tokens) > 1 else "")
        if action == "resume" and len(tokens) == 2:
            return await self.resume(tokens[1])
        if action == "cancel":
            return await self.cancel(tokens[1] if len(tokens) > 1 else "")
        if action == "answer" and len(tokens) >= 4:
            return await self.answer(tokens[1], tokens[2], " ".join(tokens[3:]))
        objective = " ".join(tokens[1:] if action == "start" else tokens).strip()
        if not objective:
            return "Usage: /goal <objective>"
        return await self.start(objective)

    async def start(self, objective):
        objective = str(objective or "").strip()
        if not objective:
            raise ValueError("Goal objective must not be empty")
        git = GoalGit(self.runtime.root, "goal_" + uuid.uuid4().hex[:12])
        base_commit = git.preflight()
        goal_id = git.goal_id
        created = datetime.now(timezone.utc)
        goal = {
            "schema_version": 1,
            "goal_id": goal_id,
            "objective": objective,
            "status": "planning",
            "created_at": created.isoformat(),
            "updated_at": created.isoformat(),
            "deadline_at": (created + timedelta(seconds=DEFAULT_GOAL_SECONDS)).isoformat(),
            "workspace_root": str(self.runtime.root),
            "base_commit": base_commit,
            "integration_branch": git.integration_branch,
            "integration_worktree": str(git.integration_path),
            "integration_commit": "",
            "graph_revision": 0,
            "node_spec_revisions": {},
            "replan_count": 0,
            "max_replans": MAX_REPLANS,
            "max_runtime_seconds": DEFAULT_GOAL_SECONDS,
            "max_total_tokens": _configured_goal_token_budget(),
            "token_usage": {},
            "execution_window_count": 1,
            "nodes": [],
            "retired_nodes": [],
            "verifiers": [],
            "attempts": [],
            "queries": [],
            "parent_answers": {},
            "verifier_runs": [],
            "critic_rounds": 0,
            "critic_findings": [],
            "failure": "",
        }
        self.store.save(goal)
        self._event(goal, "goal_created", {"base_commit": base_commit})
        self._schedule(goal_id)
        return (
            f"Goal {goal_id} accepted at base {base_commit[:12]}; background planning started. "
            f"Wall-clock budget: {DEFAULT_GOAL_SECONDS // 60} minutes. Use /goal status {goal_id}."
        )

    async def resume(self, goal_id):
        goal = self.store.load(goal_id)
        if Path(goal.get("workspace_root", "")).resolve() != Path(self.runtime.root).resolve():
            return f"error: Goal {goal_id} belongs to a different workspace"
        if goal_id in self._active and not self._active[goal_id].done():
            return f"Goal {goal_id} is already running."
        if goal.get("status") in {"completed", "cancelled"}:
            return f"Goal {goal_id} is {goal.get('status')} and cannot be resumed."
        if any(
            query.get("status") == "waiting_for_parent" and not query.get("answer")
            for query in goal.get("queries", [])
        ):
            return f"Goal {goal_id} is waiting for a parent decision; use /goal answer {goal_id} <query-id> <answer>."
        self._recover_interrupted_attempts(goal)
        budget = self._renew_execution_window(goal)
        previous_token_budget = int(
            goal.get("max_total_tokens", DEFAULT_GOAL_TOKEN_BUDGET)
        )
        configured_token_budget = _configured_goal_token_budget()
        if configured_token_budget > previous_token_budget:
            goal["max_total_tokens"] = configured_token_budget
            budget["token_budget_increased_to"] = configured_token_budget
        goal["status"] = "running"
        goal["failure"] = ""
        self._save(goal, "goal_resumed", budget)
        self._schedule(goal_id)
        return f"Goal {goal_id} resumed from graph revision {goal.get('graph_revision', 0)}."

    async def cancel(self, goal_id=""):
        goal_id = goal_id or self._latest_id()
        if not goal_id:
            return "No saved Goal."
        goal = self.store.load(goal_id)
        if Path(goal.get("workspace_root", "")).resolve() != Path(self.runtime.root).resolve():
            return f"error: Goal {goal_id} belongs to a different workspace"
        task = self._active.get(goal_id)
        if task is None or task.done():
            if goal.get("status") in {"completed", "failed", "blocked", "cancelled", "budget_exhausted"}:
                return f"Goal {goal_id} is already {goal.get('status')}."
            goal["status"] = "cancelled"
            goal["updated_at"] = now()
            self.store.save(goal)
            self._event(goal, "goal_cancelled", {})
            return f"Goal {goal_id} marked cancelled."
        event = self._cancel.get(goal_id)
        if event is not None:
            event.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        goal = self.store.load(goal_id)
        if goal.get("status") not in {"completed", "failed", "blocked", "cancelled", "budget_exhausted"}:
            self._finish(goal, "cancelled", "Goal cancelled by user")
            goal = self.store.load(goal_id)
        return f"Goal {goal_id} cancellation recorded ({goal.get('status')})."

    async def answer(self, goal_id, query_id, answer):
        goal = self.store.load(goal_id)
        if Path(goal.get("workspace_root", "")).resolve() != Path(self.runtime.root).resolve():
            return f"error: Goal {goal_id} belongs to a different workspace"
        answer = str(answer or "").strip()
        if not answer:
            return "Parent answer must not be empty."
        query = next((item for item in goal.get("queries", []) if item.get("query_id") == query_id), None)
        if query is None or query.get("status") != "waiting_for_parent":
            return f"No unanswered parent query {query_id} in Goal {goal_id}."
        query["answer"] = answer
        query["status"] = "answered"
        query["answered_at"] = now()
        goal.setdefault("parent_answers", {})[query["node_id"]] = answer
        for node in goal.get("nodes", []):
            if node.get("id") == query.get("node_id") and node.get("status") == "waiting_for_parent":
                node["status"] = "pending"
                node["attempts_used"] = int(node.get("attempts_used", 0))
        budget = self._renew_execution_window(goal)
        goal["status"] = "running"
        self._save(goal, "parent_answer_received", {"query_id": query_id, **budget})
        self._schedule(goal_id)
        return f"Answer recorded; Goal {goal_id} is resuming with a new Worker attempt."

    async def wait(self, goal_id=""):
        goal_id = goal_id or self._latest_id()
        if not goal_id:
            return "No saved Goal."
        try:
            goal = self.store.load(goal_id)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return f"error: could not load Goal: {exc}"
        if Path(goal.get("workspace_root", "")).resolve() != Path(self.runtime.root).resolve():
            return f"error: Goal {goal_id} belongs to a different workspace"
        task = self._active.get(goal_id)
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        return self.status_text(goal_id)

    def status_text(self, goal_id=""):
        goal_id = goal_id or self._latest_id()
        if not goal_id:
            return "No saved Goal."
        try:
            goal = self.store.load(goal_id)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return f"error: could not load Goal: {exc}"
        counts = {}
        for node in goal.get("nodes", []):
            state = node.get("status", "unknown")
            counts[state] = counts.get(state, 0) + 1
        lines = [
            f"goal: {goal_id}",
            f"status: {goal.get('status', 'unknown')}",
            f"objective: {goal.get('objective', '')}",
            f"graph revision: {goal.get('graph_revision', 0)} (replans {goal.get('replan_count', 0)}/{goal.get('max_replans', MAX_REPLANS)})",
            "nodes: " + (", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "not planned"),
            f"base: {str(goal.get('base_commit', ''))[:12]}",
            f"integration: {str(goal.get('integration_commit', ''))[:12] or 'not started'}",
            "tokens: " + self._token_usage_text(goal),
            f"candidate: {goal.get('integration_branch', '-')}",
            f"failure: {goal.get('failure', '') or '-'}",
        ]
        for query in goal.get("queries", []):
            if query.get("status") == "waiting_for_parent":
                lines.append(
                    f"parent query: {query['query_id']} node={query['node_id']} "
                    f"impact={query['impact']} missing={clip(query['missing'], 300)} "
                    f"why={clip(query['why'], 300)} options={json.dumps(query['options'], ensure_ascii=False)}"
                )
        return "\n".join(lines)

    @staticmethod
    def _token_usage_text(goal):
        usage = goal.get("token_usage", {}) or {}
        limit = int(goal.get("max_total_tokens", DEFAULT_GOAL_TOKEN_BUDGET) or 0)
        used = int(usage.get("total_tokens", 0) or 0)
        reserved = sum(
            int(item.get("reserved_tokens", 0) or 0)
            for item in (usage.get("pending_reservations", {}) or {}).values()
            if isinstance(item, dict)
        )
        by_role = usage.get("by_role", {}) or {}
        roles = ", ".join(
            f"{role}={int(details.get('total_tokens', 0) or 0)}"
            for role, details in sorted(by_role.items())
        ) or "none"
        estimated = int(usage.get("estimated_calls", 0) or 0)
        return (
            f"{used}/{limit} total (reserved {reserved}, "
            f"available {max(0, limit - used - reserved)}, "
            f"input {int(usage.get('input_tokens', 0) or 0)}, "
            f"cached {int(usage.get('cached_tokens', 0) or 0)}, "
            f"output {int(usage.get('output_tokens', 0) or 0)}, "
            f"calls {int(usage.get('model_calls', 0) or 0)}, "
            f"estimated {estimated}; roles: {roles})"
        )

    def list_text(self):
        items = self.store.list()
        if not items:
            return "No saved Goal."
        return "\n".join(
            f"{item.get('goal_id')}  {item.get('status')}  {clip(item.get('objective', ''), 100)}"
            for item in items[:20]
        )

    def _latest_id(self):
        rows = self.store.list()
        return str(rows[0].get("goal_id", "")) if rows else ""

    def _schedule(self, goal_id):
        previous = self._active.get(goal_id)
        if previous is not None:
            if not previous.done():
                raise RuntimeError(f"Goal {goal_id} is already running")
            self._observe_task(goal_id, previous)
        self._leases[goal_id] = self.store.acquire_lease(goal_id)
        task = asyncio.create_task(self._run_goal(goal_id), name=f"gencode-goal-{goal_id}")
        self._active[goal_id] = task
        self._cancel[goal_id] = asyncio.Event()
        task.add_done_callback(lambda done, key=goal_id: self._observe_task(key, done))

    def _observe_task(self, goal_id, task):
        if not task.cancelled():
            task.exception()
        self._active.pop(goal_id, None)
        self._cancel.pop(goal_id, None)
        self._token_budgets.pop(goal_id, None)
        lease = self._leases.pop(goal_id, None)
        if lease is not None:
            lease.release()

    async def _run_goal(self, goal_id):
        goal = self.store.load(goal_id)
        goal.setdefault("max_total_tokens", DEFAULT_GOAL_TOKEN_BUDGET)
        goal.setdefault("token_usage", {})
        self._token_budgets[goal_id] = GoalTokenBudget(goal, self.store)
        if Path(goal.get("workspace_root", "")).resolve() != Path(self.runtime.root).resolve():
            self._finish(goal, "blocked", "Goal workspace path does not match the active Runtime")
            return
        git = GoalGit(goal["workspace_root"], goal_id)
        remaining = self._remaining(goal)
        if remaining <= 0:
            self._finish(goal, "budget_exhausted", "30-minute Goal wall-clock budget expired")
            return
        try:
            async with asyncio.timeout(remaining):
                if not goal.get("nodes"):
                    plan = await self._request_plan(goal, reason="initial plan")
                    self._apply_plan(goal, plan, initial=True)
                    self._save(goal, "goal_plan_created", {"nodes": len(goal["nodes"])})
                if not goal.get("integration_commit"):
                    goal["integration_commit"] = git.create_integration(goal["base_commit"])
                else:
                    goal["integration_commit"] = git.reopen_integration(goal["integration_branch"])
                goal["status"] = "running"
                self._save(goal, "integration_worktree_ready", {"head": goal["integration_commit"]})

                while True:
                    if self._cancel.get(goal_id) and self._cancel[goal_id].is_set():
                        self._finish(goal, "cancelled", "Goal cancelled by user")
                        return
                    active_nodes = [node for node in goal["nodes"] if node.get("status") != "obsolete"]
                    if active_nodes and all(node.get("status") == "completed" for node in active_nodes):
                        verify = await self._run_verifiers(goal, git)
                        if not verify["passed"]:
                            goal["verifier_runs"].append(verify)
                            self._save(goal, "verifier_failed", {"failure": verify.get("summary", "")})
                            if not await self._replan(goal, "post-merge verifier failed", evidence=verify):
                                self._finish(goal, "blocked", verify.get("summary", "verifier failed"))
                                return
                            continue
                        goal["verifier_runs"].append(verify)
                        self._save(goal, "verifier_passed", {"candidate_commit": git.integration_head()})
                        review = await self._run_critic(goal, git, verify)
                        if review.get("verdict") == "budget_exhausted":
                            self._finish(goal, "budget_exhausted", review.get("error", "Goal token budget exhausted during Critic review"))
                            return
                        if review.get("verdict") in {"unavailable", "invalid"}:
                            reason = str(review.get("error", "Critic did not return a valid review"))
                            event = (
                                "critic_unavailable"
                                if review["verdict"] == "unavailable"
                                else "critic_response_invalid"
                            )
                            self._save(
                                goal,
                                event,
                                {
                                    "candidate_commit": git.integration_head(),
                                    "reason": reason,
                                },
                            )
                            self._finish(
                                goal,
                                "blocked",
                                f"{reason}; candidate preserved for /goal resume",
                            )
                            return
                        goal["critic_rounds"] = int(goal.get("critic_rounds", 0)) + 1
                        goal["critic_findings"].extend(review.get("findings", []))
                        self._save(goal, "critic_completed", {"verdict": review.get("verdict", "invalid")})
                        if review.get("verdict") == "pass" and not review.get("findings"):
                            goal["integration_commit"] = git.integration_head()
                            goal["candidate_files"] = git.changed_files(goal["base_commit"], goal["integration_commit"])
                            self._finish(goal, "completed", "candidate passed deterministic verifiers and read-only Critic")
                            self._cleanup_successful_workers(goal, git)
                            return
                        if goal["critic_rounds"] >= MAX_CRITIC_ROUNDS:
                            self._finish(goal, "blocked", "Critic did not approve after the bounded review rounds")
                            return
                        if not await self._replan(goal, "Critic requested changes", evidence=review):
                            self._finish(goal, "blocked", "Critic findings could not be converted into a changed DAG within the replan limit")
                            return
                        continue

                    ready = self._ready_nodes(goal)
                    if not ready:
                        unresolved = [node for node in goal["nodes"] if node.get("status") in {"pending", "failed", "waiting_for_parent"}]
                        evidence = {"nodes": unresolved, "graph_revision": goal["graph_revision"]}
                        if not await self._replan(goal, "DAG has no runnable node", evidence=evidence):
                            self._finish(goal, "blocked", "DAG is stuck or its bounded replan budget is exhausted")
                            return
                        continue

                    wave_revision = int(goal["graph_revision"])
                    wave_results = await asyncio.gather(
                        *(self._execute_node(goal, node, git, wave_revision) for node in ready),
                        return_exceptions=True,
                    )
                    wave = []
                    changed = False
                    token_budget_exhausted = False
                    replan_queries = []
                    for node_spec, result in zip(ready, wave_results):
                        if isinstance(result, BaseException):
                            if not isinstance(result, Exception):
                                raise result
                            node = self._node(goal, node_spec["id"])
                            node["last_error"] = clip(str(result), 1200)
                            node["status"] = (
                                "failed"
                                if int(node.get("attempts_used", 0)) >= MAX_ATTEMPTS_PER_NODE
                                else "pending"
                            )
                            running = next(
                                (
                                    item
                                    for item in reversed(goal.get("attempts", []))
                                    if item.get("node_id") == node["id"]
                                    and item.get("status") == "running"
                                ),
                                None,
                            )
                            if running is not None:
                                self._attempt_update(
                                    goal,
                                    running["attempt_id"],
                                    status="failed",
                                    error=node["last_error"],
                                )
                            self._save(
                                goal,
                                "worker_dispatch_failed",
                                {"node_id": node["id"], "error": node["last_error"]},
                            )
                            changed = True
                            continue
                        wave.append(result)
                    for result in wave:
                        node = self._node(goal, result["node_id"])
                        if (
                            int(goal["graph_revision"]) != result["graph_revision"]
                            or int(node.get("spec_revision", 1)) != result["node_spec_revision"]
                            or node_fingerprint(node) != result["node_fingerprint"]
                        ):
                            self._attempt_update(goal, result["attempt_id"], status="stale_result", error="DAG or node contract changed before integration")
                            if node.get("status") != "completed":
                                node["status"] = "pending"
                            self._save(goal, "stale_worker_result_rejected", {"attempt_id": result["attempt_id"]})
                            changed = True
                            continue
                        if result["status"] == "waiting_for_parent":
                            node["status"] = "waiting_for_parent"
                            self._finish(goal, "waiting_for_parent", result.get("error", "Worker needs a parent decision"))
                            return
                        if result["status"] == "replan_requested":
                            node["status"] = "pending"
                            self._attempt_update(
                                goal,
                                result["attempt_id"],
                                status="stale_result",
                                error="Worker requested a DAG change; parent will replan serially",
                            )
                            replan_queries.append(result["query"])
                            self._save(goal, "worker_requested_dag_replan", {"attempt_id": result["attempt_id"]})
                            changed = True
                            continue
                        if result["status"] == "budget_exhausted":
                            node["status"] = (
                                "failed"
                                if int(node.get("attempts_used", 0)) >= MAX_ATTEMPTS_PER_NODE
                                else "pending"
                            )
                            node["last_error"] = result.get("error", "Goal token budget exhausted")
                            self._attempt_update(
                                goal,
                                result["attempt_id"],
                                status="budget_exhausted",
                                error=node["last_error"],
                            )
                            token_budget_exhausted = True
                            changed = True
                            continue
                        if result["status"] != "completed":
                            node["status"] = "failed" if int(node.get("attempts_used", 0)) >= MAX_ATTEMPTS_PER_NODE else "pending"
                            node["last_error"] = result.get("error", "worker failed")
                            self._attempt_update(goal, result["attempt_id"], status="failed", error=node["last_error"])
                            self._save(goal, "worker_attempt_failed", {"node_id": node["id"], "attempt_id": result["attempt_id"]})
                            changed = True
                            continue
                        try:
                            candidate_commit = git.commit_worker_changes(
                                result["worktree"],
                                f"gencode goal {goal_id}: {node['id']} attempt {result['attempt_id']}",
                            )
                            if candidate_commit == result["input_commit_sha"]:
                                error = "worker completed without producing workspace changes"
                                node["status"] = (
                                    "failed"
                                    if int(node.get("attempts_used", 0)) >= MAX_ATTEMPTS_PER_NODE
                                    else "pending"
                                )
                                node["last_error"] = error
                                self._attempt_update(
                                    goal,
                                    result["attempt_id"],
                                    status="failed",
                                    error=error,
                                )
                                self._save(
                                    goal,
                                    "worker_no_changes",
                                    {"node_id": node["id"], "attempt_id": result["attempt_id"]},
                                )
                                changed = True
                                continue
                            if candidate_commit != result["input_commit_sha"]:
                                integrated = git.merge_worker(candidate_commit, node["id"])
                            else:
                                integrated = git.integration_head()
                        except GoalGitError as exc:
                            node["status"] = "failed" if int(node.get("attempts_used", 0)) >= MAX_ATTEMPTS_PER_NODE else "pending"
                            node["last_error"] = str(exc)
                            self._attempt_update(goal, result["attempt_id"], status="merge_failed", error=str(exc))
                            self._save(goal, "worker_integration_failed", {"node_id": node["id"], "error": str(exc)})
                            changed = True
                            continue
                        node.update({
                            "status": "completed",
                            "result": clip(result.get("result", ""), 3000),
                            "integrated_commit": integrated,
                            "completed_at": now(),
                            "last_error": "",
                        })
                        goal["integration_commit"] = integrated
                        self._attempt_update(goal, result["attempt_id"], status="integrated", commit=candidate_commit, integration_commit=integrated)
                        self._save(goal, "worker_integrated", {"node_id": node["id"], "commit": integrated})
                        changed = True
                    if token_budget_exhausted:
                        self._finish(goal, "budget_exhausted", self._budget_failure_text(goal))
                        return
                    if replan_queries:
                        if not await self._replan(goal, "Worker query changes prerequisites or node contract", evidence=replan_queries):
                            self._finish(goal, "blocked", "Worker requested a DAG change that could not be safely replanned")
                            return
                        for query in replan_queries:
                            query["status"] = "replanned"
                        self._save(goal, "worker_queries_resolved_by_replan", {"query_ids": [item["query_id"] for item in replan_queries]})
                        continue
                    exhausted = [node for node in goal["nodes"] if node.get("status") == "failed"]
                    if exhausted:
                        if not await self._replan(goal, "Worker exhausted its fresh-worktree attempts", evidence=exhausted):
                            self._finish(goal, "blocked", "Worker attempts exhausted and bounded DAG replanning made no safe progress")
                            return
                    elif not changed:
                        await asyncio.sleep(0)
        except asyncio.TimeoutError:
            self._finish(goal, "budget_exhausted", "30-minute Goal wall-clock budget expired; persisted state can be inspected")
        except GoalBudgetExceeded as exc:
            self._finish(goal, "budget_exhausted", str(exc))
        except asyncio.CancelledError:
            self._finish(goal, "cancelled", "Goal cancelled; completed commits and evidence were preserved")
        except (GoalGitError, GoalPlanError, OSError, RuntimeError, ValueError) as exc:
            self._finish(goal, "blocked", str(exc))
        except Exception as exc:  # noqa: BLE001
            self._finish(goal, "failed", f"Goal orchestration error: {type(exc).__name__}: {exc}")

    async def _request_plan(self, goal, *, reason, evidence=None):
        planner = self._role_runtime(
            "Explore", self.runtime.workspace, goal_id=goal["goal_id"], budget_role="planner"
        )
        current = {
            "graph_revision": goal.get("graph_revision", 0),
            "nodes": goal.get("nodes", []),
            "completed_evidence": [
                {key: node.get(key) for key in ("id", "title", "status", "result", "integrated_commit")}
                for node in goal.get("nodes", []) if node.get("status") == "completed"
            ],
        }
        prompt = (
            "You are the Goal planner. Produce a small dependency DAG for the user's objective. "
            "Return only JSON with keys nodes and verifiers. Each node has id, title, prompt, depends_on, "
            "write_scope (relative repository paths, never . or a broad root), and acceptance (non-empty list). "
            "verifiers must be a list of argv arrays, for example [[\"python\",\"-m\",\"pytest\",\"-q\"]]; "
            "do not wrap them in objects. Use only pytest/python -m pytest|unittest, ruff check|format, "
            "mypy, pyright, npm/pnpm/yarn test, cargo test, go test, or dotnet test. Never use shell operators. "
            "Keep nodes independently implementable, use explicit prerequisites, and preserve completed nodes and "
            "their contracts unless verifier/Critic evidence requires a revision; preserve stable IDs when revising, "
            "never remove completed nodes, and keep the verifier contract unchanged. Do not invent user permissions "
            "or acceptance criteria.\n\n"
            f"Goal objective: {goal['objective']}\nFrozen base commit: {goal['base_commit']}\n"
            f"Current integration commit: {goal.get('integration_commit') or goal['base_commit']}\n"
            f"Reason for this plan: {reason}\nCurrent graph: {json.dumps(current, ensure_ascii=False)}\n"
            f"Failure/query/review evidence: {json.dumps(evidence or {}, ensure_ascii=False)[:12000]}\n"
            "Return a complete replacement plan, including still-needed nodes."
        )
        response = await planner.ask_async(prompt)
        if getattr(getattr(planner, "current_task_state", None), "stop_reason", "") == STOP_REASON_GOAL_TOKEN_BUDGET_EXHAUSTED:
            raise GoalBudgetExceeded(self._budget_failure_text(goal))
        return validate_plan(_parse_json(response))

    def _apply_plan(self, goal, plan, *, initial=False):
        active_nodes = list(plan["nodes"])
        old = {node["id"]: node for node in goal.get("nodes", [])}
        new_ids = {node["id"] for node in active_nodes}
        for node_id, previous in old.items():
            if previous.get("status") == "completed":
                candidate = next((item for item in active_nodes if item["id"] == node_id), None)
                if candidate is None:
                    raise GoalPlanError(f"replacement plan cannot remove completed node {node_id}")
        if not initial and plan["verifiers"] != goal.get("verifiers", []):
            raise GoalPlanError("verifier contract is frozen after initial planning")
        applied = []
        changed = initial or set(old) != new_ids
        for spec in active_nodes:
            prior = old.get(spec["id"])
            if prior is None:
                node = {**spec, "status": "pending", "spec_revision": 1, "attempts_used": 0, "attempt_history": [], "result": "", "integrated_commit": "", "last_error": ""}
                changed = True
            elif node_fingerprint(prior) == node_fingerprint(spec):
                node = {**prior, **spec}
            else:
                revision_history = list(prior.get("revision_history", []))
                if prior.get("status") == "completed":
                    revision_history.append({
                        "spec_revision": int(prior.get("spec_revision", 1)),
                        "node_fingerprint": node_fingerprint(prior),
                        "result": prior.get("result", ""),
                        "integrated_commit": prior.get("integrated_commit", ""),
                        "status": "completed",
                    })
                node = {**prior, **spec, "status": "pending", "spec_revision": int(prior.get("spec_revision", 1)) + 1, "attempts_used": 0, "revision_history": revision_history, "last_error": "", "result": "", "integrated_commit": ""}
                changed = True
            applied.append(node)
            goal.setdefault("node_spec_revisions", {})[spec["id"]] = int(node.get("spec_revision", 1))
        removed = [node for node_id, node in old.items() if node_id not in new_ids]
        if any(node.get("status") == "completed" for node in removed):
            raise GoalPlanError("replacement plan cannot remove completed nodes")
        if removed:
            goal.setdefault("retired_nodes", []).extend([{**node, "status": "obsolete"} for node in removed])
            changed = True
        goal["nodes"] = applied
        if initial:
            goal["verifiers"] = plan["verifiers"]
        if not initial:
            if not changed:
                raise GoalPlanError("planner returned no DAG change")
            goal["replan_count"] = int(goal.get("replan_count", 0)) + 1
        goal["graph_revision"] = int(goal.get("graph_revision", 0)) + 1
        return changed

    async def _replan(self, goal, reason, evidence=None):
        if int(goal.get("replan_count", 0)) >= int(goal.get("max_replans", MAX_REPLANS)):
            goal["failure"] = f"replan limit reached ({goal.get('max_replans', MAX_REPLANS)}): {reason}"
            self._save(goal, "replan_limit_reached", {"reason": reason})
            return False
        try:
            plan = await self._request_plan(goal, reason=reason, evidence=evidence)
            self._apply_plan(goal, plan)
            self._save(goal, "dag_replanned", {"graph_revision": goal["graph_revision"], "reason": reason})
            return True
        except (GoalPlanError, ValueError, RuntimeError) as exc:
            if isinstance(exc, GoalBudgetExceeded):
                raise
            goal["failure"] = f"DAG replan rejected: {exc}"
            self._save(goal, "dag_replan_rejected", {"reason": str(exc)})
            return False

    async def _execute_node(self, goal, node, git, graph_revision):
        async with self._semaphore:
            node = self._node(goal, node["id"])
            spec_attempt = int(node.get("attempts_used", 0)) + 1
            node["attempts_used"] = spec_attempt
            previous_attempt_numbers = [
                int(item.get("attempt_number", 0))
                for item in goal.get("attempts", [])
                if item.get("node_id") == node["id"]
            ]
            number = max(previous_attempt_numbers, default=0) + 1
            attempt_id = f"{node['id']}-a{number:02d}-{uuid.uuid4().hex[:6]}"
            input_sha = git.integration_head()
            worktree = git.create_worker(node["id"], number, input_sha)
            attempt = {
                "attempt_id": attempt_id,
                "dispatch_key": hashlib.sha256(f"{goal['goal_id']}:{node['id']}:{node.get('spec_revision', 1)}:{number}:{input_sha}".encode()).hexdigest(),
                "goal_id": goal["goal_id"],
                "node_id": node["id"],
                "graph_revision": int(graph_revision),
                "node_spec_revision": int(node.get("spec_revision", 1)),
                "node_fingerprint": node_fingerprint(node),
                "attempt_number": number,
                "spec_attempt_number": spec_attempt,
                "input_commit_sha": input_sha,
                "worktree": worktree["path"],
                "branch": worktree["branch"],
                "status": "running",
                "started_at": now(),
                "result": "",
                "error": "",
            }
            goal["attempts"].append(attempt)
            node["status"] = "running"
            self._save(goal, "worker_dispatched", {key: attempt[key] for key in ("attempt_id", "node_id", "graph_revision", "input_commit_sha")})
            workspace = WorkspaceContext.build(worktree["path"], repo_root_override=worktree["path"])
            child = self._role_runtime(
                "worker",
                workspace,
                write_scope=node["write_scope"],
                goal_id=goal["goal_id"],
                budget_role="worker",
            )
            dependencies = [self._node(goal, dep) for dep in node.get("depends_on", [])]
            context = {
                "goal_id": goal["goal_id"],
                "objective": goal["objective"],
                "base_commit": goal["base_commit"],
                "graph_revision": graph_revision,
                "node": {key: node.get(key) for key in ("id", "title", "prompt", "acceptance", "write_scope", "spec_revision")},
                "dependency_evidence": [{"id": dep["id"], "result": dep.get("result", ""), "commit": dep.get("integrated_commit", "")} for dep in dependencies],
                "prior_parent_answers": goal.get("parent_answers", {}).get(node["id"], ""),
                "prior_failure": node.get("last_error", ""),
            }
            prompt = (
                "You are a Goal Worker operating only in this attempt's isolated Git worktree and write_scope. "
                "Complete the assigned node and return a concise outcome with paths and validation evidence. "
                "Read-only describes the requested command's runtime behavior, not your coding permissions. "
                "The node's write_scope authorizes write_file and patch_file on those paths; "
                "do not ask for extra permission for in-scope edits. run_shell is unavailable to Workers; "
                "the parent runs declared verifiers after integration. Never edit outside write_scope. "
                "Keep repository exploration bounded: read only the files needed for this node, avoid broad or "
                "repeated searches, and start editing as soon as the relevant code is understood. "
                "If prior_failure reports step_limit_reached with changed_paths=none, do not repeat exploration; "
                "make the scoped edit before any further search. "
                "Do not change the Goal objective, permissions, or acceptance contract. If essential information "
                "is missing, your FINAL response must be exactly <goal_query>{JSON}</goal_query>, where JSON has "
                "missing, why, options (list), and impact (info|dependency|scope|permission|acceptance|safety). "
                "Never guess about safety, permissions, user intent, or acceptance criteria.\n\n"
                f"Goal/node context:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
            )
            result_text = ""
            try:
                result_text = await child.ask_async(prompt)
                query_count = 0
                while _is_goal_query(result_text):
                    query_count += 1
                    if query_count > MAX_QUERY_ROUNDS:
                        return await self._wait_for_parent(goal, node, attempt, result_text, "Worker exceeded query limit")
                    response = await self._resolve_parent_query(goal, node, attempt, result_text, child.workspace)
                    if response["kind"] == "wait":
                        return await self._wait_for_parent(goal, node, attempt, result_text, response["reason"], response.get("query"))
                    if response["kind"] == "replan":
                        return {
                            **attempt,
                            "status": "replan_requested",
                            "query": response["query"],
                            "error": "Worker requested a DAG change; parent will replan serially",
                            "worktree": worktree["path"],
                            "branch": worktree["branch"],
                            "input_commit_sha": input_sha,
                        }
                    result_text = await child.ask_async(response["answer"])
                task_state = getattr(child, "current_task_state", None)
                if task_state is not None and (
                    task_state.status != "completed"
                    or task_state.stop_reason != "final_answer_returned"
                ):
                    if task_state.stop_reason == STOP_REASON_GOAL_TOKEN_BUDGET_EXHAUSTED:
                        attempt["status"] = "budget_exhausted"
                        attempt["error"] = self._budget_failure_text(goal)
                        attempt["finished_at"] = now()
                        self._save(
                            goal,
                            "worker_token_budget_exhausted",
                            {"attempt_id": attempt_id, "node_id": node["id"]},
                        )
                        return {
                            **attempt,
                            "status": "budget_exhausted",
                            "worktree": worktree["path"],
                            "branch": worktree["branch"],
                            "input_commit_sha": input_sha,
                        }
                    if (
                        task_state.stop_reason == "step_limit_reached"
                        and _has_only_scoped_changes(
                            task_state.changed_paths, node["write_scope"]
                        )
                    ):
                        attempt["status"] = "candidate_ready"
                        attempt["completion_mode"] = "step_limit_with_scoped_changes"
                        attempt["finished_at"] = now()
                        attempt["result"] = clip(result_text, 3000)
                        self._save(
                            goal,
                            "worker_candidate_ready",
                            {
                                "attempt_id": attempt_id,
                                "stop_reason": task_state.stop_reason,
                                "changed_paths": list(task_state.changed_paths),
                            },
                        )
                        return {
                            **attempt,
                            "status": "completed",
                            "result": result_text,
                            "worktree": worktree["path"],
                            "branch": worktree["branch"],
                            "input_commit_sha": input_sha,
                        }
                    return {**attempt, "status": "failed", "result": result_text, "error": _worker_failure_summary(task_state)}
                attempt["result"] = clip(result_text, 3000)
                attempt["status"] = "completed"
                attempt["finished_at"] = now()
                self._save(goal, "worker_attempt_completed", {"attempt_id": attempt_id})
                return {**attempt, "status": "completed", "result": result_text, "worktree": worktree["path"], "branch": worktree["branch"], "input_commit_sha": input_sha}
            except asyncio.CancelledError:
                attempt["status"] = "cancelled"
                attempt["error"] = "Goal was cancelled"
                attempt["finished_at"] = now()
                self._save(goal, "worker_attempt_cancelled", {"attempt_id": attempt_id})
                raise
            except GoalBudgetExceeded as exc:
                attempt["status"] = "budget_exhausted"
                attempt["error"] = str(exc)[:1200]
                attempt["finished_at"] = now()
                self._save(goal, "worker_token_budget_exhausted", {"attempt_id": attempt_id, "node_id": node["id"]})
                return {**attempt, "status": "budget_exhausted", "error": attempt["error"], "worktree": worktree["path"], "branch": worktree["branch"], "input_commit_sha": input_sha}
            except Exception as exc:  # noqa: BLE001
                attempt["status"] = "failed"
                attempt["error"] = str(exc)[:1200]
                attempt["finished_at"] = now()
                self._save(goal, "worker_attempt_failed", {"attempt_id": attempt_id, "error": attempt["error"]})
                return {**attempt, "status": "failed", "error": attempt["error"], "worktree": worktree["path"], "branch": worktree["branch"], "input_commit_sha": input_sha}

    async def _resolve_parent_query(self, goal, node, attempt, raw, workspace):
        query = _parse_goal_query(raw)
        query_id = "query_" + uuid.uuid4().hex[:10]
        query.update({"query_id": query_id, "goal_id": goal["goal_id"], "node_id": node["id"], "attempt_id": attempt["attempt_id"], "graph_revision": goal["graph_revision"], "status": "resolving", "created_at": now()})
        goal["queries"].append(query)
        self._save(goal, "worker_query_received", {"query_id": query_id, "node_id": node["id"], "impact": query["impact"]})
        if query["impact"] in {"dependency", "scope", "permission", "acceptance", "safety"}:
            if query["impact"] == "dependency" and int(goal.get("replan_count", 0)) < int(goal.get("max_replans", MAX_REPLANS)):
                query["status"] = "replan_requested"
                return {"kind": "replan", "query": query}
            query["status"] = "waiting_for_parent"
            self._save(goal, "goal_waiting_for_parent", {"query_id": query_id, "impact": query["impact"]})
            return {"kind": "wait", "query": query, "reason": f"Worker needs an explicit parent decision about {query['impact']}"}
        resolver = self._role_runtime(
            "Explore", workspace, goal_id=goal["goal_id"], budget_role="query_resolver"
        )
        prompt = (
            "Resolve a low-risk informational query from a Goal Worker. Answer only using the objective, "
            "node contract, repository evidence, and dependency artifacts below. If unsupported or if this "
            "would decide user intent, permissions, scope, or acceptance, return disposition=block. "
            "Return JSON only: {\"disposition\":\"answer|block\",\"answer\":\"...\",\"basis\":\"...\"}.\n"
            f"Goal objective: {goal['objective']}\nNode: {json.dumps(node, ensure_ascii=False)}\n"
            f"Query: {json.dumps(query, ensure_ascii=False)}\nRepository root: {workspace.repo_root}"
        )
        resolution_text = await resolver.ask_async(prompt)
        if getattr(getattr(resolver, "current_task_state", None), "stop_reason", "") == STOP_REASON_GOAL_TOKEN_BUDGET_EXHAUSTED:
            raise GoalBudgetExceeded(self._budget_failure_text(goal))
        resolution = _parse_json(resolution_text)
        if resolution.get("disposition") != "answer" or not str(resolution.get("answer", "")).strip():
            query["status"] = "waiting_for_parent"
            self._save(goal, "goal_waiting_for_parent", {"query_id": query_id, "reason": "query cannot be resolved from available evidence"})
            return {"kind": "wait", "query": query, "reason": "Parent could not safely resolve the Worker query"}
        query["status"] = "answered"
        query["answer"] = str(resolution["answer"]).strip()
        query["basis"] = str(resolution.get("basis", "")).strip()
        query["answered_at"] = now()
        self._save(goal, "parent_answered_worker_query", {"query_id": query_id})
        return {"kind": "answer", "answer": query["answer"], "query": query}

    async def _wait_for_parent(self, goal, node, attempt, raw, reason, query=None):
        if query is None:
            query = _parse_goal_query(raw, allow_empty=True)
            query.update({"query_id": "query_" + uuid.uuid4().hex[:10], "goal_id": goal["goal_id"], "node_id": node["id"], "attempt_id": attempt["attempt_id"], "graph_revision": goal["graph_revision"], "status": "waiting_for_parent", "created_at": now()})
            goal["queries"].append(query)
        self._attempt_update(goal, attempt["attempt_id"], status="waiting_for_parent", error=reason)
        self._save(goal, "worker_waiting_for_parent", {"attempt_id": attempt["attempt_id"], "query_id": query["query_id"]})
        return {**attempt, "status": "waiting_for_parent", "error": reason, "query_id": query["query_id"]}

    async def _run_verifiers(self, goal, git):
        records = []
        for argv in goal.get("verifiers", []):
            started = time.monotonic()
            timeout = min(MAX_VERIFIER_SECONDS, max(1, int(self._remaining(goal))))
            process = await asyncio.create_subprocess_exec(
                *_verifier_process_args(argv),
                cwd=str(git.integration_path),
                env=self.runtime.shell_env(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=1)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                raise
            record = {
                "argv": list(argv),
                "exit_code": int(process.returncode),
                "stdout": clip(stdout.decode("utf-8", errors="replace"), 8000),
                "stderr": clip(stderr.decode("utf-8", errors="replace"), 8000),
                "duration_ms": int((time.monotonic() - started) * 1000),
                "candidate_commit": git.integration_head(),
            }
            records.append(record)
        passed = bool(records) and all(item["exit_code"] == 0 for item in records)
        return {"passed": passed, "commands": records, "summary": "; ".join(f"{'PASS' if item['exit_code'] == 0 else 'FAIL'} {' '.join(item['argv'])} (exit {item['exit_code']})" for item in records)}

    async def _run_critic(self, goal, git, verifier):
        head = git.integration_head()
        workspace = WorkspaceContext.build(git.integration_path, repo_root_override=git.integration_path)
        critic = self._role_runtime(
            "Explore", workspace, goal_id=goal["goal_id"], budget_role="critic"
        )
        prompt = (
            "Act as a read-only adversarial code-review Critic for the exact integrated candidate below. "
            "Try to find a concrete defect, missing acceptance requirement, unsafe behavior, stale dependency, "
            "or test gap. Inspect files with read-only tools if needed. Do not request cosmetic changes. "
            "Return JSON only: {\"verdict\":\"pass|revise\",\"findings\":[{\"path\":\"relative path\",\"line\":1,\"reason\":\"...\",\"evidence\":\"...\"}]}. "
            "Pass only when findings is empty and the implementation satisfies the Goal.\n"
            f"Goal: {goal['objective']}\nAcceptance by node: {json.dumps([{ 'id': n['id'], 'acceptance': n['acceptance']} for n in goal['nodes']], ensure_ascii=False)}\n"
            f"Base commit: {goal['base_commit']}\nCandidate commit: {head}\n"
            f"Changed files: {json.dumps(git.changed_files(goal['base_commit'], head), ensure_ascii=False)}\n"
            f"Diff (may be clipped):\n{git.diff(goal['base_commit'], head)}\n"
            f"Verifier evidence: {json.dumps(verifier, ensure_ascii=False)[:12000]}"
        )
        try:
            response = await critic.ask_async(prompt)
        except Exception as exc:  # noqa: BLE001
            return {
                "verdict": "unavailable",
                "findings": [],
                "error": f"Critic request failed ({type(exc).__name__})",
                "candidate_commit": head,
            }
        task_state = getattr(critic, "current_task_state", None)
        if getattr(task_state, "stop_reason", "") == STOP_REASON_GOAL_TOKEN_BUDGET_EXHAUSTED:
            return {
                "verdict": "budget_exhausted",
                "findings": [],
                "error": self._budget_failure_text(goal),
                "candidate_commit": head,
            }
        if task_state is not None and (
            task_state.status != "completed"
            or task_state.stop_reason != "final_answer_returned"
        ):
            return {
                "verdict": "unavailable",
                "findings": [],
                "error": _critic_execution_failure(critic),
                "candidate_commit": head,
            }
        try:
            result = _parse_json(response)
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            return {
                "verdict": "invalid",
                "findings": [],
                "error": f"Critic response was invalid ({type(exc).__name__})",
                "candidate_commit": head,
            }
        verdict = result.get("verdict")
        findings = result.get("findings")
        if verdict not in {"pass", "revise"} or not isinstance(findings, list):
            return {
                "verdict": "invalid",
                "findings": [],
                "error": "Critic response violated the verdict/findings schema",
                "candidate_commit": head,
            }
        normalized = []
        for finding in findings:
            if not isinstance(finding, dict) or not str(finding.get("reason", "")).strip() or not str(finding.get("evidence", "")).strip():
                return {
                    "verdict": "invalid",
                    "findings": [],
                    "error": "Critic finding lacked concrete evidence",
                    "candidate_commit": head,
                }
            normalized.append({"path": str(finding.get("path", "")), "line": int(finding.get("line", 0) or 0), "reason": str(finding["reason"]).strip(), "evidence": str(finding["evidence"]).strip(), "candidate_commit": head})
        if verdict == "pass" and normalized:
            verdict = "revise"
        return {"verdict": verdict, "findings": normalized, "candidate_commit": head}

    def _role_runtime(
        self, role, workspace, write_scope=None, *, goal_id="", budget_role=""
    ):
        child = build_child_runtime(self.runtime, role, write_scope or [], workspace=workspace, read_only=role == "Explore")
        child.max_turn_seconds = min(int(self.runtime.max_turn_seconds), DEFAULT_GOAL_SECONDS)
        budget_role = budget_role or ("worker" if role == "worker" else "planner")
        role_limits = {
            "planner": (MAX_PLANNER_STEPS, MAX_PLANNER_OUTPUT_TOKENS),
            "worker": (MAX_WORKER_STEPS_PER_ATTEMPT, MAX_WORKER_OUTPUT_TOKENS),
            "critic": (MAX_CRITIC_STEPS, MAX_CRITIC_OUTPUT_TOKENS),
            "query_resolver": (MAX_QUERY_RESOLVER_STEPS, MAX_QUERY_RESOLVER_OUTPUT_TOKENS),
        }
        step_cap, output_cap = role_limits[budget_role]
        child.max_steps = min(int(self.runtime.max_steps), step_cap)
        child.max_new_tokens = min(int(self.runtime.max_new_tokens), output_cap)
        child.goal_role = budget_role
        child.goal_token_budget = self._token_budgets.get(goal_id)
        return child

    def _budget_failure_text(self, goal):
        usage = goal.get("token_usage", {}) or {}
        reason = str(usage.get("budget_exhausted_reason", "total token budget reached"))
        role = str(usage.get("budget_exhausted_role", ""))
        if role:
            reason = f"{role} token allocation reached"
        return (
            f"Goal token budget exhausted ({int(usage.get('total_tokens', 0) or 0)}/"
            f"{int(goal.get('max_total_tokens', DEFAULT_GOAL_TOKEN_BUDGET) or 0)} tokens; {reason})"
        )

    def _ready_nodes(self, goal):
        by_id = {node["id"]: node for node in goal["nodes"]}
        ready = []
        for node_id in topological_order(goal["nodes"]):
            node = by_id[node_id]
            if node.get("status") not in {"pending", "failed"}:
                continue
            if int(node.get("attempts_used", 0)) >= MAX_ATTEMPTS_PER_NODE:
                node["status"] = "failed"
                continue
            if all(by_id[dep].get("status") == "completed" for dep in node.get("depends_on", [])):
                ready.append(node)
        return ready

    @staticmethod
    def _node(goal, node_id):
        for node in goal.get("nodes", []):
            if node.get("id") == node_id:
                return node
        raise GoalPlanError(f"unknown node: {node_id}")

    @staticmethod
    def _attempt_update(goal, attempt_id, **updates):
        for attempt in goal.get("attempts", []):
            if attempt.get("attempt_id") == attempt_id:
                attempt.update(updates)
                attempt["updated_at"] = now()
                return attempt
        raise GoalPlanError(f"unknown attempt: {attempt_id}")

    def _recover_interrupted_attempts(self, goal):
        for attempt in goal.get("attempts", []):
            if attempt.get("status") == "running":
                attempt["status"] = "interrupted"
                attempt["error"] = "process ended before the attempt was integrated"
                self._node(goal, attempt["node_id"])["status"] = "pending"
        self.store.save(goal)

    def _cleanup_successful_workers(self, goal, git):
        cleaned = []
        for attempt in goal.get("attempts", []):
            if attempt.get("status") == "integrated":
                try:
                    git.remove_worker(attempt.get("worktree", ""), attempt.get("branch", ""))
                    attempt["worktree_cleaned"] = True
                    cleaned.append(attempt["attempt_id"])
                except GoalGitError as exc:
                    attempt["cleanup_error"] = str(exc)
        self._save(goal, "worker_worktrees_cleaned", {"attempt_ids": cleaned})

    def _remaining(self, goal):
        try:
            deadline = datetime.fromisoformat(goal["deadline_at"])
        except (KeyError, ValueError, TypeError):
            return 0
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        return (deadline - datetime.now(timezone.utc)).total_seconds()

    @staticmethod
    def _renew_execution_window(goal):
        try:
            seconds = int(goal.get("max_runtime_seconds", DEFAULT_GOAL_SECONDS))
        except (TypeError, ValueError):
            seconds = DEFAULT_GOAL_SECONDS
        seconds = max(1, min(seconds, DEFAULT_GOAL_SECONDS))
        previous_deadline = goal.get("deadline_at", "")
        deadline = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        goal["deadline_at"] = deadline.isoformat()
        goal["execution_window_count"] = int(goal.get("execution_window_count", 1) or 1) + 1
        return {
            "previous_deadline_at": previous_deadline,
            "deadline_at": goal["deadline_at"],
            "window_seconds": seconds,
            "window_number": goal["execution_window_count"],
        }

    def _save(self, goal, event, payload):
        goal["updated_at"] = now()
        self.store.save(goal)
        self._event(goal, event, payload)

    def _event(self, goal, event, payload):
        record = {"at": now(), "goal_id": goal["goal_id"], "graph_revision": goal.get("graph_revision", 0), "event": event, **dict(payload or {})}
        self.store.append_event(goal["goal_id"], record)
        self.runtime.session_event_bus.emit("goal_event", record)

    def _finish(self, goal, status, failure):
        goal["status"] = status
        goal["failure"] = str(failure or "")
        goal["updated_at"] = now()
        self.store.save(goal)
        self._event(goal, "goal_finished", {"status": status, "failure": goal["failure"], "integration_commit": goal.get("integration_commit", "")})


def _parse_json(text):
    raw = str(text or "").strip()
    for match in re.finditer(r"\{", raw):
        start = match.start()
        depth = 0
        quoted = False
        escaped = False
        for index in range(start, len(raw)):
            char = raw[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    parsed = json.loads(raw[start : index + 1])
                    if isinstance(parsed, dict):
                        return parsed
                    break
    raise ValueError("model response did not contain a JSON object")


def _worker_failure_summary(task_state):
    changed_paths = ",".join(str(path) for path in task_state.changed_paths if str(path)) or "none"
    last_tool = str(task_state.last_tool or "none")
    return (
        f"worker run ended as {task_state.status}: {task_state.stop_reason}; "
        f"tool_steps={task_state.tool_steps}; changed_paths={changed_paths}; last_tool={last_tool}"
    )


def _has_only_scoped_changes(changed_paths, write_scope):
    paths = [str(path or "").replace("\\", "/").strip("/") for path in changed_paths or []]
    scopes = [str(scope or "").replace("\\", "/").strip("/") for scope in write_scope or []]
    if not paths or not scopes:
        return False
    for path in paths:
        parsed = PurePosixPath(path)
        if not path or parsed.is_absolute() or ".." in parsed.parts:
            return False
        if not any(path == scope or path.startswith(f"{scope}/") for scope in scopes):
            return False
    return True


def _verifier_process_args(argv):
    command = list(argv)
    executable = str(command[0]).replace("\\", "/").rsplit("/", 1)[-1].lower()
    if os.name == "nt" and executable in {"npm", "pnpm", "yarn"}:
        return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", " ".join(command)]
    return command


def _is_goal_query(text):
    return bool(re.search(r"<goal_query>.*?</goal_query>", str(text or ""), re.DOTALL | re.IGNORECASE))


def _critic_execution_failure(critic):
    task_state = getattr(critic, "current_task_state", None)
    details = []
    if task_state is not None:
        stop_reason = str(getattr(task_state, "stop_reason", "") or "")
        if stop_reason:
            details.append(f"stop_reason={stop_reason}")
    error = (getattr(critic, "last_completion_metadata", {}) or {}).get("provider_error", {})
    if isinstance(error, dict):
        for key in ("code", "http_status"):
            value = error.get(key)
            if value not in (None, ""):
                details.append(f"{key}={value}")
    return "Critic execution failed" + (f" ({', '.join(details)})" if details else "")


def _parse_goal_query(text, allow_empty=False):
    match = re.search(r"<goal_query>(.*?)</goal_query>", str(text or ""), re.DOTALL | re.IGNORECASE)
    if match is None:
        if allow_empty:
            return {"missing": "", "why": "worker query protocol was invalid", "options": [], "impact": "acceptance"}
        raise ValueError("worker query must use <goal_query> JSON </goal_query>")
    value = json.loads(match.group(1).strip())
    if not isinstance(value, dict):
        raise TypeError("worker query must be a JSON object")
    impact = str(value.get("impact", "info"))
    allowed = {"info", "dependency", "scope", "permission", "acceptance", "safety"}
    if impact not in allowed:
        raise ValueError(f"invalid Worker query impact: {impact}")
    missing = str(value.get("missing", "")).strip()
    why = str(value.get("why", "")).strip()
    options = value.get("options", [])
    if not missing or not why or not isinstance(options, list):
        raise ValueError("worker query requires missing, why, and options")
    return {"missing": missing[:1000], "why": why[:1000], "options": [str(item)[:500] for item in options[:8]], "impact": impact}
