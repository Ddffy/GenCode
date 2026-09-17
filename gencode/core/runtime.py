"""Agent runtime state and composition.

GenCode owns session state, workspace context, memory, checkpoints, and persistence.
The turn control loop lives in core.engine; tool execution and model-output
parsing live in focused helper modules.
"""

import json
import os
import textwrap
import uuid
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..features import memory as memorylib, skills as skillslib
from ..features import repomap as repomaplib
from ..features.sandbox import SandboxConfig, SandboxRunner
from .compact import CompactManager
from .context_manager import ContextManager
from .context_orchestrator import ContextOrchestrator
from .engine import Engine
from .git_integration import GitIntegration
from . import model_output, tool_executor
from .native_messages import native_prompt_contract
from .model_router import ModelClientRouter
from .plan_mode import PlanModeController
from .permissions import PermissionChecker
from .run_store import RunStore
from .runtime_consumers import default_runtime_consumers
from .runtime_checkpoints import RuntimeCheckpointsMixin
from .runtime_knowledge import RuntimeKnowledgeMixin
from .runtime_retrieval import RuntimeRetrievalMixin
from .runtime_events import build_runtime_event
from .runtime_secrets import REDACTED_VALUE, RuntimeSecretsMixin
from .session_events import SessionEventBus
from .session_lifecycle import clear_runtime_session, resume_runtime_session
from .session_store import SessionStore as SessionStore  # noqa: F401
from .tool_repetition import is_repeated_tool_call
from .tool_profiles import build_tool_profiles
from .todo_ledger import TodoLedger
from .turn_history import TurnHistoryBuilder
from .worker_manager import WorkerManager
from ..tools import registry as toolkit
from .workspace import MAX_HISTORY, WorkspaceContext, clip, now

DEFAULT_SHELL_ENV_ALLOWLIST = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "PATH",
    "PWD",
    "SHELL",
    "TERM",
    "TMPDIR",
    "TMP",
    "TEMP",
    "USER",
)
DEFAULT_FEATURE_FLAGS = dict(
    memory=True, relevant_memory=True, typed_knowledge=True,
    context_reduction=True, prompt_cache=True, repo_map=True, hybrid_retrieval=True
)
CHECKPOINT_SCHEMA_VERSION = "phase1-v1"
CHECKPOINT_NONE_STATUS = "no-checkpoint"
CHECKPOINT_FULL_VALID_STATUS = "full-valid"
CHECKPOINT_PARTIAL_STALE_STATUS = "partial-stale"
CHECKPOINT_WORKSPACE_MISMATCH_STATUS = "workspace-mismatch"
CHECKPOINT_SCHEMA_MISMATCH_STATUS = "schema-mismatch"
@dataclass
class PromptPrefix:
    # prefix 除了文本本身，还带一小份元数据，
    # 这样 runtime 才能明确判断 prefix 是否可以复用。
    text: str
    hash: str
    workspace_fingerprint: str
    tool_signature: str
    built_at: str


class GenCode(RuntimeSecretsMixin, RuntimeCheckpointsMixin, RuntimeKnowledgeMixin, RuntimeRetrievalMixin):
    def __init__(
        self,
        model_client,
        workspace,
        session_store,
        session=None,
        run_store=None,
        approval_policy="ask",
        max_steps=50,
        max_new_tokens=8192,
        depth=0,
        max_depth=1,
        read_only=False,
        shell_env_allowlist=None,
        secret_env_names=None,
        feature_flags=None,
        write_scope=None,
        memory_dir=None,
        auto_dream=True,
        dream_interval_hours=24.0,
        dream_min_sessions=5,
        model_client_factory=None,
        model_client_router=None,
        sandbox_config=None,
        ask_user_callback=None,
        allowed_tools=None,
        final_readiness_mode="warn",
        before_final_hooks=None,
        git_auto_commit=True,
        git_auto_undo=True,
    ):
        self.model_client = model_client
        self.model_client_factory = model_client_factory
        self.model_client_router = model_client_router or ModelClientRouter(model_client)
        self.abort_requested = False
        self.ask_user_callback = ask_user_callback
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.sandbox_runner = SandboxRunner(
            self.sandbox_config,
            emit_event=lambda event, payload: self.session_event_bus.emit(
                event, payload
            ),
        )
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.shell_env_allowlist = tuple(
            shell_env_allowlist or DEFAULT_SHELL_ENV_ALLOWLIST
        )
        self.secret_env_names = {str(name).upper() for name in (secret_env_names or ())}
        if isinstance(write_scope, str):
            write_scope = [write_scope]
        self.write_scope = tuple(
            str(path) for path in (write_scope or ()) if str(path).strip()
        )
        self.feature_flags = dict(DEFAULT_FEATURE_FLAGS)
        if feature_flags:
            self.feature_flags.update(
                {str(key): bool(value) for key, value in feature_flags.items()}
            )
        self.memory_dir = self._resolve_memory_dir(memory_dir)
        memorylib.ensure_memory_dir(self.memory_dir)
        self.auto_dream = bool(auto_dream)
        self.dream_interval_hours = float(dream_interval_hours)
        self.dream_min_sessions = int(dream_min_sessions)
        self.allowed_tools = self._normalize_allowed_tools(allowed_tools)
        self.final_readiness_mode = str(final_readiness_mode or "warn")
        self.before_final_hooks = tuple(before_final_hooks or ())
        self.git_auto_commit = bool(git_auto_commit)
        self.git_auto_undo = bool(git_auto_undo)
        self.run_store = run_store or RunStore(
            Path(workspace.repo_root) / ".gencode" / "runs"
        )
        self.session = session or {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": now(),
            "workspace_root": workspace.repo_root,
            "history": [],
            "memory": memorylib.default_memory_state(),
        }
        self._ensure_session_shape()
        self.session_event_bus = SessionEventBus(
            self.session["id"],
            self.session_store.event_path(self.session["id"]),
            redact=self.redact_artifact,
        )
        if (
            not self.session_event_bus.path.exists()
            or self.session_event_bus.path.stat().st_size == 0
        ):
            self.session_event_bus.emit(
                "session_started", {"workspace_root": workspace.repo_root}
            )
        self.git = GitIntegration(
            self.root, auto_commit=self.git_auto_commit, auto_undo=self.git_auto_undo, event_sink=self.session_event_bus.emit
        )
        self.initialize_knowledge()
        self.plan_mode = PlanModeController(self)
        self.engine = Engine(self)
        self.memory = memorylib.LayeredMemory(
            self.session.setdefault("memory", memorylib.default_memory_state()),
            workspace_root=self.root,
        )
        self.session["memory"] = self.memory.to_dict()
        self.self_authored_file_freshness = {}
        self.todo_ledger = TodoLedger(self)
        self.worker_manager = WorkerManager(self)
        self.skills = skillslib.discover_skills(self.root)
        self.tools = self._apply_tool_allowlist(self.build_tools())
        self.tool_profiles = build_tool_profiles(self.tools)
        self._active_tool_profile_name = (
            "plan"
            if self.runtime_mode == "plan"
            else "readonly"
            if self.read_only
            else "default"
        )
        self.permission_checker = PermissionChecker(self)
        self.prefix_state = self.build_prefix()
        self.prefix = self.prefix_state.text
        self.current_turn_id = ""
        self.current_run_id = ""
        self._trace_seq = 0
        self._last_trace_span_id = {}
        self.turn_history = TurnHistoryBuilder(self)
        self.compact_manager = CompactManager(self)
        self.runtime_consumers = default_runtime_consumers()
        self.context_manager = ContextManager(
            self,
            context_window=int(getattr(self.model_client, "context_window", 0) or 0),
        )
        self.context_orchestrator = ContextOrchestrator(self)
        self.resume_state = self.evaluate_resume_state()
        self.session_path = self.session_store.save(self.session)
        self.current_task_state = None
        self.current_run_dir = None
        self.last_prompt_metadata = {}
        self.last_completion_metadata = {}
        self.last_durable_promotions = []
        self.last_durable_rejections = []
        self.last_durable_superseded = []
        self.last_memory_maintenance = memorylib.default_memory_maintenance_audit(
            auto_dream=self.auto_dream
        )
        self.last_dream_changed_files = []
        self._memory_maintenance_thread = None
        self._last_tool_result_metadata = {}
        self._pending_tool_result_metadata = {}
        self._last_prefix_refresh = {
            "workspace_changed": False,
            "prefix_changed": False,
        }

    @classmethod
    def from_session(cls, model_client, workspace, session_store, session_id, **kwargs):
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    def _resolve_memory_dir(self, memory_dir):
        if memory_dir:
            path = Path(memory_dir).expanduser()
            path = path if path.is_absolute() else self.root / path
        else:
            path = self.root / ".gencode" / "memory"
        resolved = path.resolve()
        if os.path.commonpath([str(self.root), str(resolved)]) != str(self.root):
            raise ValueError(f"memory_dir must stay inside workspace: {memory_dir}")
        return resolved

    def _ensure_session_shape(self):
        self.session.setdefault("history", [])
        self.session.setdefault("memory", memorylib.default_memory_state())
        self.ensure_knowledge_session_shape()
        checkpoints = self.session.setdefault("checkpoints", {})
        if not isinstance(checkpoints, dict):
            checkpoints = {}
            self.session["checkpoints"] = checkpoints
        checkpoints.setdefault("current_id", "")
        checkpoints.setdefault("items", {})
        runtime_identity = self.session.setdefault("runtime_identity", {})
        if not isinstance(runtime_identity, dict):
            self.session["runtime_identity"] = {}
        resume_state = self.session.setdefault("resume_state", {})
        if not isinstance(resume_state, dict):
            self.session["resume_state"] = {}
        runtime_mode = self.session.setdefault("runtime_mode", {"mode": "default"})
        if not isinstance(runtime_mode, dict):
            self.session["runtime_mode"] = {"mode": "default"}

    def current_runtime_identity(self):
        return {
            "session_id": self.session.get("id", ""),
            "cwd": str(self.root),
            "model": str(getattr(self.model_client, "model", "")),
            "model_client": self.model_client.__class__.__name__,
            "approval_policy": self.approval_policy,
            "read_only": bool(self.read_only),
            "max_steps": int(self.max_steps),
            "max_new_tokens": int(self.max_new_tokens),
            "feature_flags": dict(self.feature_flags),
            "shell_env_allowlist": list(self.shell_env_allowlist),
            "workspace_fingerprint": getattr(
                getattr(self, "prefix_state", None),
                "workspace_fingerprint",
                self.workspace.fingerprint(),
            ),
            "tool_signature": self.tool_signature(),
        }

    def checkpoint_state(self):
        self._ensure_session_shape()
        return self.session["checkpoints"]

    def current_checkpoint(self):
        state = self.checkpoint_state()
        checkpoint_id = str(state.get("current_id", "")).strip()
        if not checkpoint_id:
            return None
        return state.get("items", {}).get(checkpoint_id)

    def invalidate_stale_memory(self):
        invalidated = self.memory.invalidate_stale_file_summaries()
        self.session["memory"] = self.memory.to_dict()
        return invalidated

    def evaluate_resume_state(self):
        previous_resume_state = dict(self.session.get("resume_state", {}) or {})
        invalidated = self.invalidate_stale_memory()
        checkpoint = self.current_checkpoint()
        status = CHECKPOINT_NONE_STATUS
        stale_paths = list(invalidated)
        mismatch_fields = []
        if checkpoint:
            if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                status = CHECKPOINT_SCHEMA_MISMATCH_STATUS
            else:
                for item in checkpoint.get("key_files", []):
                    path = str(item.get("path", "")).strip()
                    if not path:
                        continue
                    expected = item.get("freshness")
                    current = memorylib.file_freshness(path, self.root)
                    if expected != current and path not in stale_paths:
                        stale_paths.append(path)
                saved_identity = dict(
                    checkpoint.get("runtime_identity", {})
                    or self.session.get("runtime_identity", {})
                    or {}
                )
                current_identity = self.current_runtime_identity()
                identity_keys = (
                    "cwd",
                    "model",
                    "model_client",
                    "approval_policy",
                    "read_only",
                    "max_steps",
                    "max_new_tokens",
                    "feature_flags",
                    "shell_env_allowlist",
                    "workspace_fingerprint",
                    "tool_signature",
                )
                for key in identity_keys:
                    if key not in saved_identity:
                        continue
                    if saved_identity.get(key) != current_identity.get(key):
                        mismatch_fields.append(key)
                mismatch_fields.sort()
                if stale_paths:
                    status = CHECKPOINT_PARTIAL_STALE_STATUS
                elif mismatch_fields:
                    status = CHECKPOINT_WORKSPACE_MISMATCH_STATUS
                else:
                    status = CHECKPOINT_FULL_VALID_STATUS

        resume_state = {
            "status": status,
            "stale_paths": stale_paths,
            "runtime_identity_mismatch_fields": mismatch_fields,
            "stale_summary_invalidations": max(
                len(invalidated),
                int(previous_resume_state.get("stale_summary_invalidations", 0))
                if status == CHECKPOINT_PARTIAL_STALE_STATUS
                else 0,
            ),
        }
        self.session["resume_state"] = resume_state
        self.session["runtime_identity"] = self.current_runtime_identity()
        return resume_state

    def render_checkpoint_text(self):
        checkpoint = self.current_checkpoint()
        if not checkpoint:
            return ""
        lines = [
            "Task checkpoint:",
            f"- Resume status: {self.resume_state.get('status', CHECKPOINT_NONE_STATUS)}",
            f"- Current goal: {checkpoint.get('current_goal', '-') or '-'}",
            f"- Current blocker: {checkpoint.get('current_blocker', '-') or '-'}",
            f"- Next step: {checkpoint.get('next_step', '-') or '-'}",
        ]
        key_files = [
            str(item.get("path", "")).strip()
            for item in checkpoint.get("key_files", [])
            if str(item.get("path", "")).strip()
        ]
        lines.append(f"- Key files: {', '.join(key_files) or '-'}")
        if checkpoint.get("completed"):
            lines.append(
                "- Completed: "
                + " | ".join(str(item) for item in checkpoint.get("completed", []))
            )
        if checkpoint.get("excluded"):
            lines.append(
                "- Excluded: "
                + " | ".join(str(item) for item in checkpoint.get("excluded", []))
            )
        if self.resume_state.get("stale_paths"):
            lines.append(
                "- Stale paths: " + ", ".join(self.resume_state["stale_paths"])
            )
        summary = str(checkpoint.get("summary", "")).strip()
        if summary:
            lines.append(f"- Summary: {summary}")
        return "\n".join(lines)

    @staticmethod
    def remember(bucket, item, limit):
        if not item:
            return
        if item in bucket:
            bucket.remove(item)
        bucket.append(item)
        del bucket[:-limit]

    def build_tools(self):
        return toolkit.build_tool_registry(self)

    @staticmethod
    def _normalize_allowed_tools(allowed_tools):
        if allowed_tools is None:
            return None
        normalized = tuple(str(name).strip() for name in allowed_tools)
        if not normalized or any(not name for name in normalized):
            raise ValueError("allowed_tools must be a non-empty sequence of tool names")
        return normalized

    def _apply_tool_allowlist(self, tools):
        if self.allowed_tools is None:
            return tools
        unknown = [name for name in self.allowed_tools if name not in tools]
        if unknown:
            raise ValueError(f"unknown allowed tool: {', '.join(unknown)}")
        allowed = set(self.allowed_tools)
        return {name: tool for name, tool in tools.items() if name in allowed}

    @property
    def active_tool_profile(self):
        return self.tool_profiles[self._active_tool_profile_name]

    def set_tool_profile(self, name):
        if name not in self.tool_profiles:
            raise ValueError(f"unknown tool profile: {name}")
        self._active_tool_profile_name = name

    def available_tools(self):
        profile = self.active_tool_profile
        return {name: tool for name, tool in self.tools.items() if profile.allows(name)}

    def native_tool_definitions(self):
        """Return real JSON Schemas when the provider supports tool calling."""
        if not getattr(self.model_client, "supports_native_tool_calling", False):
            return []
        return toolkit.build_native_tool_definitions(self)

    def tool_signature(self):
        payload = []
        for name in sorted(self.available_tools()):
            tool = self.available_tools()[name]
            payload.append(
                {
                    "name": name,
                    "schema": tool["schema"],
                    "risky": tool["risky"],
                    "description": tool["description"],
                }
            )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def build_prefix(self):
        tool_lines = []
        for name, tool in self.available_tools().items():
            fields = ", ".join(
                f"{key}: {value}" for key, value in tool["schema"].items()
            )
            risk = "approval required" if tool["risky"] else "safe"
            tool_lines.append(f"- {name}({fields}) [{risk}] {tool['description']}")
        native_tools = bool(self.native_tool_definitions())
        tool_text, tool_protocol_rules, examples, final_rule, edit_rule = native_prompt_contract(
            self, native_tools, tool_lines
        )
        # prefix 可以理解成 agent 的“工作手册”：
        # 它是谁、工具怎么调用、当前仓库是什么状态，都写在这里。
        text = textwrap.dedent(
            f"""\
            You are gencode, a small local coding agent working inside a local repository.

            Rules:
            - Use tools instead of guessing about the workspace.
            {tool_protocol_rules}
            {edit_rule}
            {final_rule}
            - Never invent tool results.
            - Keep answers concise and concrete.
            - If the path is clear, write or patch directly; for multi-file deliverables, batch related writes in one response or one shell script and do not read back files you just wrote.
            - Before writing tests for existing code, read the implementation first.
            - When writing tests, match the current implementation unless the user explicitly asked you to change the code.
            - New files should be complete and runnable, including obvious imports.
            - Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or return a final answer.
            - Required tool arguments must not be empty. Do not call read_file, write_file, patch_file, run_shell, or agent with args={{}}. Git changes auto-commit when available; /undo reverts the latest safe change.
            - Use agent for bounded subagents. Explore is read-only; worker writes must stay inside write_scope.
            - Use send_message to continue an existing worker instead of spawning a fresh worker with missing context.
            - {skillslib.SKILL_FILE_CREATION_GUIDE}

            {self.runtime_mode_text()}

            Tools:
            {tool_text}

            Valid response examples:
            {examples}

            {self.workspace.text()}
            """
        ).strip()
        return PromptPrefix(
            text=text,
            hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            workspace_fingerprint=self.workspace.fingerprint(),
            tool_signature=self.tool_signature(),
            built_at=now(),
        )

    def _apply_prefix_state(self, prefix_state):
        self.prefix_state = prefix_state
        self.prefix = prefix_state.text

    def refresh_prefix(self, force=False):
        if not force and self.__dict__.pop("_reuse_workspace_snapshot_once", False):
            return dict(self._last_prefix_refresh)
        previous_hash = getattr(getattr(self, "prefix_state", None), "hash", None)
        previous_workspace_fingerprint = getattr(
            getattr(self, "prefix_state", None), "workspace_fingerprint", None
        )

        # 工作区事实相对稳定，所以这里按整体刷新；
        # 只有这些事实真的变化了，才重建完整 prefix。
        refreshed_workspace = WorkspaceContext.build(self.root)
        refreshed_workspace_fingerprint = refreshed_workspace.fingerprint()
        workspace_changed = (
            force or refreshed_workspace_fingerprint != previous_workspace_fingerprint
        )
        if workspace_changed:
            self.workspace = refreshed_workspace

        prefix_state = (
            self.build_prefix()
            if workspace_changed or force or previous_hash is None
            else self.prefix_state
        )
        prefix_changed = force or previous_hash != prefix_state.hash
        if prefix_changed:
            self._apply_prefix_state(prefix_state)

        self._last_prefix_refresh = {
            "workspace_changed": workspace_changed,
            "prefix_changed": prefix_changed,
        }
        return dict(self._last_prefix_refresh)

    def memory_text(self):
        return self.memory.render_memory_text()

    @property
    def runtime_mode(self):
        return str(
            self.session.get("runtime_mode", {}).get("mode", "default") or "default"
        )

    def runtime_mode_text(self):
        return self.plan_mode.prompt_text()

    def enter_plan_mode(self, topic, path=None):
        return self.plan_mode.enter(topic, path=path)

    def exit_plan_mode(self):
        return self.plan_mode.exit()

    def history_text(self):
        history = self.session["history"]
        if not history:
            return "- empty"

        lines = []
        seen_reads = set()
        recent_start = max(0, len(history) - 6)
        for index, item in enumerate(history):
            recent = index >= recent_start
            if item["role"] == "tool" and item["name"] == "read_file" and not recent:
                path = str(item["args"].get("path", ""))
                if path in seen_reads:
                    continue
                seen_reads.add(path)

            if item["role"] == "tool":
                limit = 900 if recent else 180
                lines.append(
                    f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}"
                )
                lines.append(clip(item["content"], limit))
            else:
                limit = 900 if recent else 220
                lines.append(f"[{item['role']}] {clip(item['content'], limit)}")

        return clip("\n".join(lines), MAX_HISTORY)

    def feature_enabled(self, name):
        return bool(self.feature_flags.get(str(name), False))

    def repo_map_builder(self):
        # 懒加载：只有真正生成 map 时才扫描仓库，未装 language-pack 时零开销。
        builder = getattr(self, "_repo_map_builder", None)
        if builder is None:
            builder = repomaplib.RepoMapBuilder(self.root)
            self._repo_map_builder = builder
        return builder

    def build_repo_map(self, query="", budget_chars=6000, recent_paths=None):
        """Build the repo map; returns ("", meta) when disabled or unsupported."""
        if not self.feature_enabled("repo_map"):
            return "", {"enabled": False, "reason": "feature_disabled"}
        return self.repo_map_builder().build(
            query=str(query or ""),
            budget_chars=budget_chars,
            recent_paths=list(recent_paths or ()),
        )

    def build_repo_map_section(self, user_message):
        return self.build_retrieval_section(user_message)

    def prompt(self, user_message):
        prompt, _ = self._build_prompt_and_metadata(user_message)
        return prompt

    def record(self, item):
        self.session["history"].append(self.turn_history.enrich(item))
        self.session_path = self.session_store.save(self.session)

    def prompt_metadata(self, user_message, prompt):
        _, metadata = self._build_prompt_and_metadata(user_message)
        return metadata

    def _build_prompt_and_metadata(self, user_message):
        refresh = self.refresh_prefix()
        self.resume_state = self.evaluate_resume_state()
        snapshot = self.context_orchestrator.snapshot(user_message, prefix_refresh=refresh)
        result = self.context_orchestrator.build(snapshot)
        return result.prompt, result.metadata

    def compact_history(self, trigger="manual", keep_recent_turns=2, summary_mode="deterministic"):
        return self.compact_manager.compact(
            trigger=trigger, keep_recent_turns=keep_recent_turns, summary_mode=summary_mode
        )

    def durable_memory_index_text(self):
        return memorylib.load_memory_index_text(self.memory_dir)

    def remember_durable_note(self, text):
        path = memorylib.append_to_daily_log(self.memory_dir, text)
        if path:
            self.session_event_bus.emit(
                "memory_note_appended",
                {
                    "source": "slash_command",
                    "path": memorylib._agent_relative_path(self, path),
                    "chars": len(str(text).strip()),
                },
            )
        return path

    def memory_command_text(self):
        index = self.durable_memory_index_text()
        if index:
            return index
        return "No durable memories yet. Use /remember <text> and /dream to consolidate daily logs."

    def run_dream(self, quiet=False, session_ids=None):
        return memorylib.run_dream(self, quiet=quiet, session_ids=session_ids)

    def maintain_memory_after_turn(self, final_answer):
        return memorylib.maintain_memory_after_turn(self, final_answer)

    def wait_for_memory_maintenance(self, timeout=None):
        thread = self._memory_maintenance_thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def emit_trace(self, task_state, event, payload=None):
        payload = self.redact_artifact(payload or {})
        for path in payload.get("affected_paths", []) or []:
            if path not in task_state.changed_paths:
                task_state.changed_paths.append(path)
        payload = build_runtime_event(self, task_state, event, payload)
        self.run_store.append_trace(task_state, payload)
        for consumer in self.runtime_consumers:
            try:
                consumer.handle(self, task_state, payload)
            except Exception as exc:
                error = {
                    "consumer": consumer.__class__.__name__,
                    "event": str(event),
                    "span_id": str(payload.get("span_id", "")),
                    "message": clip(str(exc), 200),
                    "critical": bool(getattr(consumer, "critical", False)),
                }
                task_state.evidence_summaries.setdefault("runtime_consumer_errors", []).append(error)
                if error["critical"]:
                    task_state.evidence_summaries.setdefault("consumer_errors", []).append(error)
        self.run_store.write_task_state(task_state)
        return payload

    def infer_next_step(self, task_state):
        if task_state.status == "completed":
            return "No next step recorded."
        if task_state.stop_reason == "step_limit_reached":
            return "Resume from the latest checkpoint and continue the task."
        if task_state.last_tool:
            return f"Decide the next action after {task_state.last_tool}."
        return "Continue the task from the latest checkpoint."

    def update_memory_after_tool(self, name, args, result):
        """把少量高价值工具结果沉淀到 working memory。

        为什么存在：
        并不是每个工具结果都值得长期带进下一轮 prompt。完整结果已经进了
        `history`，这里只挑少量“下一轮大概率还会用到”的事实做提纯，
        例如最近读写过哪些文件、某个文件读出来的短摘要。

        输入 / 输出：
        - 输入：工具名 `name`、参数 `args`、执行结果 `result`
        - 输出：无显式返回值，副作用是更新 `self.memory`

        在 agent 链路里的位置：
        它发生在 `run_tool()` 真正执行完工具之后、下一轮 prompt 组装之前。
        也就是说：工具结果先进入完整历史，再由这个函数择优沉淀成轻量记忆。
        """
        path = args.get("path")
        if not path:
            return

        canonical_path = self.memory.canonical_path(path)
        if name in {"write_file", "patch_file"}:
            freshness = memorylib.file_freshness(canonical_path, self.root)
            if freshness:
                self.self_authored_file_freshness[canonical_path] = freshness
        # file_summaries 既是 prompt 上下文，也是 tool policy 的 prior-read 凭证。
        # 即使 memory feature flag 关掉（如 dream agent），也必须维护 freshness，
        # 否则 patch_file/write_file 会被 prior_read_required 误拒。
        if name == "read_file":
            summary = memorylib.summarize_read_result(result)
            self.memory.set_file_summary(canonical_path, summary)
        if not self.feature_enabled("memory"):
            return
        if name in {"read_file", "write_file", "patch_file"}:
            self.memory.remember_file(canonical_path)
        if name == "read_file":
            self.memory.append_note(
                summary, tags=(canonical_path,), source=canonical_path
            )
        elif name in {"write_file", "patch_file"}:
            self.memory.invalidate_file_summary(canonical_path)

    def note_tool(self, name, args, result):
        self.update_memory_after_tool(name, args, result)

    def record_process_note_for_tool(self, name, metadata):
        status = str(metadata.get("tool_status", "")).strip()
        if status not in {"partial_success", "error", "rejected"}:
            return
        affected_paths = [
            str(path).strip()
            for path in metadata.get("affected_paths", [])
            if str(path).strip()
        ]
        path_text = ", ".join(affected_paths) or "workspace"
        if status == "partial_success":
            text = f"{name} partial_success on {path_text}; inspect diff before retry"
        elif status == "error":
            text = f"{name} error on {path_text}; check the failure before retry"
        else:
            text = f"{name} rejected; choose a different action before retry"
        tags = ["process", status, *affected_paths]
        self.memory.append_note(text, tags=tuple(tags), source=name, kind="process")
        self.session["memory"] = self.memory.to_dict()

    def reject_durable_reason(self, note_text):
        return memorylib.reject_durable_reason(note_text, redacted_value=REDACTED_VALUE)

    def extract_durable_promotions(self, user_message, final_answer):
        return memorylib.extract_durable_promotions(
            user_message, final_answer, redacted_value=REDACTED_VALUE
        )

    def promote_durable_memory(self, user_message, final_answer):
        return memorylib.promote_durable_memory(self, user_message, final_answer)

    def ask(self, user_message):
        return self.engine.ask(user_message)

    def abort_current_turn(self):
        self.abort_requested = True
        abort = getattr(self.model_client, "abort", None)
        if callable(abort):
            try:
                abort()
            except Exception:
                pass

    def ask_user(self, question, choices=None):
        if self.ask_user_callback is None:
            return "error: ask_user requires interactive mode"
        choices = [str(choice) for choice in (choices or [])]
        return str(self.ask_user_callback(str(question), choices))

    def resume_session(self, session_id):
        return resume_runtime_session(self, session_id)

    def clear_session(self):
        return clear_runtime_session(self)

    def run_tool(self, name, args):
        return tool_executor.run_tool(self, name, args)

    def repeated_tool_call(self, name, args):
        return is_repeated_tool_call(self.session["history"], name, args)

    @staticmethod
    def new_task_id():
        return (
            "task_"
            + datetime.now().strftime("%Y%m%d-%H%M%S")
            + "-"
            + uuid.uuid4().hex[:6]
        )

    @staticmethod
    def new_run_id():
        return (
            "run_"
            + datetime.now().strftime("%Y%m%d-%H%M%S")
            + "-"
            + uuid.uuid4().hex[:6]
        )

    def build_report(self, task_state):
        # report 是一次运行的最终摘要；
        return {
            "run_id": task_state.run_id,
            "task_id": task_state.task_id,
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "final_answer": task_state.final_answer,
            "tool_steps": task_state.tool_steps,
            "attempts": task_state.attempts,
            "checkpoint_id": task_state.checkpoint_id,
            "resume_status": task_state.resume_status,
            "task_state": task_state.to_dict(),
            "prompt_metadata": self.last_prompt_metadata,
            "durable_promotions": list(self.last_durable_promotions),
            "durable_rejections": list(self.last_durable_rejections),
            "durable_superseded": list(self.last_durable_superseded),
            "memory_maintenance": dict(self.last_memory_maintenance),
            **self.knowledge_report_fields(),
            "redacted_env": self.detected_secret_env_summary(),
            "compactions": list(self.session.get("compactions", [])),
            "artifact_graph": dict(task_state.artifact_graph),
            "verifier_suggestions": list(task_state.verifier_suggestions),
            "runtime_reminders": list(task_state.runtime_reminders),
            "todos": self.todo_ledger.to_dict(),
            "todo_changes": list(task_state.todo_changes),
            "evidence_summaries": dict(task_state.evidence_summaries),
            "workers": self.worker_manager.to_dict(),
            "git": self.git.report(),
        }

    def tool_example(self, name):
        return toolkit.tool_example(name)

    def validate_tool(self, name, args):
        """把通用工具校验和 runtime 级额外约束串起来。"""
        toolkit.validate_tool(self, name, args)

    def tool_run_shell(self, args):
        return toolkit.tool_run_shell(self, args)

    def approve(self, name, args):
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(
                f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] "
            )
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    parse = staticmethod(model_output.parse)
    retry_notice = staticmethod(model_output.retry_notice)
    parse_xml_tool = staticmethod(model_output.parse_xml_tool)
    parse_attrs = staticmethod(model_output.parse_attrs)
    extract = staticmethod(model_output.extract)
    extract_raw = staticmethod(model_output.extract_raw)

    def reset(self):
        self.session["history"] = []
        self.reset_knowledge_state()
        self.session["memory"].clear()
        self.session["memory"].update(memorylib.default_memory_state())
        self.memory = memorylib.LayeredMemory(
            self.session["memory"], workspace_root=self.root
        )
        self.self_authored_file_freshness.clear()
        self.session_store.save(self.session)

    def undo_last_git_commit(self):
        result = self.git.undo_last_commit(automatic=False)
        if result.get("git_undo_performed"):
            self.invalidate_stale_memory(); self.session_store.save(self.session)
            return result.get("git_undo_message", "[git] undo complete")
        return f"[git] undo skipped: {result.get('git_undo_reason', 'unknown')}"

    def path(self, raw_path):
        path = Path(raw_path)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        # 所有文件类工具都被锚定在 workspace root 之下。
        # 这样既能防住 "../" 逃逸，也能防住符号链接解析后跳出仓库。
        if os.path.commonpath([str(self.root), str(resolved)]) != str(self.root):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved
