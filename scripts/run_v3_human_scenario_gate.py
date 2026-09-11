#!/usr/bin/env python3
"""Run the prioritized GenCode human-scenario gate.

The runner intentionally drives GenCode through its public CLI entrypoint. It does
not import the GenCode runtime; verification reads only the files GenCode writes.
"""

from __future__ import annotations

import argparse
import json
import os
import pty
import select
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from gencode.evaluation.run_evidence import RunEvidence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / ".gencode.toml"
SUMMARY_JSON = "summary.json"
SUMMARY_MD = "summary.md"


@dataclass
class CommandRecord:
    name: str
    command: list[str]
    returncode: int
    duration_ms: int
    stdout_path: str
    stderr_path: str


@dataclass
class ScenarioResult:
    id: str
    title: str
    driver: str
    status: str
    workspace: str
    duration_ms: int
    checks: list[dict] = field(default_factory=list)
    commands: list[CommandRecord] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    error: str = ""


class HumanScenarioRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = Path(args.output_dir or Path("/tmp") / "gencode-human-scenarios" / stamp).resolve()
        if _is_relative_to(self.output_dir, ROOT):
            raise ValueError(
                "output-dir must be outside the GenCode repo; otherwise GenCode discovers "
                "the parent git root and the scenario no longer runs in an isolated workspace"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.workspaces_dir = self.output_dir / "workspaces"
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)
        self.config = Path(args.config).expanduser().resolve()

    def run(self) -> dict:
        selected = set(self.args.scenarios or [])
        gate_scenarios = [
            self.r01_student_management,
            self.r02_order_pricing_bugfix,
            self.r04_incident_resume_fix,
            self.r05_approval_inventory_importer,
            self.s07_repl_help,
            self.s15_plan_mode_active_artifact,
            self.s21_prior_read_required,
            self.s26_long_shell_output_artifact,
            self.s32_project_skill_arguments,
            self.s37_explore_subagent,
            self.s43_remember_daily_log,
            self.s50_path_traversal_and_redaction,
        ]
        full_scenarios = [
            self.r01_student_management,
            self.r02_order_pricing_bugfix,
            self.r03_release_readiness_skill,
            self.r04_incident_resume_fix,
            self.r05_approval_inventory_importer,
            self.s06_tty_default_tui,
            self.s07_repl_help,
            self.s08_prompt_one_shot,
            self.s09_piped_stdin_repl,
            self.s10_slash_suggestion_registry,
            self.s11_session_status,
            self.s12_usage_metadata,
            self.s13_model_runtime_switch,
            self.s14_clear_new_session,
            self.s15_plan_mode_active_artifact,
            self.s16_plan_final_gate,
            self.s17_absolute_plan_path,
            self.s18_plan_path_escape_rejected,
            self.s19_plan_allows_explore,
            self.s20_plan_rejects_worker_write,
            self.s21_prior_read_required,
            self.s22_new_file_then_overwrite_requires_read,
            self.s23_self_authored_patch,
            self.s24_shell_search_denied,
            self.s25_pipe_output_management_allowed,
            self.s26_long_shell_output_artifact,
            self.s27_approval_never_rejects_risky_tool,
            self.s28_sandbox_required_fails_closed,
            self.s29_sandbox_best_effort_degrades,
            self.s30_skills_list_local,
            self.s31_builtin_review_with_arguments,
            self.s32_project_skill_arguments,
            self.s33_skill_allowed_tools_restricts_write,
            self.s34_fork_skill_keeps_parent_history,
            self.s35_prompt_only_skill,
            self.s36_invalid_skill_frontmatter_diagnostic,
            self.s37_explore_subagent,
            self.s38_worker_write_scope,
            self.s39_worker_continuation,
            self.s40_running_worker_send_guard,
            self.s41_task_stop_worker,
            self.s42_clear_stops_worker,
            self.s43_remember_daily_log,
            self.s44_dream_writes_memory,
            self.s45_secret_memory_rejected,
            self.s46_manual_compact,
            self.s47_resume_workspace_mismatch,
            self.s48_provider_profiles,
            self.s49_provider_error_metadata,
            self.s50_path_traversal_and_redaction,
        ]
        scenarios = full_scenarios if self.args.suite == "full" else gate_scenarios
        results = []
        for scenario in scenarios:
            scenario_id = scenario.__name__.split("_", 1)[0].upper()
            if selected and scenario_id not in selected:
                continue
            started = time.monotonic()
            try:
                result = scenario()
            except Exception as exc:  # noqa: BLE001 - scenario runner must record failures.
                workspace = self.workspaces_dir / scenario_id.lower()
                result = ScenarioResult(
                    id=scenario_id,
                    title=scenario.__name__,
                    driver="unknown",
                    status="failed",
                    workspace=self._rel(workspace),
                    duration_ms=int((time.monotonic() - started) * 1000),
                    checks=[check("scenario_exception", False, str(exc))],
                    error=str(exc),
                )
            results.append(result)
            self._write_incremental_summary(results)
        return self._write_summary(results)

    def r01_student_management(self) -> ScenarioResult:
        workspace = self._fresh_workspace("r01")
        prompt = (
            "你在一个空 Python workspace。严格按步骤执行，每次只返回一个 <tool> 或最后一个 <final>："
            "1) write_file students.py，内容实现 Student dataclass、StudentStore，并提供 add_student/get_student/update_student/delete_student/list_students。"
            "2) write_file tests/test_students.py，覆盖增删改查和不存在学生返回 None/False。"
            "3) run_shell `uv run --with pytest python -m pytest -q`。"
            "4) 测试 passed 后 final。不要改其他文件。"
        )
        command = self.run_gencode("R01", workspace, prompt=prompt, max_steps=10, max_new_tokens=2048, timeout=420)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("students_py_exists", (workspace / "students.py").is_file()),
            check("student_tests_exist", (workspace / "tests" / "test_students.py").is_file()),
            check("external_pytest_passes", self.external_pytest(workspace)),
        ]
        checks.extend(self.run_artifact_checks(workspace, require_completed=True, require_changed_paths=True))
        return self.result("R01", "学生管理系统 CRUD 脚手架", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def r02_order_pricing_bugfix(self) -> ScenarioResult:
        workspace = self._fresh_workspace("r02")
        src = workspace / "src"
        tests = workspace / "tests"
        src.mkdir()
        tests.mkdir()
        (src / "__init__.py").write_text("", encoding="utf-8")
        (src / "order_pricing.py").write_text(
            "def calculate_total(subtotal, discount, tax):\n"
            "    return round(subtotal + discount + tax, 2)\n",
            encoding="utf-8",
        )
        (tests / "test_order_pricing.py").write_text(
            "from src.order_pricing import calculate_total\n\n\n"
            "def test_discount_is_subtracted_before_tax_is_added():\n"
            "    assert calculate_total(100, 15, 8.5) == 93.5\n",
            encoding="utf-8",
        )
        prompt = (
            "订单总价折扣计算错了。严格按步骤执行，每次只返回一个 <tool> 或最后一个 <final>："
            "1) read_file tests/test_order_pricing.py start=1 end=80。"
            "2) read_file src/order_pricing.py start=1 end=80。"
            "3) patch_file src/order_pricing.py，把 `return round(subtotal + discount + tax, 2)` "
            "替换为 `return round(subtotal - discount + tax, 2)`。"
            "4) run_shell `uv run --with pytest python -m pytest -q`。"
            "5) 测试 passed 后 final。不要改其他文件。"
        )
        command = self.run_gencode("R02", workspace, prompt=prompt, max_steps=9, max_new_tokens=1536, timeout=420)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("pricing_fixed", "subtotal - discount + tax" in (src / "order_pricing.py").read_text(encoding="utf-8")),
            check("external_pytest_passes", self.external_pytest(workspace)),
        ]
        checks.extend(self.trace_has_tools(workspace, ["read_file", "patch_file", "run_shell"]))
        return self.result("R02", "订单价格折扣 bugfix", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def r03_release_readiness_skill(self) -> ScenarioResult:
        workspace = self._fresh_workspace("r03")
        (workspace / "README.md").write_text("# Billing API\n\nRelease candidate.\n", encoding="utf-8")
        (workspace / ".env.example").write_text("DATABASE_URL=\nSTRIPE_API_KEY=\n", encoding="utf-8")
        (workspace / "deploy.md").write_text("- migrations applied\n- rollback owner assigned\n", encoding="utf-8")
        skill_dir = workspace / ".gencode" / "skills" / "release"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            """---
name: release
description: Release readiness reviewer
allowed-tools: read_file, write_file
---
严格按步骤执行，每次只返回一个 <tool> 或最后一个 <final>：
1) read_file README.md start=1 end=80。
2) read_file .env.example start=1 end=80。
3) read_file deploy.md start=1 end=80。
4) write_file reports/release-readiness.md，内容包含 Billing API、DATABASE_URL、STRIPE_API_KEY、rollback。
5) final。
""",
            encoding="utf-8",
        )
        command = self.run_gencode("R03", workspace, repl_input="/release billing-api\n/exit\n", max_steps=8, max_new_tokens=1536, timeout=420)
        report = workspace / "reports" / "release-readiness.md"
        text = report.read_text(encoding="utf-8") if report.exists() else ""
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("release_report_written", report.is_file()),
            check("release_report_mentions_config", "DATABASE_URL" in text and "STRIPE_API_KEY" in text),
            check("business_files_unchanged", "PAYMENT_WEBHOOK_SECRET" not in (workspace / ".env.example").read_text(encoding="utf-8")),
        ]
        checks.extend(self.events_have(workspace, "skill_invoked"))
        checks.extend(self.events_have(workspace, "skill_completed"))
        return self.result("R03", "发布就绪审查报告", "REPL project skill / DeepSeek", workspace, [command], checks)

    def r04_incident_resume_fix(self) -> ScenarioResult:
        workspace = self._fresh_workspace("r04")
        src = workspace / "src"
        tests = workspace / "tests"
        src.mkdir()
        tests.mkdir()
        (src / "__init__.py").write_text("", encoding="utf-8")
        (src / "incident_router.py").write_text(
            "def classify_latency(ms):\n"
            "    return 'ok' if ms < 1000 else 'page'\n",
            encoding="utf-8",
        )
        (tests / "test_incident_router.py").write_text(
            "from src.incident_router import classify_latency\n\n\n"
            "def test_degraded_threshold_routes_before_page():\n"
            "    assert classify_latency(750) == 'degraded'\n"
            "    assert classify_latency(1500) == 'page'\n",
            encoding="utf-8",
        )
        first_prompt = (
            "线上延迟告警误分级。严格按步骤执行，每次只返回一个 <tool> 或最后一个 <final>："
            "1) todo_add content='Fix latency incident routing' status='in_progress' priority='high'。"
            "2) read_file tests/test_incident_router.py start=1 end=80。"
            "3) read_file src/incident_router.py start=1 end=80。"
            "4) final，说明已定位，等待恢复继续。不要改代码。"
        )
        first = self.run_gencode("R04-first", workspace, prompt=first_prompt, max_steps=6, max_new_tokens=1536, timeout=360)
        first_session = self.latest_session_id(workspace)
        second_prompt = (
            "继续刚才的事故修复。严格按步骤执行，每次只返回一个 <tool> 或最后一个 <final>："
            "1) patch_file src/incident_router.py，把 `return 'ok' if ms < 1000 else 'page'` "
            "替换为 `return 'ok' if ms < 500 else 'degraded' if ms < 1000 else 'page'`。"
            "2) run_shell `uv run --with pytest python -m pytest -q`。"
            "3) todo_update todo_id='todo_1' status='done' note='threshold fixed and tests verified'。"
            "4) 测试 passed 后 final。不要改其他文件。"
        )
        second = self.run_gencode(
            "R04-resume",
            workspace,
            prompt=second_prompt,
            extra=["--resume", "latest"],
            max_steps=8,
            max_new_tokens=1536,
            timeout=420,
        )
        latest_session = self.latest_session_id(workspace)
        report = self.latest_report(workspace)
        todos = ((report or {}).get("todos") or {}).get("items", [])
        checks = [
            check("first_command_exit_0", first.returncode == 0),
            check("resume_command_exit_0", second.returncode == 0),
            check("same_session", first_session and first_session == latest_session, f"{first_session} -> {latest_session}"),
            check("todo_done", any(item.get("status") == "done" for item in todos), todos),
            check("incident_fixed", "degraded" in (src / "incident_router.py").read_text(encoding="utf-8")),
            check("external_pytest_passes", self.external_pytest(workspace)),
        ]
        return self.result("R04", "线上事故续接修复", "two one-shot CLI runs / DeepSeek", workspace, [first, second], checks)

    def r05_approval_inventory_importer(self) -> ScenarioResult:
        workspace = self._fresh_workspace("r05")
        prompt = (
            "写库存 CSV 导入器。严格按步骤执行，每次只返回一个 <tool> 或最后一个 <final>："
            "1) write_file inventory_importer.py，实现 import_inventory(path)，读取 sku,qty CSV，跳过空 sku、非数字或负数数量，返回 dict。"
            "2) write_file tests/test_inventory_importer.py，覆盖有效行、坏行跳过。"
            "3) run_shell `uv run --with pytest python -m pytest -q`。"
            "4) 测试 passed 后 final。不要改其他文件。"
        )
        command = self.run_gencode(
            "R05",
            workspace,
            prompt=prompt,
            approval="ask",
            stdin_text="y\ny\ny\n",
            max_steps=10,
            max_new_tokens=2048,
            timeout=420,
        )
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("inventory_importer_exists", (workspace / "inventory_importer.py").is_file()),
            check("inventory_tests_exist", (workspace / "tests" / "test_inventory_importer.py").is_file()),
            check("external_pytest_passes", self.external_pytest(workspace)),
        ]
        checks.extend(self.events_have(workspace, "permission_decision"))
        return self.result("R05", "库存 CSV 导入器审批路径", "one-shot CLI approval prompt / DeepSeek", workspace, [command], checks)

    def s07_repl_help(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s07")
        command = self.run_gencode("S07", workspace, repl_input="/help\n/exit\n", timeout=120)
        stdout = self.read_log(command.stdout_path)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("help_lists_commands", "Commands:" in stdout),
            check("help_lists_memory", "/memory" in stdout),
            check("help_lists_subagent", "/subagent" in stdout),
        ]
        return self.result("S07", "--repl + /help", "PTY-style stdin REPL", workspace, [command], checks)

    def s06_tty_default_tui(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s06")
        command = self.run_gencode_tty_smoke("S06", workspace, timeout=6)
        stdout = self.read_log(command.stdout_path)
        stderr = self.read_log(command.stderr_path)
        checks = [
            check("tui_process_started", command.returncode in {0, -2, 130, 143, 124}, command.returncode),
            check("no_traceback", "Traceback" not in stdout + stderr),
            check("mentions_gencode_or_tui", "gencode" in (stdout + stderr).lower() or "Textual" in stdout + stderr),
        ]
        return self.result("S06", "TTY 默认进入 TUI", "PTY TUI smoke", workspace, [command], checks)

    def s08_prompt_one_shot(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s08")
        (workspace / "README.md").write_text("# One shot\n\nGenCode one-shot fixture.\n", encoding="utf-8")
        prompt = "请只读 README 并返回 final。先 read_file README.md start=1 end=20，然后 <final>one-shot ok</final>。"
        command = self.run_gencode("S08", workspace, prompt=prompt, max_steps=4, max_new_tokens=768, timeout=240)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("stdout_has_answer", "one-shot" in self.read_log(command.stdout_path).lower()),
        ]
        checks.extend(self.run_artifact_checks(workspace, require_completed=True))
        checks.extend(self.trace_has_tools(workspace, ["read_file"]))
        return self.result("S08", "prompt 参数走 one-shot", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def s09_piped_stdin_repl(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s09")
        command = self.run_gencode("S09", workspace, repl_input="/help\n/exit\n", timeout=120)
        stdout = self.read_log(command.stdout_path)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("help_rendered", "Commands:" in stdout),
            check("exited_cleanly", "Traceback" not in stdout + self.read_log(command.stderr_path)),
        ]
        return self.result("S09", "piped stdin 使用 REPL", "piped stdin REPL", workspace, [command], checks)

    def s10_slash_suggestion_registry(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s10")
        command = self.run_python(
            "S10",
            workspace,
            "from gencode.commands.slash import suggest_commands\n"
            "items = suggest_commands('/sub')\n"
            "print(items[0].name if items else '')\n",
        )
        stdout = self.read_log(command.stdout_path)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("subagent_suggested", "subagent" in stdout),
        ]
        return self.result("S10", "TUI slash suggestion", "slash registry check", workspace, [command], checks)

    def s11_session_status(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s11")
        command = self.run_gencode("S11", workspace, repl_input="/plan refactor-auth\n/session\n/exit\n", timeout=120)
        stdout = self.read_log(command.stdout_path)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("session_id_printed", "session id:" in stdout),
            check("runtime_mode_plan", "runtime mode: plan" in stdout),
            check("worker_summary_printed", "worker summary:" in stdout),
        ]
        return self.result("S11", "/session 展示 runtime 状态", "PTY REPL slash command", workspace, [command], checks)

    def s12_usage_metadata(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s12")
        first = self.run_gencode(
            "S12-task",
            workspace,
            prompt="不要调用工具，直接返回 <final>usage seed</final>。",
            max_steps=2,
            max_new_tokens=512,
            timeout=180,
        )
        second = self.run_gencode("S12-usage", workspace, repl_input="/usage\n/exit\n", extra=["--resume", "latest"], timeout=120)
        stdout = self.read_log(second.stdout_path)
        checks = [
            check("task_exit_0", first.returncode == 0),
            check("usage_exit_0", second.returncode == 0),
            check("usage_has_provider", "provider protocol:" in stdout and "model:" in stdout),
            check("usage_redacts_key", "api_key" not in stdout.lower()),
        ]
        return self.result("S12", "/usage 展示 provider metadata", "one-shot + REPL resume", workspace, [first, second], checks)

    def s13_model_runtime_switch(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s13")
        command = self.run_gencode("S13", workspace, repl_input="/model gpt-test-local\n/model\n/exit\n", timeout=120)
        stdout = self.read_log(command.stdout_path)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("model_switched", "model: gpt-test-local" in stdout),
            check("no_workspace_config_written", not (workspace / ".gencode.toml").exists()),
        ]
        return self.result("S13", "/model 只改当前 runtime", "PTY REPL slash command", workspace, [command], checks)

    def s14_clear_new_session(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s14")
        command = self.run_gencode("S14", workspace, repl_input="/session\n/clear\n/session\n/exit\n", timeout=120)
        stdout = self.read_log(command.stdout_path)
        session_ids = [line.split("session id:", 1)[1].strip() for line in stdout.splitlines() if "session id:" in line]
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("two_session_ids_printed", len(session_ids) >= 2, session_ids),
            check("session_changed", len(session_ids) >= 2 and session_ids[0] != session_ids[-1], session_ids),
        ]
        return self.result("S14", "/clear 开新 session", "PTY REPL slash command", workspace, [command], checks)

    def s15_plan_mode_active_artifact(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s15")
        prompt = (
            "请验证 plan mode 写保护。你必须先返回这个工具调用，不要解释："
            "<tool name=\"write_file\" path=\"src/auth.py\"><content>print('no')\\n</content></tool>"
            "如果被拒绝，再写 active plan artifact .gencode/plans/auth-refactor-plan.md，内容为 # Plan。最后 final。"
        )
        repl = f"/plan auth-refactor\n{prompt}\n/exit\n"
        command = self.run_gencode("S15", workspace, repl_input=repl, max_steps=5, max_new_tokens=1536, timeout=420)
        commands = [command]
        plan_path = workspace / ".gencode" / "plans" / "auth-refactor-plan.md"
        if not plan_path.is_file():
            continue_prompt = (
                "继续上一个 plan mode。不要再写 src/auth.py。"
                "只调用一次 write_file，path 必须是 .gencode/plans/auth-refactor-plan.md，"
                "content 为 `# Plan\\n- Verified plan mode write guard.\\n`。然后 final。"
            )
            commands.append(
                self.run_gencode(
                    "S15-resume",
                    workspace,
                    prompt=continue_prompt,
                    extra=["--resume", "latest"],
                    max_steps=4,
                    max_new_tokens=1024,
                    timeout=300,
                )
            )
        stdout = "\n".join(self.read_log(item.stdout_path) for item in commands)
        checks = [
            check("command_exit_0", all(item.returncode == 0 for item in commands), [item.returncode for item in commands]),
            check("source_write_blocked", not (workspace / "src" / "auth.py").exists()),
            check("plan_written", plan_path.is_file()),
            check("stdout_mentions_plan_guard", "plan mode" in stdout.lower()),
        ]
        checks.extend(self.events_have(workspace, "permission_decision", reason="plan_mode_path_mismatch"))
        return self.result("S15", "plan mode 只能写 active plan", "REPL + resume / DeepSeek", workspace, commands, checks)

    def s16_plan_final_gate(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s16")
        prompt = (
            "先只返回 <final>plan verbally complete</final>。如果 runtime 提醒不能 final，"
            "再 write_file .gencode/plans/cache-plan.md，内容为 # Cache Plan，然后 final。"
        )
        command = self.run_gencode("S16", workspace, repl_input=f"/plan cache\n{prompt}\n/exit\n", max_steps=5, max_new_tokens=1024, timeout=360)
        stdout = self.read_log(command.stdout_path)
        plan_path = workspace / ".gencode" / "plans" / "cache-plan.md"
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("plan_gate_satisfied", "requires writing the active plan artifact" in stdout or plan_path.is_file()),
            check("plan_written_before_completion", plan_path.is_file()),
        ]
        return self.result("S16", "未写计划不能 final", "PTY REPL / DeepSeek", workspace, [command], checks)

    def s17_absolute_plan_path(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s17")
        absolute_plan = workspace / ".gencode" / "plans" / "student-plan.md"
        prompt = "write_file .gencode/plans/student-plan.md，内容为 # Student Plan，然后 final。"
        command = self.run_gencode(
            "S17",
            workspace,
            repl_input=f"/plan student {absolute_plan}\n{prompt}\n/exit\n",
            max_steps=4,
            max_new_tokens=1024,
            timeout=300,
        )
        stdout = self.read_log(command.stdout_path)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("absolute_path_normalized", "plan path: .gencode/plans/student-plan.md" in stdout),
            check("plan_written", absolute_plan.is_file()),
        ]
        return self.result("S17", "absolute plan path 自动归一", "PTY REPL / DeepSeek", workspace, [command], checks)

    def s18_plan_path_escape_rejected(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s18")
        command = self.run_gencode("S18", workspace, repl_input="/plan student .gencode/plans/../escape.md\n/exit\n", timeout=120)
        stdout = self.read_log(command.stdout_path)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("escape_rejected", "plan path must stay under .gencode/plans/" in stdout),
            check("outside_not_written", not (workspace / ".gencode" / "escape.md").exists()),
        ]
        return self.result("S18", "越界 plan path 被拒", "PTY REPL slash command", workspace, [command], checks)

    def s19_plan_allows_explore(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s19")
        (workspace / "README.md").write_text("# Payments\n\nExplore me.\n", encoding="utf-8")
        prompt = (
            "严格按步骤执行：先调用 agent 工具，description='Inspect README'，"
            "prompt='Read README.md and summarize it.'，subagent_type='Explore'。然后 final。"
        )
        command = self.run_gencode("S19", workspace, repl_input=f"/plan payments\n{prompt}\n/exit\n", max_steps=4, max_new_tokens=1536, timeout=420)
        report = self.evidence(workspace).report
        workers = ((report.get("workers") or {}).get("items") or [])
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("worker_recorded", bool(workers), workers),
            check("explore_worker", any(item.get("subagent_type") == "Explore" for item in workers), workers),
            check("plan_mode_was_entered", self.evidence(workspace).has_session_event("runtime_mode_changed", mode="plan")),
        ]
        return self.result("S19", "plan mode 允许 Explore 子 agent", "PTY REPL / DeepSeek", workspace, [command], checks)

    def s20_plan_rejects_worker_write(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s20")
        command = self.run_gencode(
            "S20",
            workspace,
            repl_input="/plan payments\n/subagent worker --scope src change code\n/exit\n",
            timeout=120,
        )
        stdout = self.read_log(command.stdout_path)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("worker_rejected", "plan mode only allows Explore agents" in stdout),
            check("src_not_created", not (workspace / "src").exists()),
        ]
        return self.result("S20", "plan mode 禁止 worker 写入", "PTY REPL slash command", workspace, [command], checks)

    def s21_prior_read_required(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s21")
        (workspace / "README.md").write_text("hello world\n", encoding="utf-8")
        prompt = (
            "验证改文件前必须先读。严格按步骤执行，每次只返回一个 <tool> 或最后一个 <final>："
            "1) 先尝试 patch_file README.md old_text='world' new_text='gencode'，不要先 read_file。"
            "2) 如果被拒绝，再 read_file README.md start=1 end=1。"
            "3) 再 patch_file README.md old_text='world' new_text='gencode'。"
            "4) final。"
        )
        command = self.run_gencode("S21", workspace, prompt=prompt, max_steps=7, max_new_tokens=1536, timeout=420)
        text = (workspace / "README.md").read_text(encoding="utf-8")
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("readme_patched_after_read", text == "hello gencode\n", text),
        ]
        checks.extend(self.report_has_runtime_reminder(workspace, "prior_read_required"))
        return self.result("S21", "改文件前必须先读", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def s22_new_file_then_overwrite_requires_read(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s22")
        (workspace / "README.md").write_text("old readme\n", encoding="utf-8")
        prompt = (
            "严格按步骤执行：1) write_file notes.txt 内容 note。"
            "2) 先尝试 write_file README.md 内容 overwrite，不要先读。"
            "3) 如果被拒绝，read_file README.md start=1 end=5。"
            "4) 再 write_file README.md 内容 overwrite。5) final。"
        )
        command = self.run_gencode("S22", workspace, prompt=prompt, max_steps=8, max_new_tokens=1536, timeout=420)
        notes_text = (workspace / "notes.txt").read_text(encoding="utf-8") if (workspace / "notes.txt").exists() else ""
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("new_file_written", notes_text == "note", notes_text),
            check("readme_overwritten", (workspace / "README.md").read_text(encoding="utf-8") == "overwrite"),
        ]
        checks.extend(self.report_has_runtime_reminder(workspace, "prior_read_required"))
        return self.result("S22", "新文件可直接写，覆盖必须读", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def s23_self_authored_patch(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s23")
        prompt = (
            "严格按步骤执行：1) write_file scripts/check.py 内容 `VALUE = False\\n`。"
            "2) patch_file scripts/check.py old_text='False' new_text='True'。3) final。"
        )
        command = self.run_gencode("S23", workspace, prompt=prompt, max_steps=5, max_new_tokens=1024, timeout=300)
        target = workspace / "scripts" / "check.py"
        text = target.read_text(encoding="utf-8") if target.exists() else ""
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("patch_succeeded", "VALUE = True" in text, text),
            check("no_prior_read_reminder", not self.evidence(workspace).runtime_reminder_contains("prior_read_required")),
        ]
        return self.result("S23", "自己刚写的文件可 patch", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def s24_shell_search_denied(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s24")
        (workspace / "README.md").write_text("TODO: search target\n", encoding="utf-8")
        prompt = "严格先调用 run_shell 命令 `grep -R TODO .`。如果被拒绝，调用 search pattern='TODO' path='.'，然后 final。"
        command = self.run_gencode("S24", workspace, prompt=prompt, max_steps=5, max_new_tokens=1024, timeout=300)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("shell_search_rejected", "shell_search_should_use_tool" in self.evidence(workspace).tool_error_codes("run_shell")),
        ]
        checks.extend(self.events_have(workspace, "tool_policy_decision", reason="shell_search_should_use_tool"))
        return self.result("S24", "shell 搜索类命令被拒", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def s25_pipe_output_management_allowed(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s25")
        command_text = f"{json.dumps(sys.executable)} --version 2>&1 | head -3"
        prompt = (
            "严格调用 run_shell 执行这个输出管理命令，然后 final："
            f"<tool>{{\"name\":\"run_shell\",\"args\":{{\"command\":{json.dumps(command_text)},\"timeout\":20}}}}</tool>"
        )
        command = self.run_gencode("S25", workspace, prompt=prompt, max_steps=3, max_new_tokens=768, timeout=240)
        trace_text, _ = self.latest_trace_and_report_text(workspace)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("run_shell_executed", self.evidence(workspace).has_tools("run_shell")),
            check("not_search_denied", "shell_search_should_use_tool" not in trace_text),
        ]
        return self.result("S25", "pipe 后 head/tail/grep 用于输出管理允许", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def s26_long_shell_output_artifact(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s26")
        command_text = f"{json.dumps(sys.executable)} -c \"print('x'*6000)\""
        prompt = (
            "严格返回 run_shell 工具调用执行这个命令，然后 final："
            f"<tool>{{\"name\":\"run_shell\",\"args\":{{\"command\":{json.dumps(command_text)},\"timeout\":20}}}}</tool>"
        )
        command = self.run_gencode("S26", workspace, prompt=prompt, max_steps=3, max_new_tokens=1024, timeout=240)
        artifact = self.latest_full_output_artifact(workspace)
        full_text = (workspace / artifact).read_text(encoding="utf-8") if artifact and (workspace / artifact).exists() else ""
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("full_output_artifact_recorded", bool(artifact), artifact),
            check("artifact_contains_full_output", "x" * 6000 in full_text),
        ]
        return self.result("S26", "长 shell 输出落 artifact", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def s27_approval_never_rejects_risky_tool(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s27")
        prompt = "严格调用 write_file denied.txt 内容 no，然后 final。"
        command = self.run_gencode("S27", workspace, prompt=prompt, approval="never", max_steps=3, max_new_tokens=768, timeout=240)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("file_not_written", not (workspace / "denied.txt").exists()),
            check("approval_denied", self.evidence(workspace).has_session_event("permission_decision", decision="deny", reason="approval_denied")),
        ]
        return self.result("S27", "approval never 拒绝 risky tool", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def s28_sandbox_required_fails_closed(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s28")
        prompt = "严格调用 run_shell `echo hi`，然后 final。"
        command = self.run_gencode(
            "S28",
            workspace,
            prompt=prompt,
            extra=["--sandbox", "required", "--sandbox-backend", "bubblewrap"],
            max_steps=3,
            max_new_tokens=768,
            timeout=240,
        )
        trace_text, report_text = self.latest_trace_and_report_text(workspace)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("sandbox_unavailable_visible", "sandbox required but unavailable" in trace_text + report_text),
            check("tool_failed", "tool_failed" in trace_text + report_text),
        ]
        return self.result("S28", "sandbox required 缺 backend fail closed", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def s29_sandbox_best_effort_degrades(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s29")
        prompt = "严格调用 run_shell `echo hi`，然后 final。"
        command = self.run_gencode(
            "S29",
            workspace,
            prompt=prompt,
            extra=["--sandbox", "best_effort", "--sandbox-backend", "bubblewrap"],
            max_steps=3,
            max_new_tokens=768,
            timeout=240,
        )
        trace_text, report_text = self.latest_trace_and_report_text(workspace)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("command_succeeded", "exit_code: 0" in trace_text + report_text),
            check("sandbox_event", self.evidence(workspace).has_session_event("sandbox_unavailable")),
        ]
        return self.result("S29", "sandbox best_effort degrade 可见", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def s30_skills_list_local(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s30")
        command = self.run_gencode("S30", workspace, repl_input="/skills\n/exit\n", timeout=120)
        stdout = self.read_log(command.stdout_path)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("skills_listed", "/review" in stdout and "/test" in stdout and "/commit" in stdout),
            check("no_run_created", not any((workspace / ".gencode" / "runs").glob("run_*")) if (workspace / ".gencode" / "runs").exists() else True),
        ]
        return self.result("S30", "/skills 不调用模型", "PTY REPL slash command", workspace, [command], checks)

    def s31_builtin_review_with_arguments(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s31")
        command = self.run_gencode(
            "S31",
            workspace,
            repl_input="/review focus auth\n/exit\n",
            max_steps=2,
            max_new_tokens=768,
            timeout=240,
        )
        stdout = self.read_log(command.stdout_path)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("review_answer", bool(stdout.strip()), stdout[-300:]),
            check("skill_invoked", self.evidence(workspace).has_session_event("skill_invoked", skill="review")),
        ]
        return self.result("S31", "内置 /review 带参数", "PTY REPL builtin skill / DeepSeek", workspace, [command], checks)

    def s32_project_skill_arguments(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s32")
        skill_dir = workspace / ".gencode" / "skills" / "deploy"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            """---
name: deploy
description: Deploy checklist
argument-hint: target
---
请只返回 <final>deploy checked $ARGUMENTS from ${GENCODE_SKILL_DIR}</final>，不要调用工具。
""",
            encoding="utf-8",
        )
        command = self.run_gencode("S32", workspace, repl_input="/deploy staging\n/exit\n", max_steps=3, max_new_tokens=512, timeout=240)
        stdout = self.read_log(command.stdout_path)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("stdout_mentions_argument", "staging" in stdout),
            check("stdout_mentions_skill_dir", str(skill_dir) in stdout),
        ]
        checks.extend(self.events_have(workspace, "skill_invoked"))
        return self.result("S32", "项目 skill 参数替换", "REPL slash skill / DeepSeek", workspace, [command], checks)

    def s33_skill_allowed_tools_restricts_write(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s33")
        skill_dir = workspace / ".gencode" / "skills" / "readonly"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            """---
name: readonly
description: Read-only checker
allowed-tools: read_file
---
严格先调用 write_file blocked.txt 内容 blocked。如果被拒绝，read_file README.md start=1 end=5，然后 final。
""",
            encoding="utf-8",
        )
        command = self.run_gencode("S33", workspace, repl_input="/readonly now\n/exit\n", max_steps=5, max_new_tokens=1024, timeout=300)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("blocked_file_absent", not (workspace / "blocked.txt").exists()),
            check("tool_not_allowed", self.evidence(workspace).has_session_event("permission_decision", decision="deny", reason="tool_not_allowed")),
        ]
        return self.result("S33", "allowed-tools 限制写操作", "PTY REPL project skill / DeepSeek", workspace, [command], checks)

    def s34_fork_skill_keeps_parent_history(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s34")
        skill_dir = workspace / ".gencode" / "skills" / "inspect"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            """---
name: inspect
description: Forked inspector
context: fork
---
不要调用工具，直接返回 <final>fork inspected $ARGUMENTS</final>。
""",
            encoding="utf-8",
        )
        first = self.run_gencode("S34-first", workspace, prompt="不要调用工具，直接返回 <final>parent seed</final>。", max_steps=2, max_new_tokens=512, timeout=180)
        second = self.run_gencode("S34-skill", workspace, repl_input="/inspect README.md\n/exit\n", extra=["--resume", "latest"], max_steps=2, max_new_tokens=512, timeout=240)
        events = self.evidence(workspace).session_events
        checks = [
            check("first_exit_0", first.returncode == 0),
            check("skill_exit_0", second.returncode == 0),
            check("skill_fork_completed", any(event.get("event") == "skill_fork_completed" for event in events), events[-5:]),
        ]
        return self.result("S34", "fork skill 不污染主 history", "one-shot + REPL fork skill", workspace, [first, second], checks)

    def s35_prompt_only_skill(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s35")
        skill_dir = workspace / ".gencode" / "skills" / "template"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            """---
name: template
description: Prompt-only template
disable-model-invocation: true
---
hello $ARGUMENTS from prompt only
""",
            encoding="utf-8",
        )
        command = self.run_gencode("S35", workspace, repl_input="/template world\n/exit\n", timeout=120)
        stdout = self.read_log(command.stdout_path)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("rendered_without_model", "hello world from prompt only" in stdout),
            check("no_run_created", not any((workspace / ".gencode" / "runs").glob("run_*")) if (workspace / ".gencode" / "runs").exists() else True),
            check("skill_completed_prompt_only", self.evidence(workspace).has_session_event("skill_completed", status="prompt_only")),
        ]
        return self.result("S35", "prompt-only skill 不发模型请求", "PTY REPL prompt-only skill", workspace, [command], checks)

    def s36_invalid_skill_frontmatter_diagnostic(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s36")
        skill_dir = workspace / ".gencode" / "skills" / "bad"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: bad\n---\nBad skill still loadable.\n", encoding="utf-8")
        command = self.run_gencode("S36", workspace, repl_input="/skills\n/exit\n", timeout=120)
        stdout = self.read_log(command.stdout_path)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("session_did_not_crash", "Traceback" not in stdout + self.read_log(command.stderr_path)),
            check("bad_skill_visible_with_default_description", "/bad" in stdout),
        ]
        return self.result("S36", "invalid skill frontmatter 可诊断", "PTY REPL slash command", workspace, [command], checks)

    def s37_explore_subagent(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s37")
        (workspace / "README.md").write_text("# Demo\n\nSubagent target.\n", encoding="utf-8")
        prompt = (
            "严格按步骤执行，每次只返回一个 <tool> 或最后一个 <final>："
            "1) agent description='Inspect README' prompt='Read README.md and summarize it in one sentence.' subagent_type='Explore'。"
            "2) 等待 worker notification 后 final。"
        )
        command = self.run_gencode("S37", workspace, prompt=prompt, max_steps=5, max_new_tokens=1536, timeout=420)
        report = self.latest_report(workspace) or {}
        workers = ((report.get("workers") or {}).get("items") or [])
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("worker_recorded", bool(workers), workers),
            check("worker_is_explore", any(item.get("subagent_type") == "Explore" for item in workers), workers),
        ]
        checks.extend(self.events_have(workspace, "worker_started"))
        return self.result("S37", "Explore 子 agent 只读探索", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def s38_worker_write_scope(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s38")
        prompt = (
            "严格调用 agent 工具：description='Write notes'，subagent_type='worker'，write_scope=['notes']，"
            "prompt='write_file notes/first.txt content first\\n and final'。然后 final。"
        )
        command = self.run_gencode("S38", workspace, prompt=prompt, max_steps=4, max_new_tokens=1536, timeout=420)
        report = self.evidence(workspace).report
        workers = ((report.get("workers") or {}).get("items") or [])
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("worker_recorded", bool(workers), workers),
            check("write_scope_notes", any(item.get("write_scope") == ["notes"] for item in workers), workers),
        ]
        return self.result("S38", "worker 只能写 scope 内", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def s39_worker_continuation(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s39")
        prompt = (
            "严格按步骤执行："
            "1) agent description='Notes worker' subagent_type='worker' write_scope=['notes'] "
            "prompt='write_file notes/first.txt content first\\n then final'。"
            "2) send_message to='agent_1' message='write_file notes/second.txt content second\\n then final'。"
            "3) final。"
        )
        command = self.run_gencode("S39", workspace, prompt=prompt, max_steps=6, max_new_tokens=1536, timeout=540)
        report = self.evidence(workspace).report
        workers = ((report.get("workers") or {}).get("items") or [])
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("same_worker_recorded", any(item.get("id") == "agent_1" for item in workers), workers),
            check("send_message_attempted", "send_message" in self.evidence(workspace).tool_names()),
        ]
        return self.result("S39", "worker 续接同一个 child context", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def s40_running_worker_send_guard(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s40")
        prompt = (
            "严格按步骤执行："
            "1) agent description='Slow worker' subagent_type='worker' write_scope=['notes'] "
            "prompt='run_shell python -c \"import time; time.sleep(5); print(1)\" then final'。"
            "2) 立刻 send_message to='agent_1' message='continue now'。3) final。"
        )
        command = self.run_gencode("S40", workspace, prompt=prompt, max_steps=5, max_new_tokens=1536, timeout=420)
        trace_text, report_text = self.latest_trace_and_report_text(workspace)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("send_guard_visible", "worker is running" in trace_text + report_text or "send_message" in self.evidence(workspace).tool_names()),
        ]
        return self.result("S40", "running worker 不能 send_message", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def s41_task_stop_worker(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s41")
        prompt = (
            "严格按步骤执行："
            "1) agent description='Stop target' subagent_type='worker' write_scope=['notes'] "
            "prompt='run_shell python -c \"import time; time.sleep(10)\" then final'。"
            "2) task_stop task_id='agent_1'。3) final。"
        )
        command = self.run_gencode("S41", workspace, prompt=prompt, max_steps=5, max_new_tokens=1536, timeout=420)
        report = self.evidence(workspace).report
        workers = ((report.get("workers") or {}).get("items") or [])
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("task_stop_attempted", "task_stop" in self.evidence(workspace).tool_names()),
            check("worker_status_recorded", bool(workers), workers),
        ]
        return self.result("S41", "task_stop 中止 worker", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def s42_clear_stops_worker(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s42")
        first = self.run_gencode(
            "S42-start",
            workspace,
            prompt=(
                "严格调用 agent description='Background' subagent_type='worker' write_scope=['notes'] "
                "prompt='run_shell python -c \"import time; time.sleep(10)\" then final'，然后 final。"
            ),
            max_steps=3,
            max_new_tokens=1024,
            timeout=240,
        )
        second = self.run_gencode("S42-clear", workspace, repl_input="/clear\n/agents\n/exit\n", extra=["--resume", "latest"], timeout=120)
        stdout = self.read_log(second.stdout_path)
        checks = [
            check("start_exit_0", first.returncode == 0),
            check("clear_exit_0", second.returncode == 0),
            check("worker_summary_none_after_clear", "worker summary: none" in stdout),
        ]
        return self.result("S42", "/clear 停掉后台 worker", "one-shot + REPL clear", workspace, [first, second], checks)

    def s43_remember_daily_log(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s43")
        command = self.run_gencode("S43", workspace, repl_input="/remember 这个项目用 pytest，不用 unittest\n/exit\n", timeout=120)
        logs = list((workspace / ".gencode" / "memory" / "logs").rglob("*.md"))
        text = "\n".join(path.read_text(encoding="utf-8") for path in logs)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("daily_log_created", bool(logs), [str(path) for path in logs]),
            check("daily_log_contains_note", "pytest" in text and "unittest" in text),
        ]
        checks.extend(self.events_have(workspace, "memory_note_appended"))
        return self.result("S43", "/remember 写 daily log", "PTY-style stdin REPL", workspace, [command], checks)

    def s44_dream_writes_memory(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s44")
        repl = (
            "/remember Project convention: use pytest for tests\n"
            "/remember Decision: keep GenCode artifacts under .gencode/runs\n"
            "/remember Project convention: prefer small files\n"
            "/dream\n"
            "/exit\n"
        )
        command = self.run_gencode("S44", workspace, repl_input=repl, max_steps=5, max_new_tokens=2048, timeout=420)
        memory_root = workspace / ".gencode" / "memory"
        memory_text = "\n".join(path.read_text(encoding="utf-8") for path in memory_root.rglob("*.md")) if memory_root.exists() else ""
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("memory_written", "pytest" in memory_text or "artifacts" in memory_text, memory_text[:300]),
            check("no_memory_escape", not (workspace.parent / "MEMORY.md").exists()),
        ]
        return self.result("S44", "/dream 写 topic 和 MEMORY.md", "PTY REPL / DeepSeek", workspace, [command], checks)

    def s45_secret_memory_rejected(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s45")
        prompt = (
            "请记住这些稳定事实到 durable memory，并把下面事实作为最终回答返回，不要调用工具："
            "<final>Dependency: API key is sk-live-secret-abc.\n"
            "Project convention: Use pytest for tests.</final>"
        )
        command = self.run_gencode("S45", workspace, prompt=prompt, max_steps=2, max_new_tokens=1024, timeout=240)
        memory_root = workspace / ".gencode" / "memory"
        memory_text = "\n".join(path.read_text(encoding="utf-8") for path in memory_root.rglob("*.md")) if memory_root.exists() else ""
        report = self.evidence(workspace).report
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("secret_not_promoted", "sk-live-secret-abc" not in memory_text),
            check("durable_rejection_recorded", any("secret_shaped" in item for item in report.get("durable_rejections", [])), report.get("durable_rejections")),
        ]
        return self.result("S45", "secret-shaped 记忆拒绝", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def s46_manual_compact(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s46")
        long_turns = "\n".join(f"第 {i} 轮：{'padding ' * 60}" for i in range(8))
        command = self.run_gencode(
            "S46",
            workspace,
            repl_input=f"{long_turns}\n/compact\n/exit\n",
            max_steps=2,
            max_new_tokens=512,
            timeout=300,
        )
        stdout = self.read_log(command.stdout_path)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("compact_output_json", "pre_tokens" in stdout and "post_tokens" in stdout),
            check("compaction_event", self.evidence(workspace).has_session_event("compaction_created")),
        ]
        return self.result("S46", "/compact 手动压缩历史", "PTY REPL slash command", workspace, [command], checks)

    def s47_resume_workspace_mismatch(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s47")
        first = self.run_gencode(
            "S47-first",
            workspace,
            prompt="read_file README.md start=1 end=20，然后 final。",
            max_steps=3,
            max_new_tokens=768,
            timeout=240,
        )
        (workspace / "README.md").write_text("# changed after checkpoint\n", encoding="utf-8")
        second = self.run_gencode(
            "S47-resume",
            workspace,
            prompt="不要改文件，只返回 <final>resume checked</final>。",
            extra=["--resume", "latest"],
            max_steps=2,
            max_new_tokens=512,
            timeout=180,
        )
        report = self.evidence(workspace).report
        checks = [
            check("first_exit_0", first.returncode == 0),
            check("resume_exit_0", second.returncode == 0),
            check("resume_status_visible", report.get("resume_status") in {"partial-stale", "workspace-mismatch", "full-valid"}, report.get("resume_status")),
        ]
        return self.result("S47", "resume 检测 workspace mismatch", "one-shot + resume", workspace, [first, second], checks)

    def s48_provider_profiles(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s48")
        commands = []
        checks = []
        for provider in ("openai", "anthropic", "deepseek"):
            command = self.run_gencode(
                f"S48-{provider}",
                workspace,
                prompt="/usage",
                extra=["--provider", provider],
                max_steps=1,
                max_new_tokens=128,
                timeout=120,
            )
            stdout = self.read_log(command.stdout_path)
            commands.append(command)
            checks.append(check(f"{provider}_exit_0", command.returncode == 0, command.returncode))
            checks.append(check(f"{provider}_usage_mentions_model", "model:" in stdout, stdout[-300:]))
            checks.append(check(f"{provider}_no_key_leak", "api_key" not in stdout.lower() and "sk-" not in stdout.lower()))
        return self.result("S48", "provider profile 切换", "one-shot slash /usage", workspace, commands, checks)

    def s49_provider_error_metadata(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s49")
        command = self.run_gencode(
            "S49",
            workspace,
            prompt="不要调用工具，直接返回 <final>provider ok</final>。",
            extra=["--base-url", "https://127.0.0.1:9/v1", "--openai-timeout", "2"],
            max_steps=1,
            max_new_tokens=64,
            timeout=60,
        )
        report = self.evidence(workspace).report
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("model_error_reported", report.get("status") == "failed" and report.get("stop_reason") == "model_error", report),
            check("provider_error_metadata", bool((report.get("prompt_metadata") or {}).get("provider_error")), report.get("prompt_metadata")),
        ]
        return self.result("S49", "provider 错误进入审计", "one-shot CLI bad endpoint", workspace, [command], checks)

    def s50_path_traversal_and_redaction(self) -> ScenarioResult:
        workspace = self._fresh_workspace("s50")
        outside = self.output_dir / "outside-secret.txt"
        outside.write_text("outside\n", encoding="utf-8")
        secret = "human-scenario-secret-12345"
        prompt = (
            "严格按步骤执行，每次只返回一个 <tool> 或最后一个 <final>："
            "1) read_file ../outside-secret.txt start=1 end=20。"
            "2) run_shell `printf 'human-scenario-secret-12345'`。"
            "3) final 时只说 safety checked，不要复述 secret。"
        )
        env = dict(os.environ)
        env["GENCODE_HUMAN_SECRET"] = secret
        command = self.run_gencode(
            "S50",
            workspace,
            prompt=prompt,
            extra=["--secret-env-name", "GENCODE_HUMAN_SECRET"],
            env=env,
            max_steps=5,
            max_new_tokens=1536,
            timeout=360,
        )
        trace_text, report_text = self.latest_trace_and_report_text(workspace)
        checks = [
            check("command_exit_0", command.returncode == 0),
            check("path_escape_recorded", "path escapes workspace" in trace_text or "path_escape" in trace_text),
            check("secret_not_in_trace_report", secret not in trace_text and secret not in report_text),
            check("redacted_marker_present", "<redacted>" in trace_text or "<redacted>" in report_text),
        ]
        return self.result("S50", "path traversal 与 secret redaction", "one-shot CLI / DeepSeek", workspace, [command], checks)

    def run_gencode(
        self,
        name: str,
        workspace: Path,
        *,
        prompt: str | None = None,
        repl_input: str | None = None,
        stdin_text: str = "",
        approval: str = "auto",
        extra: list[str] | None = None,
        env: dict | None = None,
        max_steps: int = 8,
        max_new_tokens: int = 1024,
        timeout: int = 300,
    ) -> CommandRecord:
        args = [
            "uv",
            "run",
            "gencode",
            "--cwd",
            str(workspace),
            "--config",
            str(self.config),
            "--provider",
            self.args.provider,
            "--approval",
            approval,
            "--no-auto-dream",
            "--max-steps",
            str(max_steps),
            "--max-new-tokens",
            str(max_new_tokens),
            "--temperature",
            "0",
        ]
        if extra:
            args.extend(extra)
        input_text = stdin_text
        if repl_input is not None:
            args.append("--repl")
            input_text = repl_input
        if prompt is not None:
            args.append(prompt)
        started = time.monotonic()
        try:
            proc = subprocess.run(
                args,
                cwd=ROOT,
                input=input_text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                env=env,
                check=False,
            )
            returncode = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired as exc:
            returncode = 124
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\nTIMEOUT after {timeout}s"
        duration_ms = int((time.monotonic() - started) * 1000)
        stdout_path = self.log_dir / f"{name}.stdout.txt"
        stderr_path = self.log_dir / f"{name}.stderr.txt"
        stdout_path.write_text(str(stdout), encoding="utf-8")
        stderr_path.write_text(str(stderr), encoding="utf-8")
        command_path = self.log_dir / f"{name}.command.json"
        command_path.write_text(
            json.dumps(
                {
                    "command": redact_command(args),
                    "returncode": returncode,
                    "duration_ms": duration_ms,
                    "workspace": str(workspace),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return CommandRecord(
            name=name,
            command=redact_command(args),
            returncode=returncode,
            duration_ms=duration_ms,
            stdout_path=self._rel(stdout_path),
            stderr_path=self._rel(stderr_path),
        )

    def run_gencode_tty_smoke(self, name: str, workspace: Path, *, timeout: int = 6) -> CommandRecord:
        args = [
            "uv",
            "run",
            "gencode",
            "--cwd",
            str(workspace),
            "--config",
            str(self.config),
            "--provider",
            self.args.provider,
            "--approval",
            "auto",
            "--no-auto-dream",
        ]
        started = time.monotonic()
        master_fd, slave_fd = pty.openpty()
        stdout_chunks: list[bytes] = []
        stderr = ""
        returncode = 124
        proc = None
        try:
            env = dict(os.environ)
            env.setdefault("TERM", "xterm-256color")
            proc = subprocess.Popen(
                args,
                cwd=ROOT,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                env=env,
                start_new_session=True,
            )
            os.close(slave_fd)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                ready, _, _ = select.select([master_fd], [], [], 0.2)
                if ready:
                    try:
                        data = os.read(master_fd, 4096)
                    except OSError:
                        break
                    if not data:
                        break
                    stdout_chunks.append(data)
                    if b"Traceback" in data:
                        break
                if proc.poll() is not None:
                    break
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=2)
            returncode = proc.returncode
        except Exception as exc:  # noqa: BLE001 - smoke runner records startup failures.
            stderr = str(exc)
            if proc and proc.poll() is None:
                proc.kill()
        finally:
            try:
                os.close(master_fd)
            except OSError:
                pass
            try:
                os.close(slave_fd)
            except OSError:
                pass
        duration_ms = int((time.monotonic() - started) * 1000)
        stdout_path = self.log_dir / f"{name}.stdout.txt"
        stderr_path = self.log_dir / f"{name}.stderr.txt"
        stdout_path.write_text(b"".join(stdout_chunks).decode("utf-8", errors="replace"), encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        (self.log_dir / f"{name}.command.json").write_text(
            json.dumps(
                {
                    "command": redact_command(args),
                    "returncode": returncode,
                    "duration_ms": duration_ms,
                    "workspace": str(workspace),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return CommandRecord(
            name=name,
            command=redact_command(args),
            returncode=returncode,
            duration_ms=duration_ms,
            stdout_path=self._rel(stdout_path),
            stderr_path=self._rel(stderr_path),
        )

    def run_python(self, name: str, workspace: Path, code: str, *, timeout: int = 120) -> CommandRecord:
        args = ["uv", "run", "python", "-c", code]
        started = time.monotonic()
        proc = subprocess.run(
            args,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        duration_ms = int((time.monotonic() - started) * 1000)
        stdout_path = self.log_dir / f"{name}.stdout.txt"
        stderr_path = self.log_dir / f"{name}.stderr.txt"
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        (self.log_dir / f"{name}.command.json").write_text(
            json.dumps(
                {
                    "command": redact_command(args),
                    "returncode": proc.returncode,
                    "duration_ms": duration_ms,
                    "workspace": str(workspace),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return CommandRecord(
            name=name,
            command=redact_command(args),
            returncode=proc.returncode,
            duration_ms=duration_ms,
            stdout_path=self._rel(stdout_path),
            stderr_path=self._rel(stderr_path),
        )

    def external_pytest(self, workspace: Path) -> bool:
        proc = subprocess.run(
            ["uv", "run", "--with", "pytest", "python", "-m", "pytest", "-q"],
            cwd=workspace,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
        )
        path = self.log_dir / f"{workspace.name}-external-pytest.txt"
        path.write_text(proc.stdout, encoding="utf-8")
        return proc.returncode == 0

    def run_artifact_checks(self, workspace: Path, *, require_completed: bool = False, require_changed_paths: bool = False) -> list[dict]:
        evidence = self.evidence(workspace)
        report = evidence.report
        checks = [
            check("report_exists", evidence.report_path is not None),
            check("trace_exists", evidence.trace_path is not None),
            check("task_state_exists", evidence.task_state_path is not None),
            check("session_events_exist", evidence.session_event_path is not None),
        ]
        if report and require_completed:
            checks.append(check("report_completed", evidence.status() == "completed", evidence.status()))
        if report and require_changed_paths:
            changed = evidence.changed_paths()
            checks.append(check("report_changed_paths", bool(changed), changed))
        return checks

    def trace_has_tools(self, workspace: Path, tools: list[str]) -> list[dict]:
        seen = self.evidence(workspace).tool_names()
        return [check(f"trace_has_{tool}", tool in seen, seen) for tool in tools]

    def latest_full_output_artifact(self, workspace: Path) -> str:
        artifacts = self.evidence(workspace).full_output_artifacts()
        return artifacts[-1] if artifacts else ""

    def events_have(self, workspace: Path, event_name: str, *, reason: str | None = None) -> list[dict]:
        events = self.evidence(workspace).session_events
        matched = [
            event
            for event in events
            if event.get("event") == event_name and (reason is None or event.get("reason") == reason)
        ]
        label = f"events_have_{event_name}" + (f"_{reason}" if reason else "")
        return [check(label, bool(matched), matched[:3])]

    def report_has_runtime_reminder(self, workspace: Path, code: str) -> list[dict]:
        reminders = self.evidence(workspace).report.get("runtime_reminders") or []
        return [check(f"runtime_reminder_{code}", any(code in json.dumps(item, ensure_ascii=False) for item in reminders), reminders)]

    def latest_trace_and_report_text(self, workspace: Path) -> tuple[str, str]:
        trace_path = self.latest_trace(workspace)
        report_path = self.latest_report_path(workspace)
        return (
            trace_path.read_text(encoding="utf-8") if trace_path else "",
            report_path.read_text(encoding="utf-8") if report_path else "",
        )

    def latest_report(self, workspace: Path) -> dict | None:
        path = self.latest_report_path(workspace)
        if path is None:
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def latest_report_path(self, workspace: Path) -> Path | None:
        run = self.latest_run_dir(workspace)
        path = run / "report.json" if run else None
        return path if path and path.exists() else None

    def latest_trace(self, workspace: Path) -> Path | None:
        run = self.latest_run_dir(workspace)
        path = run / "trace.jsonl" if run else None
        return path if path and path.exists() else None

    def latest_trace_jsonl(self, workspace: Path) -> list[dict]:
        path = self.latest_trace(workspace)
        return read_jsonl(path) if path else []

    def latest_events_path(self, workspace: Path) -> Path | None:
        events = sorted((workspace / ".gencode" / "sessions").glob("*.events.jsonl"), key=lambda path: path.stat().st_mtime)
        return events[-1] if events else None

    def latest_events_jsonl(self, workspace: Path) -> list[dict]:
        path = self.latest_events_path(workspace)
        return read_jsonl(path) if path else []

    def latest_run_dir(self, workspace: Path) -> Path | None:
        runs_dir = workspace / ".gencode" / "runs"
        if not runs_dir.exists():
            return None
        runs = [path for path in runs_dir.iterdir() if path.is_dir()]
        if not runs:
            return None
        return max(runs, key=lambda path: path.stat().st_mtime)

    def evidence(self, workspace: Path) -> RunEvidence:
        return RunEvidence.latest(workspace)

    def latest_session_id(self, workspace: Path) -> str:
        sessions = sorted((workspace / ".gencode" / "sessions").glob("*.json"), key=lambda path: path.stat().st_mtime)
        return sessions[-1].stem if sessions else ""

    def read_log(self, rel_path: str) -> str:
        return (self.output_dir / rel_path).read_text(encoding="utf-8")

    def result(self, scenario_id: str, title: str, driver: str, workspace: Path, commands: list[CommandRecord], checks: list[dict]) -> ScenarioResult:
        status = "passed" if all(item["status"] == "passed" for item in checks) else "failed"
        duration_ms = sum(command.duration_ms for command in commands)
        evidence = {
            "latest_report": self._rel(self.latest_report_path(workspace)) if self.latest_report_path(workspace) else "",
            "latest_trace": self._rel(self.latest_trace(workspace)) if self.latest_trace(workspace) else "",
            "latest_events": self._rel(self.latest_events_path(workspace)) if self.latest_events_path(workspace) else "",
        }
        return ScenarioResult(
            id=scenario_id,
            title=title,
            driver=driver,
            status=status,
            workspace=self._rel(workspace),
            duration_ms=duration_ms,
            checks=checks,
            commands=commands,
            evidence=evidence,
        )

    def _fresh_workspace(self, name: str) -> Path:
        workspace = self.workspaces_dir / name
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)
        (workspace / "README.md").write_text(f"# {name}\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
        return workspace

    def _write_incremental_summary(self, results: list[ScenarioResult]) -> None:
        self._write_summary(results, incremental=True)

    def _write_summary(self, results: list[ScenarioResult], incremental: bool = False) -> dict:
        summary = {
            "status": "passed" if results and all(item.status == "passed" for item in results) else "failed",
            "scenario_count": len(results),
            "passed": sum(1 for item in results if item.status == "passed"),
            "failed": sum(1 for item in results if item.status == "failed"),
            "suite": self.args.suite,
            "provider": self.args.provider,
            "config_path": str(self.config),
            "output_dir": str(self.output_dir),
            "results": [to_jsonable(item) for item in results],
        }
        (self.output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (self.output_dir / SUMMARY_MD).write_text(render_markdown(summary) + "\n", encoding="utf-8")
        if not incremental:
            print(json.dumps({"status": summary["status"], "passed": summary["passed"], "failed": summary["failed"], "output_dir": str(self.output_dir)}, ensure_ascii=False, sort_keys=True))
        return summary

    def _rel(self, path: Path | None) -> str:
        if path is None:
            return ""
        return Path(path).resolve().relative_to(self.output_dir).as_posix()


def check(name: str, condition: bool, detail="") -> dict:
    return {"name": name, "status": "passed" if condition else "failed", "detail": detail}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def redact_command(command: list[str]) -> list[str]:
    redacted = []
    skip_next = False
    for index, item in enumerate(command):
        if skip_next:
            redacted.append("<redacted>")
            skip_next = False
            continue
        redacted.append(item)
        if item == "--api-key" and index < len(command) - 1:
            skip_next = True
    return redacted


def to_jsonable(result: ScenarioResult) -> dict:
    return {
        "id": result.id,
        "title": result.title,
        "driver": result.driver,
        "status": result.status,
        "workspace": result.workspace,
        "duration_ms": result.duration_ms,
        "checks": result.checks,
        "commands": [command.__dict__ for command in result.commands],
        "evidence": result.evidence,
        "error": result.error,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# GenCode Human Scenario Gate",
        "",
        f"- status: `{summary['status']}`",
        f"- suite: `{summary['suite']}`",
        f"- provider: `{summary['provider']}`",
        f"- scenarios: `{summary['scenario_count']}`",
        f"- passed: `{summary['passed']}`",
        f"- failed: `{summary['failed']}`",
        "",
        "| ID | Status | Driver | Workspace | Evidence |",
        "|---|---|---|---|---|",
    ]
    for item in summary["results"]:
        evidence = item.get("evidence", {})
        evidence_text = "<br>".join(value for value in (evidence.get("latest_report"), evidence.get("latest_trace"), evidence.get("latest_events")) if value)
        lines.append(f"| {item['id']} | {item['status']} | {item['driver']} | `{item['workspace']}` | {evidence_text} |")
        failed = [check for check in item["checks"] if check["status"] != "passed"]
        for failed_check in failed:
            lines.append(f"| {item['id']} | failed-check | `{failed_check['name']}` |  | `{str(failed_check.get('detail', ''))[:180]}` |")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GenCode human-scenario release gate.")
    parser.add_argument("--suite", choices=("gate", "full"), default="gate", help="Run the 12-scenario release gate or all 50 designed scenarios.")
    parser.add_argument("--output-dir", default="", help="Output directory for logs, workspaces, and summary. Must be outside this git repo.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="GenCode config file. Defaults to this repo's ignored .gencode.toml.")
    parser.add_argument("--provider", default="deepseek", help="Provider profile to pass to GenCode.")
    parser.add_argument("--scenario", dest="scenarios", action="append", default=[], help="Run only one scenario id, repeatable.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    runner = HumanScenarioRunner(args)
    summary = runner.run()
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
