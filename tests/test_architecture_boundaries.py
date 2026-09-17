"""Architecture budget tests for runtime module boundaries."""

from pathlib import Path


def test_core_modules_stay_below_entropy_budget():
    root = Path(__file__).resolve().parents[1]
    budgets = {
        # Typed knowledge adds only assembly hooks here; policy lives in the
        # bounded mixin below rather than growing the runtime monolith.
        "gencode/core/runtime.py": 960,
        "gencode/core/runtime_knowledge.py": 130,
        # Citation validation lives here so the knowledge mixin stays an assembly
        # hook instead of growing the runtime monolith.
        "gencode/core/citation_report.py": 60,
        "gencode/core/before_final_hooks.py": 140,
        "gencode/core/evidence_summaries.py": 90,
        "gencode/core/final_readiness.py": 120,
        "gencode/core/final_readiness_artifacts.py": 160,
        "gencode/core/final_readiness_context.py": 60,
        "gencode/core/final_readiness_reasons.py": 60,
        "gencode/core/final_readiness_tools.py": 100,
        "gencode/core/governance.py": 80,
        "gencode/core/runtime_events.py": 90,
        "gencode/core/runtime_consumers.py": 90,
        "gencode/core/artifacts.py": 130,
        "gencode/core/task_state.py": 140,
        "gencode/core/todo_ledger.py": 120,
        "gencode/core/worker_manager.py": 220,
        "gencode/core/context_manager.py": 420,
        "gencode/core/knowledge_context.py": 60,
        "gencode/core/knowledge_governance.py": 60,
        "gencode/core/context_budget_summary.py": 130,
        "gencode/core/context_handoff.py": 240,
        "gencode/core/context_orchestrator.py": 210,
        "gencode/core/context_pressure.py": 140,
        "gencode/core/context_report.py": 140,
        "gencode/core/context_retention.py": 90,
        "gencode/core/context_replacements.py": 160,
        "gencode/core/context_sections.py": 190,
        "gencode/core/context_usage.py": 130,
        "gencode/core/compact.py": 250,
        "gencode/core/compact_summary.py": 130,
        "gencode/core/completion_governance.py": 240,
        "gencode/core/engine.py": 470,
        "gencode/core/model_errors.py": 100,
        "gencode/core/model_router.py": 40,
        "gencode/core/permissions.py": 140,
        "gencode/core/tool_policy.py": 90,
        "gencode/core/plan_mode.py": 140,
        "gencode/core/tool_executor.py": 181,
        "gencode/core/tool_profiles.py": 80,
        "gencode/core/tool_result_artifacts.py": 60,
        "gencode/core/turn_transitions.py": 90,
        # Includes cross-platform command classification for Windows paths.
        "gencode/core/verification.py": 100,
        "gencode/evaluation/commands.py": 80,
        "gencode/core/turn_history.py": 280,
        "gencode/core/media_history.py": 20,
        "gencode/features/skills.py": 220,
        "gencode/features/skills_bundled.py": 120,
        "gencode/features/skills_runtime.py": 140,
        "gencode/features/repomap/__init__.py": 120,
        "gencode/features/repomap/tags.py": 320,
        "gencode/features/repomap/fallback.py": 100,
        "gencode/features/repomap/graph.py": 220,
        # Hybrid BM25 + personalized PageRank implementation.
        "gencode/features/repomap/rank.py": 360,
        "gencode/features/repomap/render.py": 60,
        # Registry includes the built-in Repo Map and native-tool metadata.
        "gencode/tools/registry.py": 420,
        "gencode/tools/repomap.py": 60,
        "gencode/tools/todos.py": 80,
        "gencode/tools/agents.py": 90,
    }

    for relative_path, max_lines in budgets.items():
        line_count = len((root / relative_path).read_text(encoding="utf-8").splitlines())
        assert line_count <= max_lines, f"{relative_path} has {line_count} lines, budget is {max_lines}"
