"""Context cost experiment helpers."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import shutil
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType

from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.config import default_max_tokens_for_provider, resolve_provider_config
from gencode.core.run_store import RunStore
from gencode.providers import AnthropicCompatibleModelClient, OpenAICompatibleModelClient
from gencode.testing import ScriptedModelClient
from .commands import portable_verifier_command


@dataclass(frozen=True)
class ProviderPricing:
    input_per_1m: float
    cached_input_per_1m: float
    output_per_1m: float


@dataclass(frozen=True)
class CostUsage:
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    usage_source: str
    model_call_count: int

    @property
    def uncached_input_tokens(self) -> int:
        return max(0, int(self.input_tokens) - int(self.cached_tokens))


@dataclass(frozen=True)
class ExperimentRow:
    task_id: str
    layer: str
    variant: str
    repeat: int
    status: str
    verification_status: str
    tool_steps: int
    attempts: int
    prompt_estimated_tokens: int
    usage: CostUsage
    cost_usd: float
    saved_chars: int
    replacement_cache_hits: int
    summary_called: bool
    summary_delta_event_count: int
    report_path: str
    trace_path: str
    compact_summary_mode: str = ""
    compact_call_input_tokens: int = 0
    compact_call_output_tokens: int = 0
    compact_net_benefit_tokens: int | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["usage"] = asdict(self.usage)
        return payload


DEFAULT_PROXY_PRICING = ProviderPricing(
    input_per_1m=2.0,
    cached_input_per_1m=0.2,
    output_per_1m=8.0,
)

EXPERIMENT_VARIANTS = {
    "no_context_reduction": {
        "description": "Baseline: no context reduction features",
        "context_orchestrator_enabled": False,
    },
    "full_orchestrator": {
        "description": "All context reduction features, deterministic compact",
        "context_orchestrator_enabled": True,
        "compact_summary_mode": "deterministic",
    },
    "full_orchestrator_with_llm_handoff": {
        "description": "All features + LLM handoff compact when triggered",
        "context_orchestrator_enabled": True,
        "compact_summary_mode": "llm",
    },
}


def compute_cost_usd(usage: CostUsage, pricing: ProviderPricing) -> float:
    return (
        usage.uncached_input_tokens * pricing.input_per_1m
        + int(usage.cached_tokens) * pricing.cached_input_per_1m
        + int(usage.output_tokens) * pricing.output_per_1m
    ) / 1_000_000


def extract_usage_from_artifacts(
    report_path,
    trace_path,
    *,
    task_id,
    layer,
    variant,
    repeat,
    pricing,
    verification_status=None,
    allow_verification_override=False,
) -> ExperimentRow:
    report_path = Path(report_path)
    trace_path = Path(trace_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    trace_usage = _usage_from_trace(trace_path)
    compact_metrics = _compact_metrics_from_trace(trace_path)
    summary = dict(
        (report.get("evidence_summaries", {}) or {}).get("context_budget_summary", {})
        or {}
    )
    orchestrator = dict((report.get("prompt_metadata", {}) or {}).get("context_orchestrator", {}) or {})
    compact_call_usage = dict(
        compact_metrics.get("compact_call_usage")
        or summary.get("compact_call_usage")
        or orchestrator.get("compact_call_usage")
        or {}
    )
    if compact_call_usage and variant == "full_orchestrator_with_llm_handoff":
        trace_usage["usage"] = _usage_with_compact_call(
            trace_usage["usage"], compact_call_usage
        )
    derived_verification = _verification_status(report)
    if allow_verification_override and verification_status is not None:
        derived_verification = str(verification_status)
    return ExperimentRow(
        task_id=str(task_id),
        layer=str(layer),
        variant=str(variant),
        repeat=int(repeat),
        status=str(report.get("status", "")),
        verification_status=derived_verification,
        tool_steps=int(report.get("tool_steps", 0) or 0),
        attempts=int(report.get("attempts", 0) or 0),
        prompt_estimated_tokens=trace_usage["estimated_input_tokens"],
        usage=trace_usage["usage"],
        cost_usd=compute_cost_usd(trace_usage["usage"], pricing) if pricing else 0.0,
        saved_chars=int(summary.get("saved_chars", 0) or 0),
        replacement_cache_hits=int(summary.get("replacement_cache_hits", 0) or 0),
        summary_called=bool(compact_metrics.get("summary_called") or summary.get("summary_called", False)),
        summary_delta_event_count=int(
            compact_metrics.get("summary_delta_event_count")
            or summary.get("summary_delta_event_count", 0)
            or 0
        ),
        compact_summary_mode=str(compact_metrics.get("compact_summary_mode") or orchestrator.get("summary_mode", "") or ""),
        compact_call_input_tokens=int(compact_call_usage.get("input_tokens", 0) or 0),
        compact_call_output_tokens=int(compact_call_usage.get("output_tokens", 0) or 0),
        compact_net_benefit_tokens=(
            compact_metrics.get("compact_net_benefit_tokens")
            if compact_metrics.get("compact_net_benefit_tokens") is not None
            else summary.get("compact_net_benefit_tokens")
        ),
        report_path=report_path.as_posix(),
        trace_path=trace_path.as_posix(),
    )


def summarize_paired_rows(
    rows, *, treatment="full_orchestrator", control="no_context_reduction"
):
    rows = list(rows)
    pairs = _paired_rows(rows, treatment=treatment, control=control)
    actual_pairs = [
        pair for pair in pairs if _pair_usage_source(pair, treatment, control) == "actual"
    ]
    proxy_pairs = [
        pair
        for pair in pairs
        if _pair_usage_source(pair, treatment, control) == "estimated_proxy"
    ]
    mixed_pairs = [
        pair
        for pair in pairs
        if _pair_usage_source(pair, treatment, control) == "mixed_or_invalid"
    ]
    return {
        "actual_only": _summarize_pair_bucket(
            actual_pairs, treatment=treatment, control=control
        ),
        "estimated_proxy_only": _summarize_pair_bucket(
            proxy_pairs, treatment=treatment, control=control
        ),
        "mixed_or_invalid": _summarize_pair_bucket(
            mixed_pairs, treatment=treatment, control=control
        ),
        "real_usage_row_count": sum(
            1 for row in rows if row.usage.usage_source == "actual"
        ),
        "estimated_proxy_row_count": sum(
            1 for row in rows if row.usage.usage_source == "estimated_proxy"
        ),
    }


def run_deterministic_prompt_experiment(output_dir, repetitions=1, pricing=None):
    pricing = pricing or DEFAULT_PROXY_PRICING
    output_dir = Path(output_dir)
    rows = []
    for repeat in range(int(repetitions)):
        for variant, context_reduction in (
            ("full_orchestrator", True),
            ("no_context_reduction", False),
        ):
            workspace = output_dir / "runs" / "prompt-only" / variant / str(repeat)
            workspace.mkdir(parents=True, exist_ok=True)
            agent = _build_synthetic_agent(workspace, context_reduction=context_reduction)
            prompt, prompt_metadata = agent._build_prompt_and_metadata(  # noqa: SLF001
                "Summarize this workspace."
            )
            del prompt
            report_path = workspace / "report.json"
            trace_path = workspace / "trace.jsonl"
            _write_prompt_only_trace(trace_path, prompt_metadata)
            _write_prompt_only_report(report_path, prompt_metadata)
            rows.append(
                extract_usage_from_artifacts(
                    report_path,
                    trace_path,
                    task_id="prompt-only",
                    layer="deterministic",
                    variant=variant,
                    repeat=repeat,
                    pricing=pricing,
                    verification_status="passed",
                    allow_verification_override=True,
                )
            )
    return build_result_payload(rows, pricing_profile="proxy", pricing=pricing)


def run_scripted_e2e_experiment(output_dir, repetitions=1, pricing=None):
    pricing = pricing or DEFAULT_PROXY_PRICING
    output_dir = Path(output_dir)
    rows = []
    for repeat in range(int(repetitions)):
        for variant, context_reduction in (
            ("full_orchestrator", True),
            ("no_context_reduction", False),
        ):
            workspace = output_dir / "runs" / "scripted-large-read" / variant / str(repeat)
            workspace.mkdir(parents=True, exist_ok=True)
            agent = _build_scripted_agent(workspace, context_reduction=context_reduction)
            answer = agent.ask("Read large.txt and summarize it.")
            if answer != "done":
                raise AssertionError(f"unexpected scripted answer: {answer}")
            trace_events = _read_jsonl(agent.current_run_dir / "trace.jsonl")
            if not any(
                event.get("event") == "tool_executed"
                and _tool_name(event) == "read_file"
                for event in trace_events
            ):
                raise AssertionError("scripted task did not execute read_file")
            report = json.loads(
                (agent.current_run_dir / "report.json").read_text(encoding="utf-8")
            )
            verification = _verification_status(report)
            if verification == "unknown":
                verification = "passed"
            rows.append(
                extract_usage_from_artifacts(
                    agent.current_run_dir / "report.json",
                    agent.current_run_dir / "trace.jsonl",
                    task_id="scripted-large-read",
                    layer="scripted",
                    variant=variant,
                    repeat=repeat,
                    pricing=pricing,
                    verification_status=verification,
                    allow_verification_override=True,
                )
            )
    return build_result_payload(rows, pricing_profile="scripted-proxy", pricing=pricing)


def run_paired_experiment(
    tasks,
    *,
    variants=None,
    mode="scripted",
    provider=None,
    repetitions=1,
    output_dir=None,
    pricing=None,
    provider_client_factory=None,
):
    mode = str(mode)
    if mode not in {"scripted", "live"}:
        raise ValueError(f"unsupported experiment mode: {mode}")
    pricing = pricing or DEFAULT_PROXY_PRICING
    variants = list(variants or ["full_orchestrator", "full_orchestrator_with_llm_handoff"])
    for variant in variants:
        if variant not in EXPERIMENT_VARIANTS:
            raise ValueError(f"unknown experiment variant: {variant}")
    output_dir = Path(output_dir or "artifacts/llm-handoff-benchmark/work")
    rows = []
    for repeat in range(int(repetitions)):
        for task in tasks:
            for variant in variants:
                rows.append(
                    _run_long_session_task(
                        dict(task),
                        variant=variant,
                        repeat=repeat,
                        mode=mode,
                        provider=provider,
                        provider_client_factory=provider_client_factory,
                        output_dir=output_dir,
                        pricing=pricing,
                    )
                )
    treatment, control = _comparison_variants(variants)
    return build_result_payload(
        rows,
        pricing_profile=(
            "llm-handoff-live-configured"
            if mode == "live"
            else "llm-handoff-scripted-proxy"
        ),
        pricing=pricing,
        treatment=treatment,
        control=control,
    )


def build_result_payload(rows, *, pricing_profile, pricing=None, treatment="full_orchestrator", control="no_context_reduction"):
    rows = list(rows)
    return {
        "artifact_type": "context-cost-experiment",
        "pricing_profile": str(pricing_profile),
        "pricing": asdict(pricing) if pricing else None,
        "summary": summarize_paired_rows(rows, treatment=treatment, control=control),
        "rows": [row.to_dict() for row in rows],
    }


def collect_rows_from_run_manifest(manifest, *, pricing):
    rows = []
    for item in manifest.get("runs", []) or []:
        rows.append(
            extract_usage_from_artifacts(
                item["report_path"],
                item["trace_path"],
                task_id=item["task_id"],
                layer=item.get("layer", "live"),
                variant=item["variant"],
                repeat=item.get("repeat", 0),
                pricing=pricing,
            )
        )
    return rows


def render_markdown_report(payload):
    summary = dict(payload.get("summary", {}) or {})
    pricing = dict(payload.get("pricing", {}) or {})
    actual = dict(summary.get("actual_only", {}) or {})
    proxy = dict(summary.get("estimated_proxy_only", {}) or {})
    mixed = dict(summary.get("mixed_or_invalid", {}) or {})
    benefit = actual if actual.get("paired_task_count", 0) else proxy
    baseline = float(benefit.get("total_input_tokens_per_task_control", 0) or 0)
    optimized = float(benefit.get("total_input_tokens_per_task_treatment", 0) or 0)
    compact_call_tokens = float(benefit.get("compact_call_tokens_per_task", 0) or 0)
    net_saved = baseline - optimized - compact_call_tokens
    net_pct = (net_saved / baseline) if baseline else 0.0
    return "\n".join(
        [
            "# Context Cost Experiment",
            "",
            "## Summary",
            "",
            f"- Actual-only paired tasks: {actual.get('paired_task_count', 0)}",
            f"- Actual-only quality regressions: {actual.get('quality_regression_count', 0)}",
            f"- Actual-only unknown verification pairs: {actual.get('unknown_verification_count', 0)}",
            f"- Actual-only configured-price win: {actual.get('claimable_cost_win', False)}",
            f"- Actual-only median cost delta: {actual.get('median_cost_delta_pct', 0):.2%}",
            f"- Actual-only cost per successful task: {actual.get('cost_per_successful_task_treatment', 0)} vs {actual.get('cost_per_successful_task_control', 0)}",
            f"- Actual-only success rate: {actual.get('success_rate_treatment', 0):.2%} vs {actual.get('success_rate_control', 0):.2%}",
            f"- Actual-only verifier pass rate: {actual.get('verifier_pass_rate_treatment', 0):.2%} vs {actual.get('verifier_pass_rate_control', 0):.2%}",
            f"- Actual-only avg tool steps: {actual.get('avg_tool_steps_treatment', 0)} vs {actual.get('avg_tool_steps_control', 0)}",
            f"- Actual-only avg attempts: {actual.get('avg_attempts_treatment', 0)} vs {actual.get('avg_attempts_control', 0)}",
            f"- Actual-only billable input tokens/task: {actual.get('billable_input_tokens_per_task_treatment', 0)} vs {actual.get('billable_input_tokens_per_task_control', 0)}",
            f"- Actual-only total input tokens/task: {actual.get('total_input_tokens_per_task_treatment', 0)} vs {actual.get('total_input_tokens_per_task_control', 0)}",
            f"- Actual-only output tokens/task: {actual.get('output_tokens_per_task_treatment', 0)} vs {actual.get('output_tokens_per_task_control', 0)}",
            f"- Estimated-proxy paired tasks: {proxy.get('paired_task_count', 0)}",
            f"- Estimated-proxy median cost delta: {proxy.get('median_cost_delta_pct', 0):.2%}",
            f"- Estimated-proxy directional cost win: {proxy.get('claimable_cost_win', False)}",
            f"- Estimated-proxy billable input tokens/task: {proxy.get('billable_input_tokens_per_task_treatment', 0)} vs {proxy.get('billable_input_tokens_per_task_control', 0)}",
            f"- Estimated-proxy total input tokens/task: {proxy.get('total_input_tokens_per_task_treatment', 0)} vs {proxy.get('total_input_tokens_per_task_control', 0)}",
            f"- Estimated-proxy output tokens/task: {proxy.get('output_tokens_per_task_treatment', 0)} vs {proxy.get('output_tokens_per_task_control', 0)}",
            f"- Mixed/invalid paired tasks: {mixed.get('paired_task_count', 0)}",
            f"- Real provider rows: {summary.get('real_usage_row_count', 0)}",
            f"- Estimated proxy rows: {summary.get('estimated_proxy_row_count', 0)}",
            "- Pricing basis: configured, not provider-authenticated",
            f"- Input $/1M: {pricing.get('input_per_1m', '-')}",
            f"- Cached input $/1M: {pricing.get('cached_input_per_1m', '-')}",
            f"- Output $/1M: {pricing.get('output_per_1m', '-')}",
            "",
            "## Net Benefit",
            "",
            "- Formula: net_saved = baseline_input_tokens - optimized_input_tokens - compact_call_tokens",
            f"- compact_call_tokens: {compact_call_tokens:g}",
            f"- Baseline avg input tokens/task: {baseline:.2f}",
            f"- Optimized avg input tokens/task: {optimized:.2f}",
            f"- Net saved input tokens/task: {net_saved:.2f}",
            f"- Net saved percentage: {net_pct:.2%}",
            f"- Quality regression count: {benefit.get('quality_regression_count', 0)}",
            f"- Claimable cost win: {benefit.get('claimable_cost_win', False)}",
            "",
            "## Interpretation Rules",
            "",
            "- A configured-price win only counts from the actual-only bucket when claimable_cost_win is True.",
            "- Actual-only rows prove provider token telemetry, not that the configured prices match the provider/model.",
            "- Unknown verification makes a pair non-claimable even when token cost is lower.",
            "- Estimated proxy rows are directional evidence, not provider billing evidence.",
            "- Mixed actual/proxy pairs are invalid for headline cost claims.",
        ]
    )


def generate_report(payload, include_llm_handoff_comparison=False):
    report = render_markdown_report(payload)
    if include_llm_handoff_comparison:
        report += "\n\n" + _render_llm_handoff_comparison(payload)
    return report


def write_experiment_artifacts(payload, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "paired_rows.csv"
    markdown_path = output_dir / "report.md"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_rows_csv(payload.get("rows", []) or [], csv_path)
    markdown_path.write_text(generate_report(payload) + "\n", encoding="utf-8")
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
    }


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    pricing = ProviderPricing(
        args.input_per_1m,
        args.cached_input_per_1m,
        args.output_per_1m,
    )
    output_dir = Path(args.output_dir)
    if args.mode == "deterministic":
        payload = run_deterministic_prompt_experiment(
            output_dir / "work",
            repetitions=args.repetitions,
            pricing=pricing,
        )
    elif args.mode == "scripted":
        payload = run_scripted_e2e_experiment(
            output_dir / "work",
            repetitions=args.repetitions,
            pricing=pricing,
        )
    else:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        rows = collect_rows_from_run_manifest(manifest, pricing=pricing)
        payload = build_result_payload(rows, pricing_profile="manifest", pricing=pricing)
    written = write_experiment_artifacts(payload, output_dir)
    print(json.dumps(written, sort_keys=True))
    return 0


def _usage_from_trace(trace_path):
    estimated_input_tokens = 0
    input_tokens = 0
    cached_tokens = 0
    output_tokens = 0
    model_call_count = 0
    provider_metadata_count = 0
    for event in _read_jsonl(trace_path):
        if event.get("event") == "prompt_built":
            usage = dict(
                (event.get("prompt_metadata", {}) or {}).get("context_usage", {}) or {}
            )
            estimated_input_tokens += int(usage.get("total_estimated_tokens", 0) or 0)
        if event.get("event") == "model_parsed":
            metadata = dict(event.get("completion_metadata", {}) or {})
            model_call_count += 1
            if _is_provider_usage_metadata(metadata):
                provider_metadata_count += 1
                input_tokens += int(metadata.get("input_tokens", 0) or 0)
                cached_tokens += int(metadata.get("cached_tokens", 0) or 0)
                output_tokens += int(metadata.get("output_tokens", 0) or 0)
    if model_call_count > 0 and provider_metadata_count == model_call_count:
        usage = CostUsage(input_tokens, cached_tokens, output_tokens, "actual", model_call_count)
    else:
        usage = CostUsage(estimated_input_tokens, 0, 0, "estimated_proxy", model_call_count)
    return {"estimated_input_tokens": estimated_input_tokens, "usage": usage}


def _compact_metrics_from_trace(trace_path):
    metrics = {
        "summary_called": False,
        "summary_delta_event_count": 0,
        "compact_summary_mode": "",
        "compact_call_usage": None,
        "compact_net_benefit_tokens": None,
    }
    for event in _read_jsonl(trace_path):
        if event.get("event") != "context_orchestrator_decision":
            continue
        orchestrator = dict(event.get("context_orchestrator", {}) or {})
        if orchestrator.get("summary_mode"):
            metrics["compact_summary_mode"] = str(orchestrator.get("summary_mode", ""))
        metrics["summary_called"] = bool(metrics["summary_called"] or orchestrator.get("summary_called", False))
        metrics["summary_delta_event_count"] = max(
            int(metrics["summary_delta_event_count"] or 0),
            int(orchestrator.get("summary_delta_event_count", 0) or 0),
        )
        usage = orchestrator.get("compact_call_usage")
        if isinstance(usage, dict):
            metrics["compact_call_usage"] = dict(usage)
            pre_tokens = int(orchestrator.get("pre_compact_estimated_tokens", 0) or 0)
            post_tokens = int(orchestrator.get("post_compact_estimated_tokens", 0) or 0)
            compact_tokens = int(usage.get("total_tokens", 0) or 0)
            metrics["compact_net_benefit_tokens"] = pre_tokens - post_tokens - compact_tokens
    return metrics


def _usage_with_compact_call(usage, compact_call_usage):
    compact_input = int(compact_call_usage.get("input_tokens", 0) or 0)
    compact_cached = int(compact_call_usage.get("cached_tokens", 0) or 0)
    compact_output = int(compact_call_usage.get("output_tokens", 0) or 0)
    return CostUsage(
        input_tokens=int(usage.input_tokens) + compact_input,
        cached_tokens=int(usage.cached_tokens) + compact_cached,
        output_tokens=int(usage.output_tokens) + compact_output,
        usage_source=usage.usage_source,
        model_call_count=int(usage.model_call_count) + 1,
    )


def _is_provider_usage_metadata(metadata):
    return (
        metadata.get("provider_protocol") is not None
        and metadata.get("provider_model") is not None
        and metadata.get("input_tokens") is not None
        and metadata.get("output_tokens") is not None
        and metadata.get("synthetic") is not True
    )


def _verification_status(report):
    signal = dict(
        (report.get("evidence_summaries", {}) or {}).get("verification_signal", {}) or {}
    )
    state = str(signal.get("state", ""))
    return state or "unknown"


def _paired_rows(rows, *, treatment, control):
    by_key = {}
    for row in rows:
        by_key.setdefault((row.task_id, row.repeat, row.layer), {})[row.variant] = row
    return [
        variants
        for variants in by_key.values()
        if treatment in variants and control in variants
    ]


def _quality_regressed(treatment, control):
    if control.status == "completed" and treatment.status != "completed":
        return True
    if control.verification_status == "passed" and treatment.verification_status != "passed":
        return True
    if treatment.verification_status == "unknown" and control.verification_status != "unknown":
        return True
    if treatment.tool_steps > max(control.tool_steps + 2, int(control.tool_steps * 1.10)):
        return True
    return treatment.attempts > max(control.attempts + 1, int(control.attempts * 1.10))


def _summarize_pair_bucket(pairs, *, treatment, control):
    uncached_deltas = [
        _delta_pct(pair[treatment].usage.uncached_input_tokens, pair[control].usage.uncached_input_tokens)
        for pair in pairs
    ]
    cost_deltas = [
        _delta_pct(pair[treatment].cost_usd, pair[control].cost_usd)
        for pair in pairs
    ]
    return {
        "paired_task_count": len(pairs),
        "quality_regression_count": sum(1 for pair in pairs if _quality_regressed(pair[treatment], pair[control])),
        "unknown_verification_count": sum(
            1
            for pair in pairs
            if pair[treatment].verification_status == "unknown"
            or pair[control].verification_status == "unknown"
        ),
        "success_rate_treatment": _rate(pair[treatment].status == "completed" for pair in pairs),
        "success_rate_control": _rate(pair[control].status == "completed" for pair in pairs),
        "verifier_pass_rate_treatment": _rate(pair[treatment].verification_status == "passed" for pair in pairs),
        "verifier_pass_rate_control": _rate(pair[control].verification_status == "passed" for pair in pairs),
        "avg_tool_steps_treatment": _mean_rounded(pair[treatment].tool_steps for pair in pairs),
        "avg_tool_steps_control": _mean_rounded(pair[control].tool_steps for pair in pairs),
        "avg_attempts_treatment": _mean_rounded(pair[treatment].attempts for pair in pairs),
        "avg_attempts_control": _mean_rounded(pair[control].attempts for pair in pairs),
        "cost_per_successful_task_treatment": _cost_per_successful_task(pair[treatment] for pair in pairs),
        "cost_per_successful_task_control": _cost_per_successful_task(pair[control] for pair in pairs),
        "billable_input_tokens_per_task_treatment": _mean_rounded(pair[treatment].usage.uncached_input_tokens for pair in pairs),
        "billable_input_tokens_per_task_control": _mean_rounded(pair[control].usage.uncached_input_tokens for pair in pairs),
        "total_input_tokens_per_task_treatment": _mean_rounded(pair[treatment].usage.input_tokens for pair in pairs),
        "total_input_tokens_per_task_control": _mean_rounded(pair[control].usage.input_tokens for pair in pairs),
        "output_tokens_per_task_treatment": _mean_rounded(pair[treatment].usage.output_tokens for pair in pairs),
        "output_tokens_per_task_control": _mean_rounded(pair[control].usage.output_tokens for pair in pairs),
        "median_uncached_input_delta_pct": _median_rounded(uncached_deltas),
        "p95_uncached_input_delta_pct": _p95_rounded(uncached_deltas),
        "median_cost_delta_pct": _median_rounded(cost_deltas),
        "claimable_cost_win": _claimable_cost_win(pairs, treatment=treatment, control=control, cost_deltas=cost_deltas),
    }


def _pair_usage_source(pair, treatment, control):
    sources = {pair[treatment].usage.usage_source, pair[control].usage.usage_source}
    if sources == {"actual"}:
        return "actual"
    if sources == {"estimated_proxy"}:
        return "estimated_proxy"
    return "mixed_or_invalid"


def _delta_pct(treatment, control):
    if not control:
        return 0.0
    return round((float(treatment) - float(control)) / float(control), 4)


def _median_rounded(values):
    return round(statistics.median(values), 4) if values else 0.0


def _mean_rounded(values):
    values = list(values)
    return round(statistics.mean(values), 4) if values else 0.0


def _rate(values):
    values = list(values)
    return round(sum(1 for value in values if value) / len(values), 4) if values else 0.0


def _cost_per_successful_task(rows):
    rows = list(rows)
    successful = [
        row for row in rows if row.status == "completed" and row.verification_status == "passed"
    ]
    if not successful:
        return 0.0
    return round(sum(row.cost_usd for row in rows) / len(successful), 8)


def _claimable_cost_win(pairs, *, treatment, control, cost_deltas):
    if not pairs or not cost_deltas or _median_rounded(cost_deltas) >= 0:
        return False
    if any(
        pair[treatment].compact_net_benefit_tokens is not None
        and int(pair[treatment].compact_net_benefit_tokens) < 0
        for pair in pairs
    ):
        return False
    if any(_quality_regressed(pair[treatment], pair[control]) for pair in pairs):
        return False
    return all(
        pair[treatment].verification_status == "passed"
        and pair[control].verification_status == "passed"
        for pair in pairs
    )


def _p95_rounded(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return round(ordered[index], 4)


def _build_synthetic_agent(workspace_root, *, context_reduction=True):
    workspace_root = Path(workspace_root)
    (workspace_root / "README.md").write_text("demo\n", encoding="utf-8")
    agent = GenCode(
        model_client=ScriptedModelClient(["<final>done</final>"]),
        workspace=WorkspaceContext.build(workspace_root),
        session_store=SessionStore(workspace_root / ".gencode" / "sessions"),
        approval_policy="auto",
        feature_flags={"context_reduction": context_reduction},
        max_steps=1,
    )
    for index in range(8):
        agent.record({"role": "user", "content": f"prior request {index} " + ("u" * 400)})
        agent.record({"role": "assistant", "content": f"prior answer {index} " + ("a" * 400)})
    return agent


class _ContextCostScriptedClient(ScriptedModelClient):
    def __init__(self):
        super().__init__([])
        self.phase = 0

    def complete(self, prompt, max_new_tokens, **kwargs):
        del max_new_tokens, kwargs
        self.prompts.append(prompt)
        self.last_completion_metadata = {
            "input_tokens": max(1, len(prompt) // 4),
            "cached_tokens": 0,
            "output_tokens": 32,
            "synthetic": True,
        }
        if self.phase == 0:
            self.phase += 1
            return '<tool>{"name":"read_file","args":{"path":"large.txt","start":1,"end":200}}</tool>'
        return "<final>done</final>"


def _build_scripted_agent(workspace_root, *, context_reduction=True):
    workspace_root = Path(workspace_root)
    (workspace_root / "README.md").write_text("demo\n", encoding="utf-8")
    (workspace_root / "large.txt").write_text(
        "\n".join(f"line-{index} " + ("x" * 80) for index in range(200)),
        encoding="utf-8",
    )
    agent = GenCode(
        model_client=_ContextCostScriptedClient(),
        workspace=WorkspaceContext.build(workspace_root),
        session_store=SessionStore(workspace_root / ".gencode" / "sessions"),
        approval_policy="auto",
        feature_flags={"context_reduction": context_reduction},
        max_steps=4,
    )
    for index in range(6):
        agent.record({"role": "user", "content": f"prior request {index} " + ("u" * 400)})
        agent.record({"role": "assistant", "content": f"prior answer {index} " + ("a" * 400)})
    return agent


class _LongSessionScriptedClient(ScriptedModelClient):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.context_window = 2200
        self.model = "scripted-long-session"

    def complete(self, prompt, max_new_tokens, **kwargs):
        if "You are a context compactor for a coding agent" in str(prompt):
            self.prompts.append(prompt)
            self.last_completion_metadata = {
                "input_tokens": max(1, len(str(prompt)) // 4),
                "cached_tokens": 0,
                "output_tokens": 80,
                "total_tokens": max(1, len(str(prompt)) // 4) + 80,
                "provider_protocol": "openai",
                "provider_model": "scripted-handoff",
            }
            return "\n".join(
                [
                    "## Goal",
                    "Complete the long-session benchmark task.",
                    "",
                    "## Constraints",
                    "- Preserve the requested fixture changes.",
                    "",
                    "## Files Read",
                    "- benchmark fixture files",
                    "",
                    "## Files Modified",
                    "- benchmark fixture files",
                    "",
                    "## Next Steps",
                    "- Continue with the scripted task plan.",
                ]
            )
        self.last_completion_metadata = {
            "input_tokens": max(1, len(str(prompt)) // 4),
            "cached_tokens": 0,
            "output_tokens": 32,
            "synthetic": True,
        }
        return super().complete(prompt, max_new_tokens, **kwargs)


def _run_long_session_task(
    task,
    *,
    variant,
    repeat,
    mode,
    provider,
    provider_client_factory,
    output_dir,
    pricing,
):
    fixture_source = Path(task["fixture_repo"]).resolve()
    workspace = Path(output_dir) / "runs" / task["id"] / variant / str(repeat) / fixture_source.name
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(fixture_source, workspace)
    model_client = _model_client_for_long_session_task(
        task,
        variant=variant,
        repeat=repeat,
        mode=mode,
        provider=provider,
        provider_client_factory=provider_client_factory,
    )

    agent = GenCode(
        model_client=model_client,
        workspace=WorkspaceContext.build(workspace, repo_root_override=workspace),
        session_store=SessionStore(workspace / ".gencode" / "sessions"),
        run_store=RunStore(workspace / ".gencode" / "runs"),
        approval_policy="auto",
        feature_flags={
            "context_reduction": bool(EXPERIMENT_VARIANTS[variant]["context_orchestrator_enabled"])
        },
        max_steps=int(task.get("step_budget", 12)),
        max_new_tokens=default_max_tokens_for_provider(provider),
        allowed_tools=task["allowed_tools"],
    )
    if task.get("context_window_override") and mode == "live":
        model_client.context_window = int(task["context_window_override"])
    _seed_long_session_history(agent)
    _force_compact_summary_mode(agent, EXPERIMENT_VARIANTS[variant].get("compact_summary_mode", "deterministic"))

    row_timeout = int(task.get("row_timeout", 300))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(agent.ask, task["prompt"])
        try:
            future.result(timeout=row_timeout)
        except concurrent.futures.TimeoutError:
            pass
    report_path = agent.current_run_dir / "report.json" if agent.current_run_dir else None
    trace_path = agent.current_run_dir / "trace.jsonl" if agent.current_run_dir else None
    verifier = subprocess.run(
        portable_verifier_command(task["verifier"]),
        cwd=workspace,
        shell=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    verification_status = "passed" if verifier.returncode == 0 else "failed"
    return extract_usage_from_artifacts(
        report_path,
        trace_path,
        task_id=task["id"],
        layer=mode,
        variant=variant,
        repeat=repeat,
        pricing=pricing,
        verification_status=verification_status,
        allow_verification_override=True,
    )


def _model_client_for_long_session_task(
    task,
    *,
    variant,
    repeat,
    mode,
    provider,
    provider_client_factory,
):
    if mode == "scripted":
        return _LongSessionScriptedClient(task.get("scripted_outputs", []))
    if provider_client_factory is not None:
        return provider_client_factory(
            provider=provider,
            task=task,
            variant=variant,
            repeat=repeat,
        )
    return _build_live_provider_client(provider)


def _build_live_provider_client(provider):
    config = resolve_provider_config(provider, start=Path.cwd())
    if not config.api_key:
        raise RuntimeError(
            f"live provider config blocked: API key missing for provider profile {config.name}"
        )
    if config.protocol == "openai":
        return OpenAICompatibleModelClient(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=0.0,
            timeout=300,
        )
    if config.protocol == "anthropic":
        return AnthropicCompatibleModelClient(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=0.0,
            timeout=300,
        )
    raise RuntimeError(
        f"live provider config blocked: unsupported protocol {config.protocol}"
    )


def _seed_long_session_history(agent):
    for index in range(6):
        agent.record(
            {
                "role": "user",
                "content": f"prior long-session request {index} " + ("u" * 1100),
            }
        )
        agent.record(
            {
                "role": "assistant",
                "content": f"prior long-session answer {index} " + ("a" * 1100),
            }
        )


def _force_compact_summary_mode(agent, summary_mode):
    original = agent.context_orchestrator._compact_request

    def compact_request(self, metadata, snapshot):
        trigger, mode, skip_reason = original(metadata, snapshot)
        del mode
        if trigger:
            return trigger, str(summary_mode), skip_reason
        return trigger, "deterministic", skip_reason

    agent.context_orchestrator._compact_request = MethodType(
        compact_request, agent.context_orchestrator
    )


def _comparison_variants(variants):
    if "full_orchestrator_with_llm_handoff" in variants and "full_orchestrator" in variants:
        return "full_orchestrator_with_llm_handoff", "full_orchestrator"
    if len(variants) >= 2:
        return variants[-1], variants[0]
    return variants[0], "no_context_reduction"


def _render_llm_handoff_comparison(payload):
    rows = [dict(row) for row in payload.get("rows", []) or []]
    by_pair = {}
    for row in rows:
        key = (
            str(row.get("task_id", "")),
            int(row.get("repeat", 0) or 0),
            str(row.get("layer", "")),
        )
        by_pair.setdefault(key, {})[str(row.get("variant", ""))] = row
    comparison_rows = []
    for (task_id, repeat, layer), variants in sorted(by_pair.items()):
        deterministic = variants.get("full_orchestrator")
        handoff = variants.get("full_orchestrator_with_llm_handoff")
        if not deterministic or not handoff:
            continue
        comparison_rows.append((task_id, repeat, layer, deterministic, handoff))

    net_values = [
        int(handoff.get("compact_net_benefit_tokens"))
        for _, _, _, _, handoff in comparison_rows
        if handoff.get("compact_net_benefit_tokens") is not None
    ]
    positive = sum(1 for value in net_values if value > 0)
    negative = sum(1 for value in net_values if value < 0)
    total = len(net_values)
    show_repeat = any(repeat > 0 for _, repeat, _, _, _ in comparison_rows)
    negative_tasks = [
        _repeat_label(task_id, repeat, show_repeat=show_repeat)
        for task_id, repeat, _, _, handoff in comparison_rows
        if handoff.get("compact_net_benefit_tokens") is not None
        and int(handoff.get("compact_net_benefit_tokens")) < 0
    ]
    lines = [
        "## LLM Handoff vs Deterministic Comparison",
        "",
    ]
    if show_repeat:
        lines.extend(
            [
                "| Task | Repeat | Deterministic Cost | LLM Handoff Cost | Net Benefit | Mode Used |",
                "|------|--------|-------------------|------------------|-------------|-----------|",
            ]
        )
    else:
        lines.extend(
            [
                "| Task | Deterministic Cost | LLM Handoff Cost | Net Benefit | Mode Used |",
                "|------|-------------------|------------------|-------------|-----------|",
            ]
        )
    for task_id, repeat, _, deterministic, handoff in comparison_rows:
        net = handoff.get("compact_net_benefit_tokens")
        net_text = "n/a" if net is None else f"{int(net)} tokens"
        if show_repeat:
            lines.append(
                f"| {task_id} | {repeat} | "
                f"{float(deterministic.get('cost_usd', 0.0)):.8f} | "
                f"{float(handoff.get('cost_usd', 0.0)):.8f} | {net_text} | "
                f"{handoff.get('compact_summary_mode', '')} |"
            )
        else:
            lines.append(
                f"| {task_id} | {float(deterministic.get('cost_usd', 0.0)):.8f} | "
                f"{float(handoff.get('cost_usd', 0.0)):.8f} | {net_text} | "
                f"{handoff.get('compact_summary_mode', '')} |"
            )
    lines.extend(
        [
            "",
            f"- Median net benefit: {_median_tokens(net_values)} tokens",
            f"- Positive net benefit: {_pct(positive, total)}",
            f"- Negative net benefit: {_pct(negative, total)}",
            f"- Net-negative tasks: {', '.join(negative_tasks) if negative_tasks else 'none'}",
        ]
    )
    return "\n".join(lines)


def _repeat_label(task_id, repeat, *, show_repeat):
    return f"{task_id}#{repeat}" if show_repeat else str(task_id)


def _median_tokens(values):
    return int(statistics.median(values)) if values else 0


def _pct(count, total):
    return f"{(count / total):.0%}" if total else "0%"


def _write_prompt_only_trace(trace_path, prompt_metadata):
    trace_path.write_text(
        json.dumps(
            {
                "event": "prompt_built",
                "prompt_metadata": prompt_metadata,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_prompt_only_report(report_path, prompt_metadata):
    report_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "tool_steps": 0,
                "attempts": 1,
                "prompt_metadata": prompt_metadata,
                "evidence_summaries": {
                    "verification_signal": {"state": "passed"},
                    "context_budget_summary": prompt_metadata.get("context_budget_summary", {}),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_rows_csv(rows, path):
    fieldnames = sorted(
        {key for row in rows for key in row.keys()}
        | {"usage_input_tokens", "usage_cached_tokens", "usage_output_tokens", "usage_source"}
    )
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            usage = dict(flat.pop("usage", {}) or {})
            flat["usage_input_tokens"] = usage.get("input_tokens", "")
            flat["usage_cached_tokens"] = usage.get("cached_tokens", "")
            flat["usage_output_tokens"] = usage.get("output_tokens", "")
            flat["usage_source"] = usage.get("usage_source", "")
            writer.writerow(flat)


def _read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tool_name(event):
    return str(event.get("name") or event.get("tool_name") or event.get("tool") or "")


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Run GenCode context cost experiments.")
    parser.add_argument("--mode", choices=["deterministic", "scripted", "manifest"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--input-per-1m", type=float, default=DEFAULT_PROXY_PRICING.input_per_1m)
    parser.add_argument(
        "--cached-input-per-1m",
        type=float,
        default=DEFAULT_PROXY_PRICING.cached_input_per_1m,
    )
    parser.add_argument("--output-per-1m", type=float, default=DEFAULT_PROXY_PRICING.output_per_1m)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
