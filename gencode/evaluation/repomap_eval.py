"""Reproducible Repo Map retrieval and live token-efficiency experiments."""

from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
import time
from pathlib import Path

from gencode.config import resolve_provider_config
from gencode.core.context_usage import detect_content_type, estimate_tokens_typed
from gencode.features.repomap import RepoMapBuilder
from gencode.features.repomap import graph as graphlib
from gencode.features.repomap import rank as ranklib
from gencode.features.repomap import render as renderlib
from gencode.providers import AnthropicCompatibleModelClient, OpenAICompatibleModelClient

DEFAULT_BUDGET_CHARS = 4000
TOP_KS = (1, 3, 5, 10)
RETRIEVAL_VARIANTS = ("static_map", "legacy_personalized_map", "personalized_map")
PATH_RE = re.compile(r"(?:^|[\s`'\"(])((?:gencode|tests)/[A-Za-z0-9_./-]+\.py)")


def load_repomap_tasks(path):
    tasks = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Repo Map task file must contain a non-empty JSON list")
    ids = set()
    for task in tasks:
        task_id = str(task.get("id", "")).strip()
        gold_files = task.get("gold_files") or []
        if not task_id or task_id in ids:
            raise ValueError(f"invalid or duplicate task id: {task_id!r}")
        if not str(task.get("query", "")).strip() or not gold_files:
            raise ValueError(f"task {task_id} requires query and gold_files")
        ids.add(task_id)
    return tasks


def run_retrieval_experiment(
    repo_root, tasks, *, budget_chars=DEFAULT_BUDGET_CHARS, cache_dir=None
):
    """Compare static PageRank with GenCode's query/recent-path personalization."""
    repo_root = Path(repo_root).resolve()
    cache_dir = Path(cache_dir or repo_root / ".gencode" / "repomap-eval-cache")
    started = time.perf_counter()
    tags_by_file = graphlib.load_tags(repo_root, cache_dir=cache_dir)
    _defines, edge_weights = graphlib.build_reference_graph(tags_by_file)
    nodes = set(tags_by_file)
    nodes.update(src for src, _dst in edge_weights)
    nodes.update(dst for _src, dst in edge_weights)
    index_ms = (time.perf_counter() - started) * 1000

    rows = []
    for task in tasks:
        for variant in RETRIEVAL_VARIANTS:
            seeds = {}
            if variant == "legacy_personalized_map":
                seeds = ranklib.build_legacy_seeds(
                    task["query"],
                    tags_by_file,
                    recent_paths=task.get("recent_paths", []),
                )
                scores = ranklib.personalized_pagerank(nodes, edge_weights, seeds)
            elif variant == "personalized_map":
                scores, seeds = ranklib.hybrid_scores(
                    task["query"],
                    nodes,
                    edge_weights,
                    tags_by_file,
                    recent_paths=task.get("recent_paths", []),
                )
            else:
                scores = ranklib.personalized_pagerank(nodes, edge_weights, seeds)
            ranked = [
                path
                for path in sorted(
                    nodes, key=lambda item: (-scores.get(item, 0.0), item)
                )
                if path in tags_by_file and tags_by_file[path].definitions
            ]
            text = renderlib.render_repo_map(ranked, tags_by_file, int(budget_chars))
            rendered_paths = _paths_from_map(text)
            rows.append(
                _retrieval_row(
                    task,
                    variant,
                    rendered_paths,
                    map_chars=len(text),
                    seed_count=len(seeds),
                )
            )
    return {
        "artifact_type": "repomap-retrieval-experiment-v1",
        "repo_root": repo_root.as_posix(),
        "repo_fingerprint": _repo_fingerprint(repo_root),
        "task_count": len(tasks),
        "budget_chars": int(budget_chars),
        "files_tagged": len(tags_by_file),
        "edge_count": len(edge_weights),
        "index_ms": round(index_ms, 3),
        "summary": _summarize_retrieval(rows),
        "rows": rows,
    }


def run_live_ablation(
    repo_root,
    tasks,
    *,
    output_dir,
    repetitions=3,
    provider=None,
    model=None,
    budget_chars=DEFAULT_BUDGET_CHARS,
    max_new_tokens=256,
    resume=True,
):
    """Run real-provider Repo Map off/on pairs, persisting every trial."""
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "live_rows.jsonl"
    config = resolve_provider_config(provider, start=repo_root, model=model)
    if not config.api_key:
        raise RuntimeError(f"API key missing for provider profile {config.name}")
    completed = _completed_live_keys(rows_path) if resume else set()
    rows = _read_jsonl(rows_path) if resume else []

    live_tasks = [task for task in tasks if task.get("live", False)]
    for repeat in range(int(repetitions)):
        for task_index, task in enumerate(live_tasks):
            variants = ["repo_map_off", "repo_map_on"]
            if (repeat + task_index) % 2:
                variants.reverse()
            for variant in variants:
                key = (str(task["id"]), variant, repeat)
                if key in completed:
                    continue
                row = _run_live_trial(
                    repo_root,
                    task,
                    variant=variant,
                    repeat=repeat,
                    output_dir=output_dir,
                    config=config,
                    budget_chars=budget_chars,
                    max_new_tokens=max_new_tokens,
                )
                with rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                rows.append(row)
                completed.add(key)

    payload = {
        "artifact_type": "repomap-live-ablation-v1",
        "repo_root": repo_root.as_posix(),
        "repo_fingerprint": _repo_fingerprint(repo_root),
        "provider": config.name,
        "protocol": config.protocol,
        "model": config.model,
        "task_count": len(live_tasks),
        "repetitions": int(repetitions),
        "budget_chars": int(budget_chars),
        "max_new_tokens": int(max_new_tokens),
        "summary": summarize_live_rows(rows),
        "rows": rows,
    }
    (output_dir / "live_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def write_retrieval_artifacts(payload, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "retrieval_results.json"
    report_path = output_dir / "retrieval_report.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_retrieval_report(payload) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(report_path)}


def write_live_report(payload, output_dir):
    path = Path(output_dir) / "live_report.md"
    path.write_text(render_live_report(payload) + "\n", encoding="utf-8")
    return str(path)


def render_retrieval_report(payload):
    summaries = payload["summary"]
    lines = [
        "# GenCode Repo Map Retrieval Experiment",
        "",
        f"- Tasks: {payload['task_count']}",
        f"- Fixed map budget: {payload['budget_chars']} characters",
        f"- Parsed files: {payload['files_tagged']}",
        f"- Reference edges: {payload['edge_count']}",
        "",
        "| Variant | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR | Avg rendered files |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in RETRIEVAL_VARIANTS:
        item = summaries[variant]
        lines.append(
            f"| {variant} | {item['hit_at_1']:.2%} | {item['hit_at_3']:.2%} | "
            f"{item['hit_at_5']:.2%} | {item['recall_at_5']:.2%} | {item['mrr']:.4f} | "
            f"{item['avg_rendered_files']:.2f} |"
        )
    delta = summaries["paired_delta"]
    lines.extend(
        [
            "",
            f"- Hit@5 change: {delta['hit_at_5_pp']:+.2f} percentage points",
            f"- Recall@5 change: {delta['recall_at_5_pp']:+.2f} percentage points",
            f"- MRR change: {delta['mrr']:+.4f}",
        ]
    )
    fix_delta = summaries.get("fix_delta")
    if fix_delta:
        lines.extend(
            [
                "",
                f"- Fixed-vs-legacy Hit@5 change: {fix_delta['hit_at_5_pp']:+.2f} percentage points",
                f"- Fixed-vs-legacy Recall@5 change: {fix_delta['recall_at_5_pp']:+.2f} percentage points",
                f"- Fixed-vs-legacy MRR change: {fix_delta['mrr']:+.4f}",
            ]
        )
    return "\n".join(lines)


def render_live_report(payload):
    summary = payload["summary"]
    lines = [
        "# GenCode Repo Map Live A/B Experiment",
        "",
        f"- Provider/model: {payload['provider']} / {payload['model']}",
        f"- Tasks x repetitions: {payload['task_count']} x {payload['repetitions']}",
        "- Only the repo_map feature flag differs between paired trials.",
        "",
        "| Variant | Trials | Pass rate | Input tokens/success | Total tokens/success | Avg search calls | Avg read calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("repo_map_off", "repo_map_on"):
        item = summary[variant]
        lines.append(
            f"| {variant} | {item['trials']} | {item['pass_rate']:.2%} | "
            f"{item['input_tokens_per_success']:.2f} | {item['total_tokens_per_success']:.2f} | "
            f"{item['avg_search_calls']:.2f} | {item['avg_read_calls']:.2f} |"
        )
    paired = summary["paired"]
    lines.extend(
        [
            "",
            f"- Complete pairs: {paired['pair_count']}",
            f"- Both-pass pairs: {paired['both_pass_count']}",
            f"- Quality gains/regressions: {paired['quality_gain_count']} / {paired['quality_regression_count']}",
            f"- Unit-success input-token saving: {paired['unit_success_input_token_saving_pct']:.2%}",
            f"- Unit-success total-token saving: {paired['unit_success_total_token_saving_pct']:.2%}",
            f"- Median paired total-token saving on both-pass pairs: {paired['median_both_pass_total_token_saving_pct']:.2%}",
            f"- Bootstrap 95% CI for mean both-pass token delta (off - on): [{paired['mean_token_delta_ci95'][0]:.2f}, {paired['mean_token_delta_ci95'][1]:.2f}] tokens",
            f"- Actual provider-usage coverage: {paired['actual_usage_rate']:.2%}",
        ]
    )
    return "\n".join(lines)


def summarize_live_rows(rows):
    rows = list(rows)
    result = {}
    for variant in ("repo_map_off", "repo_map_on"):
        selected = [row for row in rows if row.get("variant") == variant]
        successes = sum(bool(row.get("passed")) for row in selected)
        input_tokens = sum(int(row.get("input_tokens", 0) or 0) for row in selected)
        output_tokens = sum(int(row.get("output_tokens", 0) or 0) for row in selected)
        result[variant] = {
            "trials": len(selected),
            "successes": successes,
            "pass_rate": _ratio(successes, len(selected)),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_per_success": _ratio(input_tokens, successes),
            "total_tokens_per_success": _ratio(input_tokens + output_tokens, successes),
            "avg_model_calls": _mean(row.get("model_calls", 0) for row in selected),
            "avg_tool_calls": _mean(row.get("tool_calls", 0) for row in selected),
            "avg_search_calls": _mean(row.get("search_calls", 0) for row in selected),
            "avg_read_calls": _mean(row.get("read_calls", 0) for row in selected),
            "avg_duration_ms": _mean(row.get("duration_ms", 0) for row in selected),
            "avg_repo_map_prompt_tokens_estimated": _mean(
                row.get("repo_map_prompt_tokens_estimated", 0) for row in selected
            ),
        }

    grouped = {}
    for row in rows:
        key = (str(row.get("task_id", "")), int(row.get("repeat", 0) or 0))
        grouped.setdefault(key, {})[str(row.get("variant", ""))] = row
    pairs = [
        pair
        for pair in grouped.values()
        if "repo_map_off" in pair and "repo_map_on" in pair
    ]
    both_pass = [
        pair
        for pair in pairs
        if pair["repo_map_off"].get("passed") and pair["repo_map_on"].get("passed")
    ]
    deltas = [
        int(pair["repo_map_off"].get("total_tokens", 0))
        - int(pair["repo_map_on"].get("total_tokens", 0))
        for pair in both_pass
    ]
    saving_pcts = [
        _ratio(delta, int(pair["repo_map_off"].get("total_tokens", 0)))
        for pair, delta in zip(both_pass, deltas)
        if int(pair["repo_map_off"].get("total_tokens", 0)) > 0
    ]
    off = result["repo_map_off"]
    on = result["repo_map_on"]
    actual_rows = [row for row in rows if row.get("usage_source") == "actual"]
    result["paired"] = {
        "pair_count": len(pairs),
        "both_pass_count": len(both_pass),
        "quality_gain_count": sum(
            bool(pair["repo_map_on"].get("passed"))
            and not bool(pair["repo_map_off"].get("passed"))
            for pair in pairs
        ),
        "quality_regression_count": sum(
            bool(pair["repo_map_off"].get("passed"))
            and not bool(pair["repo_map_on"].get("passed"))
            for pair in pairs
        ),
        "unit_success_input_token_saving_pct": _saving_pct(
            off["input_tokens_per_success"], on["input_tokens_per_success"]
        ),
        "unit_success_total_token_saving_pct": _saving_pct(
            off["total_tokens_per_success"], on["total_tokens_per_success"]
        ),
        "median_both_pass_total_token_saving_pct": statistics.median(saving_pcts)
        if saving_pcts
        else 0.0,
        "mean_both_pass_total_token_delta": _mean(deltas),
        "mean_token_delta_ci95": _bootstrap_mean_ci(deltas),
        "actual_usage_rate": _ratio(len(actual_rows), len(rows)),
    }
    return result


def _run_live_trial(
    repo_root,
    task,
    *,
    variant,
    repeat,
    output_dir,
    config,
    budget_chars,
    max_new_tokens,
):
    trial_root = output_dir / "runs" / variant / f"repeat-{repeat}" / str(task["id"])
    trial_root.mkdir(parents=True, exist_ok=True)
    client = _provider_client(config)
    map_enabled = variant == "repo_map_on"
    map_text = ""
    map_meta = {"enabled": False, "reason": "control"}
    map_build_ms = 0.0
    if map_enabled:
        map_started = time.perf_counter()
        map_text, map_meta = RepoMapBuilder(
            repo_root, cache_dir=output_dir / "tag-cache"
        ).build(
            task["query"],
            budget_chars=budget_chars,
            recent_paths=task.get("recent_paths", []),
        )
        map_build_ms = (time.perf_counter() - map_started) * 1000

    prompt = _controlled_initial_prompt(task, map_text)
    started = time.perf_counter()
    error = ""
    completion_metadata = []
    search_query = ""
    search_results = ""
    stop_reason = ""
    try:
        first = client.complete(prompt, int(max_new_tokens))
        completion_metadata.append(dict(client.last_completion_metadata or {}))
        directive, value = _parse_controlled_output(first)
        if directive == "final":
            answer = value
            stop_reason = "direct_final"
        else:
            search_query = value or str(task["query"])
            search_results = _search_source(repo_root, search_query, budget_chars)
            second_prompt = _controlled_final_prompt(
                task, map_text, search_query, search_results
            )
            second = client.complete(second_prompt, int(max_new_tokens))
            completion_metadata.append(dict(client.last_completion_metadata or {}))
            answer = _parse_final_path(second)
            stop_reason = "search_then_final"
    except Exception as exc:  # noqa: BLE001 - preserve evidence for the rest of the paired run
        answer = ""
        error = f"{type(exc).__name__}: {exc}"
        stop_reason = "provider_error"
    duration_ms = int((time.perf_counter() - started) * 1000)
    usage = _metadata_usage(completion_metadata)
    gold_files = {_normalize_path(path) for path in task["gold_files"]}
    answer_paths = set(_extract_paths(answer))
    passed = bool(answer_paths & gold_files)
    rendered_paths = _paths_from_map(map_text)
    row = {
        "task_id": str(task["id"]),
        "category": str(task.get("category", "")),
        "variant": variant,
        "repeat": int(repeat),
        "passed": passed,
        "answer": str(answer),
        "answer_paths": sorted(answer_paths),
        "gold_files": sorted(gold_files),
        "stop_reason": stop_reason,
        "error": error,
        "input_tokens": usage["input_tokens"],
        "reported_input_tokens": usage["reported_input_tokens"],
        "cached_tokens": usage["cached_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["input_tokens"] + usage["output_tokens"],
        "usage_source": usage["usage_source"],
        "model_calls": usage["model_calls"],
        "tool_calls": 1 if search_query else 0,
        "search_calls": 1 if search_query else 0,
        "read_calls": 0,
        "listed_calls": 0,
        "gold_file_read": False,
        "search_query": search_query,
        "search_result_chars": len(search_results),
        "repo_map_prompt_chars": len(map_text) * usage["model_calls"],
        "repo_map_prompt_tokens_estimated": estimate_tokens_typed(
            map_text, detect_content_type(map_text)
        )
        * usage["model_calls"],
        "repo_map_rendered_files": len(rendered_paths),
        "initial_gold_visible": bool(set(rendered_paths) & gold_files),
        "map_build_ms": round(map_build_ms, 3),
        "map_meta": map_meta,
        "duration_ms": duration_ms,
    }
    artifact_path = trial_root / "trial.json"
    artifact_path.write_text(
        json.dumps(
            {
                **row,
                "initial_prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "map_text": map_text,
                "search_results": search_results,
                "completion_metadata": completion_metadata,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    row["artifact_path"] = artifact_path.as_posix()
    return row


def _controlled_initial_prompt(task, map_text):
    recent = task.get("recent_paths") or []
    recent_hint = f" The file currently being changed is {recent[0]}." if recent else ""
    context = map_text or "(No initial repository map is available.)"
    return (
        "You are performing a controlled code-localization benchmark.\n"
        "Choose exactly one action and output exactly one line:\n"
        "FINAL gencode/path/to/file.py\n"
        "or, only when the context is insufficient:\n"
        "SEARCH one literal code identifier or phrase\n"
        "Do not use XML, JSON, markdown, DSML, or explanations.\n\n"
        f"Question: {task['query']}{recent_hint}\n\n"
        f"Repository context:\n{context}"
    )


def _controlled_final_prompt(task, map_text, search_query, search_results):
    context = map_text or "(No initial repository map was available.)"
    return (
        "You are completing a controlled code-localization benchmark.\n"
        "Return exactly one repository-relative Python path as: FINAL gencode/path/to/file.py\n"
        "Do not use XML, JSON, markdown, DSML, or explanations.\n\n"
        f"Question: {task['query']}\n\n"
        f"Initial repository context:\n{context}\n\n"
        f"Search query: {search_query}\n"
        f"Search results:\n{search_results or '(no matches)'}"
    )


def _parse_controlled_output(text):
    text = str(text).strip()
    paths = _extract_paths(text)
    if paths and (
        text.upper().startswith("FINAL") or not text.upper().startswith("SEARCH")
    ):
        return "final", paths[0]
    match = re.search(r"(?im)^\s*SEARCH\s+(.+?)\s*$", text)
    if match:
        return "search", match.group(1).strip(" `\"'")
    return "search", text.strip(" `\"'")


def _parse_final_path(text):
    paths = _extract_paths(text)
    return paths[0] if paths else str(text).strip()


def _search_source(repo_root, query, budget_chars):
    query = str(query).strip()
    lowered = query.lower()
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query)
        if token.lower() not in {"search", "find", "where", "file", "module", "code"}
    ]
    exact = []
    fallback = []
    for rel_path in graphlib.collect_source_files(repo_root):
        if not rel_path.endswith(".py"):
            continue
        try:
            lines = (
                (Path(repo_root) / rel_path)
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
            )
        except OSError:
            continue
        for line_no, line in enumerate(lines, 1):
            line_lower = line.lower()
            rendered = f"{rel_path}:{line_no}:{line.strip()}"
            if lowered and lowered in line_lower:
                exact.append(rendered)
            elif tokens and any(token in line_lower for token in tokens):
                fallback.append(rendered)
    matches = exact or fallback
    return "\n".join(matches)[: int(budget_chars)]


def _provider_client(config):
    kwargs = {
        "model": config.model,
        "base_url": config.base_url,
        "api_key": config.api_key,
        "temperature": 0.0,
        "timeout": 300,
    }
    if config.protocol == "openai":
        return OpenAICompatibleModelClient(**kwargs)
    if config.protocol == "anthropic":
        return AnthropicCompatibleModelClient(**kwargs)
    raise RuntimeError(f"unsupported provider protocol: {config.protocol}")


def _retrieval_row(task, variant, rendered_paths, *, map_chars, seed_count):
    gold = {_normalize_path(path) for path in task["gold_files"]}
    ranks = [index + 1 for index, path in enumerate(rendered_paths) if path in gold]
    first_rank = min(ranks) if ranks else None
    row = {
        "task_id": str(task["id"]),
        "category": str(task.get("category", "")),
        "variant": variant,
        "gold_files": sorted(gold),
        "rendered_files": len(rendered_paths),
        "map_chars": int(map_chars),
        "seed_count": int(seed_count),
        "first_gold_rank": first_rank,
        "mrr": _ratio(1, first_rank) if first_rank else 0.0,
    }
    for k in TOP_KS:
        selected = set(rendered_paths[:k])
        row[f"hit_at_{k}"] = bool(selected & gold)
        row[f"recall_at_{k}"] = _ratio(len(selected & gold), len(gold))
    return row


def _summarize_retrieval(rows):
    result = {}
    for variant in RETRIEVAL_VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        item = {
            "tasks": len(selected),
            "mrr": _mean(row["mrr"] for row in selected),
            "avg_rendered_files": _mean(row["rendered_files"] for row in selected),
            "avg_map_chars": _mean(row["map_chars"] for row in selected),
        }
        for k in TOP_KS:
            item[f"hit_at_{k}"] = _mean(row[f"hit_at_{k}"] for row in selected)
            item[f"recall_at_{k}"] = _mean(row[f"recall_at_{k}"] for row in selected)
        result[variant] = item
    static = result["static_map"]
    personalized = result["personalized_map"]
    legacy = result["legacy_personalized_map"]
    result["paired_delta"] = {
        "hit_at_5_pp": (personalized["hit_at_5"] - static["hit_at_5"]) * 100,
        "recall_at_5_pp": (personalized["recall_at_5"] - static["recall_at_5"]) * 100,
        "mrr": personalized["mrr"] - static["mrr"],
    }
    result["fix_delta"] = {
        "hit_at_5_pp": (personalized["hit_at_5"] - legacy["hit_at_5"]) * 100,
        "recall_at_5_pp": (personalized["recall_at_5"] - legacy["recall_at_5"]) * 100,
        "mrr": personalized["mrr"] - legacy["mrr"],
    }
    result["category"] = {}
    for category in sorted({row["category"] for row in rows}):
        result["category"][category] = {}
        for variant in RETRIEVAL_VARIANTS:
            selected = [
                row
                for row in rows
                if row["category"] == category and row["variant"] == variant
            ]
            result["category"][category][variant] = {
                "tasks": len(selected),
                "hit_at_5": _mean(row["hit_at_5"] for row in selected),
                "mrr": _mean(row["mrr"] for row in selected),
            }
    return result


def _paths_from_map(text):
    paths = []
    for line in str(text).splitlines():
        if not line.startswith("- "):
            continue
        path = line[2:].split(" (", 1)[0].strip()
        if path:
            paths.append(_normalize_path(path))
    return paths


def _extract_paths(text):
    normalized = str(text).replace("\\", "/")
    return [
        _normalize_path(match.group(1).rstrip(".,;:)]}"))
        for match in PATH_RE.finditer(normalized)
    ]


def _live_usage(events):
    input_tokens = reported_input_tokens = 0
    cached_tokens = output_tokens = model_calls = provider_rows = 0
    estimated = 0
    for event in events:
        if event.get("event") == "prompt_built":
            estimated += int(
                ((event.get("prompt_metadata") or {}).get("context_usage") or {}).get(
                    "total_estimated_tokens", 0
                )
                or 0
            )
        if event.get("event") != "model_parsed":
            continue
        model_calls += 1
        metadata = event.get("completion_metadata") or {}
        if (
            metadata.get("input_tokens") is not None
            and metadata.get("output_tokens") is not None
        ):
            provider_rows += 1
            reported = int(metadata.get("input_tokens", 0) or 0)
            cached = int(metadata.get("cached_tokens", 0) or 0)
            reported_input_tokens += reported
            cached_tokens += cached
            # Anthropic usage reports cache-read tokens separately from
            # input_tokens. OpenAI-compatible usage includes cached tokens in
            # input_tokens_details, so only the Anthropic path adds them.
            input_tokens += (
                reported + cached
                if metadata.get("provider_protocol") == "anthropic"
                else reported
            )
            output_tokens += int(metadata.get("output_tokens", 0) or 0)
    if model_calls and provider_rows == model_calls:
        return {
            "input_tokens": input_tokens,
            "reported_input_tokens": reported_input_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "model_calls": model_calls,
            "usage_source": "actual",
        }
    return {
        "input_tokens": estimated,
        "reported_input_tokens": estimated,
        "cached_tokens": 0,
        "output_tokens": 0,
        "model_calls": model_calls,
        "usage_source": "estimated_proxy",
    }


def _metadata_usage(metadata_rows):
    input_tokens = reported_input_tokens = cached_tokens = output_tokens = 0
    actual = True
    for metadata in metadata_rows:
        if (
            metadata.get("input_tokens") is None
            or metadata.get("output_tokens") is None
        ):
            actual = False
            continue
        reported = int(metadata.get("input_tokens", 0) or 0)
        cached = int(metadata.get("cached_tokens", 0) or 0)
        reported_input_tokens += reported
        cached_tokens += cached
        input_tokens += (
            reported + cached
            if metadata.get("provider_protocol") == "anthropic"
            else reported
        )
        output_tokens += int(metadata.get("output_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "reported_input_tokens": reported_input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "model_calls": len(metadata_rows),
        "usage_source": "actual" if metadata_rows and actual else "estimated_proxy",
    }


def _prompt_section_total(events, section, field):
    total = 0
    for event in events:
        if event.get("event") != "prompt_built":
            continue
        usage = (event.get("prompt_metadata") or {}).get("context_usage") or {}
        total += int(
            ((usage.get("sections") or {}).get(section) or {}).get(field, 0) or 0
        )
    return total


def _report_value(path, key):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get(key, "")
    except (OSError, ValueError):
        return ""


def _completed_live_keys(path):
    return {
        (
            str(row.get("task_id", "")),
            str(row.get("variant", "")),
            int(row.get("repeat", 0) or 0),
        )
        for row in _read_jsonl(path)
    }


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _repo_fingerprint(root):
    digest = hashlib.sha256()
    for path in graphlib.collect_source_files(root):
        digest.update(path.encode("utf-8"))
        try:
            digest.update((Path(root) / path).read_bytes())
        except OSError:
            continue
    return digest.hexdigest()


def _normalize_path(path):
    return str(path).strip().replace("\\", "/").lstrip("./")


def _ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _mean(values):
    values = [float(value) for value in values]
    return statistics.mean(values) if values else 0.0


def _saving_pct(control, treatment):
    return _ratio(float(control) - float(treatment), float(control)) if control else 0.0


def _bootstrap_mean_ci(values, *, samples=10000, seed=20260910):
    values = [float(value) for value in values]
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    means = sorted(
        statistics.mean(rng.choice(values) for _ in values) for _ in range(int(samples))
    )
    low = means[int(0.025 * (len(means) - 1))]
    high = means[int(0.975 * (len(means) - 1))]
    return [round(low, 3), round(high, 3)]
