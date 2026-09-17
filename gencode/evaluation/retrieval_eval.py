"""Deterministic sparse/dense/hybrid retrieval evaluation."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

VARIANTS = {
    "sparse": ("sparse",),
    "dense": ("dense",),
    "hybrid": ("sparse", "dense"),
}


def load_retrieval_tasks(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("tasks", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise TypeError("retrieval benchmark must contain a task list")
    return [dict(row) for row in rows]


def run_retrieval_evaluation(
    service,
    tasks,
    *,
    variants=("sparse", "dense", "hybrid"),
    default_source_types=("wiki",),
    artifact_path=None,
):
    rows = []
    for task in tasks:
        expected = {str(value) for value in task.get("expected_source_ids", [])}
        forbidden = {str(value) for value in task.get("forbidden_source_ids", [])}
        source_types = tuple(task.get("source_types", default_source_types))
        for variant in variants:
            channels = VARIANTS[str(variant)]
            started = time.perf_counter()
            response = service.retrieve(
                task.get("query", ""),
                source_types=source_types,
                top_k=int(task.get("top_k", 5)),
                allowed_paths=tuple(task.get("allowed_paths", ())),
                channels=channels,
                use_cache=False,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            retrieved = [hit.chunk.source_id for hit in response.hits]
            unique_retrieved = list(dict.fromkeys(retrieved))
            relevant = [source_id for source_id in unique_retrieved if source_id in expected]
            first_rank = next(
                (index for index, source_id in enumerate(unique_retrieved, 1) if source_id in expected),
                0,
            )
            no_evidence_expected = bool(task.get("expect_no_evidence", False))
            rows.append(
                {
                    "task_id": str(task.get("id", "")),
                    "variant": str(variant),
                    "query": str(task.get("query", "")),
                    "retrieved_source_ids": unique_retrieved,
                    "expected_source_ids": sorted(expected),
                    "recall_at_k": len(set(relevant)) / max(len(expected), 1),
                    "precision_at_k": len(relevant) / max(len(unique_retrieved), 1),
                    "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
                    "forbidden_exposure": int(bool(forbidden & set(unique_retrieved))),
                    "evidence_expected": int(bool(expected) and not no_evidence_expected),
                    "no_evidence_correct": int(
                        response.insufficient_evidence == no_evidence_expected
                    ),
                    "citation_count": len(response.metrics.get("citations", {})),
                    "rejected": list(response.rejected),
                    "latency_ms": round(elapsed_ms, 3),
                    "strategy": response.strategy,
                    "errors": list(response.metrics.get("errors", [])),
                }
            )
    summary = _aggregate(rows)
    artifact = {
        "artifact_type": "gencode.hybrid_retrieval_eval.v1",
        "task_count": len(tasks),
        "trial_count": len(rows),
        "variants": summary,
        "rows": rows,
        "index": service.stats(),
    }
    if artifact_path:
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return artifact


def _aggregate(rows):
    result = {}
    for variant in sorted({row["variant"] for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        result[variant] = {
            "trials": len(selected),
            "recall_at_k": _mean(row["recall_at_k"] for row in selected),
            "precision_at_k": _mean(row["precision_at_k"] for row in selected),
            "mrr": _mean(row["reciprocal_rank"] for row in selected),
            "forbidden_exposure_rate": _mean(
                row["forbidden_exposure"] for row in selected
            ),
            "no_evidence_accuracy": _mean(
                row["no_evidence_correct"] for row in selected
            ),
            "citation_coverage": _mean(
                int(row["citation_count"] > 0)
                for row in selected
                if row["evidence_expected"]
            ),
            "citation_contract_accuracy": _mean(
                int(
                    (row["citation_count"] > 0)
                    if row["evidence_expected"]
                    else row["citation_count"] == 0
                )
                for row in selected
            ),
            "latency_p50_ms": _percentile(
                [row["latency_ms"] for row in selected], 0.50
            ),
            "latency_p95_ms": _percentile(
                [row["latency_ms"] for row in selected], 0.95
            ),
        }
    return result


def _mean(values):
    rows = list(values)
    return statistics.fmean(rows) if rows else 0.0


def _percentile(values, quantile):
    rows = sorted(float(value) for value in values)
    if not rows:
        return 0.0
    index = min(len(rows) - 1, max(0, round((len(rows) - 1) * float(quantile))))
    return rows[index]
