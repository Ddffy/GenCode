#!/usr/bin/env python3
"""Paired benchmark: deterministic vs LLM handoff context compaction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gencode.evaluation.context_cost import generate_report, run_paired_experiment  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scripted", "live"], default="scripted")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--output-dir", default="artifacts/llm-handoff-benchmark")
    parser.add_argument("--tasks", default="benchmarks/long_session_tasks.json")
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))["tasks"]

    results = run_paired_experiment(
        tasks=tasks,
        variants=["full_orchestrator", "full_orchestrator_with_llm_handoff"],
        mode=args.mode,
        provider=args.provider if args.mode == "live" else None,
        repetitions=args.repetitions,
        output_dir=output_dir / "work",
    )

    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        generate_report(results, include_llm_handoff_comparison=True) + "\n",
        encoding="utf-8",
    )

    print(f"Results: {output_dir / 'results.json'}")
    print(f"Report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
