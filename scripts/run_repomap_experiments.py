#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gencode.evaluation.repomap_eval import (
    load_repomap_tasks,
    run_live_ablation,
    run_retrieval_experiment,
    write_live_report,
    write_retrieval_artifacts,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run GenCode Repo Map retrieval and live A/B experiments."
    )
    parser.add_argument("--mode", choices=("retrieval", "live", "all"), default="all")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument(
        "--tasks", default=str(ROOT / "benchmarks" / "repomap_eval_tasks.json")
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "artifacts" / "repomap-experiments")
    )
    parser.add_argument("--budget-chars", type=int, default=4000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    tasks = load_repomap_tasks(args.tasks)
    output_dir = Path(args.output_dir)
    written = {}
    if args.mode in {"retrieval", "all"}:
        retrieval = run_retrieval_experiment(
            args.repo_root,
            tasks,
            budget_chars=args.budget_chars,
            cache_dir=output_dir / "tag-cache",
        )
        written["retrieval"] = write_retrieval_artifacts(retrieval, output_dir)
        print(
            json.dumps(
                {"retrieval_summary": retrieval["summary"]},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    if args.mode in {"live", "all"}:
        live = run_live_ablation(
            args.repo_root,
            tasks,
            output_dir=output_dir,
            repetitions=args.repetitions,
            provider=args.provider,
            model=args.model,
            budget_chars=args.budget_chars,
            max_new_tokens=args.max_new_tokens,
            resume=not args.no_resume,
        )
        written["live"] = {
            "json": str(output_dir / "live_summary.json"),
            "jsonl": str(output_dir / "live_rows.jsonl"),
            "markdown": write_live_report(live, output_dir),
        }
        print(
            json.dumps(
                {"live_summary": live["summary"]}, ensure_ascii=False, sort_keys=True
            )
        )
    print(json.dumps({"written": written}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
