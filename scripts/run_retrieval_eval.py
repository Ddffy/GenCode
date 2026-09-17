"""Run a hybrid retrieval benchmark against reviewed GenCode Wiki records."""

import argparse
import json
from pathlib import Path

from gencode.config import load_project_env, resolve_project_retrieval_config
from gencode.evaluation.retrieval_eval import (
    load_retrieval_tasks,
    run_retrieval_evaluation,
)
from gencode.features.knowledge import KnowledgeStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", default="artifacts/retrieval-eval.json")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    # 与 cli.py 的装配顺序保持一致：先把项目 .env 落到 os.environ，
    # 否则 retrieval_config_from_env 读不到 GENCODE_EMBEDDING_* /
    # GENCODE_RERANKER_*，会静默回退到 hashing + deterministic。
    load_project_env(repo, override=False)
    store = KnowledgeStore(
        repo / ".gencode" / "knowledge",
        repo,
        retrieval_config=resolve_project_retrieval_config(start=repo),
    )
    records = store.list_records(kind="wiki", include_inactive=True)
    store.retrieval.sync_wiki(records, force=True)
    result = run_retrieval_evaluation(
        store.retrieval,
        load_retrieval_tasks(args.tasks),
        artifact_path=repo / args.output,
    )
    print(json.dumps(result["variants"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
