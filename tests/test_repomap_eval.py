from gencode.evaluation.repomap_eval import (
    run_retrieval_experiment,
    summarize_live_rows,
)


def test_retrieval_experiment_reports_personalized_lift(tmp_path):
    (tmp_path / "caller.py").write_text(
        "from target import locate_widget\n\ndef run():\n    return locate_widget()\n",
        encoding="utf-8",
    )
    (tmp_path / "target.py").write_text(
        "def locate_widget():\n    return 'ok'\n",
        encoding="utf-8",
    )
    (tmp_path / "noise.py").write_text(
        "def unrelated():\n    return 0\n", encoding="utf-8"
    )
    tasks = [
        {
            "id": "locate",
            "category": "symbol_lookup",
            "query": "Where is locate_widget implemented?",
            "gold_files": ["target.py"],
        }
    ]

    payload = run_retrieval_experiment(
        tmp_path,
        tasks,
        budget_chars=400,
        cache_dir=tmp_path / "cache",
    )

    assert payload["task_count"] == 1
    assert payload["summary"]["personalized_map"]["hit_at_1"] == 1.0


def test_live_summary_uses_tokens_per_success_and_pairs():
    rows = [
        {
            "task_id": "a",
            "repeat": 0,
            "variant": "repo_map_off",
            "passed": True,
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": 110,
            "usage_source": "actual",
            "model_calls": 2,
            "tool_calls": 1,
            "search_calls": 1,
            "read_calls": 0,
        },
        {
            "task_id": "a",
            "repeat": 0,
            "variant": "repo_map_on",
            "passed": True,
            "input_tokens": 80,
            "output_tokens": 10,
            "total_tokens": 90,
            "usage_source": "actual",
            "model_calls": 1,
            "tool_calls": 0,
            "search_calls": 0,
            "read_calls": 0,
        },
        {
            "task_id": "b",
            "repeat": 0,
            "variant": "repo_map_off",
            "passed": False,
            "input_tokens": 50,
            "output_tokens": 5,
            "total_tokens": 55,
            "usage_source": "actual",
            "model_calls": 1,
            "tool_calls": 0,
            "search_calls": 0,
            "read_calls": 0,
        },
        {
            "task_id": "b",
            "repeat": 0,
            "variant": "repo_map_on",
            "passed": True,
            "input_tokens": 70,
            "output_tokens": 5,
            "total_tokens": 75,
            "usage_source": "actual",
            "model_calls": 1,
            "tool_calls": 0,
            "search_calls": 0,
            "read_calls": 0,
        },
    ]

    summary = summarize_live_rows(rows)

    assert summary["repo_map_off"]["pass_rate"] == 0.5
    assert summary["repo_map_on"]["pass_rate"] == 1.0
    assert summary["paired"]["pair_count"] == 2
    assert summary["paired"]["quality_gain_count"] == 1
