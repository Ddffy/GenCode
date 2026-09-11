import json

from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.testing import ScriptedModelClient


def build_agent(tmp_path, outputs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".gencode" / "sessions")
    return GenCode(
        model_client=ScriptedModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_retrieval_trace_event_records_selected_and_rejected_without_prompt_leak(tmp_path):
    agent = build_agent(tmp_path, ["<final>Done.</final>"])
    for index in range(4):
        agent.memory.append_note(
            f"alpha note {index}",
            tags=("alpha",),
            created_at=f"2026-04-07T10:0{index}:00+00:00",
        )

    assert agent.ask("alpha") == "Done."

    trace_events = read_jsonl(agent.current_run_dir / "trace.jsonl")
    retrieval_events = [event for event in trace_events if event["event"] == "memory.retrieval"]
    assert len(retrieval_events) == 1
    payload = retrieval_events[0]
    assert len(payload["query_hash"]) == 12
    assert len(payload["workspace_fingerprint"]) == 12
    assert [note["text"] for note in payload["selected"]] == ["alpha note 3", "alpha note 2", "alpha note 1"]
    assert payload["rejected"][0]["text"] == "alpha note 0"
    assert payload["rejected"][0]["reject_reason"] == "below_limit"
    assert "alpha note 0" not in agent.model_client.prompts[-1]


def test_memory_file_read_trace_event_records_memory_paths(tmp_path):
    agent = build_agent(tmp_path, ["<final>Done.</final>"])
    topic_dir = tmp_path / ".gencode" / "memory" / "topics"
    topic_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".gencode" / "memory" / "MEMORY.md").write_text(
        "# Durable Memory Index\n\n- [test-topic](topics/test-topic.md): Test Topic\n",
        encoding="utf-8",
    )
    (topic_dir / "test-topic.md").write_text("# Test Topic\n\n## Notes\n- alpha durable note\n", encoding="utf-8")

    assert agent.ask("alpha") == "Done."

    trace_events = read_jsonl(agent.current_run_dir / "trace.jsonl")
    file_reads = [event for event in trace_events if event["event"] == "memory.file_read"]
    assert {event["reason"] for event in file_reads} == {"retrieval"}
    assert {event["path"] for event in file_reads} >= {
        ".gencode/memory/MEMORY.md",
        ".gencode/memory/topics/test-topic.md",
    }
