from types import SimpleNamespace

from gencode.core.context_replacements import commit_proposed_replacements
from gencode.core.turn_history import TurnHistoryBuilder


def fake_agent(history, context_replacements=None, changed_paths=None):
    session = {"history": history}
    if context_replacements is not None:
        session["context_replacements"] = context_replacements
    memory = SimpleNamespace(to_dict=lambda: {"file_summaries": {}})
    task_state = SimpleNamespace(changed_paths=list(changed_paths or []))
    return SimpleNamespace(session=session, memory=memory, current_task_state=task_state)


def old_tool_item(**overrides):
    item = {
        "role": "tool",
        "name": "read_file",
        "args": {"path": "src/app.py", "start": 1, "end": 200},
        "content": "full output saved: runs/current/read_file-output.txt\n"
        + "\n".join(f"FULL-CONTENT-LINE-{index}" for index in range(80)),
        "turn_id": "t0",
        "event_id": "event-read",
        "artifact_ref": "runs/current/read_file-output.txt",
        "original_chars": 6000,
        "content_sha256": "sha-current",
    }
    item.update(overrides)
    return item


def later_turns(count=4):
    history = []
    for index in range(1, count + 1):
        history.append({"role": "user", "content": f"later user {index}", "turn_id": f"t{index}"})
        history.append({"role": "assistant", "content": f"later answer {index}", "turn_id": f"t{index}"})
    return history


def render_history(agent):
    return TurnHistoryBuilder(agent).render_section(60000)


def test_turn_history_proposes_replacement_records_without_writing_session_ledger():
    history = [old_tool_item(), *later_turns()]
    agent = fake_agent(history)

    prompt, metadata = render_history(agent)

    assert "read_file output saved: runs/current/read_file-output.txt" in prompt
    assert "FULL-CONTENT-LINE-79" not in prompt
    assert "context_replacements" not in agent.session
    assert metadata["replacement_cache_hits"] == 0
    assert metadata["replacement_records_created"] == 1
    assert metadata["replacement_saved_chars"] > 0
    assert metadata["proposed_replacements"] == [
        {
            "event_id": "event-read",
            "content_sha256": "sha-current",
            "replacement_text": "read_file output saved: runs/current/read_file-output.txt (6000 chars)",
            "saved_chars": metadata["replacement_saved_chars"],
            "tool_name": "read_file",
            "artifact_ref": "runs/current/read_file-output.txt",
            "render_mode": "artifact_stub",
            "original_chars": 6000,
            "created_at": "",
            "pressure_tier": "",
        }
    ]


def test_turn_history_uses_existing_matching_replacement_ledger_record():
    ledger = {
        "records": [
            {
                "event_id": "event-read",
                "content_sha256": "sha-current",
                "replacement_text": "CACHED read_file stub",
                "saved_chars": 444,
                "tool_name": "read_file",
                "artifact_ref": "runs/current/read_file-output.txt",
            }
        ]
    }
    agent = fake_agent([old_tool_item(), *later_turns()], context_replacements=ledger)

    prompt, metadata = render_history(agent)

    assert "CACHED read_file stub" in prompt
    assert "read_file output saved: runs/current/read_file-output.txt" not in prompt
    assert metadata["replacement_cache_hits"] == 1
    assert metadata["replacement_records_created"] == 0
    assert metadata["replacement_saved_chars"] == 444
    assert metadata["proposed_replacements"] == []


def test_turn_history_uses_event_id_keyed_replacement_ledger_record():
    ledger = {
        "event-read": {
            "content_sha256": "sha-current",
            "replacement_text": "KEYED cached read_file stub",
            "saved_chars": 555,
            "tool_name": "read_file",
            "artifact_ref": "runs/current/read_file-output.txt",
            "render_mode": "artifact_stub",
            "original_chars": 6000,
            "created_at": "2026-06-19T00:00:00Z",
            "pressure_tier": "tier2_prune",
        }
    }
    agent = fake_agent([old_tool_item(), *later_turns()], context_replacements=ledger)

    prompt, metadata = render_history(agent)

    assert "KEYED cached read_file stub" in prompt
    assert "read_file output saved: runs/current/read_file-output.txt" not in prompt
    assert metadata["replacement_cache_hits"] == 1
    assert metadata["replacement_records_created"] == 0
    assert metadata["replacement_saved_chars"] == 555
    assert metadata["proposed_replacements"] == []


def test_turn_history_reads_mixed_legacy_records_and_event_id_keyed_ledger():
    ledger = {
        "records": [
            {
                "event_id": "legacy-event",
                "content_sha256": "legacy-sha",
                "replacement_text": "LEGACY cached stub",
                "saved_chars": 111,
            }
        ],
        "event-read": {
            "content_sha256": "sha-current",
            "replacement_text": "KEYED cached read_file stub",
            "saved_chars": 555,
        },
    }
    agent = fake_agent([old_tool_item(), *later_turns()], context_replacements=ledger)

    prompt, metadata = render_history(agent)

    assert "KEYED cached read_file stub" in prompt
    assert metadata["replacement_cache_hits"] == 1
    assert metadata["replacement_saved_chars"] == 555
    assert metadata["proposed_replacements"] == []


def test_turn_history_ignores_stale_replacement_hash_and_proposes_fresh_record():
    ledger = {
        "records": [
            {
                "event_id": "event-read",
                "content_sha256": "sha-stale",
                "replacement_text": "STALE cached read_file stub",
                "saved_chars": 444,
            }
        ]
    }
    agent = fake_agent([old_tool_item(), *later_turns()], context_replacements=ledger)

    prompt, metadata = render_history(agent)

    assert "STALE cached read_file stub" not in prompt
    assert "read_file output saved: runs/current/read_file-output.txt" in prompt
    assert metadata["replacement_cache_hits"] == 0
    assert metadata["replacement_records_created"] == 1
    assert metadata["proposed_replacements"][0]["content_sha256"] == "sha-current"


def test_turn_history_never_ledger_controls_old_items_without_event_id_and_content_hash():
    ledger = {
        "records": [
            {
                "event_id": "event-read",
                "content_sha256": "sha-current",
                "replacement_text": "CACHED read_file stub",
                "saved_chars": 444,
            }
        ]
    }
    agent = fake_agent(
        [
            old_tool_item(event_id="", content_sha256=""),
            *later_turns(),
        ],
        context_replacements=ledger,
    )

    prompt, metadata = render_history(agent)

    assert "CACHED read_file stub" not in prompt
    assert "read_file output saved: runs/current/read_file-output.txt" in prompt
    assert metadata["replacement_cache_hits"] == 0
    assert metadata["replacement_records_created"] == 0
    assert metadata["proposed_replacements"] == []


def test_existing_replacement_ledger_is_reversed_when_path_becomes_changed_across_turns():
    ledger = {
        "records": [
            {
                "event_id": "event-read",
                "content_sha256": "sha-current",
                "replacement_text": "CACHED read_file stub",
                "saved_chars": 444,
            }
        ]
    }
    agent = fake_agent([old_tool_item(), *later_turns()], context_replacements=ledger)
    first_prompt, first_metadata = render_history(agent)

    agent.current_task_state.changed_paths = ["src/app.py"]
    second_prompt, second_metadata = render_history(agent)

    assert "CACHED read_file stub" in first_prompt
    assert first_metadata["replacement_cache_hits"] == 1
    assert "CACHED read_file stub" not in second_prompt
    assert "[tool:read_file]" in second_prompt
    assert "FULL-CONTENT-LINE-0" in second_prompt
    assert second_metadata["replacement_cache_hits"] == 0
    assert second_metadata["replacement_records_created"] == 0


def test_commit_proposed_replacements_writes_event_id_keyed_session_records():
    session = {}
    metadata = {
        "history": {
            "proposed_replacements": [
                {
                    "event_id": "event-read",
                    "content_sha256": "sha-current",
                    "replacement_text": "read_file output saved: runs/current/read_file-output.txt (6000 chars)",
                    "saved_chars": 555,
                    "tool_name": "read_file",
                    "artifact_ref": "runs/current/read_file-output.txt",
                    "render_mode": "artifact_stub",
                    "original_chars": 6000,
                    "created_at": "2026-06-19T00:00:00Z",
                    "pressure_tier": "tier2_prune",
                }
            ]
        }
    }

    assert commit_proposed_replacements(session, metadata) == 1
    assert session["context_replacements"] == {
        "event-read": {
            "content_sha256": "sha-current",
            "replacement_text": "read_file output saved: runs/current/read_file-output.txt (6000 chars)",
            "saved_chars": 555,
            "tool_name": "read_file",
            "artifact_ref": "runs/current/read_file-output.txt",
            "render_mode": "artifact_stub",
            "original_chars": 6000,
            "created_at": "2026-06-19T00:00:00Z",
            "pressure_tier": "tier2_prune",
        }
    }
    assert commit_proposed_replacements(session, metadata) == 0
