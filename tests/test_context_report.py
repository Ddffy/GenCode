from gencode.core.context_manager import SectionRender
from gencode.core.context_sections import CURRENT_REQUEST_SECTION, REDUCTION_ORDER, SECTION_ORDER
from gencode.core.context_report import ContextReportBuilder
from gencode.core.context_usage import ContextUsageAnalyzer


class DummyAgent:
    prefix = "prefix"
    skills = {}

    def available_tools(self):
        return {}


def test_context_report_builder_matches_existing_metadata_contract():
    agent = DummyAgent()
    rendered = {
        "prefix": SectionRender(raw="prefix raw", budget=100, rendered="prefix rendered", details={}),
        "memory": SectionRender(raw="memory raw", budget=90, rendered="memory rendered", details={}),
        "skills": SectionRender(raw="skills raw", budget=80, rendered="skills rendered", details={}),
        "relevant_memory": SectionRender(
            raw="Relevant memory:\n- note one\n- note two",
            budget=70,
            rendered="Relevant memory:\n- note one",
            details={"rendered_notes": ["note one"], "rendered_count": 1},
        ),
        "repo_map": SectionRender(raw="", budget=40, rendered="", details={}),
        "history": SectionRender(
            raw="history raw",
            budget=60,
            rendered="history rendered",
            details={
                "older_entries_count": 2,
                "collapsed_duplicate_reads": 1,
                "reused_file_summary_count": 1,
                "summarized_tool_count": 3,
                "rendered_turns": 4,
                "microcompact_artifact_refs": ["artifact-1"],
                "microcompact_saved_chars": 123,
                "replacement_cache_hits": 1,
                "replacement_records_created": 2,
                "replacement_saved_chars": 456,
                "proposed_replacements": [
                    {
                        "event_id": "event-read",
                        "content_sha256": "sha",
                        "replacement_text": "stub",
                        "saved_chars": 456,
                    }
                ],
            },
        ),
        "current_request": SectionRender(
            raw="Current user request:\nship it",
            budget=0,
            rendered="Current user request:\nship it",
            details={},
        ),
    }
    budgets = {
        "prefix": 100,
        "memory": 90,
        "skills": 80,
        "relevant_memory": 70,
        "repo_map": 40,
        "history": 60,
    }
    selected_notes = [
        {"text": "note one", "source": "topic-one", "kind": "durable"},
        {"text": "note two", "source": "session", "kind": "episodic"},
    ]
    section_texts = {"current_request": "Current user request:\nship it"}
    prompt = "\n\n".join(section.rendered for section in rendered.values())
    reduction_log = [{"section": "relevant_memory", "before_chars": 100, "after_chars": 70, "overflow_chars": 30}]
    expected = {
        "prompt_chars": len(prompt),
        "prompt_budget_chars": 1200,
        "prompt_over_budget": False,
        "section_order": list(SECTION_ORDER),
        "section_budgets": {
            section: None if section == CURRENT_REQUEST_SECTION else budgets[section]
            for section in SECTION_ORDER
        },
        "sections": {
            "prefix": {"raw_chars": 10, "budget_chars": 100, "rendered_chars": 15},
            "memory": {"raw_chars": 10, "budget_chars": 90, "rendered_chars": 15},
            "skills": {"raw_chars": 10, "budget_chars": 80, "rendered_chars": 15},
            "relevant_memory": {"raw_chars": 38, "budget_chars": 70, "rendered_chars": 27},
            "repo_map": {"raw_chars": 0, "budget_chars": 40, "rendered_chars": 0},
            "history": {"raw_chars": 11, "budget_chars": 60, "rendered_chars": 16},
            "current_request": {"raw_chars": 29, "budget_chars": None, "rendered_chars": 29},
        },
        "budget_reductions": reduction_log,
        "reduction_order": list(REDUCTION_ORDER),
        "relevant_memory": {
            "limit": 3,
            "selected_count": 2,
            "selected_notes": ["note one", "note two"],
            "selected_sources": ["topic-one", "session"],
            "selected_kinds": ["durable", "episodic"],
            "selected_durable_count": 1,
            "raw_chars": 38,
            "rendered_chars": 27,
            "rendered_notes": ["note one"],
            "rendered_count": 1,
        },
        "history": {
            "raw_chars": 11,
            "rendered_chars": 16,
            "older_entries_count": 2,
            "recent_window": 0,
            "old_turn_line_limit": 0,
            "collapsed_duplicate_reads": 1,
            "reused_file_summary_count": 1,
            "summarized_tool_count": 3,
            "rendered_turns": 4,
            "microcompact_artifact_refs": ["artifact-1"],
            "microcompact_saved_chars": 123,
            "replacement_cache_hits": 1,
            "replacement_records_created": 2,
            "replacement_saved_chars": 456,
            "proposed_replacements": [
                {
                    "event_id": "event-read",
                    "content_sha256": "sha",
                    "replacement_text": "stub",
                    "saved_chars": 456,
                }
            ],
        },
        "skills": {"available_count": 0, "user_invocable_count": 0, "items": []},
        "current_request": {
            "text": "ship it",
            "raw_chars": 7,
            "rendered_chars": 7,
            "section_chars": 29,
        },
        "context_usage": ContextUsageAnalyzer(agent).analyze(rendered),
    }

    actual = ContextReportBuilder(agent, total_budget=1200, reduction_order=REDUCTION_ORDER).build(
        prompt=prompt,
        rendered=rendered,
        budgets=budgets,
        reduction_log=reduction_log,
        selected_notes=selected_notes,
        user_message="ship it",
        section_texts=section_texts,
    )

    assert actual == expected
    assert list(actual.keys()) == [
        "prompt_chars",
        "prompt_budget_chars",
        "prompt_over_budget",
        "section_order",
        "section_budgets",
        "sections",
        "budget_reductions",
        "reduction_order",
        "relevant_memory",
        "history",
        "skills",
        "current_request",
        "context_usage",
    ]
