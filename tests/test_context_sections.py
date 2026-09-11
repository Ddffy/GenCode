from gencode.core.context_sections import (
    DEFAULT_SECTION_BUDGETS,
    MIN_SECTION_BUDGETS,
    REDUCTION_ORDER,
    SECTION_ORDER,
    SECTION_POLICIES,
    ContextSectionPolicy,
    reduction_order,
    section_budgets,
    section_floors,
    section_order,
)
from gencode.core.context_manager import ContextManager


def test_context_section_policy_registry_preserves_order_and_budget_data():
    assert SECTION_ORDER == ("prefix", "memory", "skills", "relevant_memory", "repo_map", "history", "current_request")
    assert [policy.name for policy in SECTION_POLICIES] == list(SECTION_ORDER)
    assert all(isinstance(policy, ContextSectionPolicy) for policy in SECTION_POLICIES)
    assert DEFAULT_SECTION_BUDGETS == {
        "prefix": 12000,
        "memory": 8000,
        "skills": 4000,
        "relevant_memory": 6000,
        "repo_map": 4000,
        "history": 28000,
    }
    assert MIN_SECTION_BUDGETS == {
        "prefix": 4000,
        "memory": 1200,
        "skills": 600,
        "relevant_memory": 1000,
        "repo_map": 1500,
        "history": 6000,
    }
    assert REDUCTION_ORDER == ("relevant_memory", "repo_map", "skills", "history", "memory", "prefix")
    assert section_order() == SECTION_ORDER
    assert section_budgets() == DEFAULT_SECTION_BUDGETS
    assert section_floors() == MIN_SECTION_BUDGETS
    assert reduction_order() == REDUCTION_ORDER


def test_context_section_policy_records_sources_and_non_reducible_request():
    policies = {policy.name: policy for policy in SECTION_POLICIES}

    assert policies["prefix"].sources == ("workspace_prefix",)
    assert policies["memory"].sources == (
        "working_memory",
        "todo_ledger",
        "checkpoint_text",
        "memory_system_contract",
    )
    assert policies["skills"].sources == ("skills",)
    assert policies["relevant_memory"].sources == ("relevant_memory",)
    assert policies["repo_map"].sources == ("repo_map",)
    assert policies["history"].sources == ("history",)
    assert policies["current_request"].sources == ("current_request",)

    protected_sources = {source for policy in SECTION_POLICIES if policy.protected for source in policy.sources}
    assert protected_sources == {
        "working_memory",
        "todo_ledger",
        "checkpoint_text",
        "memory_system_contract",
        "relevant_memory",
        "history",
        "current_request",
    }
    assert policies["current_request"].budget_chars is None
    assert policies["current_request"].floor_chars is None
    assert policies["current_request"].reduction_rank is None
    assert {name: policies[name].reduction_rank for name in REDUCTION_ORDER} == {
        "relevant_memory": 0,
        "repo_map": 1,
        "skills": 2,
        "history": 3,
        "memory": 4,
        "prefix": 5,
    }


def test_context_manager_exports_legacy_section_policy_names():
    from gencode.core.context_manager import (
        DEFAULT_REDUCTION_ORDER,
        DEFAULT_SECTION_FLOORS,
        DEFAULT_TOTAL_BUDGET,
    )
    from gencode.core.context_sections import SECTION_ORDER as LEGACY_ORDER

    assert DEFAULT_SECTION_FLOORS == MIN_SECTION_BUDGETS
    assert DEFAULT_REDUCTION_ORDER == REDUCTION_ORDER
    assert LEGACY_ORDER == SECTION_ORDER
    assert DEFAULT_TOTAL_BUDGET == 60000


def test_context_manager_default_floors_match_section_registry():
    manager = ContextManager(agent=object())

    assert manager.section_floors == MIN_SECTION_BUDGETS


def test_context_manager_recomputes_floors_for_mutated_custom_budgets():
    manager = ContextManager(agent=object())
    manager.section_budgets = {"prefix": 120, "memory": 120, "relevant_memory": 120, "history": 160, "extra": 80}

    floors = manager._compute_section_floors()
    assert floors["prefix"] == MIN_SECTION_BUDGETS["prefix"]
    assert floors["memory"] == MIN_SECTION_BUDGETS["memory"]
    assert floors["relevant_memory"] == MIN_SECTION_BUDGETS["relevant_memory"]
    assert floors["history"] == MIN_SECTION_BUDGETS["history"]
    assert floors["extra"] == 20
