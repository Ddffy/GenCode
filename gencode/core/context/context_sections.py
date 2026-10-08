"""Data-only prompt section policy registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextSectionPolicy:
    name: str
    budget_chars: int | None
    floor_chars: int | None
    reduction_rank: int | None
    sources: tuple[str, ...]
    protected: bool


SECTION_ORDER = (
    "prefix",
    "memory",
    "skills",
    "relevant_memory",
    "repo_map",
    "history",
    "current_request",
)
REDUCTION_ORDER = ("relevant_memory", "repo_map", "skills", "history", "memory", "prefix")

SECTION_RATIOS = {
    "prefix": 0.20,
    "memory": 0.13,
    "skills": 0.07,
    "relevant_memory": 0.10,
    "repo_map": 0.03,
    "history": 0.47,
}

MIN_SECTION_BUDGETS = {
    "prefix": 4000,
    "memory": 1200,
    "skills": 600,
    "relevant_memory": 1000,
    "repo_map": 1500,
    "history": 6000,
}

DEFAULT_SECTION_BUDGETS = {
    "prefix": 12000,
    "memory": 8000,
    "skills": 4000,
    "relevant_memory": 6000,
    "repo_map": 4000,
    "history": 28000,
}


DEFAULT_UTILIZATION_RATIO = 0.50
OUTPUT_RESERVE_TOKENS = 16384
MIN_BUDGET_CHARS = 60000
MAX_BUDGET_CHARS = 800000


def compute_budget_chars(context_window_tokens, utilization_ratio=None):
    """Compute prompt budget in chars from model context window."""
    if not context_window_tokens or int(context_window_tokens) <= 0:
        return MIN_BUDGET_CHARS
    window = int(context_window_tokens)
    ratio = utilization_ratio if utilization_ratio is not None else DEFAULT_UTILIZATION_RATIO
    effective_tokens = max(0, window - OUTPUT_RESERVE_TOKENS)
    budget = int(effective_tokens * ratio * 4)
    if effective_tokens <= 0:
        budget = int(window * ratio * 4)
    return max(min(MIN_BUDGET_CHARS, window * 4), min(MAX_BUDGET_CHARS, budget))


def compute_budget_tokens(context_window_tokens, utilization_ratio=None):
    """Compute prompt budget in tokens from model context window."""
    if not context_window_tokens or int(context_window_tokens) <= 0:
        return MIN_BUDGET_CHARS // 4
    window = int(context_window_tokens)
    ratio = utilization_ratio if utilization_ratio is not None else DEFAULT_UTILIZATION_RATIO
    effective_tokens = max(0, window - OUTPUT_RESERVE_TOKENS)
    budget = int(effective_tokens * ratio)
    if effective_tokens <= 0:
        budget = int(window * ratio)
    return max(min(MIN_BUDGET_CHARS // 4, window), min(window, budget))


def compute_section_budgets(total_budget_chars, ratios=None):
    """Compute per-section budgets from total budget and ratios.

    Each section gets at least its MIN_SECTION_BUDGETS floor, unless
    the total budget is smaller than the sum of all floors — in that case
    budgets are purely ratio-based without floor enforcement.
    """
    ratios = ratios or SECTION_RATIOS
    floor_sum = sum(MIN_SECTION_BUDGETS.get(s, 0) for s in ratios)
    budgets = {}
    if total_budget_chars < floor_sum:
        for section, ratio in ratios.items():
            budgets[section] = int(total_budget_chars * ratio)
    else:
        for section, ratio in ratios.items():
            floor = MIN_SECTION_BUDGETS.get(section, 0)
            budgets[section] = max(floor, int(total_budget_chars * ratio))
    return budgets

_REDUCTION_RANKS = {name: rank for rank, name in enumerate(REDUCTION_ORDER)}

SECTION_POLICIES = (
    ContextSectionPolicy(
        name="prefix",
        budget_chars=DEFAULT_SECTION_BUDGETS["prefix"],
        floor_chars=MIN_SECTION_BUDGETS["prefix"],
        reduction_rank=_REDUCTION_RANKS["prefix"],
        sources=("workspace_prefix",),
        protected=False,
    ),
    ContextSectionPolicy(
        name="memory",
        budget_chars=DEFAULT_SECTION_BUDGETS["memory"],
        floor_chars=MIN_SECTION_BUDGETS["memory"],
        reduction_rank=_REDUCTION_RANKS["memory"],
        sources=("working_memory", "todo_ledger", "checkpoint_text", "memory_system_contract"),
        protected=True,
    ),
    ContextSectionPolicy(
        name="skills",
        budget_chars=DEFAULT_SECTION_BUDGETS["skills"],
        floor_chars=MIN_SECTION_BUDGETS["skills"],
        reduction_rank=_REDUCTION_RANKS["skills"],
        sources=("skills",),
        protected=False,
    ),
    ContextSectionPolicy(
        name="relevant_memory",
        budget_chars=DEFAULT_SECTION_BUDGETS["relevant_memory"],
        floor_chars=MIN_SECTION_BUDGETS["relevant_memory"],
        reduction_rank=_REDUCTION_RANKS["relevant_memory"],
        sources=("relevant_memory",),
        protected=True,
    ),
    ContextSectionPolicy(
        name="repo_map",
        budget_chars=DEFAULT_SECTION_BUDGETS["repo_map"],
        floor_chars=MIN_SECTION_BUDGETS["repo_map"],
        reduction_rank=_REDUCTION_RANKS["repo_map"],
        sources=("repo_map",),
        protected=False,
    ),
    ContextSectionPolicy(
        name="history",
        budget_chars=DEFAULT_SECTION_BUDGETS["history"],
        floor_chars=MIN_SECTION_BUDGETS["history"],
        reduction_rank=_REDUCTION_RANKS["history"],
        sources=("history",),
        protected=True,
    ),
    ContextSectionPolicy(
        name="current_request",
        budget_chars=None,
        floor_chars=None,
        reduction_rank=None,
        sources=("current_request",),
        protected=True,
    ),
)

SECTION_POLICIES_BY_NAME = {policy.name: policy for policy in SECTION_POLICIES}
CURRENT_REQUEST_SECTION = "current_request"


def section_order():
    return SECTION_ORDER


def section_budgets(total_budget_chars=None):
    if total_budget_chars is None:
        return dict(DEFAULT_SECTION_BUDGETS)
    return compute_section_budgets(total_budget_chars)


def section_floors():
    return dict(MIN_SECTION_BUDGETS)


def reduction_order():
    return REDUCTION_ORDER
