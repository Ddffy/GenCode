"""Aggregate model-token accounting for one durable Goal."""

from __future__ import annotations

import asyncio
import math
import uuid

ROLE_BUDGET_SHARES = {
    "planner": 0.10,
    "worker": 0.70,
    "critic": 0.15,
    "query_resolver": 0.05,
}


def _int(value):
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


class GoalTokenBudget:
    def __init__(self, goal, store):
        self.goal = goal
        self.store = store
        self.lock = asyncio.Lock()
        usage = goal.setdefault("token_usage", {})
        usage.setdefault("input_tokens", 0)
        usage.setdefault("uncached_input_tokens", 0)
        usage.setdefault("cached_tokens", 0)
        usage.setdefault("output_tokens", 0)
        usage.setdefault("total_tokens", 0)
        usage.setdefault("model_calls", 0)
        usage.setdefault("estimated_calls", 0)
        usage.setdefault("budget_exhausted", False)
        usage.setdefault("by_role", {})
        usage.setdefault("pending_reservations", {})
        self.max_total_tokens = max(1, int(goal.get("max_total_tokens", 500_000)))
        self.reserved_tokens = sum(
            _int(item.get("reserved_tokens"))
            for item in usage["pending_reservations"].values()
            if isinstance(item, dict)
        )
        self.reserved_by_role = {}
        for item in usage["pending_reservations"].values():
            if isinstance(item, dict):
                role = str(item.get("role") or "unknown")
                self.reserved_by_role[role] = (
                    self.reserved_by_role.get(role, 0)
                    + _int(item.get("reserved_tokens"))
                )
        self._charge_uncertain_reservations()
        self.blocked = _int(usage["total_tokens"]) >= self.max_total_tokens
        usage["budget_exhausted"] = self.blocked
        if not self.blocked:
            usage["budget_exhausted_reason"] = ""
            usage["budget_exhausted_role"] = ""
        self.store.save(self.goal)

    async def reserve(self, estimated_input_tokens, max_output_tokens, role):
        estimated_input_tokens = max(0, int(estimated_input_tokens or 0))
        max_output_tokens = max(0, int(max_output_tokens or 0))
        amount = estimated_input_tokens + max_output_tokens
        role = str(role or "unknown")
        async with self.lock:
            usage = self.goal["token_usage"]
            role_usage = usage["by_role"].get(role, {})
            role_budget = max(
                1,
                math.ceil(
                    self.max_total_tokens
                    * ROLE_BUDGET_SHARES.get(role, ROLE_BUDGET_SHARES["query_resolver"])
                ),
            )
            if (
                self.blocked
                or int(usage["total_tokens"]) + self.reserved_tokens + amount
                > self.max_total_tokens
                or _int(role_usage.get("total_tokens"))
                + self.reserved_by_role.get(role, 0)
                + amount
                > role_budget
            ):
                self.blocked = True
                usage["budget_exhausted"] = True
                usage["budget_exhausted_reason"] = (
                    "total token budget reached"
                    if int(usage["total_tokens"]) + self.reserved_tokens + amount
                    > self.max_total_tokens
                    else "role token allocation reached"
                )
                usage["budget_exhausted_role"] = role
                self.store.save(self.goal)
                return None
            self.reserved_tokens += amount
            self.reserved_by_role[role] = self.reserved_by_role.get(role, 0) + amount
            reservation = {
                "reservation_id": uuid.uuid4().hex,
                "role": role,
                "estimated_input_tokens": estimated_input_tokens,
                "max_output_tokens": max_output_tokens,
                "reserved_tokens": amount,
            }
            usage["pending_reservations"][reservation["reservation_id"]] = reservation
            self.store.save(self.goal)
        return reservation

    def _remove_reservation(self, usage, reservation):
        reservation_id = str(reservation.get("reservation_id", ""))
        stored = usage["pending_reservations"].pop(reservation_id, None)
        if stored is None:
            return
        role = str(stored.get("role") or "unknown")
        amount = _int(stored.get("reserved_tokens"))
        self.reserved_tokens = max(0, self.reserved_tokens - amount)
        self.reserved_by_role[role] = max(
            0, self.reserved_by_role.get(role, 0) - amount
        )

    def _charge_uncertain_reservations(self):
        """Charge a full reservation after a crash, when a provider call may have completed."""
        usage = self.goal["token_usage"]
        pending = list(usage["pending_reservations"].values())
        if not pending:
            return
        for reservation in pending:
            role = str(reservation.get("role") or "unknown")
            input_tokens = _int(reservation.get("estimated_input_tokens"))
            output_tokens = _int(reservation.get("max_output_tokens"))
            total = input_tokens + output_tokens
            usage["input_tokens"] += input_tokens
            usage["uncached_input_tokens"] += input_tokens
            usage["output_tokens"] += output_tokens
            usage["total_tokens"] += total
            usage["estimated_calls"] += 1
            role_usage = usage["by_role"].setdefault(
                role,
                {
                    "input_tokens": 0,
                    "uncached_input_tokens": 0,
                    "cached_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "model_calls": 0,
                    "estimated_calls": 0,
                },
            )
            role_usage["input_tokens"] += input_tokens
            role_usage["uncached_input_tokens"] += input_tokens
            role_usage["output_tokens"] += output_tokens
            role_usage["total_tokens"] += total
            role_usage["estimated_calls"] = role_usage.get("estimated_calls", 0) + 1
        usage["pending_reservations"] = {}
        self.reserved_tokens = 0
        self.reserved_by_role = {}
        self.blocked = usage["total_tokens"] >= self.max_total_tokens
        usage["budget_exhausted"] = self.blocked
        self.store.save(self.goal)

    async def release(self, reservation):
        if reservation is None:
            return
        async with self.lock:
            usage = self.goal["token_usage"]
            self._remove_reservation(usage, reservation)
            self.store.save(self.goal)

    async def record(self, reservation, role, metadata, response_text=""):
        if reservation is None:
            return self.snapshot()
        metadata = dict(metadata or {})
        cached = _int(metadata.get("cached_tokens"))
        protocol = str(metadata.get("provider_protocol") or "").lower()
        input_value = metadata.get("total_input_tokens")
        if input_value is None:
            reported_input = _int(metadata.get("input_tokens"))
            cache_created = _int(metadata.get("cache_creation_input_tokens"))
            input_value = (
                reported_input + cached + cache_created
                if protocol == "anthropic"
                else max(reported_input, cached) + cache_created
            )
        input_tokens = _int(input_value)
        output_value = metadata.get("output_tokens")
        estimated = output_value is None or metadata.get("total_input_tokens") is None
        if metadata.get("total_input_tokens") is None and not input_tokens:
            input_tokens = int(reservation["estimated_input_tokens"])
        output_tokens = (
            _int(output_value)
            if output_value is not None
            else (len(str(response_text or "")) + 3) // 4
        )
        cached = min(cached, input_tokens)
        uncached = max(0, input_tokens - cached)
        total = input_tokens + output_tokens
        role = str(role or reservation.get("role") or "unknown")

        async with self.lock:
            usage = self.goal["token_usage"]
            self._remove_reservation(usage, reservation)
            usage["input_tokens"] += input_tokens
            usage["uncached_input_tokens"] += uncached
            usage["cached_tokens"] += cached
            usage["output_tokens"] += output_tokens
            usage["total_tokens"] += total
            usage["model_calls"] += 1
            usage["estimated_calls"] += int(estimated)
            role_usage = usage["by_role"].setdefault(
                role,
                {
                    "input_tokens": 0,
                    "uncached_input_tokens": 0,
                    "cached_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "model_calls": 0,
                    "estimated_calls": 0,
                },
            )
            role_usage["input_tokens"] += input_tokens
            role_usage["uncached_input_tokens"] += uncached
            role_usage["cached_tokens"] += cached
            role_usage["output_tokens"] += output_tokens
            role_usage["total_tokens"] += total
            role_usage["model_calls"] += 1
            role_usage["estimated_calls"] = role_usage.get("estimated_calls", 0) + int(estimated)
            role_limit = max(
                1,
                math.ceil(
                    self.max_total_tokens
                    * ROLE_BUDGET_SHARES.get(role, ROLE_BUDGET_SHARES["query_resolver"])
                ),
            )
            if usage["total_tokens"] >= self.max_total_tokens:
                self.blocked = True
                usage["budget_exhausted"] = True
                usage["budget_exhausted_reason"] = "total token budget reached"
                usage["budget_exhausted_role"] = role
            elif role_usage["total_tokens"] >= role_limit:
                self.blocked = True
                usage["budget_exhausted"] = True
                usage["budget_exhausted_reason"] = "role token allocation reached"
                usage["budget_exhausted_role"] = role
            self.store.save(self.goal)
            return self.snapshot()

    def snapshot(self):
        usage = self.goal.get("token_usage", {})
        role_limits = {
            role: max(1, math.ceil(self.max_total_tokens * share))
            for role, share in ROLE_BUDGET_SHARES.items()
        }
        return {
            **usage,
            "by_role": {
                role: dict(details)
                for role, details in usage.get("by_role", {}).items()
            },
            "max_total_tokens": self.max_total_tokens,
            "max_role_tokens": role_limits,
            "blocked": self.blocked,
            "reserved_tokens": self.reserved_tokens,
            "remaining_tokens": max(
                0, self.max_total_tokens - int(usage.get("total_tokens", 0))
            ),
        }
