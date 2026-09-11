"""Context pressure accounting for prompt usage metadata."""

from __future__ import annotations

from dataclasses import dataclass


IDENTITY_KEYS = (
    "provider",
    "provider_base_url",
    "model",
    "context_window",
    "prompt_cache_key",
    "prompt_hash",
)


def _optional_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ContextPressure:
    input_tokens: int
    context_window: int
    budget_tokens: int
    actual_input_tokens: int | None = None
    last_actual_input_tokens: int | None = None
    usage_source: str = "estimated"
    calibration_source: str = "missing_last_completion_metadata"
    cached_tokens: int | None = None

    @property
    def pressure_ratio(self):
        budget = max(1, int(self.budget_tokens or 0))
        return round(max(0, int(self.input_tokens or 0)) / budget, 6)

    @property
    def window_ratio(self):
        window = max(1, int(self.context_window or 0))
        return round(max(0, int(self.input_tokens or 0)) / window, 6)

    @property
    def pressure_tier(self):
        ratio = self.pressure_ratio
        if ratio >= 0.95:
            return "tier3_summary"
        if ratio >= 0.8:
            return "tier2_prune"
        if ratio >= 0.6:
            return "tier1_snip"
        return "tier0_observe"

    def to_context_usage_fields(self):
        return {
            "pressure_ratio": self.pressure_ratio,
            "window_ratio": self.window_ratio,
            "pressure_tier": self.pressure_tier,
            "budget_tokens": self.budget_tokens,
            "usage_source": self.usage_source,
            "actual_input_tokens": self.actual_input_tokens,
            "last_actual_input_tokens": self.last_actual_input_tokens,
            "calibration_source": self.calibration_source,
            "cached_tokens": self.cached_tokens,
        }


class ContextPressureController:
    def evaluate(
        self,
        *,
        estimated_input_tokens,
        context_window,
        budget_tokens=None,
        current_identity,
        last_completion_metadata=None,
        last_identity=None,
    ):
        estimated = max(0, int(estimated_input_tokens or 0))
        window = max(1, int(context_window or 0))
        budget = int(budget_tokens or 0) or window
        metadata = dict(last_completion_metadata or {})
        last_actual = _optional_int(metadata.get("input_tokens"))
        cached = None
        calibration_source = "missing_last_completion_metadata"
        usage_source = "estimated"
        actual = None
        input_tokens = estimated

        if metadata and last_actual is None:
            calibration_source = "missing_actual_input_tokens"
        elif last_actual is not None:
            if self._identity_matches(current_identity, metadata, last_identity):
                input_tokens = last_actual
                actual = last_actual
                cached = _optional_int(metadata.get("cached_tokens"))
                usage_source = "actual"
                calibration_source = "current_identity_match"
            else:
                calibration_source = "last_completion_identity_mismatch"

        return ContextPressure(
            input_tokens=input_tokens,
            context_window=window,
            budget_tokens=budget,
            actual_input_tokens=actual,
            last_actual_input_tokens=last_actual,
            usage_source=usage_source,
            calibration_source=calibration_source,
            cached_tokens=cached,
        )

    def _identity_matches(self, current_identity, metadata, last_identity):
        current = dict(current_identity or {})
        previous = dict(last_identity or {})
        previous.update({key: metadata[key] for key in IDENTITY_KEYS if key in metadata})
        return all(
            key in current and key in previous and current.get(key) == previous.get(key)
            for key in IDENTITY_KEYS
        )
