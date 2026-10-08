"""Async Provider boundary for LLM-backed handoff summaries."""

from ...providers.base import complete_model_async


async def generate_handoff_summary(adapter, delta_text, prior_summary_text=""):
    prompt = adapter._prompt(delta_text, prior_summary_text)
    try:
        result = await complete_model_async(
            adapter.model_client, prompt, adapter.max_summary_tokens
        )
    except Exception:  # noqa: BLE001 - deterministic compaction is the fallback
        adapter.last_usage = None
        return None
    return adapter._parse_result(result)
