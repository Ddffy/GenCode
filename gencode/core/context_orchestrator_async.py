"""Async context assembly orchestration."""

import asyncio

from .context_orchestrator import ContextBuildResult


async def build_context_async(orchestrator, snapshot):
    agent = orchestrator.agent
    prompt, metadata = await asyncio.to_thread(
        agent.context_manager.build, snapshot.request
    )
    plan = None
    summary = None
    should_compact = False
    skip_reason = ""
    compact_metrics = {}
    compact_trigger, summary_mode, skip_reason = orchestrator._compact_request(
        metadata, snapshot
    )
    if compact_trigger and len(snapshot.session.get("history", [])) <= 4:
        skip_reason = "history_too_short_for_auto_compaction"
        metadata["auto_compacted"] = False
        metadata["auto_compaction_skip_reason"] = skip_reason
    elif compact_trigger:
        pre_compact_estimated_tokens = int(
            (metadata.get("context_usage", {}) or {}).get(
                "total_estimated_tokens", 0
            )
            or 0
        )
        plan = agent.compact_manager.plan(trigger=compact_trigger)
        summary = await agent.compact_manager.compact_async(
            trigger=plan.trigger,
            keep_recent_turns=plan.keep_recent_turns,
            summary_mode=summary_mode,
        )
        should_compact = bool(summary.get("summary_called", True))
        if should_compact:
            prompt, metadata = await asyncio.to_thread(
                agent.context_manager.build, snapshot.request
            )
        post_compact_estimated_tokens = int(
            (metadata.get("context_usage", {}) or {}).get(
                "total_estimated_tokens", 0
            )
            or 0
        )
        compact_metrics = {
            "pre_compact_estimated_tokens": pre_compact_estimated_tokens,
            "post_compact_estimated_tokens": post_compact_estimated_tokens,
        }
        metadata.update(
            {
                "auto_compacted": should_compact,
                "auto_compaction_plan": plan.to_dict(),
                "auto_compaction_summary": summary,
            }
        )
    orchestrator._attach_metadata(
        metadata,
        snapshot,
        plan,
        summary,
        should_compact,
        skip_reason,
        compact_metrics,
    )
    orchestrator._emit_decision(metadata)
    orchestrator._emit_usage(metadata)
    return ContextBuildResult(
        prompt=prompt,
        metadata=metadata,
        should_compact=should_compact,
        compact_trigger=(plan.trigger if plan else None),
    )
