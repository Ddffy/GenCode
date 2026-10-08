"""Async compaction path that keeps model calls off the synchronous facade."""

import asyncio

from gencode.core.context.compact_summary import summarize_compact_items
from gencode.core.context.context_handoff import (
    HandoffAdapter,
    render_delta_for_handoff,
    render_handoff_summary,
)
from gencode.core.context.context_handoff_async import generate_handoff_summary
from gencode.core.runtime.workspace_context import now


async def compact_async(manager, trigger, keep_recent_turns, summary_mode):
    plan = manager.plan(trigger=trigger, keep_recent_turns=keep_recent_turns)
    agent = manager.agent
    history = list(agent.session.get("history", []))
    selected = manager._select(keep_recent_turns)
    if agent.current_task_state:
        agent.emit_trace(
            agent.current_task_state,
            "compaction_started",
            {
                "trigger": trigger,
                "pre_tokens": manager._tokens(history),
                "plan": plan.to_dict(),
            },
        )
    if not plan.delta_event_ids:
        summary = manager._summary(
            trigger,
            history,
            history,
            "",
            plan,
            summary_called=False,
            summary_mode=summary_mode,
        )
        agent.session_event_bus.emit("compaction_created", summary)
        if agent.current_task_state:
            agent.emit_trace(
                agent.current_task_state, "compaction_finished", summary
            )
        return summary

    delta_items = [item for _, item in selected["delta_items"]]
    kept_items = [item for _, item in selected["protected_items"]]
    prior_text = str((selected["prior_summary"] or {}).get("content", "")).strip()
    summary_text, resolved_mode, compact_call_usage = await _summary_text(
        agent, delta_items, prior_text, summary_mode
    )
    summary_item = agent.turn_history.enrich(
        {
            "role": "system",
            "kind": "compact_summary",
            "content": summary_text,
            "created_at": now(),
            "source": "compact",
            "turn_id": str(
                (selected["prior_summary"] or {}).get(
                    "turn_id", "compact_summary"
                )
            ),
        }
    )
    agent.session["history"] = [summary_item, *kept_items]
    agent.session["context_summary"] = manager._context_summary(
        plan, summary_item, selected
    )
    summary = manager._summary(
        trigger,
        history,
        agent.session["history"],
        summary_text,
        plan,
        summary_mode=resolved_mode,
        compact_call_usage=compact_call_usage,
    )
    agent.session.setdefault("compactions", []).append(
        manager._persistent_summary(summary)
    )
    agent.session_path = await asyncio.to_thread(
        agent.session_store.save, agent.session
    )
    agent.session_event_bus.emit("compaction_created", summary)
    if agent.current_task_state:
        agent.emit_trace(
            agent.current_task_state, "compaction_finished", summary
        )
    return summary


async def _summary_text(agent, delta_items, prior_text, summary_mode):
    if summary_mode != "llm":
        return summarize_compact_items(delta_items, prior_text=prior_text), "deterministic", None
    adapter = HandoffAdapter(agent.model_client)
    handoff = await generate_handoff_summary(
        adapter, render_delta_for_handoff(delta_items), prior_text
    )
    if handoff is None:
        return (
            summarize_compact_items(delta_items, prior_text=prior_text),
            "deterministic_fallback",
            adapter.last_usage,
        )
    return render_handoff_summary(handoff), "llm", adapter.last_usage
