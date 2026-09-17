"""Provider-neutral message assembly for native Tool Calling clients."""

from __future__ import annotations

import json


def native_prompt_contract(agent, native_tools, legacy_tool_lines):
    """Return prompt-facing tool policy without duplicating native schemas."""
    if native_tools:
        return (
            "\n".join(f"- {name} [native schema supplied by provider]" for name in agent.available_tools()),
            """\n            - Use the provider's native tool-calling interface for tools; do not emit XML, DSML, or JSON tool tags.\n            - Return ordinary text for the final answer; tool calls are structured provider messages.\n            """,
            "Final answer example: Done.",
            "- Final answers must be ordinary text; do not wrap them in <final> tags.",
            "",
        )
    return (
        "\n".join(legacy_tool_lines),
        """\n            - Return one or more <tool>...</tool> calls, or one <final>...</final>.\n            - Tool calls must look like:\n              <tool>{\"name\":\"tool_name\",\"args\":{...}}</tool>\n            """,
        "\n".join(
            [
                '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
                '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
                '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
                '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
                '<tool>{"name":"agent","args":{"description":"Inspect auth","prompt":"Find auth entry points","subagent_type":"Explore"}}</tool>',
                "<final>Done.</final>",
            ]
        ),
        "- Final answers must look like: <final>your answer</final>.",
        '- For write_file and patch_file with multi-line text, prefer XML style: <tool name="write_file" path="file.py"><content>...</content></tool>.',
    )


def build_native_messages(agent, user_message, prompt=None):
    """Build structured messages while retaining compressed context summaries."""
    sections = dict(getattr(agent.context_manager, "last_rendered_sections", {}) or {})
    if not sections:
        return [{"role": "user", "content": str(prompt or user_message)}]

    history = list(agent.session.get("history", []) or [])
    if (
        history
        and history[-1].get("role") == "user"
        and str(history[-1].get("content", "")) == str(user_message)
    ):
        history_without_current = history[:-1]
    else:
        history_without_current = history
    canonical, canonical_count = _native_history_tail(history_without_current)

    system_parts = []
    for section in ("prefix", "memory", "skills", "relevant_memory", "repo_map"):
        value = str(sections.get(section, "") or "").strip()
        if value:
            system_parts.append(value)
    rendered_history = str(sections.get("history", "") or "").strip()
    if rendered_history and len(history_without_current) > canonical_count:
        system_parts.append(rendered_history)

    messages = []
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})
    messages.extend(canonical)
    messages.append({"role": "user", "content": str(user_message)})
    return messages


def _native_history_tail(history, max_items=16):
    """Take the most recent items without splitting a tool-call group.

    A blind positional slice can begin inside a group, leaving tool results whose
    ``tool_use`` was cut away. Anthropic rejects that with "each tool_result block
    must have a corresponding tool_use block" and OpenAI-shaped APIs reject the
    equivalent, so the window is first grown backwards over any leading tool
    results to include the assistant message that owns them. A turn that issues
    several tool calls at once is what shifts the alignment and makes this
    reachable.
    """
    items = list(history)
    start = max(0, len(items) - max_items)
    while start > 0 and str(items[start].get("role")) == "tool":
        start -= 1
    source = items[start:]
    known_call_ids = {
        str(call.get("id") or call.get("call_id") or "")
        for item in source
        if str(item.get("role")) == "assistant"
        for call in (item.get("tool_calls") or ())
    }
    result_call_ids = {
        str(item["tool_call_id"])
        for item in source
        if str(item.get("role")) == "tool" and item.get("tool_call_id")
    }
    messages = []
    for item in source:
        role = str(item.get("role", ""))
        content = item.get("content", "")
        if role == "assistant" and item.get("tool_calls"):
            calls = []
            for call in item.get("tool_calls") or ():
                call_id = str(call.get("id") or call.get("call_id") or "")
                # History recorded before the skipped-call guard existed can hold
                # calls whose results were never written (a batch cut short by the
                # step budget or an abort). A provider rejects the whole request for
                # one unpaired tool_use, so the call is dropped from the request
                # rather than sent; otherwise the session can never be resumed.
                if call_id and call_id not in result_call_ids:
                    continue
                function = call.get("function") or {}
                args = call.get("args", function.get("arguments", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw_arguments": args}
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": str(call.get("name") or function.get("name") or ""),
                            "arguments": json.dumps(args or {}, ensure_ascii=False),
                        },
                    }
                )
            if calls:
                messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            elif content:
                messages.append({"role": "assistant", "content": str(content)})
        elif role == "tool" and item.get("tool_call_id"):
            call_id = str(item["tool_call_id"])
            # Stored history can still lack the owning call (for example a crash
            # between recording the call and its results), so an orphan is dropped
            # here rather than sent as a payload every provider rejects.
            if call_id not in known_call_ids:
                continue
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": str(item.get("name", "")),
                    "content": str(content),
                }
            )
        elif role in {"user", "assistant", "system"}:
            messages.append({"role": role, "content": str(content)})
    return messages, len(messages)
