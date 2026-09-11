"""Worker notification rendering."""

from xml.sax.saxutils import escape


def render_worker_notification(item):
    result = str(item.get("result", ""))
    parts = [
        "<task-notification>",
        f"<task-id>{escape(item['id'])}</task-id>",
        f"<status>{escape(item['status'])}</status>",
        f"<summary>{escape('Agent ' + item['description'] + ' ' + item['status'])}</summary>",
    ]
    if result:
        parts.append(f"<result>{escape(result)}</result>")
    parts.extend(
        [
            "<usage>",
            f"  <tool_uses>{int(item.get('tool_steps', 0))}</tool_uses>",
            f"  <attempts>{int(item.get('attempts', 0))}</attempts>",
            f"  <duration_ms>{int(item.get('duration_ms', 0))}</duration_ms>",
            "</usage>",
            "</task-notification>",
        ]
    )
    return "\n".join(parts)
