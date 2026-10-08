"""Prompt-safe rendering for media references in session history."""


def render_media_refs(item):
    lines = []
    for ref in item.get("media_refs", []) or []:
        path = str(ref.get("path", "")).strip() or "-"
        mime_type = str(ref.get("mime_type", "")).strip() or "-"
        sha256 = str(ref.get("sha256", "")).strip()
        digest = sha256[:12] if sha256 else "-"
        artifact = str(ref.get("artifact_ref", "")).strip()
        suffix = f", artifact={artifact}" if artifact else ""
        lines.append(f"[image] path={path}, mime={mime_type}, sha256={digest}{suffix}")
    return lines
