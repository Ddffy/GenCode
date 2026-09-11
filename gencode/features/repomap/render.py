"""Render the ranked file list into a budgeted repository map text.

Output mirrors aider's compact form: one line per file, highest rank
first, with the file's symbol skeleton in parentheses. Rendering stops at
the budget, so the head of the map is always the most relevant part.
"""

from __future__ import annotations

HEADER = "Repository map (files ranked by relevance to the current request):"
MAX_TAGS_PER_FILE = 12


def render_repo_map(ranked_paths, tags_by_file, budget_chars):
    if not ranked_paths:
        return ""
    lines = []
    used = len(HEADER) + 1
    for path in ranked_paths:
        file_tags = tags_by_file.get(path)
        if file_tags is None or not file_tags.definitions:
            continue
        symbols = []
        seen = set()
        for name, _line in file_tags.definitions:
            if name in seen:
                continue
            seen.add(name)
            symbols.append(name)
            if len(symbols) >= MAX_TAGS_PER_FILE:
                break
        line = f"- {path} ({', '.join(symbols)})"
        if used + len(line) + 1 > budget_chars:
            break
        lines.append(line)
        used += len(line) + 1
    if not lines:
        return ""
    return "\n".join([HEADER, *lines])
