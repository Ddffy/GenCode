"""Bounded Skill frontmatter reads and deferred body reads."""

from pathlib import Path


def read_skill_metadata(path):
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as stream:
            prefix = stream.read(64 * 1024)
    except OSError:
        return {}
    if not prefix.startswith("---\n"):
        return {}
    from .skills import parse_frontmatter

    return parse_frontmatter(prefix)[0]


def read_skill_body(path):
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    from .skills import parse_frontmatter

    return parse_frontmatter(text)[1].strip()
