"""Quarantine rules for durable memory notes."""

import re

from .memory_lint import SECRET_PATTERNS, _SECRET_HINT

QUARANTINE_PATTERN = re.compile(
    r"ignore (?:previous|prior) instructions|</?(?:system|assistant)>|disregard all earlier|new instructions:|you are now",
    re.I,
)


def should_quarantine(note_text):
    text = str(note_text)
    if QUARANTINE_PATTERN.search(text):
        return True
    # Most source lines contain no credential context at all.  Avoid feeding
    # long identifier/blob lines into the compound secret regex unless a
    # keyword or known provider prefix makes a match plausible.
    if not _SECRET_HINT.search(text):
        return False
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)
