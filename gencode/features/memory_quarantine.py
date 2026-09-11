"""Quarantine rules for durable memory notes."""

import re

from .memory_lint import SECRET_PATTERNS

QUARANTINE_PATTERN = re.compile(
    r"ignore (?:previous|prior) instructions|</?(?:system|assistant)>|disregard all earlier|new instructions:|you are now",
    re.I,
)


def should_quarantine(note_text):
    text = str(note_text)
    return bool(QUARANTINE_PATTERN.search(text)) or any(pattern.search(text) for pattern in SECRET_PATTERNS)
