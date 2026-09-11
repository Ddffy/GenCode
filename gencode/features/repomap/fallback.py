"""Dependency-free tag extraction fallback for offline environments."""

from __future__ import annotations

import re


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DEFINITION = re.compile(
    r"^\s*(?:async\s+)?(?:def|class|function|interface|type|struct|enum|trait)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
_KEYWORDS = {
    "and", "as", "assert", "async", "await", "break", "class", "const", "def",
    "elif", "else", "enum", "except", "false", "for", "from", "function", "if",
    "import", "in", "interface", "is", "let", "match", "none", "not", "null",
    "or", "pass", "raise", "return", "struct", "trait", "true", "try", "type",
    "var", "while", "with", "yield",
}


def extract_fallback_tags(rel_path, language, source_bytes):
    """Return conservative ``(definitions, references)`` when Tree-sitter is unavailable."""
    text = source_bytes.decode("utf-8", "replace")
    definitions = []
    definition_offsets = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        match = _DEFINITION.match(line)
        if match:
            name = match.group(1)
            definitions.append((name, line_number))
            definition_offsets.add((line_number, name))
    references = []
    seen = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        for match in _IDENTIFIER.finditer(line):
            name = match.group(0)
            if name.lower() in _KEYWORDS or (line_number, name) in definition_offsets:
                continue
            item = (name, line_number)
            if item not in seen:
                seen.add(item)
                references.append(item)
    return tuple(definitions), tuple(references)
