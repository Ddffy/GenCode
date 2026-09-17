"""Deterministic query understanding and retrieval-only rewriting."""

from __future__ import annotations

import re

_WORD_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_.:/-]*|\d+|[\u4e00-\u9fff]+"
)
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_PATH_RE = re.compile(
    r"(?:[A-Za-z0-9_.-]+[/\\])+[A-Za-z0-9_.-]+"
)
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "this", "to",
    "what", "when", "where", "which", "with", "请", "一下", "这个",
    "怎么", "如何", "什么", "哪里", "一个", "我们", "项目",
}


def search_tokens(text):
    """Tokenize prose, identifiers and CJK text for local sparse retrieval."""
    tokens = []
    seen = set()
    for match in _WORD_RE.findall(str(text or "")):
        raw = match.casefold().strip("._:/-")
        candidates = [raw]
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
            candidates = [raw[index : index + 2] for index in range(len(raw) - 1)]
            if len(raw) <= 8:
                candidates.append(raw)
        else:
            candidates.extend(
                part.casefold()
                for part in _CAMEL_RE.sub(" ", match).replace("_", " ").split()
            )
        for candidate in candidates:
            if len(candidate) < 2 or candidate in _STOP_WORDS or candidate in seen:
                continue
            seen.add(candidate)
            tokens.append(candidate)
    return tokens


def understand_query(query, *, source_types=()):
    """Keep the original query and derive safe retrieval-only variants."""
    original = str(query or "").strip()
    identifiers = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", original):
        if token not in identifiers:
            identifiers.append(token)
    paths = [value.replace("\\", "/") for value in _PATH_RE.findall(original)]
    tokens = search_tokens(original)
    variants = [original]
    compact = " ".join(tokens)
    if compact and compact.casefold() != original.casefold():
        variants.append(compact)
    if identifiers:
        variants.append(" ".join(identifiers))
    return {
        "original": original,
        "variants": list(dict.fromkeys(item for item in variants if item)),
        "tokens": tokens,
        "identifiers": identifiers,
        "paths": paths,
        "source_types": list(source_types),
        "query_type": "symbol" if identifiers or paths else "semantic",
    }
