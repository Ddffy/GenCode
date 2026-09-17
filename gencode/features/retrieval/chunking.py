"""Parent-child chunking for Markdown knowledge and source code."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .types import RetrievalChunk

MAX_CHUNK_CHARS = 8000
MAX_PARENT_SUMMARY_CHARS = 500
# Section bodies are stored whole because the line-window fallback slices them by
# the matched child's line offset. Truncating a section to the child cap would make
# that offset point past the end of the stored text, so a hit in the tail would
# silently fall back to the bare child. The limit here is only a sanity bound on
# pathological generated files, not a chunking budget.
MAX_SECTION_CHARS = 200_000
_HEADING_RE = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*#*\s*$")

# Child chunks are produced by recursive splitting inside a structural unit, so a
# long function or a long section is cut on its own boundaries instead of being
# truncated. Parents stay structural (file / page).
DEFAULT_CHILD_CHUNK_TOKENS = 300
DEFAULT_CHILD_CHUNK_OVERLAP = 0.10

# Tried in order; a piece that is still too large is handed to the next separator.
# Paragraph breaks are preferred over line breaks, which are preferred over
# sentence punctuation, and a character split is the last resort.
_RECURSIVE_SEPARATORS = (
    "\n\n",
    "\n",
    "。",
    "；",
    "！",
    "？",
    ". ",
    "; ",
    ", ",
    "，",
    " ",
    "",
)


def clean_text(value, *, limit=MAX_CHUNK_CHARS):
    text = str(value or "").replace("\x00", "").replace("\r\n", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(text) > int(limit):
        text = text[: int(limit) - 1].rstrip() + "…"
    return text


def _is_wide(character):
    """True for characters that cost roughly one token on their own."""
    code = ord(character)
    return (
        0x1100 <= code <= 0x115F  # Hangul Jamo
        or 0x2E80 <= code <= 0xA4CF  # CJK radicals through Yi
        or 0xAC00 <= code <= 0xD7A3  # Hangul syllables
        or 0xF900 <= code <= 0xFAFF  # CJK compatibility ideographs
        or 0xFE30 <= code <= 0xFE6F  # CJK compatibility forms
        or 0xFF00 <= code <= 0xFF60  # fullwidth forms
        or 0xFFE0 <= code <= 0xFFE6
    )


def estimate_tokens(text):
    """Estimate a token count without shipping a tokenizer.

    Wide characters (CJK, Hangul) count as one token each and every other four
    characters count as one, which tracks the usual ratios closely enough to size
    chunks. It is deliberately an estimate: the exact count depends on the
    embedder's own tokenizer, and chunk sizing does not need to be exact.
    """
    total = 0.0
    for character in str(text or ""):
        total += 1.0 if _is_wide(character) else 0.25
    return int(total) if total == int(total) else int(total) + 1


def _hard_split(text, *, chunk_size_tokens):
    """Last-resort split for text that offers no separator at all."""
    pieces = []
    current = []
    pending = 0.0
    for character in text:
        current.append(character)
        pending += 1.0 if _is_wide(character) else 0.25
        if pending >= chunk_size_tokens:
            pieces.append("".join(current))
            current = []
            pending = 0.0
    if current:
        pieces.append("".join(current))
    return pieces


def _split_keeping_separator(text, separator, offset):
    """Split on ``separator`` keeping it attached to the piece it ends."""
    pieces = []
    cursor = 0
    while True:
        index = text.find(separator, cursor)
        if index < 0:
            break
        end = index + len(separator)
        pieces.append((text[cursor:end], offset + cursor, offset + end))
        cursor = end
    if cursor < len(text):
        pieces.append((text[cursor:], offset + cursor, offset + len(text)))
    return pieces


def split_recursive(text, *, chunk_size_tokens, separators=_RECURSIVE_SEPARATORS, offset=0):
    """Split text into boundary-aligned pieces of at most ``chunk_size_tokens``.

    Returns ``(piece, start_offset, end_offset)`` triples so callers can map a
    piece back to the line numbers it came from.
    """
    value = str(text or "")
    if not value:
        return []
    if estimate_tokens(value) <= chunk_size_tokens:
        return [(value, offset, offset + len(value))]
    for index, separator in enumerate(separators):
        if separator == "":
            break
        if separator not in value:
            continue
        pieces = []
        for part, part_start, part_end in _split_keeping_separator(value, separator, offset):
            pieces.extend(
                split_recursive(
                    part,
                    chunk_size_tokens=chunk_size_tokens,
                    separators=separators[index + 1 :],
                    offset=part_start,
                )
                or [(part, part_start, part_end)]
            )
        return pieces
    pieces = []
    cursor = offset
    for piece in _hard_split(value, chunk_size_tokens=chunk_size_tokens):
        pieces.append((piece, cursor, cursor + len(piece)))
        cursor += len(piece)
    return pieces


def pack_pieces(pieces, *, chunk_size_tokens, overlap_ratio):
    """Merge adjacent pieces into chunks up to the budget, then add overlap.

    Overlap is a ratio of the budget and is taken at piece granularity from the
    tail of the previous chunk, so the shared text starts on a boundary instead
    of cutting a sentence or a line in half.
    """
    units = [piece for piece in pieces if piece[0] and piece[0].strip()]
    if not units:
        return []
    budget = max(1, int(chunk_size_tokens))
    # Keep the overlap strictly below the budget: an overlap equal to or larger
    # than it would make consecutive chunks barely advance.
    overlap_tokens = min(
        budget - 1,
        max(0, int(round(budget * float(overlap_ratio or 0.0)))),
    )
    chunks = []
    current = []
    current_tokens = 0
    for unit in units:
        unit_tokens = estimate_tokens(unit[0])
        if current and current_tokens + unit_tokens > budget:
            chunks.append(current)
            tail = []
            tail_tokens = 0
            for previous in reversed(current):
                token_count = estimate_tokens(previous[0])
                if tail_tokens + token_count > overlap_tokens:
                    break
                tail.insert(0, previous)
                tail_tokens += token_count
            current = tail
            current_tokens = tail_tokens
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        chunks.append(current)
    packed = []
    for group in chunks:
        text = "".join(unit[0] for unit in group).strip()
        if not text:
            continue
        packed.append((text, group[0][1], group[-1][2]))
    return packed


def _stable_id(*parts):
    payload = "\x1f".join(str(part or "") for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _summary(record):
    value = str(
        record.get("summary")
        or record.get("description")
        or record.get("title")
        or ""
    ).strip()
    return re.sub(r"\s+", " ", value)[:MAX_PARENT_SUMMARY_CHARS]


def split_markdown_sections(body, title=""):
    """Return stable heading sections without changing the Markdown source."""
    lines = str(body or "").splitlines()
    sections = []
    stack = []
    current_heading = str(title or "Overview").strip() or "Overview"
    current_path = [current_heading]
    current_start = 1
    current_body = []

    def flush(end_line):
        # Section bodies are kept whole: the recursive splitter and the
        # line-window fallback both operate on this text, so truncating it here
        # would drop a long section's tail before it is ever chunked.
        text = clean_text("\n".join(current_body), limit=MAX_SECTION_CHARS)
        # Do not manufacture an empty preamble parent when a document starts
        # with a heading.  A genuinely heading-less document still gets one
        # Overview child so it remains searchable.
        if text or not sections and not any(
            _HEADING_RE.match(line.strip()) for line in lines
        ):
            sections.append(
                {
                    "heading": current_heading,
                    "heading_path": " / ".join(current_path),
                    "start_line": current_start,
                    "end_line": max(current_start, end_line),
                    "body": text,
                }
            )

    for line_number, line in enumerate(lines, 1):
        match = _HEADING_RE.match(line.strip())
        if not match:
            current_body.append(line)
            continue
        flush(line_number - 1)
        level = len(match.group("marks"))
        heading = match.group("title").strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        current_heading = heading
        current_path = [item[1] for item in stack]
        current_start = line_number
        current_body = []
    flush(len(lines))
    return sections


@dataclass(frozen=True)
class ChunkSet:
    """Result of chunking one corpus.

    ``children`` are the searchable units (recursively split, ~300 tokens).
    ``sections`` are the structural units they came from -- a function, or a
    heading section -- and exist only so retrieval can hand generation the whole
    section instead of the matched fragment. Sections are stored but never
    indexed for search: the point of small-to-big is that only the focused pieces
    are matched.
    """

    children: tuple
    sections: tuple


def _wiki_identity(record, *, parent_id, workspace_id, path):
    """Fields shared by a wiki page's section rows and its child rows."""
    record_id = str(record.get("id", "")).strip()
    return {
        "parent_id": parent_id,
        "source_id": record_id,
        "source_type": "wiki",
        "title": str(record.get("title", "")),
        "summary": _summary(record),
        "path": path,
        "workspace_id": str(record.get("workspace_fingerprint") or workspace_id),
        # Recompute from the parsed source instead of trusting the front-matter
        # hash. Manual Markdown edits intentionally invalidate the derived
        # sparse and dense indexes.
        "source_hash": _stable_id(
            record_id,
            record.get("title", ""),
            record.get("description", ""),
            record.get("summary", ""),
            record.get("body", ""),
        ),
        "revision": str(record.get("version", "")),
        "status": str(record.get("status", "candidate")),
        "scope": str(record.get("scope", "workspace")),
        "supersedes": str(record.get("supersedes", "")),
        "sensitivity": (
            "sensitive"
            if "secret_shaped" in record.get("quality_reasons", [])
            else "normal"
        ),
        "injection_flag": "prompt_injection" in record.get("quality_reasons", []),
    }


def chunk_wiki_records(
    records,
    *,
    workspace_id="",
    child_chunk_tokens=DEFAULT_CHILD_CHUNK_TOKENS,
    child_chunk_overlap=DEFAULT_CHILD_CHUNK_OVERLAP,
):
    """Chunk wiki records into searchable children plus their whole sections.

    The parent is the page and the section is the heading boundary, but a section
    that exceeds the child budget is split recursively inside itself, so a long
    section yields several boundary-aligned children instead of being truncated.
    Every child records the section it came from so retrieval can expand back to
    the whole section for generation.
    """
    chunks = []
    sections = []
    for record in records:
        record_id = str(record.get("id", "")).strip()
        if not record_id:
            continue
        parent_id = f"wiki:{record_id}"
        path = (record.get("source_paths") or [f"knowledge/wiki/{record_id}.md"])[0]
        common = _wiki_identity(
            record, parent_id=parent_id, workspace_id=workspace_id, path=path
        )
        for ordinal, section in enumerate(
            split_markdown_sections(record.get("body", ""), record.get("title", ""))
        ):
            heading_path = section.get("heading_path", "")
            body = section.get("body", "")
            section_id = f"{parent_id}#{ordinal}"
            section_start = int(section.get("start_line", 0))
            section_end = int(section.get("end_line", 0))
            sections.append(
                RetrievalChunk(
                    chunk_id=section_id,
                    section_id=section_id,
                    text=clean_text(body, limit=MAX_SECTION_CHARS),
                    heading_path=heading_path,
                    start_line=section_start,
                    end_line=section_end,
                    metadata={
                        "record_id": record_id,
                        "role": "section",
                        "ordinal": ordinal,
                    },
                    **common,
                )
            )
            packed = pack_pieces(
                split_recursive(body, chunk_size_tokens=child_chunk_tokens),
                chunk_size_tokens=child_chunk_tokens,
                overlap_ratio=child_chunk_overlap,
            )
            for piece_index, (text, piece_start, piece_end) in enumerate(packed):
                chunks.append(
                    RetrievalChunk(
                        chunk_id=_stable_id(
                            parent_id, heading_path, ordinal, piece_index
                        ),
                        section_id=section_id,
                        text=clean_text(text),
                        heading_path=heading_path,
                        start_line=section_start + body[:piece_start].count("\n"),
                        end_line=section_start + body[:piece_end].count("\n"),
                        metadata={
                            "record_id": record_id,
                            "conflict_key": (
                                f"wiki:{path}:"
                                f"{heading_path or record.get('title', '')}:{piece_index}"
                            ),
                            "tags": list(record.get("tags", [])),
                            "source_paths": list(record.get("source_paths", [])),
                            "updated_at": str(record.get("updated_at", "")),
                            "ordinal": ordinal,
                            "piece": piece_index,
                        },
                        **common,
                    )
                )
    return ChunkSet(children=tuple(chunks), sections=tuple(sections))


def chunk_code_repository(
    root,
    *,
    workspace_id="",
    cache_dir=None,
    child_chunk_tokens=DEFAULT_CHILD_CHUNK_TOKENS,
    child_chunk_overlap=DEFAULT_CHILD_CHUNK_OVERLAP,
):
    """Chunk a repository into searchable children plus their whole definitions.

    The parent is the file, the section is the definition, and a definition whose
    body exceeds the child budget is split recursively inside itself. Sections
    hold the complete definition body, which is what small-to-big hands to
    generation when a child is matched. Every child keeps the symbol name, so the
    sparse index's symbol weighting still applies to all of them.
    """
    from ..repomap import graph as graphlib

    root = Path(root).resolve()
    tags_by_file = graphlib.load_tags(root, cache_dir=cache_dir)
    chunks = []
    sections = []
    for relative, file_tags in sorted(tags_by_file.items()):
        path = root / relative
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        definitions = sorted(
            {(str(name), max(1, int(line))) for name, line in file_tags.definitions},
            key=lambda item: (item[1], item[0]),
        )
        parent_id = f"code:{relative}"
        parent_summary = _code_parent_summary(file_tags)
        if not definitions:
            definitions = [(Path(relative).stem, 1)]
        for index, (name, start) in enumerate(definitions):
            next_start = definitions[index + 1][1] if index + 1 < len(definitions) else len(lines) + 1
            end = min(len(lines), max(start, next_start - 1))
            raw_body = "\n".join(lines[start - 1 : end])
            if not raw_body.strip():
                continue
            section_id = f"{parent_id}#{name}@{start}"
            common = {
                "parent_id": parent_id,
                "source_id": relative,
                "source_type": "code",
                "title": relative,
                "summary": parent_summary,
                "path": relative,
                "workspace_id": workspace_id,
                "source_hash": file_tags.sha256,
                "revision": file_tags.sha256[:12],
                "status": "active",
                "scope": "workspace",
            }
            sections.append(
                RetrievalChunk(
                    chunk_id=section_id,
                    section_id=section_id,
                    text=clean_text(raw_body, limit=MAX_SECTION_CHARS),
                    symbol=name,
                    start_line=start,
                    end_line=end,
                    metadata={
                        "role": "section",
                        "conflict_key": f"code:{relative}:{name}@section",
                        "language": file_tags.language,
                    },
                    **common,
                )
            )
            packed = pack_pieces(
                split_recursive(raw_body, chunk_size_tokens=child_chunk_tokens),
                chunk_size_tokens=child_chunk_tokens,
                overlap_ratio=child_chunk_overlap,
            )
            for piece_index, (text, piece_start, piece_end) in enumerate(packed):
                body = clean_text(text)
                if not body:
                    continue
                chunks.append(
                    RetrievalChunk(
                        chunk_id=_stable_id(
                            parent_id, name, start, piece_index, file_tags.sha256
                        ),
                        section_id=section_id,
                        text=body,
                        symbol=name,
                        start_line=start + raw_body[:piece_start].count("\n"),
                        end_line=start + raw_body[:piece_end].count("\n"),
                        metadata={
                            # One definition can yield several children, so the
                            # conflict key needs the piece index or conflict
                            # resolution would collapse them into one.
                            "conflict_key": f"code:{relative}:{name}#{piece_index}",
                            "language": file_tags.language,
                            "definitions": [item[0] for item in definitions[:32]],
                            "references": [item[0] for item in file_tags.references[:64]],
                            "piece": piece_index,
                        },
                        **common,
                    )
                )
    return ChunkSet(children=tuple(chunks), sections=tuple(sections))


def _code_parent_summary(file_tags):
    definitions = ", ".join(name for name, _line in file_tags.definitions[:20])
    references = ", ".join(name for name, _line in file_tags.references[:20])
    value = f"language={file_tags.language}; defines={definitions}; references={references}"
    return value[:MAX_PARENT_SUMMARY_CHARS]
