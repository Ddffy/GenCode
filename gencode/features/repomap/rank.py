"""Lexical and graph ranking for the repository map.

The ranking pipeline deliberately keeps its dependencies small.  BM25 gives
the query a precise lexical signal, while personalized PageRank supplies the
cross-file dependency signal.  Both vectors are normalized before fusion so
the configured 0.8/0.2 weights have the meaning they claim to have.
"""

from __future__ import annotations

import math
import re

DAMPING = 0.85
MAX_ITERATIONS = 100
TOLERANCE = 1e-9
BM25_K1 = 1.2
BM25_B = 0.75
LEXICAL_WEIGHT = 0.8
GRAPH_WEIGHT = 0.2
MAX_SEED_FILES = 8

# Natural-language glue words and generic repository words should not seed a
# large fraction of the graph.  We still keep them in the lexical document
# representation when they occur inside a distinctive symbol.
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "locate",
    "module",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "where",
    "which",
    "with",
    "what",
    "why",
    "does",
    "implement",
    "implements",
    "find",
    "show",
    "file",
    "code",
    "function",
    "class",
    "method",
    "component",
}
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def personalized_pagerank(nodes, edge_weights, seeds=None):
    """Return {node: score}. Pure function, deterministic ordering."""
    nodes = sorted(set(nodes))
    if not nodes:
        return {}
    node_index = {node: index for index, node in enumerate(nodes)}
    count = len(nodes)

    outgoing = [[] for _ in range(count)]
    out_degree = [0.0] * count
    for (src, dst), weight in edge_weights.items():
        src_index = node_index.get(src)
        dst_index = node_index.get(dst)
        if src_index is None or dst_index is None or weight <= 0:
            continue
        outgoing[src_index].append((dst_index, weight))
        out_degree[src_index] += weight

    seed_total = sum(float(weight) for weight in (seeds or {}).values())
    if seed_total > 0:
        personal = [0.0] * count
        for node, weight in (seeds or {}).items():
            index = node_index.get(node)
            if index is not None:
                personal[index] = float(weight) / seed_total
    else:
        personal = [1.0 / count] * count

    rank = [1.0 / count] * count
    for _ in range(MAX_ITERATIONS):
        dangling_sum = sum(
            rank[index] for index in range(count) if out_degree[index] <= 0
        )
        next_rank = [0.0] * count
        for index in range(count):
            base = (1.0 - DAMPING) * personal[index] + DAMPING * dangling_sum / count
            next_rank[index] += base
        for src_index in range(count):
            if out_degree[src_index] <= 0:
                continue
            share = DAMPING * rank[src_index] / out_degree[src_index]
            for dst_index, weight in outgoing[src_index]:
                next_rank[dst_index] += share * weight
        error = sum(abs(next_rank[index] - rank[index]) for index in range(count))
        rank = next_rank
        if error < TOLERANCE:
            break

    return {node: rank[node_index[node]] for node in nodes}


def normalize_scores(scores, nodes=None):
    """Min-max normalize a score vector over *nodes* into [0, 1]."""
    nodes = sorted(set(nodes or scores))
    if not nodes:
        return {}
    values = [float(scores.get(node, 0.0)) for node in nodes]
    low, high = min(values), max(values)
    if high <= low:
        return {node: 0.0 for node in nodes}
    span = high - low
    return {node: (float(scores.get(node, 0.0)) - low) / span for node in nodes}


def _tokens(text):
    """Tokenize natural language and code identifiers deterministically."""
    result = []
    for raw in _TOKEN_RE.findall(str(text or "")):
        result.append(raw.lower())
        for part in _CAMEL_RE.sub(" ", raw).replace("_", " ").split():
            part = part.lower()
            if part and part != raw.lower():
                result.append(part)
    return result


def _query_terms(query):
    return [
        term for term in _tokens(query) if term not in _STOPWORDS and len(term) >= 2
    ]


def bm25_scores(query, tags_by_file):
    """Return BM25 scores over file paths and extracted symbol tags.

    The index is rebuilt from the already cached Tree-sitter tags, so this
    adds no vector database and remains consistent with the Repo Map cache.
    """
    paths = sorted(tags_by_file)
    if not paths:
        return {}
    documents = {}
    for path in paths:
        file_tags = tags_by_file[path]
        documents[path] = _tokens(
            path
            + " "
            + " ".join(name for name, _line in file_tags.definitions)
            + " "
            + " ".join(name for name, _line in file_tags.references)
        )
    query_terms = _query_terms(query)
    if not query_terms:
        return {path: 0.0 for path in paths}
    document_frequency = {}
    for tokens in documents.values():
        for term in set(tokens):
            document_frequency[term] = document_frequency.get(term, 0) + 1
    average_length = sum(len(tokens) for tokens in documents.values()) / len(paths)
    average_length = max(average_length, 1.0)
    scores = {}
    for path, tokens in documents.items():
        frequencies = {}
        for term in tokens:
            frequencies[term] = frequencies.get(term, 0) + 1
        length = len(tokens)
        score = 0.0
        for term in query_terms:
            df = document_frequency.get(term, 0)
            if not df:
                continue
            idf = math.log(1.0 + (len(paths) - df + 0.5) / (df + 0.5))
            tf = frequencies.get(term, 0)
            if not tf:
                continue
            denominator = tf + BM25_K1 * (
                1.0 - BM25_B + BM25_B * length / average_length
            )
            score += idf * (tf * (BM25_K1 + 1.0)) / denominator
        scores[path] = score
    return scores


def build_seeds(query, tags_by_file, recent_paths=(), ident_limit=64):
    """Seed weights from the current request and working memory.

    - identifiers mentioned in the query that appear as tags in a file: 3
    - files touched recently per working memory: 2
    """
    seeds = {}
    query_idents = _extract_identifiers(query, ident_limit)
    if query_idents:
        # A term that appears in most files is not useful as a teleport seed.
        # Keep the absolute cap generous for tiny test repositories while
        # preventing words such as ``runtime`` from seeding half a monorepo.
        file_count = max(len(tags_by_file), 1)
        max_term_files = max(8, int(file_count * 0.20))
        term_files = {term: set() for term in query_idents}
        for file_tags in tags_by_file.values():
            names = {name for name, _line in file_tags.definitions}
            names.update(name for name, _line in file_tags.references)
            for term in query_idents:
                if term in names:
                    term_files[term].add(file_tags.path)
        useful_idents = {
            term
            for term in query_idents
            if term.lower() not in _STOPWORDS
            and len(term_files[term]) <= max_term_files
        }
        matched_paths = set()
        for file_tags in tags_by_file.values():
            names = {name for name, _line in file_tags.definitions}
            names.update(name for name, _line in file_tags.references)
            if names & useful_idents:
                matched_paths.add(file_tags.path)
        # Keep the teleport vector sparse; broad vectors are indistinguishable
        # from ordinary PageRank after normalization.
        ranked = sorted(
            matched_paths,
            key=lambda path: (
                -sum(
                    1.0 / max(len(term_files[term]), 1)
                    for term in useful_idents
                    if path in term_files[term]
                ),
                path,
            ),
        )[:MAX_SEED_FILES]
        for path in ranked:
            seeds[path] = 3.0
    for path in recent_paths or ():
        path = str(path).replace("\\", "/")
        if path in tags_by_file:
            seeds[path] = seeds.get(path, 0.0) + 2.0
    return seeds


def build_legacy_seeds(query, tags_by_file, recent_paths=(), ident_limit=64):
    """Pre-fix seed construction retained for controlled ablation reports."""
    seeds = {}
    query_idents = _extract_identifiers_legacy(query, ident_limit)
    matched_paths = set()
    for file_tags in tags_by_file.values():
        names = {name for name, _line in file_tags.definitions}
        names.update(name for name, _line in file_tags.references)
        if names & query_idents:
            matched_paths.add(file_tags.path)
    for path in sorted(matched_paths):
        seeds[path] = seeds.get(path, 0.0) + 3.0
    for path in recent_paths or ():
        path = str(path).replace("\\", "/")
        if path in tags_by_file:
            seeds[path] = seeds.get(path, 0.0) + 2.0
    return seeds


def hybrid_scores(query, nodes, edge_weights, tags_by_file, recent_paths=()):
    """Fuse normalized BM25 (0.8) with normalized personalized PageRank (0.2)."""
    lexical = bm25_scores(query, tags_by_file)
    seeds = build_seeds(query, tags_by_file, recent_paths=recent_paths)
    graph = personalized_pagerank(nodes, edge_weights, seeds)
    lexical_norm = normalize_scores(lexical, nodes)
    graph_norm = normalize_scores(graph, nodes)
    return {
        node: LEXICAL_WEIGHT * lexical_norm.get(node, 0.0)
        + GRAPH_WEIGHT * graph_norm.get(node, 0.0)
        for node in nodes
    }, seeds


def _extract_identifiers(text, limit):
    idents = []
    seen = set()
    for token in str(text or "").replace("::", ".").replace("/", ".").split():
        piece = token.strip(".,;:()[]{}'\"`!?")
        for candidate in piece.split("."):
            candidate = candidate.strip()
            if (
                len(candidate) < 3
                or not candidate.replace("_", "").isalnum()
                or candidate[0].isdigit()
            ):
                continue
            if candidate not in seen:
                seen.add(candidate)
                idents.append(candidate)
                if len(idents) >= limit:
                    return set(idents)
    return set(idents)


def _extract_identifiers_legacy(text, limit):
    """The original broad extractor, used only as an evaluation baseline."""
    idents = []
    seen = set()
    for token in str(text or "").replace("::", ".").replace("/", ".").split():
        piece = token.strip(".,;:()[]{}'\"`!?")
        for candidate in piece.split("."):
            candidate = candidate.strip()
            if (
                len(candidate) < 3
                or not candidate.replace("_", "").isalnum()
                or candidate[0].isdigit()
            ):
                continue
            if candidate not in seen:
                seen.add(candidate)
                idents.append(candidate)
                if len(idents) >= limit:
                    return set(idents)
    return set(idents)
