"""Configuration for the local-first hybrid retrieval stack."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    enabled: bool = True
    # Qwen over the OpenAI-compatible endpoint is the default because it is what
    # this project runs against. `hashing` (dependency-free lexical pseudo-vectors)
    # stays available as the explicit offline fallback, and `off` disables dense
    # retrieval entirely. A missing base_url or api_key must degrade to sparse-only
    # rather than fail on every request, which is enforced in create_embedding_provider.
    embedding_provider: str = "openai-compatible"
    embedding_model: str = "qwen3.7-text-embedding-flash"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 20
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_api_key: str = ""
    # Qwen reranking is the default for the same reason. It is also the only
    # reranker whose scores support the abstention floor below.
    reranker_provider: str = "dashscope"
    reranker_model: str = "qwen3.7-text-rerank"
    reranker_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    reranker_api_key: str = ""
    reranker_timeout: int = 30
    sparse_top_k: int = 50
    dense_top_k: int = 50
    coarse_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60
    min_score: float = 0.12
    # Evidence whose rerank score falls below this floor is treated as no
    # evidence, which is what makes abstention actually reachable. 0 disables
    # the floor. The value is bound to the reranker that produced the score:
    # it only means anything for a reranker whose score is monotonic in
    # relevance (a remote reranker, a cross-encoder), never for the synthetic
    # rank-derived score of the deterministic reranker.
    abstention_min_rerank: float = 0.0
    # Parents stay structural (file / page). Children are cut recursively inside
    # a parent, so a long function or section is split on its own boundaries
    # instead of being truncated by a character cap.
    child_chunk_tokens: int = 300
    child_chunk_overlap: float = 0.10
    # Per-parent cap on the final candidate set, so one file cannot consume the
    # whole result set. It deliberately stays low now that sections expand: matching
    # one child is enough to get the whole section, so a file never needs several
    # slots. Measured, raising it to 4 cost multi-gold recall and made the
    # forbidden-exposure metric look better only because fewer distinct sources
    # survived to the tail of the list.
    parent_max_chunks: int = 2
    # Small-to-big: retrieval matches children because they are focused, and
    # generation receives the whole structural section they came from (a function,
    # a heading section). Children of one section collapse into a single evidence
    # entry so the section is not repeated per matched fragment.
    section_expand: bool = True
    # A section larger than this degrades to a line window around the hit; if no
    # window can be built it degrades to the matched child text.
    section_max_chars: int = 2400
    section_window_lines: int = 12
    code_rag_file_threshold: int = 1000
    code_context_budget_chars: int = 2800
    evidence_budget_chars: int = 4200
    query_cache_size: int = 128


# Single source for defaults: the environment resolver reads the dataclass rather
# than repeating every literal, so the two cannot drift apart.
_DEFAULTS = RetrievalConfig()


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _env_int(name, default, minimum=1):
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name, default, minimum=0.0):
    try:
        return max(float(minimum), float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return float(default)


def retrieval_config_from_env(defaults=None) -> RetrievalConfig:
    """Resolve retrieval settings without reading or mutating project secrets."""
    defaults = dict(defaults or {})

    def value(name, default):
        return defaults.get(name, default)

    def text(env_name, field):
        return os.environ.get(
            env_name, value(field, getattr(_DEFAULTS, field))
        ).strip()

    def number(env_name, field, minimum=1):
        return _env_int(env_name, value(field, getattr(_DEFAULTS, field)), minimum)

    return RetrievalConfig(
        enabled=_env_bool("GENCODE_RETRIEVAL_ENABLED", value("enabled", _DEFAULTS.enabled)),
        embedding_provider=text("GENCODE_EMBEDDING_PROVIDER", "embedding_provider"),
        embedding_model=text("GENCODE_EMBEDDING_MODEL", "embedding_model"),
        embedding_dimensions=number(
            "GENCODE_EMBEDDING_DIMENSIONS", "embedding_dimensions", 32
        ),
        embedding_batch_size=number("GENCODE_EMBEDDING_BATCH_SIZE", "embedding_batch_size"),
        embedding_base_url=text("GENCODE_EMBEDDING_BASE_URL", "embedding_base_url"),
        embedding_api_key=text("GENCODE_EMBEDDING_API_KEY", "embedding_api_key"),
        reranker_provider=text("GENCODE_RERANKER_PROVIDER", "reranker_provider"),
        reranker_model=text("GENCODE_RERANKER_MODEL", "reranker_model"),
        reranker_base_url=text("GENCODE_RERANKER_BASE_URL", "reranker_base_url"),
        reranker_api_key=text("GENCODE_RERANKER_API_KEY", "reranker_api_key"),
        reranker_timeout=number("GENCODE_RERANKER_TIMEOUT", "reranker_timeout"),
        sparse_top_k=number("GENCODE_RETRIEVAL_SPARSE_TOP_K", "sparse_top_k"),
        dense_top_k=number("GENCODE_RETRIEVAL_DENSE_TOP_K", "dense_top_k"),
        coarse_top_k=number("GENCODE_RETRIEVAL_COARSE_TOP_K", "coarse_top_k"),
        final_top_k=number("GENCODE_RETRIEVAL_FINAL_TOP_K", "final_top_k"),
        rrf_k=number("GENCODE_RETRIEVAL_RRF_K", "rrf_k"),
        min_score=_env_float(
            "GENCODE_RETRIEVAL_MIN_DENSE_SCORE", value("min_score", _DEFAULTS.min_score)
        ),
        abstention_min_rerank=_env_float(
            "GENCODE_RETRIEVAL_ABSTENTION_MIN_RERANK",
            value("abstention_min_rerank", _DEFAULTS.abstention_min_rerank),
        ),
        child_chunk_tokens=number(
            "GENCODE_RETRIEVAL_CHILD_CHUNK_TOKENS", "child_chunk_tokens", 32
        ),
        child_chunk_overlap=_env_float(
            "GENCODE_RETRIEVAL_CHILD_CHUNK_OVERLAP",
            value("child_chunk_overlap", _DEFAULTS.child_chunk_overlap),
        ),
        parent_max_chunks=number(
            "GENCODE_RETRIEVAL_PARENT_MAX_CHUNKS", "parent_max_chunks"
        ),
        section_expand=_env_bool(
            "GENCODE_RETRIEVAL_SECTION_EXPAND", value("section_expand", _DEFAULTS.section_expand)
        ),
        section_max_chars=number(
            "GENCODE_RETRIEVAL_SECTION_MAX_CHARS", "section_max_chars", 200
        ),
        section_window_lines=number(
            "GENCODE_RETRIEVAL_SECTION_WINDOW_LINES", "section_window_lines"
        ),
        code_rag_file_threshold=number(
            "GENCODE_CODE_RAG_FILE_THRESHOLD", "code_rag_file_threshold"
        ),
        code_context_budget_chars=number(
            "GENCODE_CODE_RAG_BUDGET_CHARS", "code_context_budget_chars", 400
        ),
        evidence_budget_chars=number(
            "GENCODE_RETRIEVAL_EVIDENCE_BUDGET_CHARS", "evidence_budget_chars", 400
        ),
        query_cache_size=number(
            "GENCODE_RETRIEVAL_CACHE_SIZE", "query_cache_size", 8
        ),
    )
