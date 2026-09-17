"""Pluggable retrieval for code and durable knowledge.

The package deliberately separates index-time work from query-time policy:
documents are cleaned, chunked and indexed once; a registry then chooses one
or more retrievers for each query.  The built-in stack is local-first and has
no mandatory vector dependency.
"""

from .chunking import chunk_code_repository, chunk_wiki_records
from .config import RetrievalConfig, retrieval_config_from_env
from .embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    create_embedding_provider,
)
from .pipeline import HybridRetrievalService
from .rerank import DashScopeReranker
from .types import RetrievalChunk, RetrievalHit, RetrievalResponse

__all__ = [
    "DashScopeReranker",
    "EmbeddingProvider",
    "HashingEmbeddingProvider",
    "HybridRetrievalService",
    "OpenAICompatibleEmbeddingProvider",
    "RetrievalChunk",
    "RetrievalConfig",
    "RetrievalHit",
    "RetrievalResponse",
    "SentenceTransformerEmbeddingProvider",
    "chunk_code_repository",
    "chunk_wiki_records",
    "create_embedding_provider",
    "retrieval_config_from_env",
]
