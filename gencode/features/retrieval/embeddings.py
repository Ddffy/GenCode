"""Embedding provider plugins with a deterministic offline fallback."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from .query import search_tokens


class EmbeddingProvider(Protocol):
    model_id: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def normalize_vector(vector):
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        return values
    return [value / norm for value in values]


class HashingEmbeddingProvider:
    """Dependency-free dense vectorizer used for offline operation and tests.

    It is intentionally identified as a lexical hashing embedding.  Operators
    can replace it with sentence-transformers or an OpenAI-compatible endpoint
    without changing indexing or retrieval policy.
    """

    def __init__(self, dimensions=384, model_id="hashing-embedding-v1"):
        self.dimensions = max(32, int(dimensions))
        self.model_id = str(model_id)

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        vector = [0.0] * self.dimensions
        tokens = search_tokens(text)
        features = list(tokens)
        features.extend(
            f"{tokens[index]}::{tokens[index + 1]}"
            for index in range(len(tokens) - 1)
        )
        # Identifier fragments improve path/symbol matching while remaining
        # language independent.
        for identifier in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(text)):
            features.extend(part.casefold() for part in identifier.split("_") if part)
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dimensions
            vector[index] += -1.0 if value & (1 << 63) else 1.0
        return normalize_vector(vector)


class SentenceTransformerEmbeddingProvider:
    """Optional local semantic embedding backed by sentence-transformers."""

    def __init__(self, model="BAAI/bge-small-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "sentence-transformers is not installed; install gencode[rag]"
            ) from exc
        self._model = SentenceTransformer(str(model))
        self.model_id = f"sentence-transformers:{model}"
        dimension = self._model.get_sentence_embedding_dimension()
        self.dimensions = int(dimension or 0)

    def embed_documents(self, texts):
        values = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [[float(value) for value in row] for row in values]

    def embed_query(self, text):
        return self.embed_documents([str(text)])[0]


class OpenAICompatibleEmbeddingProvider:
    """Small standard-library adapter for ``POST /embeddings`` endpoints."""

    def __init__(
        self,
        *,
        model,
        base_url,
        api_key="",
        dimensions=0,
        timeout=30,
    ):
        if not base_url:
            raise ValueError("embedding base_url is required")
        base = str(base_url).rstrip("/")
        self.endpoint = base if base.endswith("/embeddings") else base + "/embeddings"
        self.api_key = str(api_key or "")
        self.model_id = f"openai-compatible:{model}"
        self.model = str(model)
        self.dimensions = int(dimensions or 0)
        self.timeout = int(timeout)

    def embed_documents(self, texts):
        payload = {"model": self.model, "input": [str(text) for text in texts]}
        if self.dimensions > 0:
            payload["dimensions"] = self.dimensions
        headers = {"Content-Type": "application/json", "User-Agent": "gencode/0.3"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        rows = sorted(result.get("data", []), key=lambda row: int(row.get("index", 0)))
        vectors = [normalize_vector(row.get("embedding", [])) for row in rows]
        if len(vectors) != len(texts) or not all(vectors):
            raise RuntimeError("embedding provider returned an invalid response")
        if not self.dimensions:
            self.dimensions = len(vectors[0])
        return vectors

    def embed_query(self, text):
        return self.embed_documents([str(text)])[0]


class CachedEmbeddingProvider:
    """Persistent content-hash cache shared by indexing and query calls."""

    def __init__(self, provider, cache_path, *, schema_version="chunk-v1"):
        self.provider = provider
        self.model_id = str(provider.model_id)
        self.dimensions = int(provider.dimensions)
        self.cache_path = Path(cache_path)
        self.schema_version = str(schema_version)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def embed_documents(self, texts):
        values = [str(text) for text in texts]
        keys = [self._key(text) for text in values]
        cached = self._load(keys)
        missing_positions = [index for index, key in enumerate(keys) if key not in cached]
        if missing_positions:
            generated = self.provider.embed_documents(
                [values[index] for index in missing_positions]
            )
            self._save(
                [(keys[index], vector) for index, vector in zip(missing_positions, generated)]
            )
            cached.update(
                {key: vector for key, vector in zip(
                    [keys[index] for index in missing_positions], generated
                )}
            )
        return [cached[key] for key in keys]

    def embed_query(self, text):
        return self.embed_documents([str(text)])[0]

    def _key(self, text):
        payload = f"{self.model_id}\x1f{self.schema_version}\x1f{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _ensure_schema(self):
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS embedding_cache (
                cache_key TEXT PRIMARY KEY, model_id TEXT NOT NULL,
                dimensions INTEGER NOT NULL, vector BLOB NOT NULL)"""
            )

    def _load(self, keys):
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT cache_key, dimensions, vector FROM embedding_cache WHERE cache_key IN ({placeholders})",
                list(keys),
            ).fetchall()
        return {key: _unpack_vector(blob, dimensions) for key, dimensions, blob in rows}

    def _save(self, rows):
        with self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO embedding_cache VALUES (?, ?, ?, ?)",
                [
                    (key, self.model_id, len(vector), _pack_vector(vector))
                    for key, vector in rows
                ],
            )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.cache_path, timeout=30)
        try:
            yield connection
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()


def _pack_vector(vector):
    values = [float(value) for value in vector]
    return struct.pack(f"<{len(values)}f", *values)


def _unpack_vector(blob, dimensions):
    return list(struct.unpack(f"<{int(dimensions)}f", bytes(blob)))


def create_embedding_provider(config, *, cache_path=None):
    name = str(config.embedding_provider or "hashing").casefold()
    if name in {"sentence-transformers", "sentence_transformers", "local"}:
        model = config.embedding_model
        if not model or str(model).casefold() == "hashing-embedding-v1":
            model = "BAAI/bge-small-en-v1.5"
        provider = SentenceTransformerEmbeddingProvider(model)
    elif name in {"openai", "openai-compatible", "remote"}:
        # Fail here rather than on the first request. A remote provider without a
        # key would otherwise look configured and then 401 on every turn; raising
        # lets the caller degrade to sparse-only and record why.
        if not str(config.embedding_base_url or "").strip():
            raise ValueError(
                "embedding_provider=openai-compatible requires embedding_base_url"
            )
        if not str(config.embedding_api_key or "").strip():
            raise ValueError(
                "embedding_provider=openai-compatible requires embedding_api_key; "
                "set GENCODE_EMBEDDING_API_KEY, or select the offline 'hashing' "
                "provider explicitly"
            )
        provider = OpenAICompatibleEmbeddingProvider(
            model=config.embedding_model,
            base_url=config.embedding_base_url,
            api_key=config.embedding_api_key,
            dimensions=config.embedding_dimensions,
        )
    elif name in {"off", "none", "disabled"}:
        return None
    else:
        provider = HashingEmbeddingProvider(
            config.embedding_dimensions, config.embedding_model
        )
    if cache_path:
        return CachedEmbeddingProvider(provider, cache_path)
    return provider
