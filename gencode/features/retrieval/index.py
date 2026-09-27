"""SQLite sparse and dense index used by built-in retrieval plugins."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
import threading
from contextlib import contextmanager
from pathlib import Path

try:
    import hnswlib
except ImportError:  # The exact search remains available without the optional ANN package.
    hnswlib = None

from .query import search_tokens
from .types import RetrievalChunk, RetrievalHit


class SQLiteRetrievalIndex:
    """Rebuildable FTS5 + packed vectors with an HNSW search index."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._hnsw_cache = {}
        self._hnsw_error = ""
        self._last_dense_backend = "not_queried"
        self.fts_enabled = True
        self._ensure_schema()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            yield connection
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    def _ensure_schema(self):
        with self._lock, self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS retrieval_chunks (
                chunk_id TEXT PRIMARY KEY, parent_id TEXT NOT NULL,
                source_id TEXT NOT NULL, source_type TEXT NOT NULL,
                title TEXT NOT NULL, text TEXT NOT NULL, summary TEXT NOT NULL,
                path TEXT NOT NULL, symbol TEXT NOT NULL, heading_path TEXT NOT NULL,
                start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
                workspace_id TEXT NOT NULL, source_hash TEXT NOT NULL,
                revision TEXT NOT NULL, status TEXT NOT NULL, scope TEXT NOT NULL,
                supersedes TEXT NOT NULL, sensitivity TEXT NOT NULL,
                injection_flag INTEGER NOT NULL, metadata_json TEXT NOT NULL,
                chunk_json TEXT NOT NULL)"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS retrieval_vectors (
                chunk_id TEXT NOT NULL, model_id TEXT NOT NULL,
                dimensions INTEGER NOT NULL, vector BLOB NOT NULL,
                PRIMARY KEY(chunk_id, model_id))"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS retrieval_manifests (
                namespace TEXT NOT NULL, workspace_id TEXT NOT NULL,
                signature TEXT NOT NULL, model_id TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                PRIMARY KEY(namespace, workspace_id))"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS retrieval_index_generation (
                id INTEGER PRIMARY KEY CHECK (id = 1), generation INTEGER NOT NULL)"""
            )
            connection.execute(
                "INSERT OR IGNORE INTO retrieval_index_generation VALUES (1, 0)"
            )
            # Structural sections (a function, a heading section) exist so a
            # matched child can be expanded to the whole section before it is
            # handed to generation. They are stored but deliberately not indexed:
            # no FTS row, no vector, so retrieval only ever matches children.
            connection.execute(
                """CREATE TABLE IF NOT EXISTS retrieval_sections (
                chunk_id TEXT NOT NULL, namespace TEXT NOT NULL,
                workspace_id TEXT NOT NULL, payload_json TEXT NOT NULL,
                PRIMARY KEY(chunk_id, namespace))"""
            )
            try:
                connection.execute(
                    """CREATE VIRTUAL TABLE IF NOT EXISTS retrieval_fts USING fts5(
                    chunk_id UNINDEXED, title, path, symbol, heading_path,
                    summary, text, search_text)"""
                )
                self.fts_enabled = True
            except sqlite3.OperationalError:
                self.fts_enabled = False

    def sync(
        self,
        chunks,
        *,
        namespace,
        workspace_id,
        embedder=None,
        embedding_batch_size=64,
        force=False,
        sections=(),
    ):
        rows = list(chunks)
        model_id = str(getattr(embedder, "model_id", "") or "disabled")
        signature = self._signature(rows, model_id)
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT signature, model_id FROM retrieval_manifests WHERE namespace=? AND workspace_id=?",
                (str(namespace), str(workspace_id)),
            ).fetchone()
            if not force and existing == (signature, model_id):
                return {"changed": False, "chunks": len(rows), "signature": signature}

        eligible = [
            chunk
            for chunk in rows
            if chunk.status == "active"
            and chunk.sensitivity == "normal"
            and not chunk.injection_flag
            and (chunk.scope == "global" or chunk.workspace_id == str(workspace_id))
        ]
        eligible_ids = {chunk.chunk_id for chunk in eligible}
        # Gate sections by the same rules as their children, so quarantined or
        # out-of-scope material cannot leak back in through an expansion.
        eligible_section_ids = {chunk.section_id for chunk in eligible if chunk.section_id}
        section_rows = [
            section
            for section in list(sections)
            if section.chunk_id in eligible_section_ids
        ]
        vectors = {}
        if embedder is not None and eligible:
            batch_size = max(1, int(embedding_batch_size))
            for offset in range(0, len(eligible), batch_size):
                batch = eligible[offset : offset + batch_size]
                embedded = embedder.embed_documents(
                    [chunk.embedding_text() for chunk in batch]
                )
                if len(embedded) != len(batch):
                    raise RuntimeError(
                        "embedding provider returned a row count different from the chunk batch"
                    )
                vectors.update(
                    {chunk.chunk_id: vector for chunk, vector in zip(batch, embedded)}
                )

        with self._lock, self._connect() as connection:
            old_ids = [
                row[0]
                for row in connection.execute(
                    "SELECT chunk_id FROM retrieval_chunks WHERE source_type=? AND workspace_id=?",
                    (str(namespace), str(workspace_id)),
                ).fetchall()
            ]
            if old_ids:
                placeholders = ",".join("?" for _ in old_ids)
                connection.execute(
                    f"DELETE FROM retrieval_vectors WHERE chunk_id IN ({placeholders})",
                    old_ids,
                )
                if self.fts_enabled:
                    connection.execute(
                        f"DELETE FROM retrieval_fts WHERE chunk_id IN ({placeholders})",
                        old_ids,
                    )
            connection.execute(
                "DELETE FROM retrieval_chunks WHERE source_type=? AND workspace_id=?",
                (str(namespace), str(workspace_id)),
            )
            connection.execute(
                "DELETE FROM retrieval_sections WHERE namespace=? AND workspace_id=?",
                (str(namespace), str(workspace_id)),
            )
            for chunk in rows:
                self._insert_chunk(connection, chunk)
                if chunk.chunk_id in eligible_ids and self.fts_enabled:
                    self._insert_fts(connection, chunk)
                vector = vectors.get(chunk.chunk_id)
                if vector is not None:
                    connection.execute(
                        "INSERT OR REPLACE INTO retrieval_vectors VALUES (?, ?, ?, ?)",
                        (
                            chunk.chunk_id,
                            model_id,
                            len(vector),
                            _pack_vector(vector),
                        ),
                    )
            connection.execute(
                "INSERT OR REPLACE INTO retrieval_manifests VALUES (?, ?, ?, ?, ?)",
                (str(namespace), str(workspace_id), signature, model_id, len(rows)),
            )
            for section in section_rows:
                connection.execute(
                    "INSERT OR REPLACE INTO retrieval_sections VALUES (?, ?, ?, ?)",
                    (
                        section.chunk_id,
                        str(namespace),
                        str(workspace_id),
                        json.dumps(
                            section.to_dict(), ensure_ascii=False, sort_keys=True
                        ),
                    ),
                )
            connection.execute(
                "UPDATE retrieval_index_generation SET generation = generation + 1 WHERE id = 1"
            )
        with self._lock:
            self._hnsw_cache.clear()
            if hnswlib is not None and vectors:
                # SQLite remains the source of truth; the graph is rebuilt after
                # a changed sync and once after a process restart.
                try:
                    self._hnsw_index(model_id, len(next(iter(vectors.values()))))
                    self._hnsw_error = ""
                except Exception as exc:  # noqa: BLE001 - derived index failure is recoverable
                    self._hnsw_error = str(exc)
        return {
            "changed": True,
            "chunks": len(rows),
            "sections": len(section_rows),
            "signature": signature,
        }

    def sections(self, section_ids):
        """Full structural sections for the given ids, used for small-to-big.

        Sections are never returned by search; they are only fetched after a
        child has been selected, which is what keeps retrieval at child
        granularity and generation at section granularity.
        """
        ids = [str(value) for value in section_ids if str(value)]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"SELECT chunk_id, payload_json FROM retrieval_sections "
                f"WHERE chunk_id IN ({placeholders})",
                ids,
            ).fetchall()
        return {
            chunk_id: RetrievalChunk.from_dict(json.loads(payload))
            for chunk_id, payload in rows
        }

    def sparse_search(self, query, *, top_k, workspace_id, source_types=()):
        terms = search_tokens(query)
        if not terms:
            return []
        if not self.fts_enabled:
            return self._lexical_search(
                terms, top_k=top_k, workspace_id=workspace_id, source_types=source_types
            )
        match_query = " OR ".join(
            f'"{term.replace(chr(34), chr(34) * 2)}"*' for term in terms
        )
        type_sql, params = _source_type_sql(source_types)
        sql = f"""SELECT c.chunk_json,
            bm25(retrieval_fts, 0.0, 6.0, 4.0, 7.0, 5.0, 3.0, 1.0, 2.0)
            FROM retrieval_fts JOIN retrieval_chunks c USING(chunk_id)
            WHERE retrieval_fts MATCH ? AND c.status='active'
              AND c.sensitivity='normal' AND c.injection_flag=0
              AND (c.scope='global' OR c.workspace_id=?) {type_sql}
            ORDER BY 2 ASC LIMIT ?"""
        try:
            with self._lock, self._connect() as connection:
                rows = connection.execute(
                    sql, [match_query, str(workspace_id), *params, int(top_k)]
                ).fetchall()
        except sqlite3.Error:
            return self._lexical_search(
                terms, top_k=top_k, workspace_id=workspace_id, source_types=source_types
            )
        return [
            RetrievalHit(
                chunk=RetrievalChunk.from_dict(json.loads(payload)),
                sparse_score=max(0.0, -float(rank)),
                channels=["sparse"],
            )
            for payload, rank in rows
        ]

    def dense_search(
        self,
        vector,
        *,
        model_id,
        top_k,
        workspace_id,
        source_types=(),
    ):
        if hnswlib is not None and len(vector) > 0 and top_k > 0:
            try:
                result = self._dense_search_hnsw(
                    vector, model_id=model_id, top_k=top_k,
                    workspace_id=workspace_id, source_types=source_types,
                )
                self._last_dense_backend = "hnsw"
                self._hnsw_error = ""
                return result
            except Exception as exc:  # noqa: BLE001 - exact search is the fallback
                # An invalid or unavailable derived graph must not lose results.
                self._hnsw_error = str(exc)
        self._last_dense_backend = "exact"
        return self._dense_search_exact(
            vector, model_id=model_id, top_k=top_k,
            workspace_id=workspace_id, source_types=source_types,
        )

    def _dense_search_hnsw(self, vector, *, model_id, top_k, workspace_id, source_types):
        with self._lock:
            graph, chunks = self._hnsw_index(model_id, len(vector))
            if graph is None:
                return []
            wanted_types = set(source_types)
            allowed = {
                label for label, (chunk_id, source_type, owner, scope) in enumerate(chunks)
                if (not wanted_types or source_type in wanted_types)
                and (scope == "global" or owner == str(workspace_id))
            }
            if not allowed:
                return []
            graph.set_ef(max(100, int(top_k) * 2))
            labels, distances = graph.knn_query(
                [vector], k=min(int(top_k), len(allowed)), num_threads=1,
                filter=allowed.__contains__,
            )
            matches = [
                (chunks[int(label)][0], 1.0 - float(distance))
                for label, distance in zip(labels[0], distances[0])
                if 1.0 - float(distance) > 0
            ]
            if not matches:
                return []
            placeholders = ",".join("?" for _ in matches)
            type_sql, type_params = _source_type_sql(source_types)
            with self._connect() as connection:
                rows = connection.execute(
                    f"SELECT chunk_id, chunk_json FROM retrieval_chunks "
                    f"WHERE chunk_id IN ({placeholders}) AND status='active' "
                    "AND sensitivity='normal' AND injection_flag=0 "
                    f"AND (scope='global' OR workspace_id=?) {type_sql}",
                    [*[chunk_id for chunk_id, _ in matches], str(workspace_id), *type_params],
                ).fetchall()
        payloads = dict(rows)
        return [
            RetrievalHit(
                chunk=RetrievalChunk.from_dict(json.loads(payloads[chunk_id])),
                dense_score=score, channels=["dense"],
            )
            for chunk_id, score in matches if chunk_id in payloads
        ]

    def _hnsw_index(self, model_id, dimensions):
        key = (str(model_id), int(dimensions))
        with self._lock:
            with self._connect() as connection:
                connection.execute("BEGIN")  # generation and vectors share one snapshot
                generation = connection.execute(
                    "SELECT generation FROM retrieval_index_generation WHERE id = 1"
                ).fetchone()[0]
                cached = self._hnsw_cache.get(key)
                if cached is not None and cached[0] == generation:
                    return cached[1], cached[2]
                rows = connection.execute(
                    """SELECT v.chunk_id, v.vector, c.source_type, c.workspace_id, c.scope
                    FROM retrieval_vectors v JOIN retrieval_chunks c USING(chunk_id)
                    WHERE v.model_id=? AND v.dimensions=? AND c.status='active'
                      AND c.sensitivity='normal' AND c.injection_flag=0
                    ORDER BY v.chunk_id""",
                    key,
                ).fetchall()
            if not rows:
                self._hnsw_cache[key] = (generation, None, [])
                return None, []
            graph = hnswlib.Index(space="cosine", dim=int(dimensions))
            graph.init_index(max_elements=len(rows), M=16, ef_construction=200)
            graph.add_items(
                [_unpack_vector(blob, dimensions) for _, blob, *_ in rows],
                list(range(len(rows))),
            )
            chunks = [(chunk_id, source_type, owner, scope)
                      for chunk_id, _, source_type, owner, scope in rows]
            self._hnsw_cache[key] = (generation, graph, chunks)
            return graph, chunks

    def _dense_search_exact(
        self, vector, *, model_id, top_k, workspace_id, source_types=(),
    ):
        type_sql, params = _source_type_sql(source_types, prefix="c.")
        sql = f"""SELECT c.chunk_json, v.dimensions, v.vector
            FROM retrieval_vectors v JOIN retrieval_chunks c USING(chunk_id)
            WHERE v.model_id=? AND v.dimensions=? AND c.status='active'
              AND c.sensitivity='normal' AND c.injection_flag=0
              AND (c.scope='global' OR c.workspace_id=?) {type_sql}"""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                sql, [str(model_id), len(vector), str(workspace_id), *params]
            ).fetchall()
        ranked = []
        query_vector = _normalize(vector)
        for payload, dimensions, blob in rows:
            candidate = _unpack_vector(blob, dimensions)
            score = sum(left * right for left, right in zip(query_vector, candidate))
            ranked.append((score, payload))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievalHit(
                chunk=RetrievalChunk.from_dict(json.loads(payload)),
                dense_score=float(score),
                channels=["dense"],
            )
            for score, payload in ranked[: int(top_k)]
            if score > 0
        ]

    def stats(self):
        with self._lock, self._connect() as connection:
            chunks = connection.execute("SELECT COUNT(*) FROM retrieval_chunks").fetchone()[0]
            vectors = connection.execute("SELECT COUNT(*) FROM retrieval_vectors").fetchone()[0]
            manifests = connection.execute(
                "SELECT namespace, workspace_id, signature, model_id, chunk_count FROM retrieval_manifests"
            ).fetchall()
        return {
            "chunks": int(chunks),
            "vectors": int(vectors),
            "fts_enabled": bool(self.fts_enabled),
            "dense_backend": self._last_dense_backend,
            "hnsw_error": self._hnsw_error,
            "manifests": [
                {
                    "namespace": row[0],
                    "workspace_id": row[1],
                    "signature": row[2],
                    "model_id": row[3],
                    "chunk_count": row[4],
                }
                for row in manifests
            ],
        }

    def _insert_chunk(self, connection, chunk):
        payload = chunk.to_dict()
        connection.execute(
            """INSERT OR REPLACE INTO retrieval_chunks VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk.chunk_id, chunk.parent_id, chunk.source_id, chunk.source_type,
                chunk.title, chunk.text, chunk.summary, chunk.path, chunk.symbol,
                chunk.heading_path, chunk.start_line, chunk.end_line,
                chunk.workspace_id, chunk.source_hash, chunk.revision, chunk.status,
                chunk.scope, chunk.supersedes, chunk.sensitivity,
                int(chunk.injection_flag),
                json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ),
        )

    def _insert_fts(self, connection, chunk):
        search_text = " ".join(search_tokens(chunk.embedding_text()))
        connection.execute(
            "INSERT INTO retrieval_fts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                chunk.chunk_id, chunk.title, chunk.path, chunk.symbol,
                chunk.heading_path, chunk.summary, chunk.text, search_text,
            ),
        )

    def _lexical_search(self, terms, *, top_k, workspace_id, source_types):
        type_sql, params = _source_type_sql(source_types)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""SELECT chunk_json FROM retrieval_chunks
                WHERE status='active' AND sensitivity='normal' AND injection_flag=0
                  AND (scope='global' OR workspace_id=?) {type_sql}""",
                [str(workspace_id), *params],
            ).fetchall()
        ranked = []
        wanted = set(terms)
        for (payload,) in rows:
            chunk = RetrievalChunk.from_dict(json.loads(payload))
            fields = {
                "title": set(search_tokens(chunk.title)),
                "path": set(search_tokens(chunk.path)),
                "symbol": set(search_tokens(chunk.symbol)),
                "heading": set(search_tokens(chunk.heading_path)),
                "summary": set(search_tokens(chunk.summary)),
                "text": set(search_tokens(chunk.text)),
            }
            score = (
                len(wanted & fields["symbol"]) * 7
                + len(wanted & fields["title"]) * 6
                + len(wanted & fields["heading"]) * 5
                + len(wanted & fields["path"]) * 4
                + len(wanted & fields["summary"]) * 3
                + len(wanted & fields["text"])
            )
            if score:
                ranked.append((float(score), chunk))
        ranked.sort(key=lambda item: (item[0], item[1].chunk_id), reverse=True)
        return [
            RetrievalHit(chunk=chunk, sparse_score=score, channels=["lexical_fallback"])
            for score, chunk in ranked[: int(top_k)]
        ]

    @staticmethod
    def _signature(chunks, model_id):
        payload = [
            (chunk.chunk_id, chunk.source_hash, chunk.status, chunk.scope)
            for chunk in sorted(chunks, key=lambda row: row.chunk_id)
        ]
        return hashlib.sha256(
            json.dumps([model_id, payload], sort_keys=True).encode("utf-8")
        ).hexdigest()


def _source_type_sql(source_types, prefix=""):
    values = [str(value) for value in source_types if str(value)]
    if not values:
        return "", []
    placeholders = ",".join("?" for _ in values)
    return f"AND {prefix}source_type IN ({placeholders})", values


def _pack_vector(vector):
    values = [float(value) for value in vector]
    return struct.pack(f"<{len(values)}f", *values)


def _unpack_vector(blob, dimensions):
    return list(struct.unpack(f"<{int(dimensions)}f", bytes(blob)))


def _normalize(vector):
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else values
