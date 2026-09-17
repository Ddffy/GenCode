"""Rank fusion, deterministic reranking and parent-aware deduplication."""

from __future__ import annotations

import json
import urllib.request

from .query import search_tokens
from .types import RetrievalHit


def reciprocal_rank_fusion(channel_hits, *, rrf_k=60):
    merged = {}
    for channel, hits in channel_hits.items():
        for rank, hit in enumerate(hits, 1):
            item = merged.get(hit.chunk.chunk_id)
            if item is None:
                item = RetrievalHit(chunk=hit.chunk)
                merged[hit.chunk.chunk_id] = item
            item.score += 1.0 / (int(rrf_k) + rank)
            if channel not in item.channels:
                item.channels.append(channel)
            if channel == "sparse":
                item.sparse_rank = rank
                item.sparse_score = max(item.sparse_score, hit.sparse_score)
            elif channel == "dense":
                item.dense_rank = rank
                item.dense_score = max(item.dense_score, hit.dense_score)
    return sorted(
        merged.values(),
        key=lambda hit: (hit.score, hit.chunk.chunk_id),
        reverse=True,
    )


class DeterministicReranker:
    """Cheap coarse/fine ranker; replaceable by a cross-encoder plugin."""

    model_id = "deterministic"
    # Rank-derived scores are not comparable across corpora, so an abstention
    # floor can never be calibrated against them.
    score_kind = "synthetic"

    def rerank(self, query, hits, *, top_k):
        query_tokens = set(search_tokens(query))
        identifiers = {
            token.casefold()
            for token in query_tokens
            if "_" in token or "." in token or "/" in token
        }
        for hit in hits:
            chunk = hit.chunk
            title_tokens = set(search_tokens(chunk.title))
            path_tokens = set(search_tokens(chunk.path))
            symbol_tokens = set(search_tokens(chunk.symbol))
            heading_tokens = set(search_tokens(chunk.heading_path))
            body_tokens = set(search_tokens(chunk.text))
            coverage = len(
                query_tokens
                & (title_tokens | path_tokens | symbol_tokens | heading_tokens | body_tokens)
            ) / max(len(query_tokens), 1)
            exact_symbol = int(bool(query_tokens & symbol_tokens))
            exact_path = int(bool(identifiers & path_tokens))
            dual_channel = int(len(hit.channels) > 1)
            hit.rerank_score = (
                hit.score * 100
                + coverage * 4
                + exact_symbol * 2
                + exact_path * 1.5
                + dual_channel * 0.5
                + max(hit.dense_score, 0.0) * 0.25
            )
        ranked = sorted(
            hits,
            key=lambda hit: (hit.rerank_score, hit.score, hit.chunk.chunk_id),
            reverse=True,
        )
        return ranked[: max(0, int(top_k))]


class CrossEncoderReranker:
    """Optional semantic fine-ranker using sentence-transformers."""

    # Logits are monotonic in relevance, so a threshold is calibratable per model.
    score_kind = "relevance"

    def __init__(self, model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "sentence-transformers is not installed; install gencode[rag]"
            ) from exc
        self.model_id = f"cross-encoder:{model}"
        self.model = CrossEncoder(str(model))

    def rerank(self, query, hits, *, top_k):
        if not hits:
            return []
        pairs = [(str(query), hit.chunk.embedding_text()) for hit in hits]
        scores = self.model.predict(pairs, show_progress_bar=False)
        for hit, score in zip(hits, scores):
            hit.rerank_score = float(score)
        return sorted(
            hits,
            key=lambda hit: (hit.rerank_score, hit.score, hit.chunk.chunk_id),
            reverse=True,
        )[: max(0, int(top_k))]


class DashScopeReranker:
    """Remote Qwen text/multimodal reranker over the DashScope HTTP API."""

    # The API returns a relevance score, which is the only signal in this stack
    # that reliably separates answerable from unanswerable queries.
    score_kind = "relevance"

    def __init__(self, *, model, base_url, api_key, timeout=30):
        if not str(base_url or "").strip():
            raise ValueError("reranker base_url is required")
        if not str(api_key or "").strip():
            raise ValueError("reranker api_key is required")
        self.model = str(model or "qwen3.7-text-rerank")
        self.model_id = f"dashscope:{self.model}"
        self.endpoint = _dashscope_rerank_endpoint(base_url, self.model)
        self.api_key = str(api_key)
        self.timeout = max(1, int(timeout))

    def rerank(self, query, hits, *, top_k):
        candidates = list(hits)
        if not candidates:
            return []
        limit = min(len(candidates), max(0, int(top_k)))
        if limit == 0:
            return []
        documents = [hit.chunk.embedding_text() for hit in candidates]
        payload = self._payload(str(query), documents, limit)
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "gencode/0.3",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        rows = result.get("results")
        if rows is None:
            rows = dict(result.get("output", {}) or {}).get("results", [])
        ranked = []
        seen = set()
        for row in rows:
            try:
                index = int(row["index"])
                score = float(row.get("relevance_score", row.get("score", 0.0)))
            except (KeyError, TypeError, ValueError):
                continue
            if index < 0 or index >= len(candidates) or index in seen:
                continue
            seen.add(index)
            hit = candidates[index]
            hit.rerank_score = score
            ranked.append(hit)
            if len(ranked) >= limit:
                break
        if not ranked:
            raise RuntimeError("DashScope reranker returned no valid results")
        return ranked

    def _payload(self, query, documents, top_k):
        if self.model == "qwen3-rerank":
            return {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_k,
                "return_documents": False,
            }
        return {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": top_k, "return_documents": False},
        }


def _dashscope_rerank_endpoint(base_url, model):
    base = str(base_url).rstrip("/")
    if base.endswith(("/reranks", "/text-rerank")):
        return base
    if model == "qwen3-rerank":
        if base.endswith("/compatible-api/v1"):
            return base + "/reranks"
        return base + "/compatible-api/v1/reranks"
    suffix = "/services/rerank/text-rerank/text-rerank"
    if base.endswith("/api/v1"):
        return base + suffix
    return base + "/api/v1" + suffix


def parent_aware_deduplicate(hits, *, top_k, max_per_parent=2):
    selected = []
    parent_counts = {}
    seen_chunks = set()
    for hit in hits:
        if hit.chunk.chunk_id in seen_chunks:
            continue
        parent = hit.chunk.parent_id or hit.chunk.source_id
        if parent_counts.get(parent, 0) >= int(max_per_parent):
            continue
        selected.append(hit)
        seen_chunks.add(hit.chunk.chunk_id)
        parent_counts[parent] = parent_counts.get(parent, 0) + 1
        if len(selected) >= int(top_k):
            break
    return selected


def resolve_conflicts(hits):
    """Resolve competing evidence while keeping an audit record.

    A source can have two active records during a promotion/update window.  An
    explicit ``supersedes`` reference wins first; otherwise a higher numeric
    revision wins, then the more recently updated record, and rank remains the
    deterministic last resort.  The losing hit is never silently deleted from
    the trace: it is returned in the conflict list for audit and evaluation.
    """
    selected = {}
    conflicts = []
    for hit in hits:
        chunk = hit.chunk
        key = _conflict_key(chunk)
        previous = selected.get(key)
        if previous is None:
            selected[key] = hit
            continue
        winner, loser = _preferred_hit(previous, hit)
        selected[key] = winner
        conflicts.append(
            {
                "winner": winner.chunk.chunk_id,
                "discarded": loser.chunk.chunk_id,
                "reason": _conflict_reason(winner, loser),
            }
        )
    return list(selected.values()), conflicts


def _conflict_key(chunk):
    explicit = str((chunk.metadata or {}).get("conflict_key", "")).strip()
    if explicit:
        return explicit
    return ":".join(
        (
            chunk.source_type,
            chunk.source_id,
            chunk.heading_path or chunk.symbol or chunk.parent_id,
        )
    )


def _preferred_hit(left, right):
    left_refs = _supersede_refs(left)
    right_refs = _supersede_refs(right)
    left_ids = _identity_values(left)
    right_ids = _identity_values(right)
    if left_ids & right_refs:
        return right, left
    if right_ids & left_refs:
        return left, right
    return _newer_hit(left, right)


def _identity_values(hit):
    chunk = hit.chunk
    return {
        value
        for value in (chunk.chunk_id, chunk.parent_id, chunk.source_hash, chunk.source_id)
        if value
    }


def _supersede_refs(hit):
    value = str(hit.chunk.supersedes or "").strip()
    return {value} if value else set()


def _conflict_reason(winner, loser):
    """Name the rule that actually decided the conflict, not just who won."""
    if _identity_values(loser) & _supersede_refs(winner):
        return "supersedes"
    winner_revision = _as_int(winner.chunk.revision)
    loser_revision = _as_int(loser.chunk.revision)
    if (
        winner_revision is not None
        and loser_revision is not None
        and winner_revision != loser_revision
    ):
        return "newer_revision"
    if _updated_at(winner.chunk) > _updated_at(loser.chunk):
        return "newer_updated_at"
    return "lower_rank"


def _newer_hit(left, right):
    """Higher revision wins; at equal revisions the more recent record wins.

    Records that are peers at the same version are separated by ``updated_at``
    before rank, so a freshly approved page is not outranked by an older one
    that merely happened to score higher in fusion.
    """
    left_revision = _as_int(left.chunk.revision)
    right_revision = _as_int(right.chunk.revision)
    if (
        left_revision is not None
        and right_revision is not None
        and left_revision != right_revision
    ):
        return (right, left) if right_revision > left_revision else (left, right)
    left_updated = _updated_at(left.chunk)
    right_updated = _updated_at(right.chunk)
    if left_updated != right_updated:
        return (left, right) if left_updated > right_updated else (right, left)
    return (left, right) if left.rerank_score >= right.rerank_score else (right, left)


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _updated_at(chunk):
    return str((chunk.metadata or {}).get("updated_at", ""))
