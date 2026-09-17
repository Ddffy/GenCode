"""Index-time and query-time orchestration for pluggable retrieval."""

from __future__ import annotations

import copy
import json
import time
from collections import OrderedDict
from pathlib import Path

from .assembler import assemble_evidence, insufficient_evidence_message
from .chunking import chunk_code_repository, chunk_wiki_records
from .config import retrieval_config_from_env
from .embeddings import create_embedding_provider
from .filters import RetrievalFilter
from .index import SQLiteRetrievalIndex
from .plugins import DenseRetrieverPlugin, RetrieverRegistry, SparseRetrieverPlugin
from .query import understand_query
from .rerank import (
    CrossEncoderReranker,
    DashScopeReranker,
    DeterministicReranker,
    parent_aware_deduplicate,
    reciprocal_rank_fusion,
    resolve_conflicts,
)
from .router import RetrievalRouter
from .types import RetrievalResponse


class HybridRetrievalService:
    """Local-first hybrid search with replaceable retriever plugins."""

    def __init__(
        self,
        root,
        *,
        workspace_id,
        config=None,
        embedder=None,
        reranker=None,
        event_sink=None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.workspace_id = str(workspace_id)
        self.config = config or retrieval_config_from_env()
        self.event_sink = event_sink
        self.index = SQLiteRetrievalIndex(self.root / "index.db")
        self.degraded_reasons = []
        if embedder is None and self.config.enabled:
            try:
                embedder = create_embedding_provider(
                    self.config, cache_path=self.root / "embedding_cache.db"
                )
            except Exception as exc:  # noqa: BLE001 - optional provider boundary
                self.degraded_reasons.append(f"embedding_unavailable:{exc}")
                embedder = None
        self.embedder = embedder
        self.registry = RetrieverRegistry()
        self.registry.register(SparseRetrieverPlugin(self.index))
        if self.embedder is not None:
            self.registry.register(
                DenseRetrieverPlugin(
                    self.index, self.embedder, min_score=self.config.min_score
                )
            )
        self.router = RetrievalRouter(
            code_rag_file_threshold=self.config.code_rag_file_threshold
        )
        self.fallback_reranker = DeterministicReranker()
        self.fine_reranker = reranker or self._build_fine_reranker()
        self._cache = OrderedDict()

    def register_retriever(self, plugin):
        self._cache.clear()
        return self.registry.register(plugin)

    def sync_wiki(self, records, *, force=False):
        chunked = chunk_wiki_records(
            records,
            workspace_id=self.workspace_id,
            child_chunk_tokens=self.config.child_chunk_tokens,
            child_chunk_overlap=self.config.child_chunk_overlap,
        )
        if not self.config.enabled:
            return {"changed": False, "chunks": len(chunked.children), "disabled": True}
        result = self.index.sync(
            chunked.children,
            sections=chunked.sections,
            namespace="wiki",
            workspace_id=self.workspace_id,
            embedder=self.embedder,
            embedding_batch_size=self.config.embedding_batch_size,
            force=force,
        )
        self._cache.clear()
        self._emit("retrieval_index_synced", {"source_type": "wiki", **result})
        return result

    def sync_code(self, repo_root, *, cache_dir=None, force=False):
        chunked = chunk_code_repository(
            repo_root,
            workspace_id=self.workspace_id,
            cache_dir=cache_dir,
            child_chunk_tokens=self.config.child_chunk_tokens,
            child_chunk_overlap=self.config.child_chunk_overlap,
        )
        if not self.config.enabled:
            return {"changed": False, "chunks": len(chunked.children), "disabled": True}
        result = self.index.sync(
            chunked.children,
            sections=chunked.sections,
            namespace="code",
            workspace_id=self.workspace_id,
            embedder=self.embedder,
            embedding_batch_size=self.config.embedding_batch_size,
            force=force,
        )
        self._cache.clear()
        self._emit("retrieval_index_synced", {"source_type": "code", **result})
        return result

    def retrieve(
        self,
        query,
        *,
        source_types=("wiki",),
        top_k=None,
        allowed_paths=(),
        freshness_checker=None,
        channels=(),
        use_cache=True,
    ):
        started = time.perf_counter()
        source_types = tuple(str(value) for value in source_types)
        if not self.config.enabled:
            return RetrievalResponse(
                query=str(query),
                strategy="disabled",
                query_plan={"source_types": list(source_types)},
                metrics={
                    "cache_hit": False,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "channels": {},
                    "fused_count": 0,
                    "coarse_count": 0,
                    "selected_count": 0,
                    "conflicts": [],
                    "citations": {},
                    "errors": [],
                    "degraded_reasons": ["disabled_by_config"],
                    "index": self.index.stats(),
                },
                evidence_text=insufficient_evidence_message(source_types),
                insufficient_evidence=True,
            )
        plan = understand_query(query, source_types=source_types)
        enabled_channels = tuple(channels) or self._channels_for(source_types)
        cache_key = self._cache_key(query, source_types, allowed_paths, enabled_channels)
        if use_cache and freshness_checker is None and cache_key in self._cache:
            cached = self._cache.pop(cache_key)
            self._cache[cache_key] = cached
            response = copy.deepcopy(cached)
            response.metrics = {**response.metrics, "cache_hit": True}
            return response

        hard_filter = RetrievalFilter(
            allowed_paths=allowed_paths, freshness_checker=freshness_checker
        )
        channel_hits = {}
        rejected = []
        errors = []
        for channel in enabled_channels:
            plugin = self.registry.get(channel)
            if plugin is None:
                errors.append(f"{channel}_unavailable")
                continue
            limit = (
                self.config.sparse_top_k
                if channel == "sparse"
                else self.config.dense_top_k
            )
            merged = {}
            for variant in plan["variants"]:
                try:
                    hits = plugin.search(
                        variant,
                        top_k=limit,
                        workspace_id=self.workspace_id,
                        source_types=source_types,
                    )
                except Exception as exc:  # noqa: BLE001 - third-party plugin boundary
                    errors.append(f"{channel}_error:{exc}")
                    continue
                filtered, denied = hard_filter.apply(
                    hits, workspace_id=self.workspace_id
                )
                rejected.extend(denied)
                for hit in filtered:
                    previous = merged.get(hit.chunk.chunk_id)
                    if previous is None or _channel_value(hit, channel) > _channel_value(previous, channel):
                        merged[hit.chunk.chunk_id] = hit
            channel_hits[channel] = sorted(
                merged.values(),
                key=lambda hit: (_channel_value(hit, channel), hit.chunk.chunk_id),
                reverse=True,
            )[:limit]

        fused, candidates, conflicts = self._prepare_candidates(channel_hits)
        final_limit = int(top_k or self.config.final_top_k)
        reranker_requested = getattr(
            self.fine_reranker, "model_id", self.fine_reranker.__class__.__name__
        )
        active_reranker = self.fine_reranker
        try:
            selected = active_reranker.rerank(query, candidates, top_k=final_limit)
        except Exception as exc:  # noqa: BLE001 - remote reranker boundary
            errors.append(f"reranker_error:{exc}")
            active_reranker = self.fallback_reranker
            selected = active_reranker.rerank(query, candidates, top_k=final_limit)
        reranker_used = getattr(
            active_reranker, "model_id", active_reranker.__class__.__name__
        )
        selected, abstention = self._apply_abstention_floor(selected, active_reranker)
        selected = self._expand_to_sections(selected)
        evidence, citations = assemble_evidence(
            selected, budget_chars=self.config.evidence_budget_chars
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        response = RetrievalResponse(
            query=str(query),
            hits=selected,
            rejected=rejected,
            strategy=_strategy_name(channel_hits),
            query_plan=plan,
            metrics={
                "cache_hit": False,
                "latency_ms": round(elapsed_ms, 3),
                "channels": {name: len(rows) for name, rows in channel_hits.items()},
                "fused_count": len(fused),
                "coarse_count": len(candidates),
                "candidate_count": len(candidates),
                "selected_count": len(selected),
                "expansions": _expansion_counts(selected),
                "reranker_requested": reranker_requested,
                "reranker_used": reranker_used,
                "abstention": abstention,
                "conflicts": conflicts,
                "citations": citations,
                "errors": errors,
                "degraded_reasons": list(self.degraded_reasons),
                "index": self.index.stats(),
            },
            evidence_text=evidence,
            insufficient_evidence=not bool(selected),
        )
        if not selected:
            response.evidence_text = insufficient_evidence_message(source_types)
        if use_cache and freshness_checker is None:
            self._remember(cache_key, response)
        self._emit(
            "retrieval_completed",
            {
                "strategy": response.strategy,
                "source_types": list(source_types),
                "selected_count": len(selected),
                "latency_ms": round(elapsed_ms, 3),
                "errors": errors,
            },
        )
        return response

    def route(self, *, source_type, corpus_size=0, lexical_confidence=0.0):
        return self.router.route(
            source_type=source_type,
            corpus_size=corpus_size,
            lexical_confidence=lexical_confidence,
        )

    def record_feedback(self, *, query, citation_ids=(), accepted=None, details=None):
        payload = {
            "time": time.time(),
            "query_hash": __import__("hashlib").sha256(
                str(query).encode("utf-8")
            ).hexdigest()[:12],
            "citation_ids": list(citation_ids),
            "accepted": accepted,
            "details": dict(details or {}),
        }
        path = self.root / "feedback.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        return payload

    def stats(self):
        return {
            **self.index.stats(),
            "plugins": list(self.registry.names()),
            "embedding_model": getattr(self.embedder, "model_id", "disabled"),
            "reranker_model": getattr(
                self.fine_reranker,
                "model_id",
                self.fine_reranker.__class__.__name__,
            ),
            "degraded_reasons": list(self.degraded_reasons),
            "query_cache_entries": len(self._cache),
        }

    def _channels_for(self, source_types):
        channels = ["sparse"]
        if self.embedder is not None and self.registry.get("dense") is not None:
            channels.append("dense")
        return tuple(channels)

    def _prepare_candidates(self, channel_hits):
        """One gate: fuse, collapse obsolete/duplicate evidence, keep at most K."""
        fused = reciprocal_rank_fusion(channel_hits, rrf_k=self.config.rrf_k)
        conflict_free, conflicts = resolve_conflicts(fused)
        candidates = parent_aware_deduplicate(
            conflict_free,
            top_k=self.config.coarse_top_k,
            max_per_parent=self.config.parent_max_chunks,
        )
        return fused, candidates, conflicts

    def _build_fine_reranker(self):
        provider = str(self.config.reranker_provider).casefold()
        if provider in {
            "cross-encoder",
            "cross_encoder",
            "semantic",
        }:
            try:
                return CrossEncoderReranker(self.config.reranker_model)
            except Exception as exc:  # noqa: BLE001 - optional model boundary
                self.degraded_reasons.append(f"reranker_unavailable:{exc}")
        if provider in {"dashscope", "qwen", "alibaba"}:
            try:
                return DashScopeReranker(
                    model=self.config.reranker_model,
                    base_url=self.config.reranker_base_url,
                    api_key=self.config.reranker_api_key,
                    timeout=self.config.reranker_timeout,
                )
            except Exception as exc:  # noqa: BLE001 - remote provider boundary
                self.degraded_reasons.append(f"reranker_unavailable:{exc}")
        return self.fallback_reranker

    def _expand_to_sections(self, hits):
        """Collapse hits to one entry per structural section and attach the section.

        This is the second half of small-to-big: retrieval matches children because
        a focused fragment is easy to locate, but generation needs the surrounding
        section, since a matched fragment often lacks the condition the answer
        depends on. Children of one section therefore become a single evidence
        entry -- otherwise the same section would be rendered once per matched
        fragment and consume the budget several times over.
        """
        if not self.config.section_expand or not hits:
            return hits
        collapsed = []
        seen = set()
        for hit in hits:
            key = hit.chunk.section_id or hit.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            collapsed.append(hit)
        sections = self.index.sections(
            [hit.chunk.section_id for hit in collapsed if hit.chunk.section_id]
        )
        for hit in collapsed:
            section = sections.get(hit.chunk.section_id)
            if section is None:
                continue
            hit.section = section
            hit.expansion, hit.body = _expanded_body(hit.chunk, section, self.config)
        return collapsed

    def _apply_abstention_floor(self, selected, reranker):
        """Decide whether to answer at all, from the best evidence score.

        Without a floor the pipeline can only abstain when *zero* candidates
        survive, which in practice never happens: sparse lexical overlap almost
        always returns something, so ``insufficient_evidence`` stays false and
        the system answers questions its corpus does not cover.

        The floor is compared against the **best** score and abstains for the
        whole turn, rather than discarding individual hits. A per-hit floor looks
        equivalent but is not: on a multi-document question the second relevant
        document legitimately scores below the first, so a per-hit rule silently
        drops valid secondary evidence. Measured on the wiki benchmark, a per-hit
        floor cost recall on exactly the two multi-gold tasks.

        The floor is only meaningful for a reranker whose score is monotonic in
        relevance and calibrated to the corpus. The deterministic reranker's
        score is derived from RRF ranks and is nearly constant across queries, so
        applying a floor to it would be a silent no-op; that case is reported as
        ignored rather than left to look enabled.
        """
        threshold = float(self.config.abstention_min_rerank or 0.0)
        if threshold <= 0 or not selected:
            return selected, {"threshold": threshold, "applied": False, "dropped": 0}
        score_kind = str(getattr(reranker, "score_kind", "synthetic"))
        if score_kind != "relevance":
            reason = f"abstention_floor_ignored:{score_kind}"
            if reason not in self.degraded_reasons:
                self.degraded_reasons.append(reason)
            return selected, {
                "threshold": threshold,
                "applied": False,
                "dropped": 0,
                "ignored_reason": reason,
            }
        best_score = max(hit.rerank_score for hit in selected)
        if best_score >= threshold:
            return selected, {
                "threshold": threshold,
                "applied": True,
                "dropped": 0,
                "best_score": best_score,
            }
        return [], {
            "threshold": threshold,
            "applied": True,
            "dropped": len(selected),
            "best_score": best_score,
        }

    def _cache_key(self, query, source_types, allowed_paths, channels):
        manifests = self.index.stats().get("manifests", [])
        signature = tuple(
            sorted((row["namespace"], row["signature"]) for row in manifests)
        )
        return (
            str(query), tuple(source_types), tuple(allowed_paths), tuple(channels), signature
        )

    def _remember(self, key, response):
        self._cache[key] = copy.deepcopy(response)
        while len(self._cache) > int(self.config.query_cache_size):
            self._cache.popitem(last=False)

    def _emit(self, event, payload):
        if callable(self.event_sink):
            try:
                self.event_sink(event, dict(payload))
            except Exception as exc:  # noqa: BLE001 - observers cannot break retrieval
                reason = f"event_sink_error:{exc}"
                if reason not in self.degraded_reasons:
                    self.degraded_reasons.append(reason)


def _channel_value(hit, channel):
    return hit.sparse_score if channel == "sparse" else hit.dense_score


def _strategy_name(channel_hits):
    populated = [name for name, rows in channel_hits.items() if rows]
    if "sparse" in populated and "dense" in populated:
        return "sparse_dense_rrf"
    if "dense" in populated:
        return "dense_only"
    if "sparse" in populated:
        return "fts5_bm25"
    return "no_evidence"


def _expansion_counts(hits):
    """How many evidence entries came from a section, a window or a bare child."""
    counts = {"section": 0, "window": 0, "child": 0}
    for hit in hits:
        kind = str(getattr(hit, "expansion", "") or "child")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _expanded_body(child, section, config):
    """Return ``(expansion_kind, body)`` degrading section -> window -> child.

    The whole section is preferred because that is the point of small-to-big, but
    an unbounded section would eat the evidence budget, so an oversized one falls
    back to a line window around the matched child and finally to the child text.
    """
    text = str(section.text or "")
    if text and len(text) <= int(config.section_max_chars):
        return "section", text
    window = _line_window(text, child, section, config.section_window_lines)
    if window:
        return "window", window
    return "child", str(child.text or "")


def _line_window(text, child, section, window_lines):
    """Slice an oversized section around the child's own line span."""
    lines = text.splitlines()
    if not lines:
        return ""
    child_lines = max(1, int(child.end_line or 0) - int(child.start_line or 0) + 1)
    offset = max(0, int(child.start_line or 0) - int(section.start_line or 0))
    radius = max(0, int(window_lines))
    start = max(0, offset - radius)
    end = min(len(lines), offset + child_lines + radius)
    if start >= end:
        return ""
    return "\n".join(lines[start:end]).strip()
