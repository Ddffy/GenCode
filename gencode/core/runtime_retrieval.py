"""Runtime adapter for Repo Map versus large-repository hybrid retrieval."""

from __future__ import annotations

from ..features import repomap as repomaplib


class RuntimeRetrievalMixin:
    def build_retrieval_section(self, user_message):
        recent = self.memory.to_dict()["working"]["recent_files"]
        cache = getattr(self, "_retrieval_section_cache", None)
        if cache is None:
            cache = self._retrieval_section_cache = {}
        cache_key = (
            str(getattr(self, "current_run_id", "")),
            str(user_message or ""),
        )
        cached = cache.get(cache_key)
        if cached is not None:
            self.last_repo_map_metadata = dict(cached[1])
            self.last_code_retrieval = (
                dict(cached[2]) if cached[2] is not None else None
            )
            return cached[0]
        fast_read_only_qa = bool(getattr(self, "fast_read_only_qa", False))
        if fast_read_only_qa:
            text, metadata = "", {
                "enabled": False,
                "reason": "deferred_for_fast_read_only_qa",
                "files_scanned": 0,
            }
        else:
            text, metadata = self.build_repo_map(
                query=user_message,
                budget_chars=repomaplib.DEFAULT_MAP_BUDGET_CHARS,
                recent_paths=recent,
            )
        self.last_repo_map_metadata = dict(metadata or {})
        service = getattr(getattr(self, "knowledge_store", None), "retrieval", None)
        if service is None or not self.feature_enabled("hybrid_retrieval"):
            self.last_code_retrieval = None
            cache[cache_key] = (text, dict(metadata or {}), None)
            return text
        route = service.route(
            source_type="code",
            corpus_size=int(metadata.get("files_scanned", 0) or 0),
            force_hybrid=fast_read_only_qa,
        )
        if route.get("strategy") == "repo_map":
            self.last_code_retrieval = {
                "strategy": "repo_map",
                "repo_map": metadata,
            }
            cache[cache_key] = (
                text,
                dict(metadata or {}),
                dict(self.last_code_retrieval),
            )
            return text
        try:
            sync = service.sync_code(
                self.root,
                cache_dir=self.root / ".gencode" / "repomap",
                embed=False,
            )
            if not sync.get("dense_complete", False):
                service.start_background_code_sync(
                    self.root,
                    cache_dir=self.root / ".gencode" / "repomap",
                )
            response = service.retrieve(
                user_message,
                source_types=("code",),
                top_k=service.config.final_top_k,
                channels=(
                    ("sparse",)
                    if not sync.get("dense_complete", False)
                    else ()
                ),
                rerank=bool(sync.get("dense_complete", False)),
            )
            self.last_code_retrieval = response.to_dict()
            self.last_code_retrieval["index_sync"] = sync
        except Exception as exc:  # noqa: BLE001 - retrieval plugins must degrade
            self.last_code_retrieval = {
                "strategy": "repo_map_fallback",
                "error": str(exc),
            }
            cache[cache_key] = (
                text,
                dict(metadata or {}),
                dict(self.last_code_retrieval),
            )
            return text
        mini_map = _clip_lines(text, service.config.code_context_budget_chars)
        if response.insufficient_evidence:
            cache[cache_key] = (
                mini_map,
                dict(metadata or {}),
                dict(self.last_code_retrieval),
            )
            return mini_map
        section = "\n\n".join(
            value
            for value in (
                mini_map,
                "Large-repository retrieval:\n" + response.evidence_text,
            )
            if value
        )
        cache[cache_key] = (
            section,
            dict(metadata or {}),
            dict(self.last_code_retrieval),
        )
        return section

    def invalidate_retrieval_cache(self):
        self._retrieval_section_cache = {}


def _clip_lines(text, limit):
    value = str(text or "")
    if len(value) <= int(limit):
        return value
    lines = []
    used = 0
    for line in value.splitlines():
        if used + len(line) + 1 > int(limit):
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines).rstrip() + "\n...[repo map clipped for code retrieval]"
