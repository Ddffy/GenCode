"""Runtime adapter for Repo Map versus large-repository hybrid retrieval."""

from __future__ import annotations

from ..features import repomap as repomaplib


class RuntimeRetrievalMixin:
    def build_retrieval_section(self, user_message):
        recent = self.memory.to_dict()["working"]["recent_files"]
        text, metadata = self.build_repo_map(
            query=user_message,
            budget_chars=repomaplib.DEFAULT_MAP_BUDGET_CHARS,
            recent_paths=recent,
        )
        self.last_repo_map_metadata = dict(metadata or {})
        service = getattr(getattr(self, "knowledge_store", None), "retrieval", None)
        if service is None or not self.feature_enabled("hybrid_retrieval"):
            self.last_code_retrieval = None
            return text
        route = service.route(
            source_type="code",
            corpus_size=int(metadata.get("files_scanned", 0) or 0),
        )
        if route.get("strategy") == "repo_map":
            self.last_code_retrieval = {
                "strategy": "repo_map",
                "repo_map": metadata,
            }
            return text
        try:
            sync = service.sync_code(
                self.root,
                cache_dir=self.root / ".gencode" / "repomap",
            )
            response = service.retrieve(
                user_message,
                source_types=("code",),
                top_k=service.config.final_top_k,
            )
            self.last_code_retrieval = response.to_dict()
            self.last_code_retrieval["index_sync"] = sync
        except Exception as exc:  # noqa: BLE001 - retrieval plugins must degrade
            self.last_code_retrieval = {
                "strategy": "repo_map_fallback",
                "error": str(exc),
            }
            return text
        mini_map = _clip_lines(text, service.config.code_context_budget_chars)
        if response.insufficient_evidence:
            return mini_map
        return "\n\n".join(
            value
            for value in (
                mini_map,
                "Large-repository retrieval:\n" + response.evidence_text,
            )
            if value
        )


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
