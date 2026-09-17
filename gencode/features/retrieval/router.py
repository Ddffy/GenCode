"""Policy router that keeps corpus choice out of individual retrievers."""

from __future__ import annotations


class RetrievalRouter:
    def __init__(self, *, code_rag_file_threshold=1000):
        self.code_rag_file_threshold = max(1, int(code_rag_file_threshold))

    def route(self, *, source_type, corpus_size=0, lexical_confidence=0.0):
        kind = str(source_type or "").casefold()
        if kind == "spec":
            return {"strategy": "explicit_binding", "channels": ()}
        if kind == "skill":
            return {"strategy": "metadata_trigger", "channels": ()}
        if kind == "code":
            if int(corpus_size) < self.code_rag_file_threshold:
                return {"strategy": "repo_map", "channels": ("repo_map",)}
            return {
                "strategy": "mini_repo_map_plus_hybrid",
                "channels": ("sparse", "dense"),
            }
        if kind == "wiki":
            # Dense is a recall supplement.  The pipeline can still degrade to
            # sparse-only when no embedding provider is available.
            return {
                "strategy": "hybrid_rrf" if lexical_confidence <= 0 else "sparse_first_hybrid",
                "channels": ("sparse", "dense"),
            }
        return {"strategy": "hybrid_rrf", "channels": ("sparse", "dense")}
