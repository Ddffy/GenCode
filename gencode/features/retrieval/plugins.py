"""Retriever plugin contracts and built-in sparse/dense adapters."""

from __future__ import annotations

from typing import Protocol


class RetrieverPlugin(Protocol):
    name: str

    def search(
        self,
        query: str,
        *,
        top_k: int,
        workspace_id: str,
        source_types: tuple[str, ...],
    ) -> list: ...


class RetrieverRegistry:
    def __init__(self):
        self._plugins = {}

    def register(self, plugin):
        name = str(getattr(plugin, "name", "")).strip()
        if not name:
            raise ValueError("retriever plugin must declare a name")
        self._plugins[name] = plugin
        return plugin

    def unregister(self, name):
        return self._plugins.pop(str(name), None)

    def get(self, name):
        return self._plugins.get(str(name))

    def names(self):
        return tuple(sorted(self._plugins))

    def plugins(self, names=()):
        selected = self.names() if not names else tuple(str(name) for name in names)
        return [self._plugins[name] for name in selected if name in self._plugins]


class SparseRetrieverPlugin:
    name = "sparse"

    def __init__(self, index):
        self.index = index

    def search(self, query, *, top_k, workspace_id, source_types):
        return self.index.sparse_search(
            query,
            top_k=top_k,
            workspace_id=workspace_id,
            source_types=source_types,
        )


class DenseRetrieverPlugin:
    name = "dense"

    def __init__(self, index, embedder, *, min_score=0.0):
        self.index = index
        self.embedder = embedder
        self.min_score = float(min_score)

    def search(self, query, *, top_k, workspace_id, source_types):
        if self.embedder is None:
            return []
        vector = self.embedder.embed_query(query)
        rows = self.index.dense_search(
            vector,
            model_id=self.embedder.model_id,
            top_k=top_k,
            workspace_id=workspace_id,
            source_types=source_types,
        )
        return [row for row in rows if row.dense_score >= self.min_score]
