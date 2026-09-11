"""Repo map: tree-sitter tags -> reference graph -> personalized PageRank.

Facade for the three-layer repo map pipeline:

1. parse engine: tree-sitter-language-pack (optional extra `gencode[map]`).
2. symbol layer: per-file definition/reference tags + weighted file graph.
3. ranking layer: personalized PageRank seeded by the current request and
   working memory, rendered as a budgeted map.

Every layer degrades gracefully: missing language pack, no supported
files, or an empty graph all yield an empty map plus a machine-readable
reason, never an error.
"""

from __future__ import annotations

from pathlib import Path

from . import graph as graphlib
from . import rank as ranklib
from . import render as renderlib
from . import tags as tagslib

DEFAULT_MAP_BUDGET_CHARS = 6000
MAX_SEED_RECENT_PATHS = 8


class RepoMapBuilder:
    def __init__(self, root, cache_dir=None):
        self.root = Path(root)
        self.cache_dir = cache_dir

    def build(self, query="", budget_chars=DEFAULT_MAP_BUDGET_CHARS, recent_paths=()):
        query = str(query or "")
        budget_chars = max(200, int(budget_chars or DEFAULT_MAP_BUDGET_CHARS))
        meta = {
            "enabled": False,
            "reason": "",
            "files_scanned": 0,
            "files_tagged": 0,
            "edge_count": 0,
            "seeded_files": [],
            "rendered_files": 0,
            "chars": 0,
        }
        if not tagslib.pack_available():
            meta["reason"] = tagslib.availability_reason()
            return "", meta
        # 根目录护栏：workspace 解析到 home 或盘符根时，地图没有意义，
        # 扫描也只是浪费——直接空地图返回。
        try:
            home = Path.home()
        except RuntimeError:
            home = None
        if (home is not None and self.root == home) or self.root.anchor == str(self.root):
            meta["reason"] = "workspace_root_too_broad"
            return "", meta

        tags_by_file = graphlib.load_tags(self.root, cache_dir=self.cache_dir)
        meta["files_scanned"] = len(graphlib.collect_source_files(self.root))
        meta["files_tagged"] = len(tags_by_file)
        if not tags_by_file:
            meta["reason"] = "no_supported_files"
            return "", meta

        _defines, edge_weights = graphlib.build_reference_graph(tags_by_file)
        meta["edge_count"] = len(edge_weights)
        seeds = ranklib.build_seeds(
            query,
            tags_by_file,
            recent_paths=list(recent_paths or ())[:MAX_SEED_RECENT_PATHS],
        )
        meta["seeded_files"] = sorted(seeds)

        nodes = set(tags_by_file)
        nodes.update(src for src, _dst in edge_weights)
        nodes.update(dst for _src, dst in edge_weights)
        # Hybrid lexical + graph ranking.  The lexical channel is normalized
        # BM25; the graph channel is normalized personalized PageRank.  This
        # keeps the 0.8/0.2 fusion numerically meaningful and prevents a
        # broad natural-language query from flattening the teleport vector.
        scores, seeds = ranklib.hybrid_scores(
            query,
            nodes,
            edge_weights,
            tags_by_file,
            recent_paths=list(recent_paths or ())[:MAX_SEED_RECENT_PATHS],
        )

        ranked = [
            path
            for path in sorted(nodes, key=lambda item: (-scores.get(item, 0.0), item))
            if tags_by_file.get(path) is not None and tags_by_file[path].definitions
        ]
        text = renderlib.render_repo_map(ranked, tags_by_file, budget_chars)
        meta["enabled"] = bool(text)
        if not text:
            meta["reason"] = "empty_graph"
        meta["rendered_files"] = len(text.splitlines()) - 1 if text else 0
        meta["chars"] = len(text)
        return text, meta


def available():
    return tagslib.pack_available()
