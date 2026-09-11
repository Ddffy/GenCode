"""File -> file reference graph built from extracted tags.

The graph is a lexical approximation on purpose: a reference to identifier
`foo` in file A creates a weighted edge A -> every file that defines `foo`.
No scope resolution, no import parsing -- the same trade-off aider makes.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from ...core.workspace import IGNORED_PATH_NAMES
from . import tags as tagslib

CACHE_SCHEMA_VERSION = 1
MAX_FILE_BYTES = 512 * 1024
MAX_FILES = 3000
# Hard traversal bounds: the workspace root may resolve somewhere huge
# (e.g. home when git discovery escapes a temp dir), so the walk must stay
# lazy and capped instead of materializing the whole tree.
MAX_DIRS = 20000
# Common names ("main", "run", "new") define in many files; edges from them
# are noise, so identifiers with too many defining files are skipped.
MAX_DEFINERS_PER_IDENT = 32

SUPPORTED_SUFFIXES = set(tagslib.SUFFIX_LANGUAGES)


def collect_source_files(root):
    """Deterministic bounded BFS over parseable files, ignored dirs skipped."""
    root = Path(root)
    files = []
    queue = [root]
    visited_dirs = 0
    while queue and len(files) < MAX_FILES and visited_dirs < MAX_DIRS:
        current = queue.pop(0)
        visited_dirs += 1
        try:
            entries = sorted(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if len(files) >= MAX_FILES:
                break
            name = entry.name
            if name in IGNORED_PATH_NAMES or name.startswith("."):
                continue
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    queue.append(entry)
                    continue
                if (
                    not entry.is_file()
                    or entry.suffix.lower() not in SUPPORTED_SUFFIXES
                ):
                    continue
                if entry.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(entry.relative_to(root).as_posix())
    return files


def _file_sha256(abs_path):
    digest = hashlib.sha256()
    try:
        with open(abs_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


class TagCache:
    """(path, sha256)-keyed tag cache stored under .gencode/repomap/."""

    def __init__(self, root, cache_dir=None):
        self.root = Path(root)
        self.cache_dir = (
            Path(cache_dir) if cache_dir else self.root / ".gencode" / "repomap"
        )
        self.cache_path = self.cache_dir / "tags_cache.json"
        self._data = {"version": CACHE_SCHEMA_VERSION, "files": {}}
        self.loaded = False
        try:
            if self.cache_path.exists():
                stored = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if stored.get("version") == CACHE_SCHEMA_VERSION:
                    self._data["files"] = dict(stored.get("files", {}))
            self.loaded = True
        except (OSError, ValueError):
            self._data["files"] = {}

    def get(self, rel_path, sha256):
        if not sha256:
            return None
        cached = self._data["files"].get(rel_path)
        if not cached or cached.get("sha") != sha256:
            return None
        return cached

    def put(self, file_tags):
        self._data["files"][file_tags.path] = {
            "sha": file_tags.sha256,
            "language": file_tags.language,
            "definitions": [list(tag) for tag in file_tags.definitions],
            "references": [list(tag) for tag in file_tags.references],
        }

    def save(self):
        if not self.loaded:
            return False
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._data, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            return True
        except OSError:
            return False


def load_tags(root, cache=None, cache_dir=None):
    """Extract tags for all source files, reusing cache entries by sha."""
    root = Path(root)
    tag_cache = cache if cache is not None else TagCache(root, cache_dir=cache_dir)
    tags_by_file = {}
    cache_dirty = False
    for rel_path in collect_source_files(root):
        abs_path = root / rel_path
        language = tagslib.language_for_path(rel_path)
        if language is None or not tagslib.pack_available():
            continue
        sha = _file_sha256(abs_path)
        cached = tag_cache.get(rel_path, sha)
        if cached is not None:
            tags_by_file[rel_path] = tagslib.FileTags(
                path=rel_path,
                language=cached.get("language") or language,
                sha256=sha,
                definitions=tuple(
                    (name, int(line)) for name, line in cached.get("definitions", [])
                ),
                references=tuple(
                    (name, int(line)) for name, line in cached.get("references", [])
                ),
            )
            continue
        try:
            source_bytes = abs_path.read_bytes()
        except OSError:
            continue
        file_tags = tagslib.extract_file_tags(
            abs_path, rel_path, language, source_bytes
        )
        if file_tags is None:
            continue
        tag_cache.put(file_tags)
        cache_dirty = True
        tags_by_file[rel_path] = file_tags
    if cache_dirty:
        tag_cache.save()
    return tags_by_file


def build_reference_graph(tags_by_file):
    """Return (defines_index, edge_weights) for the file-level graph.

    Edge weight is the number of matching reference occurrences, split
    evenly when one identifier resolves to several defining files.
    """
    defines = defaultdict(set)
    for file_tags in tags_by_file.values():
        for name, _line in file_tags.definitions:
            defines[name].add(file_tags.path)

    edge_weights = defaultdict(float)
    for file_tags in tags_by_file.values():
        for name, _line in file_tags.references:
            targets = defines.get(name)
            if not targets:
                continue
            targets = sorted(target for target in targets if target != file_tags.path)
            if not targets or len(targets) > MAX_DEFINERS_PER_IDENT:
                continue
            share = 1.0 / len(targets)
            for target in targets:
                edge_weights[(file_tags.path, target)] += share
    return dict(defines), dict(edge_weights)
