"""Content-addressed persistence for extracted repository tags."""

import json
from pathlib import Path

CACHE_SCHEMA_VERSION = 1


class TagCache:
    """Content-addressed tags with a stat-based fast path for unchanged files."""

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

    def get_by_stat(self, rel_path, *, size, mtime_ns):
        cached = self._data["files"].get(rel_path)
        if (
            cached
            and int(cached.get("size", -1)) == int(size)
            and int(cached.get("mtime_ns", -1)) == int(mtime_ns)
        ):
            return cached
        return None

    def refresh_stat(self, rel_path, *, size, mtime_ns):
        cached = self._data["files"].get(rel_path)
        if cached is not None:
            cached["size"] = int(size)
            cached["mtime_ns"] = int(mtime_ns)

    def put(self, file_tags, *, size, mtime_ns):
        self._data["files"][file_tags.path] = {
            "sha": file_tags.sha256,
            "language": file_tags.language,
            "definitions": [list(tag) for tag in file_tags.definitions],
            "references": [list(tag) for tag in file_tags.references],
            "size": int(size),
            "mtime_ns": int(mtime_ns),
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
