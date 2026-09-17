"""Hard retrieval gates applied before evidence reaches the prompt."""

from __future__ import annotations

from pathlib import PurePosixPath


class RetrievalFilter:
    def __init__(self, *, allowed_paths=(), freshness_checker=None):
        self.allowed_paths = tuple(
            str(path).replace("\\", "/").strip("/")
            for path in allowed_paths
            if str(path).strip()
        )
        self.freshness_checker = freshness_checker

    def reject_reason(self, chunk, *, workspace_id):
        if chunk.status != "active":
            return chunk.status or "inactive"
        if chunk.scope != "global" and chunk.workspace_id != str(workspace_id):
            return "scope_mismatch"
        if chunk.sensitivity != "normal":
            return "sensitive"
        if chunk.injection_flag:
            return "prompt_injection"
        if self.allowed_paths and chunk.path and not self._path_allowed(chunk.path):
            return "path_not_allowed"
        if self.freshness_checker is not None:
            result = self.freshness_checker(chunk)
            if result is False:
                return "stale_evidence"
            if isinstance(result, str) and result:
                return result
        return ""

    def apply(self, hits, *, workspace_id):
        selected = []
        rejected = []
        for hit in hits:
            reason = self.reject_reason(hit.chunk, workspace_id=workspace_id)
            if reason:
                hit.rejection_reason = reason
                rejected.append(
                    {
                        "chunk_id": hit.chunk.chunk_id,
                        "source_id": hit.chunk.source_id,
                        "reason": reason,
                    }
                )
            else:
                selected.append(hit)
        return selected, rejected

    def _path_allowed(self, path):
        candidate = str(PurePosixPath(str(path).replace("\\", "/"))).strip("/")
        return any(
            candidate == root or candidate.startswith(root + "/")
            for root in self.allowed_paths
        )
