"""Stable value objects shared by retrieval plugins."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RetrievalChunk:
    chunk_id: str
    parent_id: str
    source_id: str
    source_type: str
    title: str
    text: str
    # Structural unit the chunk belongs to: the function for code, the heading
    # section for documents. Retrieval matches children because they are focused,
    # but generation is given the whole section, so the id has to be addressable
    # independently of the parent (which is the file/page and is used for dedup).
    section_id: str = ""
    summary: str = ""
    path: str = ""
    symbol: str = ""
    heading_path: str = ""
    start_line: int = 0
    end_line: int = 0
    workspace_id: str = ""
    source_hash: str = ""
    revision: str = ""
    status: str = "active"
    scope: str = "workspace"
    supersedes: str = ""
    sensitivity: str = "normal"
    injection_flag: bool = False
    metadata: dict = field(default_factory=dict)

    def embedding_text(self) -> str:
        fields = [
            self.title,
            self.path,
            self.symbol,
            self.heading_path,
            self.summary,
            self.text,
        ]
        return "\n".join(value for value in fields if value).strip()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> RetrievalChunk:
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value[key] for key in allowed if key in value})


@dataclass
class RetrievalHit:
    chunk: RetrievalChunk
    score: float = 0.0
    sparse_score: float = 0.0
    dense_score: float = 0.0
    sparse_rank: int | None = None
    dense_rank: int | None = None
    rerank_score: float = 0.0
    channels: list[str] = field(default_factory=list)
    rejection_reason: str = ""
    # Set when the hit was expanded from a matched child to its whole structural
    # section. ``body`` is what gets rendered into the prompt and ``expansion``
    # records which fallback produced it.
    section: "RetrievalChunk | None" = None
    body: str = ""
    expansion: str = ""

    @property
    def citation_id(self) -> str:
        location = self.chunk.heading_path or self.chunk.symbol or self.chunk.chunk_id
        return f"{self.chunk.source_type}:{self.chunk.source_id}#{location}"

    def evidence_chunk(self) -> RetrievalChunk:
        """Chunk whose span and text should be cited and rendered."""
        return self.section or self.chunk

    def to_dict(self) -> dict:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "sparse_score": self.sparse_score,
            "dense_score": self.dense_score,
            "sparse_rank": self.sparse_rank,
            "dense_rank": self.dense_rank,
            "rerank_score": self.rerank_score,
            "channels": list(self.channels),
            "rejection_reason": self.rejection_reason,
            "citation_id": self.citation_id,
            "section_id": self.chunk.section_id,
            "expansion": self.expansion,
        }


@dataclass
class RetrievalResponse:
    query: str
    hits: list[RetrievalHit] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    strategy: str = ""
    query_plan: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    evidence_text: str = ""
    insufficient_evidence: bool = False

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "hits": [hit.to_dict() for hit in self.hits],
            "rejected": list(self.rejected),
            "strategy": self.strategy,
            "query_plan": dict(self.query_plan),
            "metrics": dict(self.metrics),
            "evidence_text": self.evidence_text,
            "insufficient_evidence": self.insufficient_evidence,
        }
