"""
Vector store interface.

The service talks only to this protocol, so swapping the in-memory store for
pgvector is a config change rather than a code change. Both implementations
return the same Hit shape, and both filter on metadata the same way.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class Chunk:
    """One retrievable unit of text plus where it came from."""

    id: str
    text: str
    source: str                       # which system supplied it: mongo, files, ...
    doc_id: str                       # the parent document
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    chunk: Chunk
    score: float                      # cosine similarity, higher is closer
    # Set only when a reranker has read the (query, passage) pair. Kept separate
    # from `score` rather than replacing it, so a wrong answer can be traced to
    # retrieval missing the passage or to the reranker demoting it.
    rerank_score: float | None = None


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int: ...
    def search(
        self,
        vector: np.ndarray,
        k: int = 5,
        sources: Sequence[str] | None = None,
    ) -> list[Hit]: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...
