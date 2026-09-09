"""
In-memory vector store.

The default when no Postgres is configured. Keeps vectors in a single numpy
matrix so search is one matmul rather than a Python loop, which is fast enough
for the tens of thousands of chunks this service is scoped for.

Not durable. Restarting the process loses the index, which is the honest
trade for having zero infrastructure to run locally or in CI.
"""
from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np

from .base import Chunk, Hit


class InMemoryVectorStore:
    name = "memory"

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self._by_id: dict[str, int] = {}
        self._lock = threading.Lock()

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunk/vector mismatch: {len(chunks)} chunks, {vectors.shape[0]} vectors"
            )
        if not chunks:
            return 0

        with self._lock:
            new_chunks, new_rows = [], []
            for chunk, vec in zip(chunks, vectors, strict=True):
                idx = self._by_id.get(chunk.id)
                if idx is None:
                    self._by_id[chunk.id] = len(self._chunks) + len(new_chunks)
                    new_chunks.append(chunk)
                    new_rows.append(vec)
                else:
                    # replace in place so re-ingesting a document does not duplicate it
                    self._chunks[idx] = chunk
                    if self._matrix is not None:
                        self._matrix[idx] = vec

            if new_chunks:
                block = np.vstack(new_rows).astype(np.float32)
                self._chunks.extend(new_chunks)
                self._matrix = block if self._matrix is None else np.vstack([self._matrix, block])

            return len(chunks)

    def search(
        self,
        vector: np.ndarray,
        k: int = 5,
        sources: Sequence[str] | None = None,
    ) -> list[Hit]:
        with self._lock:
            if self._matrix is None or not self._chunks:
                return []

            q = np.asarray(vector, dtype=np.float32).reshape(-1)
            if q.shape[0] != self._matrix.shape[1]:
                raise ValueError(
                    f"query dim {q.shape[0]} does not match index dim {self._matrix.shape[1]}; "
                    "the index was probably built with a different embedder"
                )

            # vectors are stored L2-normalised, so a dot product is cosine similarity
            scores = self._matrix @ q

            allowed = set(sources) if sources else None
            order = np.argsort(-scores)

            hits: list[Hit] = []
            for i in order:
                chunk = self._chunks[int(i)]
                if allowed is not None and chunk.source not in allowed:
                    continue
                hits.append(Hit(chunk=chunk, score=float(scores[int(i)])))
                if len(hits) >= k:
                    break
            return hits

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._by_id.clear()
            self._matrix = None
