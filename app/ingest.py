"""
Ingestion pipeline: source -> chunk -> embed -> upsert.

This is the "data transfer pipeline" half of the JD. It is deliberately separate
from retrieval so the two can be scaled, scheduled and failed independently.

Chunking overlaps by default because a fact that straddles a boundary is
otherwise retrievable from neither side of it.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .embeddings import Embedder
from .sources import Document, Source, stable_id
from .stores.base import Chunk, VectorStore

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class IngestReport:
    documents: int = 0
    chunks: int = 0
    sources: dict[str, int] = None
    skipped: list[str] = None

    def __post_init__(self) -> None:
        if self.sources is None:
            self.sources = {}
        if self.skipped is None:
            self.skipped = []


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """
    Split on sentence boundaries, packing sentences up to `size` characters and
    carrying `overlap` characters of tail into the next chunk.

    Sentence-aware rather than a blind character slice, because cutting
    mid-sentence produces chunks that embed poorly and read badly as citations.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    sentences = _SENTENCE_END.split(text)
    chunks: list[str] = []
    current = ""

    for sent in sentences:
        # a single sentence longer than the window has to be hard-split
        if len(sent) > size:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(sent), size - overlap):
                chunks.append(sent[i : i + size].strip())
            continue

        if len(current) + len(sent) + 1 <= size:
            current = f"{current} {sent}".strip()
        else:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {sent}".strip()

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]


def documents_to_chunks(
    docs: Sequence[Document], size: int = 900, overlap: int = 150
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        pieces = chunk_text(doc.text, size=size, overlap=overlap)
        for i, piece in enumerate(pieces):
            out.append(
                Chunk(
                    id=stable_id(doc.source, doc.doc_id, str(i)),
                    text=piece,
                    source=doc.source,
                    doc_id=doc.doc_id,
                    metadata={**doc.metadata, "chunk_index": i, "chunk_count": len(pieces)},
                )
            )
    return out


class IngestionPipeline:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._corpus: list[str] = []

    def run(self, sources: Sequence[Source], limit: int | None = None) -> IngestReport:
        report = IngestReport()
        docs: list[Document] = []

        for src in sources:
            if not src.healthy():
                report.skipped.append(src.name)
                logger.warning("source %s unhealthy, skipping", src.name)
                continue
            fetched = list(src.fetch(limit=limit))
            docs.extend(fetched)
            report.sources[src.name] = len(fetched)
            logger.info("source %s produced %d documents", src.name, len(fetched))

        if not docs:
            logger.warning("no documents fetched from any source")
            return report

        chunks = documents_to_chunks(docs, self.chunk_size, self.chunk_overlap)
        report.documents = len(docs)
        report.chunks = len(chunks)

        # LSA has to see the whole corpus before it can encode anything, so the
        # vocabulary is refit over everything ingested so far, not just this batch.
        self._corpus.extend(c.text for c in chunks)
        self.embedder.fit(self._corpus)

        # refitting changes the vector space, so previously indexed chunks would
        # no longer be comparable. Rebuild the index rather than silently mixing
        # vectors from two different spaces.
        if getattr(self.embedder, "name", "") == "lsa" and self.store.count() > 0:
            logger.info("embedder refit; rebuilding index over %d chunks", len(self._corpus))
            self.store.clear()
            all_chunks = getattr(self, "_all_chunks", [])
            all_chunks.extend(chunks)
            self._all_chunks = all_chunks
            chunks_to_index = all_chunks
        else:
            self._all_chunks = getattr(self, "_all_chunks", []) + list(chunks)
            chunks_to_index = self._all_chunks

        vectors = self.embedder.encode([c.text for c in chunks_to_index])
        self.store.upsert(chunks_to_index, vectors)

        logger.info("ingested %d documents into %d chunks", report.documents, report.chunks)
        return report
