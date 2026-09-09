"""
PostgreSQL + pgvector store.

The production path. Same interface as the in-memory store, so nothing upstream
changes when this is switched on via DATABASE_URL.

Requires the pgvector extension. docker-compose.yml brings up a Postgres image
that already has it.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence

import numpy as np

from .base import Chunk, Hit

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id        TEXT PRIMARY KEY,
    doc_id    TEXT NOT NULL,
    source    TEXT NOT NULL,
    text      TEXT NOT NULL,
    metadata  JSONB DEFAULT '{}'::jsonb,
    embedding vector(%(dim)s)
);

CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source);
CREATE INDEX IF NOT EXISTS chunks_doc_idx    ON chunks (doc_id);
"""

# ivfflat needs rows before it can build meaningful lists, so it is created after
# the first sizeable ingest rather than up front on an empty table.
INDEX_SQL = """
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = %(lists)s);
"""


class PgVectorStore:
    name = "pgvector"

    def __init__(self, dsn: str, dim: int) -> None:
        import psycopg2  # imported here so the package stays optional

        self._psycopg2 = psycopg2
        self.dsn = dsn
        self.dim = dim
        self._ensure_schema()

    def _connect(self):
        return self._psycopg2.connect(self.dsn)

    def _ensure_schema(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(SCHEMA % {"dim": self.dim})
            conn.commit()
        logger.info("pgvector schema ready (dim=%d)", self.dim)

    @staticmethod
    def _to_literal(vec: np.ndarray) -> str:
        """pgvector accepts a bracketed list literal."""
        return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunk/vector mismatch: {len(chunks)} chunks, {vectors.shape[0]} vectors"
            )
        if not chunks:
            return 0

        rows = [
            (c.id, c.doc_id, c.source, c.text, json.dumps(c.metadata), self._to_literal(v))
            for c, v in zip(chunks, vectors, strict=True)
        ]

        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (id, doc_id, source, text, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
                ON CONFLICT (id) DO UPDATE SET
                    doc_id    = EXCLUDED.doc_id,
                    source    = EXCLUDED.source,
                    text      = EXCLUDED.text,
                    metadata  = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    def search(
        self,
        vector: np.ndarray,
        k: int = 5,
        sources: Sequence[str] | None = None,
    ) -> list[Hit]:
        lit = self._to_literal(np.asarray(vector).reshape(-1))

        # <=> is pgvector cosine distance, so similarity is 1 - distance
        sql = """
            SELECT id, doc_id, source, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
        """
        params: list = [lit]
        if sources:
            sql += " WHERE source = ANY(%s)"
            params.append(list(sources))
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params += [lit, k]

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            Hit(
                chunk=Chunk(
                    id=r[0], doc_id=r[1], source=r[2], text=r[3],
                    metadata=r[4] if isinstance(r[4], dict) else json.loads(r[4] or "{}"),
                ),
                score=float(r[5]),
            )
            for r in rows
        ]

    def build_ann_index(self) -> None:
        """Call once the table has real volume; pointless on a near-empty table."""
        n = self.count()
        if n < 1000:
            logger.info("skipping ivfflat index: only %d rows", n)
            return
        lists = max(1, min(int(np.sqrt(n)), 1000))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(INDEX_SQL % {"lists": lists})
            conn.commit()
        logger.info("ivfflat index built with %d lists over %d rows", lists, n)

    def count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks")
            return int(cur.fetchone()[0])

    def clear(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("TRUNCATE chunks")
            conn.commit()
