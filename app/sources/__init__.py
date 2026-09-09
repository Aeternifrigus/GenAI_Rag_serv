"""
Source connectors.

A source is any upstream system holding documents worth putting into the RAG
index. The JD's phrasing is "agents sourcing data from various systems", and this
is that layer: each connector knows how to reach one system and hand back a
uniform Document, so the ingestion pipeline never learns their differences.

Adding Starburst, BigQuery or Hadoop later means writing one more class here and
registering it, with nothing else in the service changing.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document as it arrives from a source, before chunking."""

    doc_id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Source(Protocol):
    name: str

    def fetch(self, limit: int | None = None) -> Iterator[Document]: ...
    def healthy(self) -> bool: ...


def stable_id(*parts: str) -> str:
    """Deterministic id, so re-ingesting the same document updates rather than duplicates."""
    return hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:16]


class FileSource:
    """Reads .txt and .md files off disk. The zero-dependency source."""

    name = "files"

    def __init__(self, root: str | Path, patterns: tuple[str, ...] = ("*.txt", "*.md")) -> None:
        self.root = Path(root)
        self.patterns = patterns

    def healthy(self) -> bool:
        return self.root.exists() and self.root.is_dir()

    def fetch(self, limit: int | None = None) -> Iterator[Document]:
        if not self.healthy():
            logger.warning("file source root missing: %s", self.root)
            return

        paths: list[Path] = []
        for pat in self.patterns:
            paths.extend(sorted(self.root.rglob(pat)))

        for i, p in enumerate(paths):
            if limit is not None and i >= limit:
                break
            try:
                text = p.read_text(encoding="utf-8").strip()
            except (UnicodeDecodeError, OSError) as exc:
                logger.warning("skipping %s: %s", p, exc)
                continue
            if not text:
                continue
            yield Document(
                doc_id=stable_id(self.name, str(p.relative_to(self.root))),
                text=text,
                source=self.name,
                metadata={"path": str(p.relative_to(self.root)), "title": p.stem},
            )


class MongoSource:
    """
    Reads documents out of a MongoDB collection.

    text_field and title_field are configurable because no two collections agree
    on their schema, and hardcoding 'body' would make this connector single-use.
    """

    name = "mongo"

    def __init__(
        self,
        uri: str,
        database: str,
        collection: str,
        text_field: str = "text",
        title_field: str = "title",
        query: dict | None = None,
    ) -> None:
        self.uri = uri
        self.database = database
        self.collection = collection
        self.text_field = text_field
        self.title_field = title_field
        self.query = query or {}
        self._client = None

    def _get_client(self):
        if self._client is None:
            from pymongo import MongoClient

            self._client = MongoClient(self.uri, serverSelectionTimeoutMS=2000)
        return self._client

    def healthy(self) -> bool:
        try:
            self._get_client().admin.command("ping")
            return True
        except Exception as exc:
            logger.info("mongo source unreachable: %s", exc)
            return False

    def fetch(self, limit: int | None = None) -> Iterator[Document]:
        if not self.healthy():
            return

        coll = self._get_client()[self.database][self.collection]
        cursor = coll.find(self.query)
        if limit is not None:
            cursor = cursor.limit(limit)

        for doc in cursor:
            text = (doc.get(self.text_field) or "").strip()
            if not text:
                continue
            mongo_id = str(doc.get("_id"))
            yield Document(
                doc_id=stable_id(self.name, mongo_id),
                text=text,
                source=self.name,
                metadata={
                    "mongo_id": mongo_id,
                    "title": doc.get(self.title_field, ""),
                    "collection": self.collection,
                },
            )


class InlineSource:
    """
    Documents handed straight to the API by a caller. Lets the service be driven
    over HTTP without any upstream system attached, which is what the tests use.
    """

    name = "inline"

    def __init__(self, documents: list[dict] | None = None) -> None:
        self.documents = documents or []

    def healthy(self) -> bool:
        return True

    def fetch(self, limit: int | None = None) -> Iterator[Document]:
        for i, d in enumerate(self.documents):
            if limit is not None and i >= limit:
                break
            text = (d.get("text") or "").strip()
            if not text:
                continue
            yield Document(
                doc_id=d.get("doc_id") or stable_id(self.name, text[:120], str(i)),
                text=text,
                source=self.name,
                metadata=d.get("metadata", {}),
            )
