"""
Configuration and singleton wiring.

Every external dependency is optional. The service picks the best available
option at startup and records what it actually resolved to, which is exposed on
/health so an operator can see the running configuration rather than infer it.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # storage
    database_url: str | None = field(default_factory=lambda: os.environ.get("DATABASE_URL"))

    # mongo source
    mongo_uri: str | None = field(default_factory=lambda: os.environ.get("MONGO_URI"))
    mongo_db: str = field(default_factory=lambda: os.environ.get("MONGO_DB", "genai"))
    mongo_collection: str = field(
        default_factory=lambda: os.environ.get("MONGO_COLLECTION", "documents")
    )

    # file source
    docs_dir: str = field(
        default_factory=lambda: os.environ.get(
            "DOCS_DIR", str(BASE_DIR / "data" / "sample_docs")
        )
    )

    # models
    embed_provider: str = field(default_factory=lambda: os.environ.get("EMBED_PROVIDER", "auto"))
    embed_dim: int = field(default_factory=lambda: int(os.environ.get("EMBED_DIM", "256")))
    generator: str = field(default_factory=lambda: os.environ.get("GENERATOR", "auto"))

    # retrieval
    top_k: int = field(default_factory=lambda: int(os.environ.get("TOP_K", "5")))
    min_score: float = field(default_factory=lambda: float(os.environ.get("MIN_SCORE", "0.05")))
    chunk_size: int = field(default_factory=lambda: int(os.environ.get("CHUNK_SIZE", "900")))
    chunk_overlap: int = field(default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP", "150")))

    # reranking
    rerank_provider: str = field(
        default_factory=lambda: os.environ.get("RERANK_PROVIDER", "auto")
    )
    rerank_candidates: int = field(
        default_factory=lambda: int(os.environ.get("RERANK_CANDIDATES", "20"))
    )
    rerank_model_path: str | None = field(
        default_factory=lambda: os.environ.get("RERANK_MODEL_PATH")
    )

    # service
    auto_ingest: bool = field(default_factory=lambda: _env_bool("AUTO_INGEST", True))
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8080")))


settings = Settings()


class Registry:
    """Holds the resolved components so the app builds them exactly once."""

    def __init__(self) -> None:
        self.embedder = None
        self.store = None
        self.reranker = None
        self.agent = None
        self.pipeline = None
        self.ingestion = None
        self.resolved: dict[str, str] = {}

    def build(self) -> None:
        from .agent import RetrievalAgent
        from .embeddings import get_embedder
        from .ingest import IngestionPipeline
        from .rag import RagPipeline, get_generator
        from .rerank import get_reranker
        from .stores.memory import InMemoryVectorStore

        self.embedder = get_embedder(settings.embed_provider, settings.embed_dim)
        self.resolved["embedder"] = self.embedder.name

        # pgvector when a DSN is configured and reachable, memory otherwise
        self.store = None
        if settings.database_url:
            try:
                from .stores.pgvector import PgVectorStore

                self.store = PgVectorStore(settings.database_url, dim=settings.embed_dim)
                self.resolved["store"] = "pgvector"
            except Exception as exc:
                logger.warning("pgvector unavailable (%s); using in-memory store", exc)
        if self.store is None:
            self.store = InMemoryVectorStore()
            self.resolved["store"] = "memory"

        generator = get_generator(settings.generator)
        self.resolved["generator"] = generator.name

        self.reranker = get_reranker(
            settings.rerank_provider, model_path=settings.rerank_model_path
        )
        self.resolved["reranker"] = self.reranker.name

        self.agent = RetrievalAgent(
            store=self.store,
            embedder=self.embedder,
            k=settings.top_k,
            min_score=settings.min_score,
            reranker=self.reranker,
            candidate_k=settings.rerank_candidates,
        )
        self.pipeline = RagPipeline(agent=self.agent, generator=generator)
        self.ingestion = IngestionPipeline(
            store=self.store,
            embedder=self.embedder,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        logger.info("registry built: %s", self.resolved)

    def default_sources(self) -> list:
        """File source always; mongo only when a URI is configured."""
        from .sources import FileSource, MongoSource

        srcs: list = [FileSource(settings.docs_dir)]
        if settings.mongo_uri:
            srcs.append(
                MongoSource(
                    uri=settings.mongo_uri,
                    database=settings.mongo_db,
                    collection=settings.mongo_collection,
                )
            )
        return srcs


registry = Registry()
