"""
Shared setup for training and evaluation.

Both need the same thing: the service's real retrieval stack, built outside the
web app so a training run does not need a server. Everything here uses the
application's own classes rather than a reimplementation, so a change to
chunking or fusion shows up in training rather than quietly diverging from it.
"""
from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.agent import RetrievalAgent
from app.config import BASE_DIR, settings
from app.embeddings import LsaEmbedder
from app.ingest import IngestionPipeline
from app.sources import FileSource
from app.stores.base import Chunk
from app.stores.memory import InMemoryVectorStore

logger = logging.getLogger(__name__)

EVAL_PATH = BASE_DIR / "data" / "eval" / "questions.json"
MODEL_DIR = BASE_DIR / "models"


@dataclass
class IndexedCorpus:
    agent: RetrievalAgent
    chunks: list[Chunk]
    title_to_doc_id: dict[str, str]

    @property
    def texts(self) -> list[str]:
        return [c.text for c in self.chunks]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def build_index(docs_dir: str | None = None, k: int = 20) -> IndexedCorpus:
    """
    Ingest the corpus exactly as the service does, with no reranker attached.

    No reranker on purpose: this index is what mines the hard negatives, and a
    reranked pool would be negatives selected by the model being trained rather
    than by the retriever it has to improve on.
    """
    embedder = LsaEmbedder(dim=settings.embed_dim)
    store = InMemoryVectorStore()
    pipeline = IngestionPipeline(
        store=store,
        embedder=embedder,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    report = pipeline.run([FileSource(docs_dir or settings.docs_dir)])
    if report.chunks == 0:
        raise RuntimeError(f"no documents ingested from {docs_dir or settings.docs_dir}")

    chunks = list(pipeline._all_chunks)
    title_to_doc_id = {c.metadata.get("title", c.doc_id): c.doc_id for c in chunks}

    agent = RetrievalAgent(
        store=store,
        embedder=embedder,
        k=k,
        min_score=settings.min_score,
        reranker=None,
    )
    logger.info("index built: %d documents, %d chunks", report.documents, report.chunks)
    return IndexedCorpus(agent=agent, chunks=chunks, title_to_doc_id=title_to_doc_id)


def load_eval_cases(path: Path | None = None) -> list[dict]:
    """
    The held-out question set.

    These are hand-written natural questions and are never used for training.
    Training queries come from the corpus itself, so no question in this file has
    ever been seen by the model being measured on it.
    """
    p = path or EVAL_PATH
    cases = json.loads(Path(p).read_text(encoding="utf-8"))
    if not cases:
        raise ValueError(f"evaluation set at {p} is empty")
    return cases


def resolve_ground_truth(cases: Sequence[dict], corpus: IndexedCorpus) -> list[dict]:
    """Map each case's document title to the doc_id the index actually assigned."""
    resolved = []
    for case in cases:
        doc_id = corpus.title_to_doc_id.get(case["title"])
        if doc_id is None:
            raise ValueError(f"evaluation case references unknown document {case['title']!r}")
        resolved.append({"question": case["question"], "doc_id": doc_id, "title": case["title"]})
    return resolved
