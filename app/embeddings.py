"""
Embedding providers.

Two implementations behind one interface:

  LsaEmbedder            deterministic, offline, no model download. TF-IDF folded
                         down to a dense vector with truncated SVD (classic LSA).
                         This is the default so the service, its tests, and CI all
                         run with no network access and no API key.

  SentenceTransformerEmbedder  real neural embeddings, used when the model is
                         available. Better retrieval quality, needs a download.

The service never calls a provider directly. It asks get_embedder() for whatever
is configured and degrades to LSA if the preferred provider cannot be loaded, so
a missing model or a cold network never takes the API down.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Anything that turns text into fixed-width dense vectors."""

    name: str
    dim: int

    def fit(self, corpus: Sequence[str]) -> None: ...
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2_normalise(m: np.ndarray) -> np.ndarray:
    """Unit-length rows, so cosine similarity reduces to a dot product."""
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class LsaEmbedder:
    """
    TF-IDF -> truncated SVD. Offline, deterministic, and good enough to prove the
    retrieval and evaluation machinery works end to end.

    Unlike a neural embedder this must see the corpus before it can encode, so
    fit() is called once during ingestion and the fitted state is reused.
    """

    name = "lsa"

    def __init__(self, dim: int = 256, seed: int = 42) -> None:
        self.dim = dim
        self.seed = seed
        self._vec = None
        self._svd = None
        self._fitted = False
        self._lock = threading.Lock()

    def fit(self, corpus: Sequence[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not corpus:
            raise ValueError("cannot fit an embedder on an empty corpus")

        with self._lock:
            self._vec = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
            )
            tfidf = self._vec.fit_transform(corpus)

            # SVD cannot ask for more components than the matrix can supply.
            n_comp = int(min(self.dim, tfidf.shape[1] - 1, max(len(corpus) - 1, 1)))
            n_comp = max(n_comp, 1)

            self._svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
            self._svd.fit(tfidf)
            self.dim = n_comp
            self._fitted = True

        logger.info("lsa embedder fitted: %d docs, %d dims", len(corpus), self.dim)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("embedder used before fit(); ingest documents first")
        tfidf = self._vec.transform(list(texts))
        return _l2_normalise(self._svd.transform(tfidf).astype(np.float32))

    @property
    def fitted(self) -> bool:
        return self._fitted


class SentenceTransformerEmbedder:
    """Neural embeddings. Preferred in production; needs the model present."""

    name = "sentence-transformers"

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model
        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def fit(self, corpus: Sequence[str]) -> None:
        """No-op. Pretrained models need no corpus, but the interface is shared."""
        return

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(list(texts), convert_to_numpy=True)
        return _l2_normalise(vecs.astype(np.float32))

    @property
    def fitted(self) -> bool:
        return True


def get_embedder(provider: str = "auto", dim: int = 256) -> Embedder:
    """
    Resolve a provider by name. 'auto' prefers the neural embedder and falls back
    to LSA, logging the reason, rather than raising at startup.
    """
    if provider in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:  # missing package, no model cache, no network
            if provider == "sentence-transformers":
                logger.warning("requested sentence-transformers but %s; using lsa", exc)
            else:
                logger.info("sentence-transformers unavailable (%s); using lsa", exc)

    return LsaEmbedder(dim=dim)
