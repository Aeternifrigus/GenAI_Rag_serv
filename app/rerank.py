"""
Reranking: a second, more expensive pass over a short candidate list.

Retrieval ends at rank fusion, which scores a passage without ever letting it
meet the query. Reranking closes that gap. The agent widens its candidate pool,
a cross-encoder reads each (query, passage) pair properly, and only the survivors
reach the generator.

Three implementations behind one interface:

  NoopReranker    truncates to k and changes nothing. The honest default when no
                  model is present, and the control arm when measuring whether
                  reranking helped.

  TorchReranker   the compact cross-encoder from crossencoder.py, trained by
                  training/. No download, no key, runs in CI.

  HfReranker      a pretrained cross-encoder through transformers. Better model,
                  needs the weights to be reachable.

Resolution follows the rest of the service: ask for the best, fall back with a
logged reason, and report on /health what actually loaded rather than what was
configured. A reranker that fails to load must never take retrieval down with
it, because unranked results still answer the question.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .stores.base import Hit

logger = logging.getLogger(__name__)

DEFAULT_HF_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: Sequence[Hit], k: int) -> list[Hit]: ...


class NoopReranker:
    """Keeps the fusion order. Present so the calling code has no branches."""

    name = "none"

    def rerank(self, query: str, hits: Sequence[Hit], k: int) -> list[Hit]:
        return list(hits[:k])


def _attach(hits: Sequence[Hit], scores: Sequence[float], k: int) -> list[Hit]:
    """
    Order by the reranker's score, keeping the retrieval score intact.

    Both numbers are worth having: when a reranked answer is wrong, the question
    is whether retrieval surfaced the passage and the reranker buried it, or
    retrieval never surfaced it at all, and that is unanswerable if the second
    score overwrote the first.
    """
    scored = [
        Hit(chunk=h.chunk, score=h.score, rerank_score=round(float(s), 6))
        for h, s in zip(hits, scores, strict=True)
    ]
    scored.sort(key=lambda h: -h.rerank_score)
    return scored[:k]


class TorchReranker:
    """The in-repo cross-encoder, optionally carrying trained LoRA adapters."""

    name = "cross-encoder-local"

    def __init__(
        self,
        model,
        tokenizer,
        max_len: int | None = None,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        import torch

        self._torch = torch
        self.model = model.eval()
        self.tokenizer = tokenizer
        # The window is a property of the checkpoint, not of this class. The
        # positional embedding table was sized at training time, so a longer
        # window here indexes past the end of it. Taking a default from the
        # constructor instead is what caused that failure once already.
        trained_len = getattr(getattr(model, "cfg", None), "max_len", None)
        self.max_len = min(max_len, trained_len) if (max_len and trained_len) else (
            max_len or trained_len or 256
        )
        self.batch_size = batch_size
        self.device = device or _best_device()
        self.model.to(self.device)

    def rerank(self, query: str, hits: Sequence[Hit], k: int) -> list[Hit]:
        if not hits:
            return []

        torch = self._torch
        pairs = [(query, h.chunk.text) for h in hits]
        scores: list[float] = []

        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = self.tokenizer.encode_batch(
                    pairs[i : i + self.batch_size], max_len=self.max_len
                )
                batch = {k_: v.to(self.device) for k_, v in batch.items()}
                logits = self.model(**batch)
                scores.extend(torch.sigmoid(logits).cpu().tolist())

        return _attach(hits, scores, k)


class HfReranker:
    """A pretrained cross-encoder from the transformers hub."""

    name = "cross-encoder-hf"

    def __init__(
        self,
        model_name: str = DEFAULT_HF_MODEL,
        adapter_path: str | None = None,
        max_len: int = 256,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size

        if adapter_path:
            from .lora import inject_lora, load_lora_state_dict

            payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
            inject_lora(
                self.model,
                targets=payload.get("targets", ("query", "value")),
                r=payload.get("r", 8),
                alpha=payload.get("alpha", 16),
            )
            load_lora_state_dict(self.model, payload["state_dict"])
            logger.info("loaded LoRA adapters from %s", adapter_path)

        self.device = device or _best_device()
        self.model.to(self.device).eval()

    def rerank(self, query: str, hits: Sequence[Hit], k: int) -> list[Hit]:
        if not hits:
            return []

        torch = self._torch
        scores: list[float] = []
        texts = [h.chunk.text for h in hits]

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                window = texts[i : i + self.batch_size]
                enc = self.tokenizer(
                    [query] * len(window),
                    window,
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                # ms-marco cross-encoders emit a single relevance logit; a
                # two-class head means the positive class is the second column
                col = logits[:, -1] if logits.shape[-1] > 1 else logits.squeeze(-1)
                scores.extend(torch.sigmoid(col).cpu().tolist())

        return _attach(hits, scores, k)


def _best_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    # Apple silicon. Checked before cpu because it is the difference between a
    # usable and an unusable rerank latency on a laptop.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_reranker(
    provider: str = "auto",
    model_path: str | None = None,
    hf_model: str = DEFAULT_HF_MODEL,
    adapter_path: str | None = None,
) -> Reranker:
    """
    Resolve a reranker.

    'auto' prefers the local trained checkpoint over the hub, because a
    checkpoint sitting in the repository is reachable with no network and was
    trained on this corpus, while the hub model is better in general but may not
    load at all. 'none' disables reranking outright, which is what the
    before-and-after measurement uses as its control.
    """
    if provider == "none":
        return NoopReranker()

    if provider in ("auto", "local"):
        path = model_path or os.environ.get("RERANK_MODEL_PATH") or _default_model_path()
        if path and Path(path).exists():
            try:
                from .crossencoder import load_checkpoint

                model, tokenizer, extra = load_checkpoint(path)
                if extra.get("adapter_state"):
                    _apply_adapters(model, extra)
                logger.info("local reranker loaded from %s", path)
                return TorchReranker(model, tokenizer)
            except Exception as exc:
                logger.warning("local reranker at %s failed to load (%s)", path, exc)
        elif provider == "local":
            logger.warning("local reranker requested but no checkpoint at %s", path)

    if provider in ("auto", "hf"):
        try:
            return HfReranker(hf_model, adapter_path=adapter_path)
        except Exception as exc:
            level = logger.warning if provider == "hf" else logger.info
            level("hf reranker unavailable (%s); reranking disabled", exc)

    return NoopReranker()


def _apply_adapters(model, extra: dict) -> None:
    """Re-inject LoRA at the recorded shape, then load the trained adapters."""
    from .lora import inject_lora, load_lora_state_dict

    inject_lora(
        model,
        targets=tuple(extra.get("targets", ("q_proj", "v_proj"))),
        r=int(extra.get("r", 8)),
        alpha=int(extra.get("alpha", 16)),
    )
    load_lora_state_dict(model, extra["adapter_state"])


def _default_model_path() -> str:
    from .config import BASE_DIR

    return str(BASE_DIR / "models" / "reranker.pt")
