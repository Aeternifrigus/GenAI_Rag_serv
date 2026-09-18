"""
Training data for the reranker.

The hand-written evaluation questions are the test set and are never trained on.
Training queries are therefore generated from the corpus itself, using the
inverse cloze construction: take a sentence out of a chunk, call it the query,
and call the chunk it came from the positive passage. The sentence is removed
from the passage, so the model cannot win by spotting the query verbatim.

Negatives are where a reranker is actually made or wasted.

  Random negatives are nearly free to separate. A model trained on them learns
  topic matching, which the bi-encoder already does, and then contributes
  nothing on the candidate list it is shown in production, where every candidate
  is already topically plausible.

  Hard negatives are the passages the real retriever ranked highly and got
  wrong. Training on those teaches the model the distinction it is actually
  asked to make at serving time.

So negatives are mined by running each generated query through the same
retrieval agent the service uses, and keeping the top-ranked passages that come
from a different document.
"""
from __future__ import annotations

import logging
import random
import re
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from .common import IndexedCorpus

logger = logging.getLogger(__name__)

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "it", "as", "at", "by", "be", "this", "that", "with", "from",
    "not", "no", "but", "if", "then", "than", "so", "such", "can", "may", "any",
    "must", "where", "which", "who", "what", "how", "when", "does", "do", "will",
}


def _drop_one_sentence(text: str, rng: random.Random) -> str:
    """
    Remove one sentence, the same operation the positive already underwent.

    Not cosmetic. The positive is built by cutting the query sentence out of its
    own chunk, which leaves it shorter than an untouched negative. On this corpus
    that gap was large enough that picking the shortest candidate alone beat
    chance by more than double, so the model could score well by measuring length
    and never reading the query. Applying the same cut to negatives removes the
    shortcut.
    """
    sentences = [x.strip() for x in _SENTENCE.split(text) if x.strip()]
    if len(sentences) < 2:
        return text
    drop = rng.randrange(len(sentences))
    return " ".join(x for i, x in enumerate(sentences) if i != drop)


@dataclass
class TrainingGroup:
    """One query, its positive passage, and the negatives it must outrank."""

    query: str
    positive: str
    negatives: list[str]

    @property
    def size(self) -> int:
        return 1 + len(self.negatives)


def _keyword_form(text: str) -> str:
    """A terse variant, mirroring the query expansion the agent does at serving."""
    words = [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2]
    return " ".join(words[:12])


def generate_queries(
    corpus: IndexedCorpus,
    per_chunk: int = 2,
    min_words: int = 6,
    keyword_variants: bool = True,
    seed: int = 42,
) -> list[tuple[str, str, str]]:
    """
    Returns (query, positive_passage, doc_id) triples.

    The positive passage is the source chunk with the query sentence removed. A
    chunk that is a single sentence yields nothing, because removing the query
    would leave no passage to retrieve.
    """
    rng = random.Random(seed)
    out: list[tuple[str, str, str]] = []

    for chunk in corpus.chunks:
        sentences = [s.strip() for s in _SENTENCE.split(chunk.text) if s.strip()]
        usable = [s for s in sentences if len(s.split()) >= min_words]
        if len(sentences) < 2 or not usable:
            continue

        rng.shuffle(usable)
        for sent in usable[:per_chunk]:
            remainder = " ".join(s for s in sentences if s != sent).strip()
            if not remainder:
                continue
            out.append((sent, remainder, chunk.doc_id))
            if keyword_variants:
                kw = _keyword_form(sent)
                if kw and kw != sent.lower():
                    out.append((kw, remainder, chunk.doc_id))

    logger.info("generated %d training queries from %d chunks", len(out), len(corpus.chunks))
    return out


@contextmanager
def _score_floor(agent, min_score: float):
    """
    Temporarily drop the retriever's score floor.

    The floor exists to keep weak passages away from the generator. A passage
    sitting just under it is exactly the near-miss worth training against, so
    mining lifts it and serving keeps it.
    """
    previous = agent.min_score
    agent.min_score = min_score
    try:
        yield agent
    finally:
        agent.min_score = previous


def mine_hard_negatives(
    corpus: IndexedCorpus,
    triples: Sequence[tuple[str, str, str]],
    n_negatives: int = 4,
    pool: int = 20,
    seed: int = 42,
    min_score: float = 0.0,
) -> list[TrainingGroup]:
    """
    Run each query through the real retriever and keep the top wrong documents.

    Wrong is judged at document level, not chunk level. Another chunk of the same
    document is usually a legitimate answer to the same question, so labelling it
    negative would train the model to contradict the ground truth the evaluation
    uses.
    """
    rng = random.Random(seed)
    all_texts = corpus.texts
    groups: list[TrainingGroup] = []
    topped_up = 0

    with _score_floor(corpus.agent, min_score) as agent:
        for query, positive, doc_id in triples:
            hits = agent.retrieve(query, k=pool).hits
            negatives = [
                _drop_one_sentence(h.chunk.text, rng)
                for h in hits
                if h.chunk.doc_id != doc_id and h.chunk.text != positive
            ][:n_negatives]

            # Top up from the rest of the corpus when the retriever returned too
            # few wrong documents to fill the group. Groups of unequal size would
            # make the softmax a different problem from one step to the next,
            # which shows up as noise in the loss rather than as signal.
            if len(negatives) < n_negatives:
                topped_up += 1
                spare = [t for t in all_texts if t != positive and t not in negatives]
                rng.shuffle(spare)
                negatives.extend(
                    _drop_one_sentence(t, rng) for t in spare[: n_negatives - len(negatives)]
                )

            if len(negatives) < n_negatives:
                continue
            groups.append(TrainingGroup(query=query, positive=positive, negatives=negatives))

    logger.info(
        "mined %d groups, %d needed topping up from random passages", len(groups), topped_up
    )
    return groups


def random_negative_groups(
    corpus: IndexedCorpus,
    triples: Sequence[tuple[str, str, str]],
    n_negatives: int = 4,
    seed: int = 42,
) -> list[TrainingGroup]:
    """
    Groups with negatives drawn uniformly at random.

    This is the easy task used for the first training stage. It teaches the model
    what a passage is and what topical relevance looks like, which is the part
    that does not need adapters and would otherwise consume them.
    """
    rng = random.Random(seed)
    by_doc: dict[str, list[str]] = {}
    for chunk in corpus.chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk.text)

    groups: list[TrainingGroup] = []
    for query, positive, doc_id in triples:
        others = [t for d, texts in by_doc.items() if d != doc_id for t in texts]
        if len(others) < n_negatives:
            continue
        groups.append(
            TrainingGroup(
                query=query,
                positive=positive,
                negatives=[_drop_one_sentence(t, rng) for t in rng.sample(others, n_negatives)],
            )
        )
    logger.info("built %d random-negative groups", len(groups))
    return groups


def split_groups(
    groups: Sequence[TrainingGroup], dev_fraction: float = 0.1, seed: int = 42
) -> tuple[list[TrainingGroup], list[TrainingGroup]]:
    """
    Split for loss monitoring only.

    The dev split here is drawn from the same generated distribution as training,
    so it tracks optimisation, not real performance. Real performance is the
    held-out question set, which nothing in this module touches.
    """
    items = list(groups)
    random.Random(seed).shuffle(items)
    cut = max(1, int(len(items) * dev_fraction))
    return items[cut:], items[:cut]
