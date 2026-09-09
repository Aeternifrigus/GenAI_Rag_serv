"""
Evaluation.

A RAG service without evaluation is a service nobody can tell has regressed. This
module scores two separable things, because they fail for different reasons and
need different fixes:

  Retrieval   did the right passages come back at all?
              Fixes live in chunking, embedding, routing, k.

  Generation  did the answer stay inside what was retrieved?
              Fixes live in the prompt, the model, the context budget.

Faithfulness here is lexical overlap against the retrieved context, not a model
grading itself. It is a weaker signal than an LLM judge but it is deterministic,
free, and cannot be gamed by the same model that produced the answer, which
makes it usable as a CI gate.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "it", "as", "at", "by", "be", "this", "that", "with", "from",
    "not", "no", "but", "if", "then", "than", "so", "such", "can", "may",
}


def tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 2}


@dataclass
class EvalCase:
    """One labelled question. relevant_doc_ids is the ground truth."""

    question: str
    relevant_doc_ids: list[str]
    expected_answer: str | None = None


@dataclass
class CaseResult:
    question: str
    retrieved_doc_ids: list[str]
    relevant_doc_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    hit: bool
    faithfulness: float | None = None
    answer: str = ""


@dataclass
class EvalReport:
    k: int
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.cases)

    def _mean(self, attr: str) -> float:
        vals = [getattr(c, attr) for c in self.cases if getattr(c, attr) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    @property
    def precision(self) -> float:
        return self._mean("precision_at_k")

    @property
    def recall(self) -> float:
        return self._mean("recall_at_k")

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank: how high the first correct passage lands."""
        return self._mean("reciprocal_rank")

    @property
    def hit_rate(self) -> float:
        if not self.cases:
            return 0.0
        return round(sum(1 for c in self.cases if c.hit) / len(self.cases), 4)

    @property
    def faithfulness(self) -> float:
        return self._mean("faithfulness")

    def summary(self) -> dict:
        return {
            "cases": self.n,
            "k": self.k,
            f"precision@{self.k}": self.precision,
            f"recall@{self.k}": self.recall,
            "mrr": self.mrr,
            "hit_rate": self.hit_rate,
            "faithfulness": self.faithfulness,
        }


def faithfulness_score(answer: str, context_texts: Sequence[str]) -> float:
    """
    Fraction of the answer's content words that appear somewhere in the retrieved
    context. Low scores mean the answer introduced material that was not retrieved,
    which is the lexical shadow of a hallucination.

    Blunt by design: it cannot detect a claim that reuses context vocabulary but
    misstates the relationship. It catches the obvious failure, not the subtle one.
    """
    a = tokens(answer)
    if not a:
        return 0.0
    ctx = set().union(*(tokens(t) for t in context_texts)) if context_texts else set()
    if not ctx:
        return 0.0
    return round(len(a & ctx) / len(a), 4)


def evaluate(
    pipeline,
    cases: Sequence[EvalCase],
    k: int = 5,
    score_answers: bool = True,
) -> EvalReport:
    report = EvalReport(k=k)

    for case in cases:
        answer, _plan = pipeline.answer(case.question, k=k)

        retrieved = [c.doc_id for c in answer.citations]
        relevant = set(case.relevant_doc_ids)

        # dedupe while keeping order, since one document can supply several chunks
        seen, unique = set(), []
        for d in retrieved:
            if d not in seen:
                seen.add(d)
                unique.append(d)

        found = [d for d in unique if d in relevant]
        precision = len(found) / len(unique) if unique else 0.0
        recall = len(found) / len(relevant) if relevant else 0.0

        rr = 0.0
        for rank, d in enumerate(unique, start=1):
            if d in relevant:
                rr = 1.0 / rank
                break

        faith = None
        if score_answers and answer.citations:
            # score against the full retrieved chunks, not the truncated excerpts,
            # or long answers are penalised for words that were in fact grounded
            context = answer.context_texts or [c.excerpt for c in answer.citations]
            faith = faithfulness_score(answer.text, context)

        report.cases.append(
            CaseResult(
                question=case.question,
                retrieved_doc_ids=unique,
                relevant_doc_ids=list(relevant),
                precision_at_k=round(precision, 4),
                recall_at_k=round(recall, 4),
                reciprocal_rank=round(rr, 4),
                hit=bool(found),
                faithfulness=faith,
                answer=answer.text[:400],
            )
        )

    logger.info("evaluated %d cases: %s", report.n, report.summary())
    return report


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """
    Normalised discounted cumulative gain. Unlike precision it rewards putting the
    right passage first rather than merely somewhere in the top k, which is what
    actually matters when the generator only reads the first few.
    """
    rel = set(relevant)
    dcg = sum(
        (1.0 / math.log2(i + 2)) for i, d in enumerate(retrieved[:k]) if d in rel
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return round(dcg / ideal, 4) if ideal else 0.0
