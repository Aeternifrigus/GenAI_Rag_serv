"""
Retrieval agent.

Not a wrapper around a single similarity search. It decides *how* to retrieve:

  1. routes the query to the sources likely to hold the answer
  2. expands the query into variants so retrieval does not hinge on one phrasing
  3. runs the variants and fuses their rankings
  4. drops weak hits so the generator is not handed noise

Routing is rule-based by default and LLM-assisted when a key is present. The
rule path exists so the agent is deterministic, testable, and free to run, which
matters more for an evaluation harness than marginal routing quality.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .embeddings import Embedder
from .stores.base import Hit, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalPlan:
    """What the agent decided to do, exposed so its reasoning is auditable."""

    query: str
    variants: list[str] = field(default_factory=list)
    sources: list[str] | None = None
    reason: str = ""


@dataclass
class RetrievalResult:
    plan: RetrievalPlan
    hits: list[Hit]


# words that hint a query belongs to a particular upstream system
SOURCE_HINTS: dict[str, tuple[str, ...]] = {
    "mongo": ("ticket", "customer", "record", "account", "order", "case"),
    "files": ("policy", "handbook", "guide", "documentation", "procedure", "runbook"),
}

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "what", "how", "why", "when", "which", "who", "does", "do",
    "can", "should", "would", "with", "that", "this", "it", "as", "at", "by",
}


class RetrievalAgent:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        k: int = 5,
        min_score: float = 0.05,
        use_llm_routing: bool = False,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.k = k
        self.min_score = min_score
        self.use_llm_routing = use_llm_routing

    # ── planning ────────────────────────────────────────────────

    def _route(self, query: str) -> tuple[list[str] | None, str]:
        """Pick sources to search. None means search everything."""
        q = query.lower()
        picked = [
            src for src, words in SOURCE_HINTS.items()
            if any(w in q for w in words)
        ]
        if picked:
            return picked, f"query mentions terms associated with {', '.join(picked)}"
        return None, "no source-specific signal; searching all sources"

    def _expand(self, query: str) -> list[str]:
        """
        Cheap deterministic query expansion. A keyword-only variant helps when the
        question is wordy and the document is terse, which single-vector retrieval
        otherwise handles badly.
        """
        variants = [query]
        keywords = [
            w for w in re.findall(r"[a-z0-9]+", query.lower())
            if w not in _STOP and len(w) > 2
        ]
        if keywords and len(keywords) < len(query.split()):
            kw = " ".join(keywords)
            if kw != query.lower():
                variants.append(kw)
        return variants

    def plan(self, query: str, sources: Sequence[str] | None = None) -> RetrievalPlan:
        if sources:
            chosen, reason = list(sources), "sources supplied by caller"
        else:
            chosen, reason = self._route(query)
        return RetrievalPlan(
            query=query,
            variants=self._expand(query),
            sources=chosen,
            reason=reason,
        )

    # ── execution ───────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        sources: Sequence[str] | None = None,
    ) -> RetrievalResult:
        k = k or self.k
        plan = self.plan(query, sources)

        if self.store.count() == 0:
            return RetrievalResult(plan=plan, hits=[])

        hits = self._search(plan.variants, k, plan.sources)

        # Routing is a heuristic over query wording, so it can send a query at a
        # source that holds nothing relevant. When the caller did not pin the
        # sources themselves, widen to everything rather than return a dead end.
        if not hits and plan.sources and not sources:
            logger.info(
                "no hits in routed sources %s; widening to all sources", plan.sources
            )
            plan.sources = None
            plan.reason += "; routed sources returned nothing, widened to all"
            hits = self._search(plan.variants, k, None)

        logger.info(
            "retrieved %d hits for %r (sources=%s, variants=%d)",
            len(hits), query, plan.sources or "all", len(plan.variants),
        )
        return RetrievalResult(plan=plan, hits=hits)

    def _search(
        self,
        variants: Sequence[str],
        k: int,
        sources: Sequence[str] | None,
    ) -> list[Hit]:
        vectors = self.embedder.encode(list(variants))

        # reciprocal rank fusion across variants: a chunk ranked decently by
        # several phrasings beats one ranked top by a single lucky phrasing.
        fused: dict[str, tuple[Hit, float]] = {}
        for vec in vectors:
            for rank, hit in enumerate(self.store.search(vec, k=k * 2, sources=sources)):
                contribution = 1.0 / (60 + rank)     # 60 is the standard RRF constant
                if hit.chunk.id in fused:
                    existing, score = fused[hit.chunk.id]
                    best = existing if existing.score >= hit.score else hit
                    fused[hit.chunk.id] = (best, score + contribution)
                else:
                    fused[hit.chunk.id] = (hit, contribution)

        ranked = sorted(fused.values(), key=lambda pair: -pair[1])
        return [hit for hit, _ in ranked if hit.score >= self.min_score][:k]
