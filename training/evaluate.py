"""
Measuring whether the reranker earned its place.

Two questions, answered separately, because a reranker that improves ranking and
triples tail latency is not obviously an improvement and the decision belongs to
whoever is paying for the latency.

  Quality  recall@k, MRR and nDCG@k on the held-out question set, with the
           reranker attached and detached, same index, same queries, same k.

  Cost     end-to-end retrieval latency percentiles for both arms. p50 says what
           a typical request costs. p95 and p99 say what the slowest requests
           cost, which is what a timeout is set against.

A candidate-pool ceiling is reported alongside. A reranker can only reorder what
retrieval handed it, so recall over the whole pool is the highest score any
reranker could reach here, and comparing against it says whether the remaining
gap is a reranking problem or a retrieval problem.
"""
from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

from app.evaluation import ndcg_at_k

from .common import IndexedCorpus

logger = logging.getLogger(__name__)


def _rank_of(doc_ids: Sequence[str], target: str) -> int | None:
    for i, d in enumerate(doc_ids, start=1):
        if d == target:
            return i
    return None


def _dedupe(doc_ids: Sequence[str]) -> list[str]:
    """One document may supply several chunks; rank it once, at its best rank."""
    seen: set[str] = set()
    out: list[str] = []
    for d in doc_ids:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def retrieval_metrics(
    corpus: IndexedCorpus,
    cases: Sequence[dict],
    reranker=None,
    k: int = 5,
    candidate_k: int = 20,
) -> dict:
    """Score the held-out questions with the given reranker attached."""
    agent = corpus.agent
    previous, previous_pool = agent.reranker, agent.candidate_k
    agent.reranker, agent.candidate_k = reranker, candidate_k

    try:
        recalls, rrs, ndcgs, precisions = [], [], [], []
        misses: list[str] = []
        failures = 0

        for case in cases:
            result = agent.retrieve(case["question"], k=k)
            if "failed" in result.plan.reranker:
                failures += 1
            hits = result.hits
            docs = _dedupe([h.chunk.doc_id for h in hits])

            rank = _rank_of(docs, case["doc_id"])
            recalls.append(1.0 if rank else 0.0)
            rrs.append(1.0 / rank if rank else 0.0)
            precisions.append((1.0 / len(docs)) if rank else 0.0)
            ndcgs.append(ndcg_at_k(docs, [case["doc_id"]], k))
            if not rank:
                misses.append(case["question"])

        n = len(cases)
        return {
            "cases": n,
            "k": k,
            f"recall@{k}": round(sum(recalls) / n, 4),
            "mrr": round(sum(rrs) / n, 4),
            f"ndcg@{k}": round(sum(ndcgs) / n, 4),
            f"precision@{k}": round(sum(precisions) / n, 4),
            "rank_1": round(sum(1 for r in rrs if r == 1.0) / n, 4),
            "misses": misses,
            "rerank_failures": failures,
        }
    finally:
        agent.reranker, agent.candidate_k = previous, previous_pool


def candidate_ceiling(
    corpus: IndexedCorpus, cases: Sequence[dict], candidate_k: int = 20
) -> dict:
    """Recall over the whole candidate pool: the best any reranker could do."""
    agent = corpus.agent
    previous = agent.reranker
    agent.reranker = None
    try:
        found = 0
        for case in cases:
            docs = {h.chunk.doc_id for h in agent.retrieve(case["question"], k=candidate_k).hits}
            found += int(case["doc_id"] in docs)
        return {
            "candidate_k": candidate_k,
            "pool_recall": round(found / len(cases), 4),
        }
    finally:
        agent.reranker = previous


def latency_profile(
    corpus: IndexedCorpus,
    cases: Sequence[dict],
    reranker=None,
    k: int = 5,
    candidate_k: int = 20,
    repeats: int = 3,
) -> dict:
    """
    Wall-clock latency of the full retrieve path, percentiles over every run.

    Timed around the agent rather than around the model, because the number that
    matters is what the request costs, not what the forward pass costs. One
    untimed warm-up pass runs first so lazy imports and the first allocation do
    not land in the distribution.
    """
    agent = corpus.agent
    previous, previous_pool = agent.reranker, agent.candidate_k
    agent.reranker, agent.candidate_k = reranker, candidate_k

    try:
        agent.retrieve(cases[0]["question"], k=k)

        samples: list[float] = []
        for _ in range(repeats):
            for case in cases:
                start = time.perf_counter()
                agent.retrieve(case["question"], k=k)
                samples.append((time.perf_counter() - start) * 1000.0)

        samples.sort()

        def pct(p: float) -> float:
            idx = min(len(samples) - 1, int(round(p * (len(samples) - 1))))
            return round(samples[idx], 2)

        return {
            "samples": len(samples),
            "mean_ms": round(statistics.fmean(samples), 2),
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
            "p99_ms": pct(0.99),
            "max_ms": round(samples[-1], 2),
        }
    finally:
        agent.reranker, agent.candidate_k = previous, previous_pool


def compare(
    corpus: IndexedCorpus,
    cases: Sequence[dict],
    reranker,
    k: int = 5,
    candidate_k: int = 20,
    latency_repeats: int = 3,
) -> dict:
    """Both arms, both dimensions, one report."""
    baseline = retrieval_metrics(corpus, cases, reranker=None, k=k)
    reranked = retrieval_metrics(corpus, cases, reranker=reranker, k=k, candidate_k=candidate_k)

    # The agent degrades to fusion order when a reranker raises, which is right
    # for a live request and wrong for a measurement: every metric then reads as
    # "the reranker changed nothing" when in fact it never ran. Refuse to report
    # a comparison built on that.
    if reranked["rerank_failures"]:
        raise RuntimeError(
            f"{reranked['rerank_failures']} of {len(cases)} queries fell back to fusion "
            "order because the reranker raised; the comparison would be meaningless. "
            "Check the warning logged by RetrievalAgent.retrieve for the cause."
        )

    ceiling = candidate_ceiling(corpus, cases, candidate_k=candidate_k)

    deltas = {
        key: round(reranked[key] - baseline[key], 4)
        for key in baseline
        if isinstance(baseline[key], int | float)
        and key not in ("cases", "k", "rerank_failures")
    }

    return {
        "baseline": baseline,
        "reranked": reranked,
        "delta": deltas,
        "ceiling": ceiling,
        "latency": {
            "baseline": latency_profile(
                corpus, cases, reranker=None, k=k, repeats=latency_repeats
            ),
            "reranked": latency_profile(
                corpus, cases, reranker=reranker, k=k,
                candidate_k=candidate_k, repeats=latency_repeats,
            ),
        },
    }


def format_comparison(report: dict) -> str:
    base, rer, delta = report["baseline"], report["reranked"], report["delta"]
    k = base["k"]
    keys = [f"recall@{k}", "mrr", f"ndcg@{k}", f"precision@{k}", "rank_1"]

    lines = [
        f"{'metric':<14}{'fusion only':>14}{'reranked':>12}{'delta':>10}",
        "-" * 50,
    ]
    for key in keys:
        lines.append(f"{key:<14}{base[key]:>14.4f}{rer[key]:>12.4f}{delta[key]:>+10.4f}")

    lines += [
        "",
        f"candidate pool recall@{report['ceiling']['candidate_k']} "
        f"(ceiling for any reranker): {report['ceiling']['pool_recall']:.4f}",
        "",
        f"{'latency':<14}{'fusion only':>14}{'reranked':>12}",
        "-" * 40,
    ]
    lb, lr = report["latency"]["baseline"], report["latency"]["reranked"]
    for key in ("p50_ms", "p95_ms", "p99_ms"):
        lines.append(f"{key:<14}{lb[key]:>14.2f}{lr[key]:>12.2f}")

    return "\n".join(lines)


def promote_if_better(
    comparison: dict,
    candidate_path: Path,
    serving_path: Path,
    metric: str = "mrr",
    min_delta: float = 0.0,
    force: bool = False,
) -> dict:
    """
    Install the candidate as the served model only if it beat fusion-only.

    A reranker that ranks worse than the retrieval it reorders is not a smaller
    win, it is a regression that also costs latency, and it looks identical to a
    working one from the outside. Leaving the decision to whoever runs the script
    means the failure mode is someone forgetting to read a table. The gate is the
    control; the table is the evidence.
    """
    delta = comparison["delta"].get(metric)
    if delta is None:
        raise ValueError(f"{metric!r} is not a metric in this comparison")

    passed = delta >= min_delta
    result = {
        "metric": metric,
        "delta": delta,
        "min_delta": min_delta,
        "passed": passed,
        "forced": bool(force and not passed),
    }

    if passed or force:
        if candidate_path.exists():
            serving_path.write_bytes(candidate_path.read_bytes())
        result["promoted"] = True
        state = "promoted" if passed else "promoted BY FORCE despite failing the gate"
        print(f"\ngate: {metric} delta {delta:+.4f} >= {min_delta:+.4f} is {passed}; {state}")
        print(f"      serving checkpoint: {serving_path}")
    else:
        # An older passing checkpoint stays exactly where it is. A failed
        # candidate must not replace a model that is currently working.
        serving_path.unlink(missing_ok=True)
        result["promoted"] = False
        print(f"\ngate: {metric} delta {delta:+.4f} < {min_delta:+.4f}; NOT promoted")
        print(f"      candidate kept at {candidate_path} for inspection")
        print("      the service will resolve its reranker to 'none' and serve fusion order")

    return result
