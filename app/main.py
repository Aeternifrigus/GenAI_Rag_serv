"""
FastAPI service.

Endpoints:
  GET  /health     resolved configuration and index size
  POST /ingest     pull from configured sources, or accept inline documents
  POST /query      retrieve and answer, with citations and the retrieval plan
  POST /retrieve   retrieval only, no generation, for debugging relevance
  POST /evaluate   score a set of labelled questions

The retrieval plan is returned on /query on purpose. When an answer looks wrong
the first question is always whether retrieval or generation caused it, and
returning the plan answers that without a second round trip.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import registry, settings
from .evaluation import EvalCase, evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.build()
    if settings.auto_ingest:
        try:
            report = registry.ingestion.run(registry.default_sources())
            logger.info("startup ingest: %d docs, %d chunks", report.documents, report.chunks)
        except Exception as exc:
            # an empty index is recoverable; a crashed service is not
            logger.warning("startup ingest failed (%s); service starting empty", exc)
    yield


app = FastAPI(
    title="GenAI Data Integration Service",
    description="Multi-source retrieval agent, RAG endpoint and evaluation harness.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── schemas ──────────────────────────────────────────────────────

class InlineDocument(BaseModel):
    text: str = Field(..., min_length=1)
    doc_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[InlineDocument] | None = Field(
        default=None,
        description="Ingest these directly. When omitted, pulls from configured sources.",
    )
    limit: int | None = Field(default=None, ge=1, description="Cap documents per source.")


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    per_source: dict[str, int]
    skipped_sources: list[str]
    index_size: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int | None = Field(default=None, ge=1, le=50)
    sources: list[str] | None = Field(
        default=None, description="Restrict to these sources. Omit to let the agent route."
    )


class CitationOut(BaseModel):
    marker: int
    chunk_id: str
    doc_id: str
    source: str
    score: float
    rerank_score: float | None = None
    excerpt: str
    metadata: dict[str, Any]


class PlanOut(BaseModel):
    query: str
    variants: list[str]
    sources: list[str] | None
    reason: str
    reranker: str = "none"
    candidates: int = 0


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    generator: str
    citations: list[CitationOut]
    plan: PlanOut


class RetrieveResponse(BaseModel):
    plan: PlanOut
    hits: list[CitationOut]


class EvalCaseIn(BaseModel):
    question: str
    relevant_doc_ids: list[str]


class EvaluateRequest(BaseModel):
    cases: list[EvalCaseIn] = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=50)


class EvaluateResponse(BaseModel):
    summary: dict[str, Any]
    per_case: list[dict[str, Any]]


# ── endpoints ────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "resolved": registry.resolved,
        "index_size": registry.store.count() if registry.store else 0,
        "embedder_fitted": getattr(registry.embedder, "fitted", True),
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    if req.documents:
        from .sources import InlineSource

        sources = [InlineSource([d.model_dump() for d in req.documents])]
    else:
        sources = registry.default_sources()

    try:
        report = registry.ingestion.run(sources, limit=req.limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(
        documents=report.documents,
        chunks=report.chunks,
        per_source=report.sources,
        skipped_sources=report.skipped,
        index_size=registry.store.count(),
    )


def _plan_out(plan) -> PlanOut:
    return PlanOut(
        query=plan.query,
        variants=plan.variants,
        sources=plan.sources,
        reason=plan.reason,
        reranker=plan.reranker,
        candidates=plan.candidates,
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if registry.store.count() == 0:
        raise HTTPException(
            status_code=409,
            detail="Index is empty. POST /ingest before querying.",
        )

    answer, plan = registry.pipeline.answer(req.question, k=req.k, sources=req.sources)
    return QueryResponse(
        answer=answer.text,
        grounded=answer.grounded,
        generator=answer.generator,
        citations=[CitationOut(**c.__dict__) for c in answer.citations],
        plan=_plan_out(plan),
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: QueryRequest) -> RetrieveResponse:
    """Retrieval without generation, for judging relevance in isolation."""
    if registry.store.count() == 0:
        raise HTTPException(status_code=409, detail="Index is empty. POST /ingest first.")

    from .rag import _build_citations

    result = registry.agent.retrieve(req.question, k=req.k, sources=req.sources)
    return RetrieveResponse(
        plan=_plan_out(result.plan),
        hits=[CitationOut(**c.__dict__) for c in _build_citations(result.hits)],
    )


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_endpoint(req: EvaluateRequest) -> EvaluateResponse:
    if registry.store.count() == 0:
        raise HTTPException(status_code=409, detail="Index is empty. POST /ingest first.")

    cases = [EvalCase(question=c.question, relevant_doc_ids=c.relevant_doc_ids) for c in req.cases]
    report = evaluate(registry.pipeline, cases, k=req.k)

    return EvaluateResponse(
        summary=report.summary(),
        per_case=[
            {
                "question": c.question,
                "retrieved": c.retrieved_doc_ids,
                "relevant": c.relevant_doc_ids,
                "precision": c.precision_at_k,
                "recall": c.recall_at_k,
                "reciprocal_rank": c.reciprocal_rank,
                "hit": c.hit,
                "faithfulness": c.faithfulness,
            }
            for c in report.cases
        ],
    )
