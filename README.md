# GenAI Data Integration Service

A retrieval service that pulls documents from several upstream systems, indexes
them into a vector store, answers questions over them with citations, and scores
its own retrieval and answer quality.

Built around the part of RAG that is usually missing from demos: knowing whether
it actually works. The `/evaluate` endpoint scores retrieval and faithfulness
separately, because they fail for different reasons and need different fixes.

```
sources ──> ingestion ──> vector store ──> retrieval agent ──> generation
 files          chunk        in-memory        route             claude
 mongo          embed        or pgvector      expand            or extractive
 inline         upsert                        fuse (RRF)        + citations
                                                    │
                                                    └──> evaluation
                                                         precision, recall,
                                                         MRR, faithfulness
```

## Running it

Nothing external is required. With no configuration the service runs fully
offline: in-memory index, LSA embeddings, extractive answers, no API key, no cost.

```bash
pip install -r requirements.txt
make run                 # http://localhost:8080/docs
```

In another shell:

```bash
curl -s localhost:8080/health

curl -s -X POST localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How long are client transaction records retained?"}'
```

Run the whole thing end to end, including evaluation:

```bash
make test    # 30 unit tests
make smoke   # every endpoint against the real app
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Resolved configuration and index size |
| POST | `/ingest` | Pull from configured sources, or post documents inline |
| POST | `/query` | Retrieve and answer, with citations and the retrieval plan |
| POST | `/retrieve` | Retrieval only, no generation, for judging relevance alone |
| POST | `/evaluate` | Score labelled questions for retrieval and faithfulness |

`/query` returns the retrieval plan alongside the answer. When an answer looks
wrong, the first question is always whether retrieval or generation caused it,
and returning the plan answers that without a second round trip.

## Design decisions

**Every external dependency is optional, and the service says which it resolved to.**
Postgres, Mongo and Anthropic are all swapped for working fallbacks when absent.
`/health` reports what is actually running rather than what was configured:

```json
{"resolved": {"embedder": "lsa", "store": "memory", "generator": "extractive"}}
```

This is not just convenience. It means the test suite and CI exercise the real
code paths with no services and no secrets, and an upstream outage degrades
answer quality instead of returning 500s.

**Citations are attached by the service, not requested from the model.**
Every citation corresponds to a chunk that was actually retrieved, so a citation
cannot be hallucinated even if the answer text is wrong.

**Retrieval fuses several query phrasings.** The agent expands the question into
variants and combines their rankings with reciprocal rank fusion, so a passage
ranked decently by several phrasings beats one ranked top by a single lucky
phrasing.

**Source routing widens rather than dead-ends.** Routing is a heuristic over query
wording, so it can send a query at a source holding nothing relevant. When that
happens the agent retries across all sources and records the change in the plan.
An explicit `sources` filter from the caller is treated as an instruction and is
never overridden.

**Chunking is sentence-aware with overlap.** A fact that straddles a chunk
boundary is otherwise retrievable from neither side of it.

**Refitting the embedder rebuilds the index.** LSA derives its vector space from
the corpus, so ingesting new documents changes that space. Vectors from two
different spaces are not comparable, so the index is rebuilt rather than silently
mixed. A pretrained neural embedder has no such constraint and skips this path.

## Configuration

Copy `.env.example` and set only what you need.

| Variable | Unset behaviour |
|---|---|
| `DATABASE_URL` | In-memory vector store |
| `MONGO_URI` | Mongo connector is not registered |
| `ANTHROPIC_API_KEY` | Extractive answers, no model call |
| `EMBED_PROVIDER` | `auto`: neural if available, else LSA |

With the backing services:

```bash
make deps-up     # postgres with pgvector, and mongo
export DATABASE_URL=postgresql://rag:rag@localhost:5432/rag
export MONGO_URI=mongodb://localhost:27017
make run
```

## Evaluation

Retrieval and generation are scored separately.

- **Retrieval**: precision@k, recall@k, MRR, hit rate, nDCG. Fixes live in
  chunking, embeddings, routing and k.
- **Faithfulness**: how much of the answer's content appears in the retrieved
  context. Fixes live in the prompt, the model and the context budget.

Faithfulness is lexical overlap, not a model grading itself. That is a weaker
signal than an LLM judge, but it is deterministic, free, and cannot be gamed by
the same model that produced the answer, which makes it usable as a CI gate.
It catches an answer inventing material that was never retrieved. It will not
catch an answer that reuses the context vocabulary but misstates the
relationship.

Current numbers on the sample corpus, from `make smoke`:

```
cases 8, k=3
recall@3      1.00
mrr           1.00
hit_rate      1.00
faithfulness  0.95
precision@3   0.40
```

Precision is low by construction, not by failure: each question has exactly one
relevant document and k is 3, so the ceiling is roughly 0.33 per case.
Recall and MRR are the meaningful numbers on this set.

## Deployment

`infra/terraform/` provisions Artifact Registry, a dedicated runtime service
account, Secret Manager for the API key, and a Cloud Run service with a startup
probe on `/health`. Scaling starts at zero instances so idle cost is nothing.

```bash
cd infra/terraform
terraform init
terraform apply -var="project_id=YOUR_PROJECT"
```

**Status: written and validated locally, not deployed to a live account.** The
Terraform has not been applied against real GCP infrastructure.

## Layout

```
app/
  main.py         FastAPI app and schemas
  config.py       settings and component resolution
  embeddings.py   LSA and sentence-transformers behind one interface
  ingest.py       chunking and the ingestion pipeline
  agent.py        routing, query expansion, rank fusion
  rag.py          generation and citation attachment
  evaluation.py   retrieval metrics and faithfulness
  sources/        file, mongo and inline connectors
  stores/         in-memory and pgvector implementations
tests/            30 unit tests
scripts/smoke.py  end-to-end endpoint exercise
infra/terraform/  Cloud Run, Artifact Registry, IAM, Secret Manager
```

## What is not built

Named because a RAG service that claims completeness usually is not.

- **Reranking.** A cross-encoder over the fused candidates would raise precision
  meaningfully. Retrieval currently ends at rank fusion.
- **Streaming responses.** Answers are returned whole.
- **Incremental index updates on pgvector.** The ANN index is built on demand
  rather than maintained.
- **Real Starburst, BigQuery or Hadoop connectors.** The source interface is
  built for them and adding one is a single class, but only file, Mongo and
  inline exist today.
- **An LLM judge in evaluation.** Faithfulness is lexical only.
