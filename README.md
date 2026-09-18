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
                                                    ├──> rerank (optional)
                                                    │    cross-encoder over
                                                    │    the candidate pool
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
make test    # 64 unit tests
make smoke   # every endpoint against the real app
```

Reranking is optional and off unless a checkpoint has passed the promotion gate:

```bash
make install-training    # adds torch
make train               # trains, measures, and gates the reranker
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
{"resolved": {"embedder": "lsa", "store": "memory",
              "generator": "extractive", "reranker": "none"}}
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
| `RERANK_PROVIDER` | `auto`: local checkpoint, else hub model, else no reranking |
| `RERANK_CANDIDATES` | Pool size shown to the reranker, default 20 |
| `RERANK_MODEL_PATH` | `models/reranker.pt` |

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

The labelled set is 65 hand-written questions over 24 documents, in
`data/eval/questions.json`. Each question has exactly one relevant document.
Nothing in the training pipeline ever reads this file as training data.

Current numbers, from `make train`:

```
cases 65, k=5
recall@5      0.9231
mrr           0.8397
ndcg@5        0.8612
precision@5   0.2882
rank_1        0.7692
faithfulness  0.9622
```

Precision is low by construction, not by failure: each question has exactly one
relevant document and k is 5, so the ceiling is 0.20 per case. Recall, MRR and
rank_1 are the meaningful numbers here.

The set was rebuilt during the reranking work. The previous one, 8 questions over
4 documents, scored recall 1.00 and MRR 1.00, which meant no change to retrieval
could be measured on it at all. An evaluation set that everything passes is not
evidence that the system is good, it is evidence that the set is exhausted.

## Reranking

Rank fusion scores a passage without ever letting it meet the query: the query
and the passage are embedded separately and compared as vectors. A cross-encoder
reads the concatenation, so every query token can attend to every passage token.
It is more expressive and far more expensive, which is why it reorders a short
candidate list rather than searching the index.

The agent widens its pool to `RERANK_CANDIDATES` fused candidates, the reranker
scores each pair, and the top k survive. Reranking only the final k would just
reorder what was already returned; the gain has to come from reaching passages
fusion ranked below the cut.

Three backends behind one interface, resolved like everything else here:

| Backend | When it is used |
|---|---|
| `none` | No checkpoint, or the checkpoint failed its gate. Fusion order stands. |
| `cross-encoder-local` | A promoted checkpoint in `models/`. No download, no key. |
| `cross-encoder-hf` | A pretrained cross-encoder, when its weights are reachable. |

Both scores survive on every citation. `score` is cosine similarity from
retrieval, `rerank_score` is the cross-encoder's. When a reranked answer is
wrong, the question is whether retrieval missed the passage or the reranker
buried it, and that is unanswerable if the second score overwrote the first.

### The model in this repository is worse than no model

It is checked in as machinery and as a measurement, not as an improvement. The
numbers on the held-out questions:

```
metric           fusion only    reranked     delta
--------------------------------------------------
recall@5              0.9231      0.8000   -0.1231
mrr                   0.8397      0.4590   -0.3807
ndcg@5                0.8612      0.5439   -0.3173
precision@5           0.2882      0.2426   -0.0456
rank_1                0.7692      0.2615   -0.5077

candidate pool recall@20 (ceiling for any reranker): 0.9692

latency          fusion only    reranked
----------------------------------------
p50_ms                  0.76       45.83
p95_ms                  1.48      100.63
p99_ms                  1.58      112.93
```

The cause is not a bug. A 1.16M-parameter transformer trained from scratch on a
3,900-word corpus cannot learn generalisable relevance. Training accuracy climbs
to 0.71 while dev accuracy stays at chance and dev loss rises, which is
memorisation with nothing underneath it. This is the reason pretrained
cross-encoders exist, and measuring it is more useful than asserting it.

So the gate refuses to install it:

```
gate: mrr delta -0.3807 < +0.0000; NOT promoted
      candidate kept at models/reranker_candidate.pt for inspection
      the service will resolve its reranker to 'none' and serve fusion order
```

Training writes a candidate; promotion is a separate decision taken on held-out
evidence. A reranker that ranks worse than the retrieval it reorders is not a
smaller win, it is a regression that also costs two orders of magnitude in
latency, and from outside it looks identical to one that works. Leaving that to
whoever reads the table makes the failure mode "someone forgot to look".

To get a reranker that does help, point it at pretrained weights:

```bash
pip install transformers
RERANK_PROVIDER=hf make run
```

## Training the reranker

Two stages, `scripts/train_reranker.py`, staged so each can be run alone.

**Stage 1, base.** The whole model trains on groups whose negatives are random
passages. That is the cheap part of the problem, telling an expenses passage
from an incident passage, and doing it first means the adapters are not spent
relearning it.

**Stage 2, adaptation.** The base is frozen, LoRA adapters are injected into the
attention projections, and only those and the scoring head train, on groups whose
negatives were mined from the real retriever. This stage trains 18,625 of
1,160,641 parameters, 1.605%.

LoRA is implemented in `app/lora.py` rather than imported. `LoRALinear` wraps a
frozen `nn.Linear` and learns `(alpha / r) * B A x` alongside it, with `B`
initialised to zero so the adapted model starts numerically identical to the one
it wraps. Injection matches target layers by name substring, which is what lets
the same code adapt the encoder in this repository and a pretrained BERT-family
cross-encoder without branching on either.

The objective is listwise. Each group of one positive and n negatives is scored
together, softmaxed and trained with cross-entropy against the positive, because
the model is used to order a candidate list rather than to classify one passage.

**Training queries never come from the evaluation set.** They are generated from
the corpus by inverse cloze: a sentence is taken out of a chunk and used as the
query, and the chunk it came from, minus that sentence, is the positive. The 65
hand-written questions are therefore held out completely.

**Negatives are mined from the live retriever.** Random negatives are nearly free
to separate, so a model trained on them learns topic matching, which the
bi-encoder already does, and contributes nothing on a candidate list where every
entry is already topically plausible. Mining drops the serving score floor, since
a passage sitting just under it is exactly the near-miss worth training against.

## Bugs found while building this

Kept because each one cost real time and each one had a cause worth naming.

**A reranker window that did not match its checkpoint.** `TorchReranker`
defaulted to a 256-token window while the checkpoint had been trained at 192, so
every forward pass indexed past the end of the positional embedding table and
raised. The window is a property of the checkpoint, not of the serving class, and
is now read from it.

**The fallback hid that failure completely.** The agent catches a raising
reranker and keeps fusion order, which is right for a live request and wrong for
a measurement. Every metric came back at exactly 0.0000 delta, which reads as
"the reranker changed nothing" rather than "the reranker never ran once". The
comparison now counts fallbacks and refuses to report a result built on them. The
identical-to-four-decimal-places deltas were the only clue, and a less suspicious
number would have shipped.

**The training construction leaked passage length.** Positives were built by
cutting the query sentence out of a chunk, so they were systematically shorter
than untouched negatives: 523 characters against 687. Picking the shortest
candidate alone scored 0.412 on a five-way task where chance is 0.200, so the
model could do well by measuring length and never reading the query. It was
invisible in training, where dev accuracy reached 0.96, and only showed up as a
collapse on real questions. Negatives now undergo the same sentence deletion,
which brings the shortcut down to 0.244. Removing it also removed the apparent
learning, which is what exposed the real finding above.

**Injecting adapters into nothing used to be silent.** A name-matched injection
that matches no layer still trains, still produces a loss curve and still writes
a checkpoint, having adapted nothing at all. `inject_lora` now raises and lists
the layer names it could see.

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
  main.py           FastAPI app and schemas
  config.py         settings and component resolution
  embeddings.py     LSA and sentence-transformers behind one interface
  ingest.py         chunking and the ingestion pipeline
  agent.py          routing, query expansion, rank fusion, reranking
  rag.py            generation and citation attachment
  evaluation.py     retrieval metrics and faithfulness
  rerank.py         reranker interface, local and hub backends
  crossencoder.py   from-scratch transformer cross-encoder and tokenizer
  lora.py           low-rank adaptation, injection and checkpointing
  sources/          file, mongo and inline connectors
  stores/           in-memory and pgvector implementations
training/
  common.py         index construction shared by training and evaluation
  data.py           inverse-cloze queries and hard-negative mining
  train.py          two-stage training loop
  evaluate.py       before-and-after comparison, latency, promotion gate
data/
  sample_docs/      24 internal-policy documents
  eval/             65 held-out labelled questions
tests/              64 unit tests
scripts/smoke.py    end-to-end endpoint exercise
scripts/train_reranker.py  train, measure, gate
infra/terraform/    Cloud Run, Artifact Registry, IAM, Secret Manager
```

## What is not built

Named because a RAG service that claims completeness usually is not.

- **A reranker good enough to switch on.** The machinery is built, measured and
  gated. The model that came out of it is worse than the retrieval it reorders,
  so the gate refuses it and the service serves fusion order. See "Reranking"
  above for the numbers and why.
- **Streaming responses.** Answers are returned whole.
- **Incremental index updates on pgvector.** The ANN index is built on demand
  rather than maintained.
- **Real Starburst, BigQuery or Hadoop connectors.** The source interface is
  built for them and adding one is a single class, but only file, Mongo and
  inline exist today.
- **An LLM judge in evaluation.** Faithfulness is lexical only.
