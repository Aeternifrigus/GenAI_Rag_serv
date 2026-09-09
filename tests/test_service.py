import numpy as np
import pytest

from app.agent import RetrievalAgent
from app.embeddings import LsaEmbedder
from app.evaluation import EvalCase, evaluate, faithfulness_score, ndcg_at_k
from app.ingest import IngestionPipeline, chunk_text, documents_to_chunks
from app.rag import ExtractiveGenerator, RagPipeline
from app.sources import FileSource, InlineSource, stable_id
from app.stores.base import Chunk
from app.stores.memory import InMemoryVectorStore

DOCS = [
    {"doc_id": "policy", "text":
        "Expense claims must be submitted within 30 days. Receipts are mandatory "
        "above 25 EUR. The daily allowance for international travel is 65 EUR."},
    {"doc_id": "runbook", "text":
        "Severity one incidents mean total unavailability. The on-call engineer "
        "acknowledges within 15 minutes. Rollback is preferred over a forward fix."},
    {"doc_id": "retention", "text":
        "Client transaction records are retained for seven years. Application logs "
        "with identifiers are kept for 90 days before deletion."},
]


# ── chunking ────────────────────────────────────────────────────

def test_short_text_is_one_chunk():
    assert chunk_text("A short sentence.", size=900) == ["A short sentence."]


def test_empty_text_yields_nothing():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_long_text_splits_and_respects_size():
    text = " ".join(f"Sentence number {i} carries some filler content." for i in range(200))
    chunks = chunk_text(text, size=400, overlap=80)
    assert len(chunks) > 1
    # overlap can push a chunk slightly past the target, but not unboundedly
    assert all(len(c) <= 400 + 80 for c in chunks)


def test_oversized_single_sentence_is_hard_split():
    text = "word" * 800  # one token, far longer than the window
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 1


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("some text", size=100, overlap=100)


def test_chunk_ids_are_stable_across_runs():
    docs = list(InlineSource(DOCS).fetch())
    a = documents_to_chunks(docs)
    b = documents_to_chunks(docs)
    assert [c.id for c in a] == [c.id for c in b]


# ── store ───────────────────────────────────────────────────────

def test_store_upsert_and_search():
    store = InMemoryVectorStore()
    chunks = [
        Chunk(id="1", text="alpha", source="files", doc_id="d1"),
        Chunk(id="2", text="beta", source="mongo", doc_id="d2"),
    ]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert store.upsert(chunks, vecs) == 2
    assert store.count() == 2

    hits = store.search(np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert hits[0].chunk.id == "1"
    assert hits[0].score > hits[1].score


def test_store_filters_by_source():
    store = InMemoryVectorStore()
    store.upsert(
        [Chunk(id="1", text="a", source="files", doc_id="d1"),
         Chunk(id="2", text="b", source="mongo", doc_id="d2")],
        np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
    )
    hits = store.search(np.array([1.0, 0.0], dtype=np.float32), k=5, sources=["mongo"])
    assert len(hits) == 1
    assert hits[0].chunk.source == "mongo"


def test_reingesting_same_chunk_updates_not_duplicates():
    store = InMemoryVectorStore()
    v = np.array([[1.0, 0.0]], dtype=np.float32)
    store.upsert([Chunk(id="1", text="first", source="files", doc_id="d1")], v)
    store.upsert([Chunk(id="1", text="second", source="files", doc_id="d1")], v)
    assert store.count() == 1


def test_dimension_mismatch_is_explicit():
    store = InMemoryVectorStore()
    store.upsert(
        [Chunk(id="1", text="a", source="files", doc_id="d1")],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="does not match index dim"):
        store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_upsert_rejects_mismatched_counts():
    store = InMemoryVectorStore()
    with pytest.raises(ValueError, match="chunk/vector mismatch"):
        store.upsert(
            [Chunk(id="1", text="a", source="files", doc_id="d1")],
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )


def test_search_on_empty_store_returns_empty():
    assert InMemoryVectorStore().search(np.array([1.0, 0.0], dtype=np.float32)) == []


# ── embedder ────────────────────────────────────────────────────

def test_embedder_requires_fit_before_encode():
    with pytest.raises(RuntimeError, match="before fit"):
        LsaEmbedder().encode(["anything"])


def test_embedder_produces_unit_vectors():
    emb = LsaEmbedder(dim=8)
    corpus = [d["text"] for d in DOCS]
    emb.fit(corpus)
    vecs = emb.encode(corpus)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embedder_rejects_empty_corpus():
    with pytest.raises(ValueError):
        LsaEmbedder().fit([])


# ── pipeline fixture ────────────────────────────────────────────

@pytest.fixture
def pipeline():
    store = InMemoryVectorStore()
    embedder = LsaEmbedder(dim=64)
    ingestion = IngestionPipeline(store, embedder, chunk_size=400, chunk_overlap=60)
    ingestion.run([InlineSource(DOCS)])
    agent = RetrievalAgent(store, embedder, k=3, min_score=0.0)
    return RagPipeline(agent, ExtractiveGenerator()), store


def test_ingestion_indexes_documents(pipeline):
    _, store = pipeline
    assert store.count() > 0


def test_query_returns_grounded_answer_with_citations(pipeline):
    rag, _ = pipeline
    answer, plan = rag.answer("How long are transaction records retained?")
    assert answer.grounded
    assert answer.citations
    assert plan.variants


def test_citation_markers_are_sequential(pipeline):
    rag, _ = pipeline
    answer, _ = rag.answer("expense claim deadline")
    assert [c.marker for c in answer.citations] == list(range(1, len(answer.citations) + 1))


def test_retrieval_finds_the_right_document(pipeline):
    rag, _ = pipeline
    answer, _ = rag.answer("How long are client transaction records kept?")
    assert "retention" in {c.doc_id for c in answer.citations}


def test_source_filter_is_honoured(pipeline):
    rag, _ = pipeline
    answer, plan = rag.answer("anything at all", sources=["inline"])
    assert plan.sources == ["inline"]
    assert all(c.source == "inline" for c in answer.citations)


def test_nonexistent_source_returns_nothing(pipeline):
    rag, _ = pipeline
    answer, _ = rag.answer("transaction records", sources=["does-not-exist"])
    assert answer.citations == []
    assert not answer.grounded


def test_routing_widens_when_routed_source_is_empty(pipeline):
    """
    'records' triggers the mongo routing hint, but this index only holds inline
    documents. The agent should widen rather than return a dead end.
    """
    rag, _ = pipeline
    answer, plan = rag.answer("How long are transaction records retained?")
    assert answer.grounded, "agent should widen past an empty routed source"
    assert plan.sources is None
    assert "widened" in plan.reason


def test_caller_pinned_sources_are_not_widened(pipeline):
    """An explicit source filter is an instruction, not a hint. Never override it."""
    rag, _ = pipeline
    answer, plan = rag.answer("transaction records", sources=["does-not-exist"])
    assert plan.sources == ["does-not-exist"]
    assert "widened" not in plan.reason
    assert answer.citations == []


# ── file source ─────────────────────────────────────────────────

def test_file_source_reads_the_sample_corpus():
    src = FileSource("data/sample_docs")
    assert src.healthy()
    docs = list(src.fetch())
    assert len(docs) >= 4
    assert all(d.source == "files" and d.text for d in docs)


def test_missing_directory_is_unhealthy_not_fatal():
    src = FileSource("data/definitely_not_here")
    assert not src.healthy()
    assert list(src.fetch()) == []


def test_stable_id_is_deterministic():
    assert stable_id("a", "b") == stable_id("a", "b")
    assert stable_id("a", "b") != stable_id("b", "a")


# ── evaluation ──────────────────────────────────────────────────

def test_faithfulness_rewards_grounded_text():
    ctx = ["Records are retained for seven years under regulation."]
    grounded = faithfulness_score("Records retained seven years", ctx)
    invented = faithfulness_score("Bananas orbit Jupiter quarterly", ctx)
    assert grounded > invented


def test_faithfulness_handles_empty_input():
    assert faithfulness_score("", ["something"]) == 0.0
    assert faithfulness_score("something", []) == 0.0


def test_ndcg_rewards_ranking_correct_doc_first():
    first = ndcg_at_k(["a", "x", "y"], ["a"], k=3)
    third = ndcg_at_k(["x", "y", "a"], ["a"], k=3)
    assert first > third
    assert first == 1.0


def test_evaluate_produces_a_full_report(pipeline):
    rag, _ = pipeline
    cases = [
        EvalCase("How long are transaction records retained?", ["retention"]),
        EvalCase("What is the expense claim deadline?", ["policy"]),
        EvalCase("Who acknowledges a severity one incident?", ["runbook"]),
    ]
    report = evaluate(rag, cases, k=3)
    s = report.summary()

    assert s["cases"] == 3
    assert 0.0 <= s["precision@3"] <= 1.0
    assert 0.0 <= s["recall@3"] <= 1.0
    assert 0.0 <= s["mrr"] <= 1.0
    assert s["hit_rate"] > 0, "retrieval should find at least one labelled document"
