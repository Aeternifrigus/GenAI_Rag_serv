import pytest

torch = pytest.importorskip("torch", reason="reranking is optional and needs torch")

from torch import nn  # noqa: E402

from app.agent import RetrievalAgent  # noqa: E402
from app.crossencoder import (  # noqa: E402
    CrossEncoderConfig,
    TinyCrossEncoder,
    WordTokenizer,
    load_checkpoint,
    save_checkpoint,
)
from app.embeddings import LsaEmbedder  # noqa: E402
from app.ingest import IngestionPipeline  # noqa: E402
from app.lora import (  # noqa: E402
    LoRALinear,
    inject_lora,
    lora_state_dict,
    mark_only_lora_trainable,
    parameter_summary,
)
from app.rerank import NoopReranker, TorchReranker, get_reranker  # noqa: E402
from app.sources import InlineSource  # noqa: E402
from app.stores.base import Chunk, Hit  # noqa: E402
from app.stores.memory import InMemoryVectorStore  # noqa: E402
from training.data import TrainingGroup, _drop_one_sentence, generate_queries  # noqa: E402
from training.evaluate import promote_if_better  # noqa: E402

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


def tiny_config(vocab_size: int = 64) -> CrossEncoderConfig:
    return CrossEncoderConfig(
        vocab_size=vocab_size, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_len=32,
        dropout=0.0,
    )


def make_chunk(cid: str, text: str, doc_id: str = "d") -> Chunk:
    return Chunk(id=cid, text=text, source="files", doc_id=doc_id)


# ── lora ────────────────────────────────────────────────────────

def test_lora_starts_as_an_exact_identity():
    """B is zero-initialised, so wrapping must not change any output."""
    base = nn.Linear(8, 4)
    x = torch.randn(3, 8)
    before = base(x).clone()
    assert torch.equal(LoRALinear(base, r=2).eval()(x), before)


def test_lora_freezes_the_layer_it_wraps():
    layer = LoRALinear(nn.Linear(8, 4), r=2)
    assert not any(p.requires_grad for p in layer.base.parameters())
    assert layer.lora_A.requires_grad and layer.lora_B.requires_grad


def test_lora_rank_must_be_positive():
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(4, 4), r=0)


def test_injection_matches_by_name_and_reports_the_count():
    model = TinyCrossEncoder(tiny_config())
    assert inject_lora(model, targets=("q_proj", "v_proj"), r=2) == 2
    adapted = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert len(adapted) == 2


def test_injection_raises_when_nothing_matches():
    """A fine-tune that adapts nothing still trains and still saves, so this must fail loudly."""
    model = TinyCrossEncoder(tiny_config())
    with pytest.raises(ValueError, match="no nn.Linear matched"):
        inject_lora(model, targets=("does_not_exist",))


def test_only_adapters_and_head_train_after_freezing():
    model = TinyCrossEncoder(tiny_config())
    inject_lora(model, targets=("q_proj", "v_proj"), r=2)
    mark_only_lora_trainable(model, train_head=True)

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all("lora_" in n or n.startswith("head.") for n in trainable)
    assert parameter_summary(model)["trainable"] < parameter_summary(model)["total"]


def test_one_step_moves_only_the_adapters():
    model = TinyCrossEncoder(tiny_config())
    inject_lora(model, targets=("q_proj", "v_proj"), r=2)
    mark_only_lora_trainable(model, train_head=False)

    before = {k: v.clone() for k, v in model.state_dict().items()}
    ids = torch.randint(0, 64, (2, 8))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model(input_ids=ids, segment_ids=torch.zeros_like(ids),
          attention_mask=torch.ones_like(ids)).sum().backward()
    opt.step()

    moved = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
    assert moved and all("lora_" in k for k in moved)


def test_adapter_checkpoint_holds_only_adapters_and_head():
    model = TinyCrossEncoder(tiny_config())
    inject_lora(model, targets=("q_proj",), r=2)
    keys = lora_state_dict(model).keys()
    assert keys and all("lora_" in k or k.startswith("head.") for k in keys)


# ── tokenizer and model ─────────────────────────────────────────

def test_tokenizer_pads_and_marks_segments():
    tok = WordTokenizer.fit(["expense claims within thirty days", "severity one incident"])
    ids, seg, mask = tok.encode_pair("expense claims", "within thirty days", max_len=24)

    assert len(ids) == len(seg) == len(mask) == 24
    assert sum(mask) < 24                       # some padding present
    assert ids[0] == tok.stoi["[CLS]"]
    assert set(seg) <= {0, 1}
    assert all(i == tok.pad_id for i, m in zip(ids, mask, strict=True) if m == 0)


def test_unknown_words_map_to_unk_rather_than_failing():
    tok = WordTokenizer.fit(["known words only"])
    ids, _, _ = tok.encode_pair("completely unseen vocabulary", "known words", max_len=16)
    assert tok.stoi["[UNK]"] in ids


def test_query_keeps_a_share_of_the_window_when_the_passage_is_long():
    tok = WordTokenizer.fit(["alpha beta gamma delta", "filler " * 50])
    ids, seg, _ = tok.encode_pair("alpha beta gamma delta", "filler " * 200, max_len=32)
    assert sum(1 for s, i in zip(seg, ids, strict=True) if s == 0) > 1


def test_forward_returns_one_score_per_pair():
    cfg = tiny_config()
    model = TinyCrossEncoder(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (3, 8))
    out = model(input_ids=ids, segment_ids=torch.zeros_like(ids),
                attention_mask=torch.ones_like(ids))
    assert out.shape == (3,)


def test_sequence_longer_than_the_trained_window_is_rejected():
    """The positional table was sized at training time; running past it is a silent corruption."""
    cfg = tiny_config()
    model = TinyCrossEncoder(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, cfg.max_len + 1))
    with pytest.raises(ValueError, match="exceeds max_len"):
        model(input_ids=ids, segment_ids=torch.zeros_like(ids),
              attention_mask=torch.ones_like(ids))


def test_padding_does_not_change_the_score():
    """Masking is applied before the softmax, so padding must be inert."""
    cfg = tiny_config()
    model = TinyCrossEncoder(cfg).eval()
    ids = torch.randint(1, cfg.vocab_size, (1, 6))
    mask = torch.ones_like(ids)

    padded_ids = torch.cat([ids, torch.zeros(1, 6, dtype=torch.long)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(1, 6, dtype=torch.long)], dim=1)

    a = model(input_ids=ids, segment_ids=torch.zeros_like(ids), attention_mask=mask)
    b = model(input_ids=padded_ids, segment_ids=torch.zeros_like(padded_ids),
              attention_mask=padded_mask)
    assert torch.allclose(a, b, atol=1e-5)


def test_dimensions_must_divide_across_heads():
    with pytest.raises(ValueError):
        CrossEncoderConfig(d_model=10, n_heads=4)


def test_checkpoint_round_trips_with_its_vocabulary(tmp_path):
    cfg = tiny_config()
    tok = WordTokenizer.fit(["some corpus text here", "another passage of text"])
    cfg.vocab_size = len(tok)
    model = TinyCrossEncoder(cfg).eval()

    path = tmp_path / "ce.pt"
    save_checkpoint(path, model, tok, extra={"stage": "test"})
    loaded, loaded_tok, extra = load_checkpoint(path)

    assert loaded_tok.state() == tok.state()
    assert extra["stage"] == "test"

    batch = tok.encode_batch([("some corpus", "another passage")], max_len=cfg.max_len)
    assert torch.allclose(model(**batch), loaded(**batch), atol=1e-6)


def test_an_adapter_only_checkpoint_refuses_to_load_alone(tmp_path):
    cfg = tiny_config()
    tok = WordTokenizer.fit(["text"])
    cfg.vocab_size = len(tok)
    model = TinyCrossEncoder(cfg)
    inject_lora(model, targets=("q_proj",), r=2)

    path = tmp_path / "adapters.pt"
    save_checkpoint(path, model, tok, adapters_only=True)
    with pytest.raises(ValueError, match="adapters only"):
        load_checkpoint(path)


# ── reranking ───────────────────────────────────────────────────

def test_noop_reranker_keeps_order_and_truncates():
    hits = [Hit(chunk=make_chunk(str(i), f"text {i}"), score=1.0 - i / 10) for i in range(5)]
    out = NoopReranker().rerank("q", hits, 2)
    assert [h.chunk.id for h in out] == ["0", "1"]


def test_reranker_reorders_and_keeps_the_retrieval_score():
    cfg = tiny_config()
    tok = WordTokenizer.fit(["alpha passage", "beta passage", "gamma passage"])
    cfg.vocab_size = len(tok)
    reranker = TorchReranker(TinyCrossEncoder(cfg), tok)

    hits = [Hit(chunk=make_chunk(str(i), f"passage {i}"), score=0.5) for i in range(4)]
    out = reranker.rerank("alpha", hits, 3)

    assert len(out) == 3
    assert all(h.rerank_score is not None for h in out)
    assert all(h.score == 0.5 for h in out)                 # retrieval score preserved
    assert out == sorted(out, key=lambda h: -h.rerank_score)


def test_reranker_window_comes_from_the_checkpoint_not_the_caller():
    """The bug this guards: a 256 default against a model trained at 192."""
    cfg = tiny_config()
    tok = WordTokenizer.fit(["text"])
    cfg.vocab_size = len(tok)
    reranker = TorchReranker(TinyCrossEncoder(cfg), tok, max_len=4096)
    assert reranker.max_len == cfg.max_len


def test_empty_candidate_list_reranks_to_nothing():
    cfg = tiny_config()
    tok = WordTokenizer.fit(["text"])
    cfg.vocab_size = len(tok)
    assert TorchReranker(TinyCrossEncoder(cfg), tok).rerank("q", [], 5) == []


def test_missing_checkpoint_resolves_to_no_reranking(tmp_path):
    assert get_reranker("local", model_path=str(tmp_path / "absent.pt")).name == "none"


def test_reranking_can_be_switched_off_outright():
    assert get_reranker("none").name == "none"


# ── agent integration ───────────────────────────────────────────

def build_agent(reranker=None, candidate_k=10) -> RetrievalAgent:
    embedder = LsaEmbedder(dim=32)
    store = InMemoryVectorStore()
    IngestionPipeline(store=store, embedder=embedder).run([InlineSource(DOCS)])
    return RetrievalAgent(
        store=store, embedder=embedder, k=2, min_score=0.0,
        reranker=reranker, candidate_k=candidate_k,
    )


class ExplodingReranker:
    name = "exploding"

    def rerank(self, query, hits, k):
        raise RuntimeError("model unavailable")


class ReversingReranker:
    name = "reversing"

    def rerank(self, query, hits, k):
        return [Hit(chunk=h.chunk, score=h.score, rerank_score=float(i))
                for i, h in enumerate(reversed(list(hits)))][:k]


def test_a_failing_reranker_degrades_to_fusion_order():
    """Losing the results entirely would be worse than serving them unranked."""
    plain = build_agent().retrieve("expense claims deadline")
    guarded = build_agent(reranker=ExplodingReranker()).retrieve("expense claims deadline")

    assert [h.chunk.id for h in guarded.hits] == [h.chunk.id for h in plain.hits]
    assert "failed" in guarded.plan.reranker


def test_the_plan_records_that_reranking_happened():
    result = build_agent(reranker=ReversingReranker()).retrieve("expense claims deadline")
    assert result.plan.reranker == "reversing"
    assert result.plan.candidates >= len(result.hits)


def test_the_candidate_pool_widens_only_when_reranking():
    without = build_agent().retrieve("expense claims deadline")
    with_rerank = build_agent(reranker=ReversingReranker()).retrieve("expense claims deadline")
    assert with_rerank.plan.candidates > without.plan.candidates
    assert len(with_rerank.hits) == len(without.hits) == 2


# ── training data construction ──────────────────────────────────

def test_dropping_a_sentence_leaves_single_sentence_text_alone():
    import random
    assert _drop_one_sentence("Only one sentence here.", random.Random(0)) == \
        "Only one sentence here."


def test_generated_queries_are_removed_from_their_own_positive():
    """Otherwise the model wins by spotting the query verbatim in the passage."""
    embedder = LsaEmbedder(dim=32)
    store = InMemoryVectorStore()
    pipeline = IngestionPipeline(store=store, embedder=embedder)
    pipeline.run([InlineSource(DOCS)])

    class _Corpus:
        chunks = None
    corpus = _Corpus()
    corpus.chunks = pipeline._all_chunks

    for query, positive, _ in generate_queries(corpus, per_chunk=1, keyword_variants=False):
        assert query not in positive


def test_a_training_group_knows_its_width():
    group = TrainingGroup(query="q", positive="p", negatives=["a", "b", "c"])
    assert group.size == 4


# ── promotion gate ──────────────────────────────────────────────

def _comparison(delta: float) -> dict:
    return {"delta": {"mrr": delta}}


def test_a_candidate_that_beats_the_baseline_is_installed(tmp_path):
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"weights")
    serving = tmp_path / "reranker.pt"

    result = promote_if_better(_comparison(0.05), candidate, serving)
    assert result["promoted"] and result["passed"]
    assert serving.read_bytes() == b"weights"


def test_a_candidate_that_loses_is_not_installed(tmp_path):
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"weights")
    serving = tmp_path / "reranker.pt"

    result = promote_if_better(_comparison(-0.38), candidate, serving)
    assert not result["promoted"]
    assert not serving.exists()
    assert candidate.exists()          # kept for inspection


def test_a_failed_candidate_removes_a_stale_serving_model(tmp_path):
    """Serving a model the gate just rejected would be worse than serving none."""
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"new")
    serving = tmp_path / "reranker.pt"
    serving.write_bytes(b"old")

    promote_if_better(_comparison(-0.1), candidate, serving)
    assert not serving.exists()


def test_forcing_promotion_is_recorded_as_forced(tmp_path):
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"weights")
    serving = tmp_path / "reranker.pt"

    result = promote_if_better(_comparison(-0.2), candidate, serving, force=True)
    assert result["promoted"] and result["forced"] and not result["passed"]


def test_the_gate_rejects_a_metric_it_was_not_given(tmp_path):
    with pytest.raises(ValueError, match="not a metric"):
        promote_if_better(_comparison(0.1), tmp_path / "c.pt", tmp_path / "s.pt", metric="nope")
