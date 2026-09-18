"""
A compact transformer cross-encoder, written out rather than downloaded.

A bi-encoder embeds the query and the passage separately, so the two never see
each other and the score is a dot product between two summaries. A cross-encoder
reads the concatenation, so every query token can attend to every passage token.
That is strictly more expressive and strictly more expensive, which is why it is
used to reorder a short candidate list rather than to search an index.

This implementation exists so the whole training and serving path runs with no
model download, no API key and no network, in line with the rest of the service.
Where a pretrained cross-encoder is available it is the better model and
rerank.py prefers it; this one is the floor, not the ceiling.

Layout is standard pre-norm encoder:

    [CLS] query tokens [SEP] passage tokens [SEP]
      |
      token + segment + position embeddings
      N x (self-attention -> feed forward), pre-norm, residual
      |
      pooled from position 0 -> linear head -> one relevance logit

Attention projections are named q_proj, k_proj, v_proj and o_proj so lora.py can
target them by name without knowing what this module is.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
from torch import nn

logger = logging.getLogger(__name__)

PAD, CLS, SEP, UNK = "[PAD]", "[CLS]", "[SEP]", "[UNK]"
SPECIALS = (PAD, CLS, SEP, UNK)

_WORD = re.compile(r"[a-z0-9]+")


# ── tokenizer ───────────────────────────────────────────────────

class WordTokenizer:
    """
    Word-level tokenizer fitted on the corpus.

    Word-level rather than subword because the corpus is small and closed: the
    vocabulary fits comfortably, and a subword model trained on a few thousand
    words would be worse than the thing it replaced. It is saved inside the
    checkpoint, so a checkpoint is never separated from the vocabulary it was
    trained against.
    """

    def __init__(self, vocab: Sequence[str] | None = None) -> None:
        self.itos: list[str] = list(vocab) if vocab else list(SPECIALS)
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def fit(cls, corpus: Sequence[str], max_vocab: int = 20000, min_count: int = 1):
        counts: Counter[str] = Counter()
        for text in corpus:
            counts.update(_WORD.findall(text.lower()))
        keep = [w for w, c in counts.most_common(max_vocab) if c >= min_count]
        tok = cls(list(SPECIALS) + keep)
        logger.info("tokenizer fitted: %d types over %d documents", len(tok), len(corpus))
        return tok

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    def _ids(self, text: str) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(w, unk) for w in _WORD.findall(text.lower())]

    def encode_pair(
        self, query: str, passage: str, max_len: int = 256
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Returns (input_ids, segment_ids, attention_mask), padded to max_len.

        The query is given a guaranteed share of the window rather than being
        truncated alongside the passage, because a truncated query changes the
        question being asked, while a truncated passage only loses evidence.
        """
        cls_id, sep_id = self.stoi[CLS], self.stoi[SEP]
        budget = max_len - 3                       # [CLS] q [SEP] p [SEP]
        q_budget = min(len(self._ids(query)), max(budget // 4, 1))

        q = self._ids(query)[:q_budget]
        p = self._ids(passage)[: budget - len(q)]

        ids = [cls_id] + q + [sep_id] + p + [sep_id]
        seg = [0] * (len(q) + 2) + [1] * (len(p) + 1)
        mask = [1] * len(ids)

        pad = max_len - len(ids)
        if pad > 0:
            ids += [self.pad_id] * pad
            seg += [1] * pad
            mask += [0] * pad
        return ids, seg, mask

    def encode_batch(
        self, pairs: Sequence[tuple[str, str]], max_len: int = 256
    ) -> dict[str, torch.Tensor]:
        ids, segs, masks = [], [], []
        for q, p in pairs:
            i, s, m = self.encode_pair(q, p, max_len=max_len)
            ids.append(i)
            segs.append(s)
            masks.append(m)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "segment_ids": torch.tensor(segs, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def state(self) -> list[str]:
        return list(self.itos)


# ── model ───────────────────────────────────────────────────────

@dataclass
class CrossEncoderConfig:
    vocab_size: int = 8000
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 384
    max_len: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}"
            )


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: CrossEncoderConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape

        def split(proj: torch.Tensor) -> torch.Tensor:
            return proj.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Padding must be masked before the softmax, not after. Masking after
        # leaves pad positions with a nonzero share of the probability mass, so
        # the row no longer sums to one over real tokens.
        pad = (attention_mask == 0)[:, None, None, :]
        scores = scores.masked_fill(pad, torch.finfo(scores.dtype).min)

        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out)


class EncoderBlock(nn.Module):
    """Pre-norm block: normalise on the way in, add the residual raw."""

    def __init__(self, cfg: CrossEncoderConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), attention_mask))
        return x + self.dropout(self.ff(self.ln2(x)))


class TinyCrossEncoder(nn.Module):
    """Query and passage read together, scored as one relevance logit."""

    def __init__(self, cfg: CrossEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.seg_emb = nn.Embedding(2, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.emb_ln = nn.LayerNorm(cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, t = input_ids.shape
        if t > self.cfg.max_len:
            raise ValueError(f"sequence of {t} exceeds max_len {self.cfg.max_len}")

        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.tok_emb(input_ids) + self.seg_emb(segment_ids) + self.pos_emb(pos)
        x = self.emb_drop(self.emb_ln(x))

        for block in self.blocks:
            x = block(x, attention_mask)

        # position 0 is [CLS], which attends over the whole pair and is never
        # masked, so it is the one position guaranteed to see both sides
        return self.head(self.ln_f(x)[:, 0]).squeeze(-1)


# ── checkpointing ───────────────────────────────────────────────

def save_checkpoint(
    path,
    model: TinyCrossEncoder,
    tokenizer: WordTokenizer,
    extra: dict | None = None,
    adapters_only: bool = False,
) -> None:
    """
    Model, config and vocabulary travel together in one file.

    Saving the vocabulary separately invites the failure where a checkpoint is
    loaded against a vocabulary fitted on a different corpus: token ids silently
    mean different words and the model appears to have forgotten everything.
    """
    from .lora import lora_state_dict

    payload = {
        "config": asdict(model.cfg),
        "vocab": tokenizer.state(),
        "state_dict": lora_state_dict(model) if adapters_only else model.state_dict(),
        "adapters_only": adapters_only,
        "extra": extra or {},
    }
    torch.save(payload, path)
    logger.info("checkpoint written: %s (adapters_only=%s)", path, adapters_only)


def load_checkpoint(path, map_location="cpu") -> tuple[TinyCrossEncoder, WordTokenizer, dict]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg = CrossEncoderConfig(**payload["config"])
    tokenizer = WordTokenizer(payload["vocab"])
    model = TinyCrossEncoder(cfg)

    if payload.get("adapters_only"):
        raise ValueError(
            "this checkpoint holds adapters only and cannot be loaded alone; "
            "load the base checkpoint first, inject LoRA, then apply this one"
        )

    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, tokenizer, payload.get("extra", {})
