"""
Low-rank adaptation, implemented directly rather than pulled from a library.

A LoRA layer wraps a frozen nn.Linear and learns a low-rank correction to it:

    y = W x + (alpha / r) * B A x        W frozen, A and B trainable

B is initialised to zeros so the adapted model starts numerically identical to
the model it wraps. That matters more than it sounds: a fine-tune that begins by
perturbing a pretrained model is a fine-tune whose first few steps are spent
undoing damage.

The parameter saving is the point. Adapting one d x d projection costs 2 * r * d
parameters instead of d^2, so at d=384 and r=8 a projection trains 6.1k
parameters instead of 147k.

This module is deliberately independent of the model it adapts. It matches
target layers by name substring, so the same code adapts the compact encoder in
crossencoder.py and a pretrained Hugging Face cross-encoder without knowing
anything about either.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import torch
from torch import nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable low-rank correction."""

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("lora rank must be positive")

        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base.in_features
        out_features = base.out_features

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # A gets the usual fan-in init, B stays at zero, so B @ A is zero at step
        # zero and the wrapped layer is unchanged until training moves it.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + delta * self.scaling

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3f}"


def _resolve_parent(model: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    targets: Sequence[str] = ("q_proj", "v_proj"),
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> int:
    """
    Replace every nn.Linear whose qualified name contains one of `targets` with a
    LoRALinear wrapping it. Returns how many layers were adapted.

    Matching on a name substring rather than a module type is what keeps this
    usable across architectures: 'q_proj' hits this repository's attention, and
    'query' hits a Hugging Face BERT-family attention, with no branching here.

    Zero matches is an error rather than a warning. A fine-tune that silently
    adapts nothing still runs, still produces a checkpoint, and still reports a
    loss curve, so it has to fail loudly at setup instead.
    """
    to_replace: list[tuple[str, nn.Linear]] = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(t in name for t in targets)
    ]

    if not to_replace:
        available = sorted({
            name.split(".")[-1]
            for name, m in model.named_modules()
            if isinstance(m, nn.Linear)
        })
        raise ValueError(
            f"no nn.Linear matched targets {list(targets)}; "
            f"leaf names present: {available}"
        )

    for name, module in to_replace:
        parent, attr = _resolve_parent(model, name)
        setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))

    logger.info("injected LoRA into %d layers (r=%d, alpha=%d)", len(to_replace), r, alpha)
    return len(to_replace)


def mark_only_lora_trainable(model: nn.Module, train_head: bool = True) -> None:
    """
    Freeze everything except the adapters, and optionally the classification head.

    The head is usually included because a frozen randomly-initialised head makes
    the adapters spend their capacity compensating for it. Where the head was
    itself pretrained, freezing it is the better choice.
    """
    for name, param in model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        is_head = train_head and name.startswith("head.")
        param.requires_grad = bool(is_lora or is_head)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Just the trainable pieces, which is the whole reason for doing this."""
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k or k.startswith("head.")
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"adapter checkpoint has unknown keys: {sorted(unexpected)[:5]}")
    # `missing` is expected and large: it is every frozen base weight, which the
    # adapter checkpoint deliberately does not carry.


def parameter_summary(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": round(100.0 * trainable / total, 3) if total else 0,
    }
