"""
Training the reranker, in two stages.

  Stage 1, base. The whole model trains on easy groups, where the negatives are
  random passages. This is the cheap part of the problem: telling an expenses
  passage from an incident passage. Doing it first means the adapters in stage 2
  are not spent relearning it.

  Stage 2, adaptation. The base is frozen, LoRA adapters are injected into the
  attention projections, and only those adapters and the scoring head train, on
  groups whose negatives were mined from the real retriever. This is the part
  that matters: separating the right passage from the four wrong passages the
  bi-encoder ranked just as highly.

The objective is listwise. Each group of one positive and n negatives is scored
together, softmaxed, and trained with cross-entropy against the positive. That
matches how the model is used, which is to order a candidate list, rather than
binary classification, which optimises a threshold nobody ever reads.

Stage 2 is where the parameter saving shows: it trains a low single-digit
percentage of the model. The same lora.py code path adapts a pretrained Hugging
Face cross-encoder, which is the better model where its weights are reachable.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from app.crossencoder import (
    CrossEncoderConfig,
    TinyCrossEncoder,
    WordTokenizer,
    save_checkpoint,
)
from app.lora import inject_lora, lora_state_dict, mark_only_lora_trainable, parameter_summary

from .data import TrainingGroup

logger = logging.getLogger(__name__)


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_groups: int = 4          # groups per optimisation step
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_len: int = 256
    grad_clip: float = 1.0
    warmup_fraction: float = 0.1
    log_every: int = 20


def _encode_groups(
    tokenizer: WordTokenizer, groups: Sequence[TrainingGroup], max_len: int, device: str
) -> tuple[dict[str, torch.Tensor], int]:
    """Flatten groups into one batch of pairs, remembering the group width."""
    width = groups[0].size
    if any(g.size != width for g in groups):
        raise ValueError("all groups in a batch must have the same number of candidates")

    pairs: list[tuple[str, str]] = []
    for g in groups:
        pairs.append((g.query, g.positive))
        pairs.extend((g.query, neg) for neg in g.negatives)

    batch = tokenizer.encode_batch(pairs, max_len=max_len)
    return {k: v.to(device) for k, v in batch.items()}, width


def _group_loss(
    model: nn.Module,
    tokenizer: WordTokenizer,
    groups: Sequence[TrainingGroup],
    max_len: int,
    device: str,
) -> tuple[torch.Tensor, int]:
    batch, width = _encode_groups(tokenizer, groups, max_len, device)
    logits = model(**batch).view(len(groups), width)
    # index 0 is the positive in every group, by construction in _encode_groups
    target = torch.zeros(len(groups), dtype=torch.long, device=device)
    loss = nn.functional.cross_entropy(logits, target)
    correct = int((logits.argmax(dim=1) == target).sum().item())
    return loss, correct


@torch.no_grad()
def evaluate_groups(
    model: nn.Module,
    tokenizer: WordTokenizer,
    groups: Sequence[TrainingGroup],
    max_len: int = 256,
    device: str = "cpu",
    batch_groups: int = 8,
) -> dict[str, float]:
    """Loss and group accuracy: how often the positive outscores every negative."""
    if not groups:
        return {"loss": 0.0, "accuracy": 0.0}

    model.eval()
    total_loss, total_correct, n = 0.0, 0, 0
    for i in range(0, len(groups), batch_groups):
        window = groups[i : i + batch_groups]
        loss, correct = _group_loss(model, tokenizer, window, max_len, device)
        total_loss += float(loss.item()) * len(window)
        total_correct += correct
        n += len(window)
    return {
        "loss": round(total_loss / n, 4),
        "accuracy": round(total_correct / n, 4),
    }


def _train(
    model: nn.Module,
    tokenizer: WordTokenizer,
    train_groups: Sequence[TrainingGroup],
    dev_groups: Sequence[TrainingGroup],
    cfg: TrainConfig,
    device: str,
    label: str,
) -> dict:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("nothing is trainable; check the freezing step")

    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = max(1, len(train_groups) // cfg.batch_groups)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(total_steps * cfg.warmup_fraction))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 1.0 - progress)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    summary = parameter_summary(model)
    logger.info(
        "%s: training %s of %s parameters (%.3f%%) on %s",
        label, f"{summary['trainable']:,}", f"{summary['total']:,}",
        summary["trainable_pct"], device,
    )

    history: list[dict] = []
    started = time.perf_counter()
    step = 0
    order = list(range(len(train_groups)))

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        torch.manual_seed(1234 + epoch)
        import random as _random

        _random.Random(1234 + epoch).shuffle(order)

        running, seen, correct = 0.0, 0, 0
        for b in range(steps_per_epoch):
            idx = order[b * cfg.batch_groups : (b + 1) * cfg.batch_groups]
            window = [train_groups[i] for i in idx]
            if not window:
                continue

            loss, n_correct = _group_loss(model, tokenizer, window, cfg.max_len, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            sched.step()

            step += 1
            running += float(loss.item()) * len(window)
            correct += n_correct
            seen += len(window)

            if cfg.log_every and step % cfg.log_every == 0:
                logger.info(
                    "%s epoch %d step %d loss %.4f acc %.3f",
                    label, epoch, step, running / seen, correct / seen,
                )

        dev = evaluate_groups(model, tokenizer, dev_groups, cfg.max_len, device)
        row = {
            "epoch": epoch,
            "train_loss": round(running / max(seen, 1), 4),
            "train_accuracy": round(correct / max(seen, 1), 4),
            "dev_loss": dev["loss"],
            "dev_accuracy": dev["accuracy"],
        }
        history.append(row)
        logger.info("%s epoch %d: %s", label, epoch, row)

    return {
        "label": label,
        "history": history,
        "parameters": summary,
        "seconds": round(time.perf_counter() - started, 2),
        "device": device,
    }


def train_base(
    tokenizer: WordTokenizer,
    train_groups: Sequence[TrainingGroup],
    dev_groups: Sequence[TrainingGroup],
    out_path: Path,
    model_config: CrossEncoderConfig | None = None,
    cfg: TrainConfig | None = None,
    device: str | None = None,
) -> dict:
    """Stage 1: everything trains, on easy groups."""
    cfg = cfg or TrainConfig()
    device = device or best_device()
    model_config = model_config or CrossEncoderConfig(vocab_size=len(tokenizer))
    model_config.vocab_size = len(tokenizer)

    model = TinyCrossEncoder(model_config).to(device)
    report = _train(model, tokenizer, train_groups, dev_groups, cfg, device, "base")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out_path, model.cpu(), tokenizer, extra={"stage": "base"})
    return report


def train_lora(
    base_path: Path,
    tokenizer: WordTokenizer,
    train_groups: Sequence[TrainingGroup],
    dev_groups: Sequence[TrainingGroup],
    out_path: Path,
    r: int = 8,
    alpha: int = 16,
    targets: Sequence[str] = ("q_proj", "v_proj"),
    cfg: TrainConfig | None = None,
    device: str | None = None,
) -> dict:
    """Stage 2: base frozen, adapters trained on hard groups."""
    from app.crossencoder import load_checkpoint

    cfg = cfg or TrainConfig()
    device = device or best_device()

    model, ckpt_tokenizer, _ = load_checkpoint(base_path)
    if ckpt_tokenizer.state() != tokenizer.state():
        raise ValueError(
            "tokenizer does not match the base checkpoint; token ids would mean "
            "different words than the base was trained on"
        )

    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    inject_lora(model, targets=targets, r=r, alpha=alpha)
    mark_only_lora_trainable(model, train_head=True)
    model.to(device)

    report = _train(model, tokenizer, train_groups, dev_groups, cfg, device, "lora")

    # The published checkpoint carries the unmodified base weights plus the
    # adapters, not the injected module tree. Saving the injected tree would
    # produce a file that only loads into a model that has already been injected
    # at exactly the same rank, which is a checkpoint that cannot be opened
    # without already knowing what is in it.
    model.cpu()
    payload_extra = {
        "stage": "lora",
        "adapter_state": lora_state_dict(model),
        "r": r,
        "alpha": alpha,
        "targets": list(targets),
        "report": report,
    }

    clean = TinyCrossEncoder(model.cfg)
    clean.load_state_dict(base_state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out_path, clean, tokenizer, extra=payload_extra)

    report["adapter_parameters"] = sum(
        v.numel() for k, v in payload_extra["adapter_state"].items()
    )
    return report
