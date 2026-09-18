"""
Train the reranker end to end and report whether it helped.

    python scripts/train_reranker.py                 # defaults, a few minutes on CPU
    python scripts/train_reranker.py --epochs 6      # longer
    python scripts/train_reranker.py --no-train      # measure an existing checkpoint

The run prints a before-and-after table on the held-out question set, which is
never trained on, and writes the full report to models/report.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.crossencoder import CrossEncoderConfig, WordTokenizer  # noqa: E402
from training.common import (  # noqa: E402
    MODEL_DIR,
    build_index,
    load_eval_cases,
    resolve_ground_truth,
    seed_everything,
)
from training.data import (  # noqa: E402
    generate_queries,
    mine_hard_negatives,
    random_negative_groups,
    split_groups,
)
from training.evaluate import compare, format_comparison, promote_if_better  # noqa: E402


def _merge_report(new: dict) -> None:
    """Merge into the existing report, so staged runs do not lose earlier stages."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / "report.json"
    merged = {}
    if path.exists():
        try:
            merged = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged = {}
    merged.update(new)
    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lora-epochs", type=int, default=4)
    ap.add_argument("--batch-groups", type=int, default=8)
    ap.add_argument("--negatives", type=int, default=4)
    ap.add_argument("--per-chunk", type=int, default=3, help="query sentences drawn per chunk")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--stage",
        choices=("all", "base", "lora", "eval"),
        default="all",
        help="run one stage at a time; stages hand over through the checkpoint files",
    )
    ap.add_argument("--no-train", action="store_true", help="measure the existing checkpoint")
    ap.add_argument("--out", default=str(MODEL_DIR / "reranker.pt"))
    ap.add_argument(
        "--gate-metric", default="mrr", help="metric the candidate must improve to be promoted"
    )
    ap.add_argument(
        "--gate-min-delta",
        type=float,
        default=0.0,
        help="minimum improvement over fusion-only required for promotion",
    )
    ap.add_argument(
        "--promote-anyway",
        action="store_true",
        help="install the candidate even if it fails the gate (it will be logged as forced)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    seed_everything(args.seed)

    corpus = build_index()
    cases = resolve_ground_truth(load_eval_cases(), corpus)
    print(f"\ncorpus: {len(corpus.chunks)} chunks   held-out questions: {len(cases)}\n")

    out_path = Path(args.out)
    base_path = out_path.with_name("reranker_base.pt")
    # Training writes a candidate, never the served path. Promotion is a separate
    # decision taken on held-out evidence, below.
    candidate_path = out_path.with_name("reranker_candidate.pt")
    reports: dict = {}

    stage = "eval" if args.no_train else args.stage

    if stage in ("all", "base", "lora"):
        from training.train import TrainConfig, train_base, train_lora

        triples = generate_queries(corpus, per_chunk=args.per_chunk, seed=args.seed)
        tokenizer = WordTokenizer.fit(corpus.texts + [q for q, _, _ in triples])

        easy = random_negative_groups(corpus, triples, n_negatives=args.negatives, seed=args.seed)
        hard = mine_hard_negatives(
            corpus, triples, n_negatives=args.negatives, pool=args.candidates, seed=args.seed
        )
        easy_train, easy_dev = split_groups(easy, seed=args.seed)
        hard_train, hard_dev = split_groups(hard, seed=args.seed)

        print(
            f"training groups: {len(easy_train)} easy, {len(hard_train)} hard "
            f"({args.negatives} negatives each)\n"
        )

        if stage in ("all", "base"):
            cfg = TrainConfig(
                epochs=args.epochs, batch_groups=args.batch_groups, max_len=args.max_len
            )
            reports["base"] = train_base(
                tokenizer, easy_train, easy_dev, base_path,
                model_config=CrossEncoderConfig(vocab_size=len(tokenizer), max_len=args.max_len),
                cfg=cfg,
            )

        if stage in ("all", "lora"):
            lora_cfg = TrainConfig(
                epochs=args.lora_epochs, batch_groups=args.batch_groups, max_len=args.max_len
            )
            reports["lora"] = train_lora(
                base_path, tokenizer, hard_train, hard_dev, candidate_path,
                r=args.rank, alpha=args.alpha, cfg=lora_cfg,
            )

    if stage == "base":
        # the adapter stage has not run yet, so there is nothing to measure
        _merge_report(reports)
        print("base stage complete; run --stage lora next")
        return 0

    from app.rerank import get_reranker

    measure_path = candidate_path if candidate_path.exists() else out_path
    reranker = get_reranker("local", model_path=str(measure_path))
    if reranker.name == "none":
        print(f"no usable checkpoint at {measure_path}", file=sys.stderr)
        return 1

    comparison = compare(
        corpus, cases, reranker, k=args.k, candidate_k=args.candidates
    )
    reports["comparison"] = comparison

    print()
    print("=" * 50)
    print(f"HELD-OUT QUESTIONS  ({reranker.name})")
    print("=" * 50)
    print(format_comparison(comparison))
    print()

    promoted = promote_if_better(
        comparison, candidate_path, out_path,
        metric=args.gate_metric, min_delta=args.gate_min_delta, force=args.promote_anyway,
    )
    reports["promotion"] = promoted

    _merge_report(reports)
    print(f"report written to {MODEL_DIR / 'report.json'}")
    return 0 if promoted["promoted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
