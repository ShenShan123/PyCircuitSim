"""Unified training CLI for DirectNet and BSIMAR Transformer.

Quick presets for fast verification:

    python -m bsimar.cli.train --model direct      --size small  --device-type nmos --cuda
    python -m bsimar.cli.train --model transformer --size medium --device-type nmos --cuda

Override individual knobs (``--epochs``, ``--batch-size``, …) to tune
beyond the preset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from bsimar.config import (
    CHECKPOINT_DIR, DATA_DIR,
    DirectNetConfig, TransformerConfig,
    LOCAL_VARIANT_CODES, VALID_TECH_SCOPES, tech_scope_vocab_size,
)
from bsimar.training.trainer import train_directnet, train_transformer
from bsimar.utils.seed import set_seed

import numpy as np


# All TSMC + ASAP7 tech names for the per-tech `--tech-scope` auto-exclude.
_ALL_TECH_NAMES = ("tsmc5", "tsmc7", "tsmc12", "tsmc16", "asap7")


# ── Loss presets (per docs/superpowers/plans/2026-05-08-…) ─────────────
# OUTPUT_COLUMN_ORDER = [id, gm, gds, gmb, qg, qd, qs, qb,
#                       cgg, cgd, cgs, cdg, cdd]
#
# B0 — uniform (baseline already shipped)
# E1 — drop qs supervision (KCL is enforced analytically anyway)
# E2 — 4-output head: only [id, qg, qd, qb] in the model output
# E3 — keep 13 outputs but down-weight non-load-bearing targets
LOSS_PRESETS = {
    "default": {"column_weights": None, "output_subset": None},
    "e1": {
        "column_weights": np.array(
            [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32),
        "output_subset": None,
    },
    "e2": {
        "column_weights": None,
        "output_subset": ["id", "qg", "qd", "qb"],
    },
    "e3": {
        "column_weights": np.array(
            [1.0, 0.1, 0.1, 0.1, 1.0, 1.0, 0.0, 1.0,
             0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float32),
        "output_subset": None,
    },
}


# (model, size) → (config dict, default save_prefix tag)
SIZE_PRESETS = {
    ("direct", "small"): dict(
        trunk_hidden=128, trunk_layers=3, batch_size=2048,
        max_epochs=80, patience=25, lr=1e-3),
    ("direct", "medium"): dict(
        trunk_hidden=256, trunk_layers=5, batch_size=2048,
        max_epochs=200, patience=40, lr=1e-3),
    ("direct", "large"): dict(
        trunk_hidden=384, trunk_layers=6, batch_size=2048,
        max_epochs=800, patience=150, lr=1e-3),
    ("transformer", "small"): dict(
        d_model=128, nhead=4, num_layers=3, dim_feedforward=512,
        dropout=0.1, batch_size=1024, max_epochs=60,
        patience=20, lr=8e-4),
    ("transformer", "medium"): dict(
        d_model=192, nhead=6, num_layers=4, dim_feedforward=768,
        dropout=0.15, batch_size=1024, max_epochs=150,
        patience=40, lr=8e-4),
    ("transformer", "large"): dict(
        d_model=256, nhead=8, num_layers=6, dim_feedforward=1024,
        dropout=0.2, batch_size=1024, max_epochs=300,
        patience=80, lr=8e-4),
}


def _resolve_data_path(args: argparse.Namespace) -> Path:
    if args.data:
        return Path(args.data)
    if args.tech_scope != "universal":
        return DATA_DIR / f"{args.tech_scope}_{args.device_type}.npz"
    return DATA_DIR / f"universal_{args.device_type}.npz"


def _make_save_prefix(args: argparse.Namespace) -> str:
    if args.exp_name:
        return f"{args.exp_name}_{args.device_type}"
    tag = "dn" if args.model == "direct" else "tf"
    suffix = ""
    if args.loss_preset != "default":
        suffix = f"_{args.loss_preset}"
    if args.tech_scope != "universal":
        # Per-tech dedicated checkpoint: tsmc{5,7}_dn_<size>[_<preset>]_<dev>.
        # The parser's preempt cascade keys off the `tsmc{5,7}_dn_` prefix.
        return f"{args.tech_scope}_{tag}_{args.size}{suffix}_{args.device_type}"
    return f"refac_{tag}_{args.size}{suffix}_{args.device_type}"


def _run(args: argparse.Namespace) -> None:
    data_path = _resolve_data_path(args)
    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        sys.exit(1)

    save_prefix = _make_save_prefix(args)
    device_str = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"

    # Per-tech scope auto-derives the exclude set + the embedding vocab.
    # Explicit --exclude-techs / --num-tech-codes still win if both are set.
    if args.tech_scope != "universal":
        auto_excl = {t for t in _ALL_TECH_NAMES if t != args.tech_scope}
        explicit_excl = (
            {t.strip().lower() for t in args.exclude_techs.split(",")}
            if args.exclude_techs else set())
        exclude = explicit_excl | auto_excl
        # Vocab = #variants(scope) + 1 UNKNOWN slot.
        if args.num_tech_codes == 18:  # untouched default
            args.num_tech_codes = tech_scope_vocab_size(args.tech_scope)
        print(f"  [tech-scope={args.tech_scope}] auto exclude={sorted(exclude)} "
              f"num_tech_codes={args.num_tech_codes}")
    else:
        exclude = (
            {t.strip().lower() for t in args.exclude_techs.split(",")}
            if args.exclude_techs else None)

    preset = dict(SIZE_PRESETS[(args.model, args.size)])
    # Per-flag overrides
    if args.epochs is not None:
        preset["max_epochs"] = args.epochs
    if args.batch_size is not None:
        preset["batch_size"] = args.batch_size
    if args.lr is not None:
        preset["lr"] = args.lr
    if args.patience is not None:
        preset["patience"] = args.patience

    loss_preset = LOSS_PRESETS[args.loss_preset]

    common = dict(
        device_type=args.device_type, device_str=device_str,
        save_prefix=save_prefix, exclude_techs=exclude,
        num_tech_codes=args.num_tech_codes, p_unknown=args.p_unknown,
        max_rows=args.max_rows, overwrite=args.overwrite,
        tech_scope=args.tech_scope,
    )

    print(f"\n=== Training {args.model} ({args.size}, "
          f"loss-preset={args.loss_preset}) → {save_prefix} ===")
    if args.model == "direct":
        cfg = DirectNetConfig(**preset)
        train_directnet(
            str(data_path), config=cfg,
            column_weights=loss_preset["column_weights"],
            output_subset=loss_preset["output_subset"],
            **common,
        )
    else:
        if (loss_preset["output_subset"] is not None
                or loss_preset["column_weights"] is not None):
            print("[warn] loss presets are DirectNet-only; "
                  "Transformer ignores them")
        cfg = TransformerConfig(**preset)
        train_transformer(str(data_path), config=cfg, **common)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Unified BSIMAR / DirectNet training CLI")
    p.add_argument("--model", choices=["direct", "transformer"],
                   default="direct")
    p.add_argument("--size", choices=["small", "medium", "large"],
                   default="medium",
                   help="Architecture-size preset (overridable below)")
    p.add_argument("--device-type", choices=["nmos", "pmos"], default="nmos")
    p.add_argument("--data", type=str, default=None,
                   help="Path to .npz dataset (auto-resolved if omitted)")

    # Per-flag overrides (None means: use the size-preset default)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap dataset rows (after filter / exclude) for "
                        "fast smoke runs")

    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # Tech-code embedding (shared by both models)
    p.add_argument("--exclude-techs", type=str, default=None)
    p.add_argument("--num-tech-codes", type=int, default=18)
    p.add_argument("--p-unknown", type=float, default=0.1)

    p.add_argument("--tech-scope",
                   choices=list(VALID_TECH_SCOPES),
                   default="universal",
                   help="Per-tech dedicated training. 'tsmc5' / 'tsmc7' "
                        "auto-set --exclude-techs (all other techs), "
                        "--num-tech-codes (per-tech vocab + UNKNOWN), "
                        "default --data path, and the save_prefix "
                        "(tsmc{5,7}_dn_<size>_<dev>) recognized by the "
                        "parser preempt cascade.")
    p.add_argument("--exp-name", type=str, default=None,
                   help="Override the auto-generated save_prefix")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--loss-preset",
                   choices=sorted(LOSS_PRESETS.keys()),
                   default="default",
                   help="DirectNet loss preset (per "
                        "2026-05-08-directnet-target-trim plan): "
                        "default=B0, e1=drop-qs, e2=4-output head, "
                        "e3=down-weight non-load-bearing targets")

    args = p.parse_args()
    set_seed(args.seed)
    _run(args)


if __name__ == "__main__":
    main()
