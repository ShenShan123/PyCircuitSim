"""Unified CLI for BSIMAR training.

Two models, one CLI:

- ``--model direct``      — DirectNet MLP with tech-code embedding
- ``--model transformer`` — BSIMAR Transformer with tech-code embedding (default)

Both models use a 7-dim continuous input plus a discrete tech-code
embedding.  The Transformer recipe is hard-wired inside
``train_transformer`` (asinh+zscore norm, MAE + LDS + Vov-LDS,
parallel_caps, grouped_inputs, AR finetune phase).  The only
caller-visible knobs are architecture (``--d-model``, ``--nhead``,
``--num-layers``, ``--dim-feedforward``, ``--dropout``), schedule
(``--epochs``, ``--batch-size``, ``--lr``, ``--patience``,
``--ar-finetune-epochs``), and checkpoint naming (``--exp-name``,
``--overwrite``).

Usage:
    # DirectNet with tech-code embedding
    python -m bsimar.cli.train \\
        --model direct --device-type nmos \\
        --epochs 800 --hidden 384 --layers 6 --batch-size 2048 --cuda

    # BSIMAR Transformer (production recipe)
    python -m bsimar.cli.train \\
        --model transformer --device-type nmos --cuda
"""

import sys
import argparse
from pathlib import Path

import torch

from bsimar.config import (
    TECH_CONFIGS,
    CHECKPOINT_DIR, DATA_DIR,
    DirectNetConfig, TransformerConfig,
)
from bsimar.utils.seed import set_seed
from bsimar.training.trainer import (
    train_directnet,
    train_transformer,
)


# ── DirectNet subcommand ────────────────────────────────────────────────────

def _run_direct(args: argparse.Namespace) -> None:
    data_path = (Path(args.data) if args.data
                 else DATA_DIR / f"universal_{args.device_type}.npz")
    save_prefix = f"v4_dn_universal_{args.device_type}"
    if args.exp_name:
        save_prefix = f"{args.exp_name}_{args.device_type}"

    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        sys.exit(1)

    config = DirectNetConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        trunk_hidden=args.hidden,
        trunk_layers=args.layers,
        patience=args.patience,
    )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    exclude = None
    if args.exclude_techs:
        exclude = set(t.strip().lower() for t in args.exclude_techs.split(","))

    train_directnet(
        str(data_path),
        device_type=args.device_type,
        config=config,
        device_str=device_str,
        save_prefix=save_prefix,
        exclude_techs=exclude,
        num_tech_codes=args.num_tech_codes,
        p_unknown=args.p_unknown,
        slope_weight=args.slope_weight,
        slope_warmup_frac=args.slope_warmup_frac,
        id_gate=not args.no_id_gate,
    )


# ── Transformer subcommand ──────────────────────────────────────────────────

def _run_transformer(args: argparse.Namespace) -> None:
    data_path = (Path(args.data) if args.data
                 else DATA_DIR / f"universal_{args.device_type}.npz")
    save_prefix = f"v4_universal_{args.device_type}"
    if args.exp_name:
        save_prefix = f"{args.exp_name}_{args.device_type}"

    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        sys.exit(1)

    config = TransformerConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    exclude = None
    if args.exclude_techs:
        exclude = set(t.strip().lower() for t in args.exclude_techs.split(","))

    train_transformer(
        str(data_path),
        save_prefix=save_prefix,
        device_type=args.device_type,
        config=config,
        device_str=device_str,
        ar_finetune_epochs=args.ar_finetune_epochs,
        overwrite=args.overwrite,
        exclude_techs=exclude,
        num_tech_codes=args.num_tech_codes,
        p_unknown=args.p_unknown,
        slope_weight=args.slope_weight,
        slope_warmup_frac=args.slope_warmup_frac,
        id_gate=not args.no_id_gate,
    )


# ── Argparse ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BSIMAR unified training CLI "
                    "(DirectNet MLP or BSIMAR Transformer, "
                    "both with tech-code embedding)")
    parser.add_argument("--model",
                        choices=["direct", "transformer"],
                        default="transformer",
                        help="Which architecture to train "
                             "(direct=DirectNet MLP, "
                             "transformer=BSIMAR Transformer)")

    # Shared data args
    parser.add_argument("--device-type", choices=["nmos", "pmos"], default="nmos")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to .npz dataset (auto-resolved if omitted)")

    # Shared optimization args
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # DirectNet-specific
    parser.add_argument("--hidden", type=int, default=384,
                        help="[direct only] MLP hidden layer dimension")
    parser.add_argument("--layers", type=int, default=6,
                        help="[direct only] MLP hidden layers")

    # Transformer-specific: architecture only
    parser.add_argument("--d-model", type=int, default=256,
                        help="[transformer] Encoder hidden dimension")
    parser.add_argument("--nhead", type=int, default=8,
                        help="[transformer] Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=6,
                        help="[transformer] Number of encoder layers")
    parser.add_argument("--dim-feedforward", type=int, default=1024,
                        help="[transformer] FFN hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name; overrides the default "
                             "save_prefix")
    parser.add_argument("--overwrite", action="store_true",
                        help="[transformer] Allow overwriting an existing "
                             "<save_prefix>_best.pt checkpoint")
    parser.add_argument("--ar-finetune-epochs", type=int, default=5,
                        help="[transformer] AR-rollout fine-tune epochs "
                             "after the cosine schedule. Default 5 "
                             "(empirically sufficient).")

    # Tech-code embedding args (shared by both models)
    parser.add_argument("--exclude-techs", type=str, default=None,
                        help="Comma-separated tech names to exclude "
                             "entirely (e.g., 'asap7')")
    parser.add_argument("--num-tech-codes", type=int, default=18,
                        help="Tech embedding vocabulary size "
                             "(default 18 = 17 TSMC + 1 UNKNOWN)")
    parser.add_argument("--p-unknown", type=float, default=0.1,
                        help="Prob of replacing tech code with UNKNOWN "
                             "during training (default 0.1)")

    # B3 — Structural Vds gate on the Id output (Sprint S-ARCH-A)
    parser.add_argument("--no-id-gate", action="store_true",
                        help="[B3] Disable the structural Vds gate on the "
                             "id output. By default the gate is ON: model "
                             "id_phys is multiplied by tanh(Vds/0.04 V) "
                             "and re-normalised before the loss, so "
                             "Id(Vds=0)=0 holds structurally. "
                             "Disable for legacy v4/v5b-style training.")

    # Slope-loss (B2) args (shared by both models)
    parser.add_argument("--slope-weight", type=float, default=0.0,
                        help="[B2] Weight on the slope-match auxiliary "
                             "loss (dId/dVg in normalised space). "
                             "Default 0 = disabled. Only active on "
                             "datasets that carry a sample_class column.")
    parser.add_argument("--slope-warmup-frac", type=float, default=0.7,
                        help="[B2] Fraction of total epochs to wait "
                             "before activating slope loss (default "
                             "0.7 = last 30%% of training).")

    args = parser.parse_args()
    set_seed(args.seed)

    if args.model == "direct":
        _run_direct(args)
    else:
        _run_transformer(args)


if __name__ == "__main__":
    main()
