"""Unified CLI entry point for BSIMAR training.

Trains either the DirectNet baseline (MLP) or the BSIM-AR Transformer.
Replaces the old `nn_model.train` and
`external_compact_models.BSIMAR.script.main` entry points.

Usage:
    # DirectNet baseline
    conda run -n pycircuitsim python -m bsimar.cli.train \
        --model direct --device-type nmos --universal --mode direct13 \
        --epochs 800 --hidden 384 --layers 6 --batch-size 2048 --cuda

    # BSIM-AR Transformer (paper config)
    conda run -n pycircuitsim python -m bsimar.cli.train \
        --model transformer --device-type nmos --universal \
        --loss mae --lds --cuda
"""

import sys
import argparse
from pathlib import Path

import torch

from bsimar.config import (
    TECH_CONFIGS, OUTPUT_COLUMNS,
    CHECKPOINT_DIR, DATA_DIR,
    DirectNetConfig, TransformerConfig,
)
from bsimar.utils.seed import set_seed
from bsimar.training.trainer import train_directnet, train_transformer


# ── DirectNet subcommand ─────────────────────────────────────────────────────

def _run_direct(args: argparse.Namespace) -> None:
    if args.universal:
        tech_name = "universal"
        data_path = (Path(args.data) if args.data
                     else DATA_DIR / f"universal_{args.device_type}.npz")
        save_prefix = f"universal_{args.device_type}"
    else:
        tech_name = args.tech.lower()
        data_path = (Path(args.data) if args.data
                     else DATA_DIR / f"{tech_name}_{args.device_type}.npz")
        save_prefix = (args.device_type if tech_name == "asap7"
                       else f"{tech_name}_{args.device_type}")

    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        print("Regenerate via: python external_compact_models/PyCMG/scripts/"
              f"generate_nn_data.py --device {args.device_type} "
              f"{'--universal' if args.universal else f'--tech {tech_name}'}")
        sys.exit(1)

    output_dim = 13 if args.mode in ("direct13", "finetune", "charge-finetune") else 4
    use_charge_consistency = (args.mode == "charge-finetune")

    config = DirectNetConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        trunk_hidden=args.hidden,
        trunk_layers=args.layers,
        patience=args.patience,
    )

    if args.mode in ("finetune", "charge-finetune"):
        config.w_charges = args.w_charges if args.w_charges is not None else 1.5
        config.w_caps = args.w_caps if args.w_caps is not None else 1.0
        if args.resume is None:
            resume = str(CHECKPOINT_DIR / f"{save_prefix}_best.pt")
            config.lr = args.lr if args.lr != 1e-3 else 1e-4
        elif args.resume.lower() == "none":
            resume = None
            config.lr = args.lr
        else:
            resume = args.resume
            config.lr = args.lr if args.lr != 1e-3 else 1e-4
    else:
        if args.w_charges is not None:
            config.w_charges = args.w_charges
        if args.w_caps is not None:
            config.w_caps = args.w_caps
        resume = args.resume

    device_str = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    train_directnet(
        str(data_path), config, device_str,
        save_prefix=save_prefix,
        output_dim=output_dim,
        resume_from=resume,
        use_charge_consistency=use_charge_consistency,
        w_consistency=args.w_consistency,
        w_cond_consistency=args.w_cond_consistency,
    )


# ── Transformer subcommand ───────────────────────────────────────────────────

def _run_transformer(args: argparse.Namespace) -> None:
    if args.norm_mode == "signedlog":
        sys.exit(
            "signedlog normalization is unstable for the autoregressive "
            "transformer (inv_signed_log amplifies AR-accumulated errors "
            "catastrophically — see CLAUDE.md smoke-test notes). Use "
            "--norm-mode zscore."
        )

    if args.universal:
        data_path = (Path(args.data) if args.data
                     else DATA_DIR / f"universal_{args.device_type}.npz")
        save_prefix = f"ar_universal_{args.device_type}"
    else:
        tech_label = args.tech.lower()
        data_path = (Path(args.data) if args.data
                     else DATA_DIR / f"{tech_label}_{args.device_type}.npz")
        save_prefix = f"ar_{tech_label}_{args.device_type}"

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
        w_curr=args.w_curr,
        w_cond=args.w_cond,
        w_charges=args.w_charges if args.w_charges is not None else 0.5,
        w_caps=args.w_caps if args.w_caps is not None else 0.3,
        w_zero_bias=args.w_zero_bias,
    )

    device_str = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_transformer(
        str(data_path),
        save_prefix=save_prefix,
        device_type=args.device_type,
        loss_name=args.loss,
        norm_mode=args.norm_mode,
        apply_filter=not args.no_filter,
        use_lds=args.lds,
        reorder=args.reorder,
        scheduled_sampling=args.scheduled_sampling,
        ss_warmup=args.ss_warmup,
        ss_max_ratio=args.ss_max_ratio,
        consistency_weight=args.consistency_weight,
        curriculum=args.curriculum,
        curriculum_warmup=args.curriculum_warmup,
        config=config,
        device_str=device_str,
        column_names=OUTPUT_COLUMNS,
        overwrite=args.overwrite,
    )


# ── Argparse ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BSIMAR unified training CLI "
                    "(DirectNet baseline or BSIM-AR Transformer)")
    parser.add_argument("--model", choices=["direct", "transformer"],
                        default="direct",
                        help="Which architecture to train "
                             "(direct=DirectNet baseline, transformer=BSIM-AR)")

    # Shared data args
    parser.add_argument("--device-type", choices=["nmos", "pmos"], default="nmos")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to .npz dataset (auto-resolved if omitted)")
    parser.add_argument("--tech", choices=list(TECH_CONFIGS.keys()), default="asap7")
    parser.add_argument("--universal", action="store_true",
                        help="Train a single universal model across all techs/variants")

    # Shared optimization args
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # DirectNet-specific
    parser.add_argument("--mode", choices=["direct4", "direct13", "finetune",
                                           "charge-finetune"],
                        default="direct13",
                        help="[direct only] Training mode")
    parser.add_argument("--hidden", type=int, default=256,
                        help="[direct only] MLP hidden layer dimension")
    parser.add_argument("--layers", type=int, default=5,
                        help="[direct only] MLP hidden layers")
    parser.add_argument("--resume", type=str, default=None,
                        help="[direct only] Checkpoint to resume from")
    parser.add_argument("--w-consistency", type=float, default=1.0,
                        help="[direct only] Weight for charge-cap autograd consistency")
    parser.add_argument("--w-cond-consistency", type=float, default=0.0,
                        help="[direct only] Weight for conductance autograd consistency")

    # Transformer-specific
    parser.add_argument("--norm-mode", choices=["zscore", "signedlog"], default="zscore",
                        help="[transformer] Normalization mode")
    parser.add_argument("--no-filter", action="store_true",
                        help="[transformer] Skip small-value data filtering")
    parser.add_argument("--loss", choices=["direct", "mae", "bni"], default="mae",
                        help="[transformer] Loss function")
    parser.add_argument("--lds", action="store_true",
                        help="[transformer] Enable LDS reweighting")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--reorder", dest="reorder", action="store_true",
                        default=True,
                        help="[transformer] Reorder outputs to paper's "
                             "Q-V → I-V → C-V order (default: on)")
    parser.add_argument("--no-reorder", dest="reorder", action="store_false",
                        help="[transformer] Disable output reordering "
                             "(falls back to OUTPUT_COLUMN_ORDER)")
    parser.add_argument("--scheduled-sampling", action="store_true")
    parser.add_argument("--ss-warmup", type=int, default=100)
    parser.add_argument("--ss-max-ratio", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--curriculum-warmup", type=int, default=50)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="[transformer] Allow overwriting an existing "
                             "<save_prefix>_best.pt checkpoint")

    # Loss weights (shared semantics, used only by their respective models)
    parser.add_argument("--w-curr", type=float, default=1.0,
                        help="[transformer DirectLoss] current weight")
    parser.add_argument("--w-cond", type=float, default=1.0,
                        help="[transformer DirectLoss] conductance weight")
    parser.add_argument("--w-charges", type=float, default=None)
    parser.add_argument("--w-caps", type=float, default=None)
    parser.add_argument("--w-zero-bias", type=float, default=5.0)

    args = parser.parse_args()
    set_seed(args.seed)

    if args.model == "direct":
        _run_direct(args)
    else:
        _run_transformer(args)


if __name__ == "__main__":
    main()
