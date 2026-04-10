"""Unified CLI for BSIMAR training.

Two models, one CLI:

- ``--model direct``      — DirectNet MLP baseline
- ``--model transformer`` — BSIMAR v3 Transformer (default)

The v3 Transformer recipe is hard-wired inside ``train_transformer``
(asinh+zscore norm, MAE + LDS + Vov-LDS, parallel_caps, grouped_inputs,
AR finetune phase). The only caller-visible knobs are architecture
(``--d-model``, ``--nhead``, ``--num-layers``, ``--dim-feedforward``,
``--dropout``), schedule (``--epochs``, ``--batch-size``, ``--lr``,
``--patience``, ``--ar-finetune-epochs``), and checkpoint naming
(``--exp-name``, ``--overwrite``).

Usage:
    # DirectNet baseline
    python -m bsimar.cli.train \\
        --model direct --device-type nmos --universal --mode direct13 \\
        --epochs 800 --hidden 384 --layers 6 --batch-size 2048 --cuda

    # BSIMAR v3 Transformer (production recipe)
    python -m bsimar.cli.train \\
        --model transformer --device-type nmos --universal --cuda
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
from bsimar.training.trainer import train_directnet, train_transformer, train_transformer_v4


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
        apply_filter=args.apply_filter,
    )


# ── Transformer subcommand ───────────────────────────────────────────────────

def _run_transformer(args: argparse.Namespace) -> None:
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
    )

    device_str = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_transformer(
        str(data_path),
        save_prefix=save_prefix,
        config=config,
        device_str=device_str,
        ar_finetune_epochs=args.ar_finetune_epochs,
        overwrite=args.overwrite,
    )


# ── Transformer v4 subcommand ────────────────────────────────────────────────

def _run_transformer_v4(args: argparse.Namespace) -> None:
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

    device_str = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    held_out = None
    if args.held_out_techs:
        held_out = set(t.strip().lower() for t in args.held_out_techs.split(","))

    train_transformer_v4(
        str(data_path),
        save_prefix=save_prefix,
        device_type=args.device_type,
        config=config,
        device_str=device_str,
        ar_finetune_epochs=args.ar_finetune_epochs,
        overwrite=args.overwrite,
        held_out_techs=held_out,
        num_tech_codes=args.num_tech_codes,
        p_unknown=args.p_unknown,
    )


# ── Argparse ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BSIMAR unified training CLI "
                    "(DirectNet baseline or BSIMAR v3 Transformer)")
    parser.add_argument("--model", choices=["direct", "transformer", "transformer-v4"],
                        default="transformer",
                        help="Which architecture to train "
                             "(direct=DirectNet baseline, "
                             "transformer=BSIMAR v3, "
                             "transformer-v4=BSIMAR v4 tech-code)")

    # Shared data args
    parser.add_argument("--device-type", choices=["nmos", "pmos"], default="nmos")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to .npz dataset (auto-resolved if omitted)")
    parser.add_argument("--tech", choices=list(TECH_CONFIGS.keys()), default="asap7")
    parser.add_argument("--universal", action="store_true",
                        help="Train a single universal model across all techs/variants")

    # Shared optimization args. Defaults reflect the v3 Transformer
    # production recipe (150 epochs, bs 1024, lr 8e-4, patience 150).
    # DirectNet typically overrides --epochs and --batch-size.
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # DirectNet-specific
    parser.add_argument("--mode",
                        choices=["direct4", "direct13", "finetune", "charge-finetune"],
                        default="direct13",
                        help="[direct only] Training mode")
    parser.add_argument("--hidden", type=int, default=384,
                        help="[direct only] MLP hidden layer dimension")
    parser.add_argument("--layers", type=int, default=6,
                        help="[direct only] MLP hidden layers")
    parser.add_argument("--resume", type=str, default=None,
                        help="[direct only] Checkpoint to resume from")
    parser.add_argument("--w-consistency", type=float, default=1.0,
                        help="[direct charge-finetune] autograd charge-cap "
                             "consistency weight")
    parser.add_argument("--w-cond-consistency", type=float, default=0.0,
                        help="[direct charge-finetune] autograd conductance "
                             "consistency weight (0=off)")
    parser.add_argument("--apply-filter", action="store_true",
                        help="[direct only] Drop sub-floor cutoff samples "
                             "(transformer always filters).")

    # Transformer-specific: architecture only. The v3 recipe is hard-
    # wired inside train_transformer -- no --loss, --norm-mode, --lds,
    # --vov-lds, --no-filter, --scheduled-sampling, --curriculum, or
    # --consistency-weight flags anymore. They either always-on or
    # removed as INFEASIBLE in the v3 sprint.
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
                        help="[transformer] Experiment name; overrides "
                             "the default save_prefix")
    parser.add_argument("--overwrite", action="store_true",
                        help="[transformer] Allow overwriting an existing "
                             "<save_prefix>_best.pt checkpoint")
    parser.add_argument("--ar-finetune-epochs", type=int, default=5,
                        help="[transformer] N3 AR-rollout fine-tune epochs "
                             "after the cosine schedule. Default 5 "
                             "(empirically sufficient).")

    # v4-specific flags
    parser.add_argument("--held-out-techs", type=str, default=None,
                        help="[transformer-v4] Comma-separated tech names "
                             "to hold out as test (e.g., 'asap7')")
    parser.add_argument("--num-tech-codes", type=int, default=18,
                        help="[transformer-v4] Tech embedding vocabulary size "
                             "(default 18 = 17 TSMC + 1 UNKNOWN)")
    parser.add_argument("--p-unknown", type=float, default=0.1,
                        help="[transformer-v4] Prob of replacing tech code "
                             "with UNKNOWN during training (default 0.1)")

    # DirectNet loss weights (transformer uses the hard-wired v3 recipe)
    parser.add_argument("--w-charges", type=float, default=None,
                        help="[direct] charges group weight")
    parser.add_argument("--w-caps", type=float, default=None,
                        help="[direct] caps group weight")

    args = parser.parse_args()
    set_seed(args.seed)

    if args.model == "direct":
        _run_direct(args)
    elif args.model == "transformer-v4":
        _run_transformer_v4(args)
    else:
        _run_transformer(args)


if __name__ == "__main__":
    main()
