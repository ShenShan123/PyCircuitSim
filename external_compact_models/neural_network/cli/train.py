"""Unified full-terminal training CLI for DirectNet and BSIM-AR."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from neural_network.config import (
    DATA_DIR,
    VALID_TECH_SCOPES,
    DirectNetConfig,
    TransformerConfig,
    tech_scope_vocab_size,
)
from neural_network.data.contracts import dataset_filename
from neural_network.data.dataset import validate_canonical_dataset
from neural_network.training.trainer import train_directnet, train_transformer
from neural_network.utils.seed import set_seed

_ALL_TECH_NAMES = ("tsmc5", "tsmc6", "tsmc7", "tsmc12", "tsmc16", "asap7")

SIZE_PRESETS = {
    ("direct", "small"): {
        "trunk_hidden": 128, "trunk_layers": 3, "batch_size": 2048,
        "max_epochs": 80, "patience": 25, "lr": 1e-3,
    },
    ("direct", "medium"): {
        "trunk_hidden": 256, "trunk_layers": 5, "batch_size": 2048,
        "max_epochs": 200, "patience": 40, "lr": 1e-3,
    },
    ("direct", "large"): {
        "trunk_hidden": 384, "trunk_layers": 6, "batch_size": 2048,
        "max_epochs": 800, "patience": 150, "lr": 1e-3,
    },
    ("direct", "xl"): {
        "trunk_hidden": 512, "trunk_layers": 8, "batch_size": 2048,
        "max_epochs": 800, "patience": 150, "lr": 1e-3,
    },
    ("transformer", "small"): {
        "d_model": 128, "nhead": 4, "num_layers": 3,
        "dim_feedforward": 512, "dropout": 0.1, "batch_size": 1024,
        "max_epochs": 60, "patience": 20, "lr": 8e-4,
    },
    ("transformer", "medium"): {
        "d_model": 192, "nhead": 6, "num_layers": 4,
        "dim_feedforward": 768, "dropout": 0.15, "batch_size": 1024,
        "max_epochs": 150, "patience": 40, "lr": 8e-4,
    },
    ("transformer", "large"): {
        "d_model": 256, "nhead": 8, "num_layers": 6,
        "dim_feedforward": 1024, "dropout": 0.2, "batch_size": 1024,
        "max_epochs": 300, "patience": 80, "lr": 8e-4,
    },
    ("transformer", "xl"): {
        "d_model": 384, "nhead": 8, "num_layers": 8,
        "dim_feedforward": 1536, "dropout": 0.2, "batch_size": 1024,
        "max_epochs": 300, "patience": 80, "lr": 6e-4,
    },
}


def _parse_class_weights(spec: str | None) -> dict[str, float] | None:
    """Parse a comma-separated ``name=weight`` mapping."""
    if not spec:
        return None
    result: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(
                f"--class-weights entry {item!r} is not name=weight"
            )
        name, value = item.split("=", 1)
        result[name.strip()] = float(value)
    return result or None


def _parse_class_names(spec: str | None) -> set[str] | None:
    """Parse a comma-separated sample-class list."""
    if not spec:
        return None
    names = {name.strip() for name in spec.split(",") if name.strip()}
    return names or None


def _resolve_data_path(args: argparse.Namespace) -> Path:
    """Resolve the explicit or canonical six-surface dataset path."""
    if args.data:
        return Path(args.data)
    return DATA_DIR / dataset_filename(args.tech_scope, args.device_type)


def _make_save_prefix(args: argparse.Namespace) -> str:
    """Return the parser-recognized full-terminal checkpoint stem."""
    if args.exp_name:
        return f"{args.exp_name}_{args.device_type}"
    tag = {"direct": "dnf", "transformer": "tff"}[args.model]
    if args.tech_scope != "universal":
        return f"{args.tech_scope}_{tag}_{args.size}_{args.device_type}"
    return f"refac_{tag}_{args.size}_{args.device_type}"


def _run(args: argparse.Namespace) -> None:
    data_path = _resolve_data_path(args)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}")
    try:
        validate_canonical_dataset(data_path)
    except ValueError as exc:
        raise SystemExit(f"ERROR: non-canonical training dataset: {exc}") from exc

    if args.cuda and not torch.cuda.is_available():
        raise SystemExit(
            "ERROR: --cuda requested but torch.cuda.is_available() is False "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}, "
            f"device_count={torch.cuda.device_count()}). Refusing CPU fallback."
        )
    if args.subthresh and args.model != "transformer":
        raise SystemExit(
            "--subthresh is supported only by the BSIM-AR-Full trainer"
        )
    if args.autoregressive_training and args.model != "transformer":
        raise SystemExit(
            "--autoregressive-training requires --model transformer"
        )
    if args.full_terminal_ar_targets is not None and args.model != "transformer":
        raise SystemExit(
            "--full-terminal-ar-targets requires --model transformer"
        )

    if args.tech_scope != "universal":
        auto_excluded = {
            tech for tech in _ALL_TECH_NAMES if tech != args.tech_scope
        }
        explicit_excluded = (
            {tech.strip().lower() for tech in args.exclude_techs.split(",")}
            if args.exclude_techs else set()
        )
        excluded = auto_excluded | explicit_excluded
    else:
        excluded = (
            {tech.strip().lower() for tech in args.exclude_techs.split(",")}
            if args.exclude_techs else None
        )
    if args.num_tech_codes is None:
        args.num_tech_codes = tech_scope_vocab_size(args.tech_scope)

    preset = dict(SIZE_PRESETS[(args.model, args.size)])
    for argument, key in (
        (args.epochs, "max_epochs"),
        (args.batch_size, "batch_size"),
        (args.lr, "lr"),
        (args.patience, "patience"),
    ):
        if argument is not None:
            preset[key] = argument

    common = {
        "device_type": args.device_type,
        "device_str": "cuda" if args.cuda else "cpu",
        "save_prefix": _make_save_prefix(args),
        "exclude_techs": excluded,
        "num_tech_codes": args.num_tech_codes,
        "p_unknown": args.p_unknown,
        "max_rows": args.max_rows,
        "overwrite": args.overwrite,
        "tech_scope": args.tech_scope,
        "swa_mode": args.swa_mode,
        "ema_decay": args.ema_decay,
        "class_weights": _parse_class_weights(args.class_weights),
        "split_mode": args.split_mode,
        "training_overlay_classes": _parse_class_names(
            args.training_overlay_classes
        ),
        "init_from": args.init_from,
        "amp": args.amp,
    }
    print(
        f"\n=== Training {args.model} ({args.size}, full-terminal) "
        f"→ {common['save_prefix']} ==="
    )
    if args.model == "direct":
        train_directnet(
            str(data_path), config=DirectNetConfig(**preset), **common,
        )
        return
    train_transformer(
        str(data_path),
        config=TransformerConfig(**preset),
        full_terminal_ar_target_dim=args.full_terminal_ar_targets,
        autoregressive_training=args.autoregressive_training,
        subthresh=args.subthresh,
        lam_subthresh=args.lam_subthresh,
        subthresh_s2=args.subthresh_s2,
        subthresh_upper=args.subthresh_upper,
        subthresh_floor=args.subthresh_floor,
        subthresh_off_floor=args.subthresh_off_floor,
        subthresh_ceiling_k=args.subthresh_ceiling_k,
        subthresh_ceiling_w=args.subthresh_ceiling_w,
        **common,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a full-terminal DirectNet or BSIM-AR compact model"
    )
    parser.add_argument(
        "--model", choices=["direct", "transformer"], default="direct",
    )
    parser.add_argument(
        "--size", choices=["small", "medium", "large", "xl"],
        default="medium",
    )
    parser.add_argument(
        "--device-type", choices=["nmos", "pmos"], default="nmos",
    )
    parser.add_argument("--data", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--split-mode", choices=["combo", "random"], default="combo",
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-techs", type=str)
    parser.add_argument("--num-tech-codes", type=int)
    parser.add_argument("--p-unknown", type=float, default=0.1)
    parser.add_argument(
        "--tech-scope", choices=list(VALID_TECH_SCOPES), default="universal",
    )
    parser.add_argument("--exp-name", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--swa-mode", choices=["none", "ema", "swa"], default="none",
    )
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--class-weights", type=str)
    parser.add_argument("--training-overlay-classes", type=str)
    parser.add_argument("--init-from", type=str)
    parser.add_argument(
        "--full-terminal-ar-targets", type=int, choices=[3, 6],
    )
    parser.add_argument("--autoregressive-training", action="store_true")
    parser.add_argument("--subthresh", action="store_true")
    parser.add_argument("--lam-subthresh", type=float, default=0.05)
    parser.add_argument("--subthresh-s2", type=float, default=1e-9)
    parser.add_argument("--subthresh-upper", type=float, default=1e-6)
    parser.add_argument("--subthresh-floor", type=float, default=1e-12)
    parser.add_argument("--subthresh-off-floor", type=float, default=1e-10)
    parser.add_argument("--subthresh-ceiling-k", type=float, default=1.0)
    parser.add_argument("--subthresh-ceiling-w", type=float, default=1.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    set_seed(args.seed)
    _run(args)


if __name__ == "__main__":
    main()
