"""Unified training CLI for DirectNet and BSIMAR Transformer.

Quick presets for fast verification:

    python -m neural_network.cli.train --model direct      --size small  --device-type nmos --cuda
    python -m neural_network.cli.train --model transformer --size medium --device-type nmos --cuda

Override individual knobs (``--epochs``, ``--batch-size``, …) to tune
beyond the preset.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

from neural_network.config import (
    CHECKPOINT_DIR, DATA_DIR,
    DirectNetConfig, TransformerConfig,
    LOCAL_VARIANT_CODES, VALID_TECH_SCOPES, tech_scope_vocab_size,
)
from neural_network.training.trainer import (
    train_directnet, train_transformer,
)
from neural_network.utils.seed import set_seed

import numpy as np

from neural_network.data.dataset import validate_canonical_dataset
from neural_network.data.contracts import (
    FULL_TERMINAL_OUTPUT_CONTRACT,
    REDUCED_OUTPUT_CONTRACT,
    dataset_filename,
)
from neural_network.data.normalize import FULL_TERMINAL_OUTPUT_COLUMN_ORDER


# All TSMC + ASAP7 tech names for the per-tech `--tech-scope` auto-exclude.
_ALL_TECH_NAMES = ("tsmc5", "tsmc6", "tsmc7", "tsmc12", "tsmc16", "asap7")


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
    # V6.5.1 — XL capacity tier (512x8, ~2.1M params). Extends the S/M/L
    # capacity curve one notch. Slightly higher weight_decay than the
    # smaller tiers (set via DirectNetConfig default 1e-5; XL relies on the
    # EMA + early-stop already in the clean recipe to bound device-surface
    # overfit the report flagged at `large`).
    ("direct", "xl"): dict(
        trunk_hidden=512, trunk_layers=8, batch_size=2048,
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
    # V6.8 — Transformer XL tier (~14.8M params), the over-fit-boundary probe
    # mirroring DirectNet's xl. Slightly lower LR for the deeper stack.
    ("transformer", "xl"): dict(
        d_model=384, nhead=8, num_layers=8, dim_feedforward=1536,
        dropout=0.2, batch_size=1024, max_epochs=300,
        patience=80, lr=6e-4),
}


def _parse_class_weights(spec: Optional[str]) -> Optional[Dict[str, float]]:
    """Parse ``"name=w,name=w"`` into {name: w}. None/empty → None."""
    if not spec:
        return None
    out: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(
                f"--class-weights entry {part!r} is not name=weight")
        name, val = part.split("=", 1)
        out[name.strip()] = float(val)
    return out or None


def _parse_class_names(spec: Optional[str]) -> Optional[set[str]]:
    """Parse a comma-separated sample-class list; empty means disabled."""
    if not spec:
        return None
    names = {name.strip() for name in spec.split(",") if name.strip()}
    return names or None


def _resolve_data_path(args: argparse.Namespace) -> Path:
    if args.data:
        return Path(args.data)
    return DATA_DIR / dataset_filename(
        args.tech_scope, args.device_type, args.output_contract,
    )


def _make_save_prefix(args: argparse.Namespace) -> str:
    if args.exp_name:
        return f"{args.exp_name}_{args.device_type}"
    if args.output_contract == FULL_TERMINAL_OUTPUT_CONTRACT:
        tag = {"direct": "dnf", "transformer": "tff"}[args.model]
    else:
        tag = {"direct": "dn", "transformer": "tf"}[args.model]
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
    try:
        validate_canonical_dataset(data_path)
    except ValueError as exc:
        print(f"ERROR: non-canonical training dataset: {exc}")
        sys.exit(1)

    save_prefix = _make_save_prefix(args)
    # --cuda is a demand, not a hint. Falling back to CPU silently costs ~50x
    # wall-clock and is invisible in a campaign log — a V7.3.0 wave lost five
    # hours to a run that had quietly degraded after a sibling GPU faulted and
    # poisoned the driver for new contexts. Note that torch enumerates by
    # CUDA_DEVICE_ORDER (FASTEST_FIRST by default), so the index pinned via
    # CUDA_VISIBLE_DEVICES need not be the nvidia-smi index: a healthy-looking
    # `nvidia-smi` is not evidence that the pinned device works.
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: --cuda requested but torch.cuda.is_available() is False "
              f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}, "
              f"device_count={torch.cuda.device_count()}). Refusing to fall "
              "back to CPU. Check the pinned device with "
              "`CUDA_VISIBLE_DEVICES=<n> python -c \"import torch; "
              "print(torch.cuda.is_available())\"` — a faulted GPU can make "
              "every new context fail while nvidia-smi still lists it.")
        sys.exit(1)
    device_str = "cuda" if args.cuda else "cpu"

    # Per-tech scope auto-derives the exclude set + the embedding vocab.
    # Explicit --exclude-techs / --num-tech-codes still win if both are set.
    if args.tech_scope != "universal":
        auto_excl = {t for t in _ALL_TECH_NAMES if t != args.tech_scope}
        explicit_excl = (
            {t.strip().lower() for t in args.exclude_techs.split(",")}
            if args.exclude_techs else set())
        exclude = explicit_excl | auto_excl
        # Vocab = #variants(scope) + 1 UNKNOWN slot, but only when the flag
        # was left unset. audit C6q: the old test was `== 18`, which also
        # swallowed an *explicit* `--num-tech-codes 18` (the vocab you want
        # when warm-starting a per-tech run from a universal checkpoint) —
        # contradicting the "explicit still wins" promise above.
        if args.num_tech_codes is None:
            args.num_tech_codes = tech_scope_vocab_size(args.tech_scope)
        print(f"  [tech-scope={args.tech_scope}] auto exclude={sorted(exclude)} "
              f"num_tech_codes={args.num_tech_codes}")
    else:
        exclude = (
            {t.strip().lower() for t in args.exclude_techs.split(",")}
            if args.exclude_techs else None)
        if args.num_tech_codes is None:
            args.num_tech_codes = tech_scope_vocab_size("universal")

    full_terminal = args.output_contract == FULL_TERMINAL_OUTPUT_CONTRACT
    if full_terminal:
        incompatible = (
            args.loss_preset != "default"
            or args.sobolev
            or args.charge_sobolev
            or args.monotonic
            or args.ekv_core
        )
        if incompatible:
            print(
                "[error] --output-contract full-terminal is a separate "
                "six-surface family and is incompatible with legacy loss "
                "presets or reduced-head auxiliary paths."
            )
            sys.exit(2)
        if args.subthresh and args.model != "transformer":
            print(
                "[error] full-terminal --subthresh is currently supported "
                "only by the Level76 Transformer family."
            )
            sys.exit(2)
        if args.apply_filter != "off":
            print(
                "[error] --output-contract full-terminal requires "
                "--apply-filter off; its current columns are i_d/i_g/i_b, "
                "not the legacy id filter contract."
            )
            sys.exit(2)
    if args.full_terminal_ar_targets is not None and (
        not full_terminal or args.model != "transformer"
    ):
        print(
            "[error] --full-terminal-ar-targets requires "
            "--model transformer --output-contract full-terminal."
        )
        sys.exit(2)
    if args.autoregressive_training and (
        not full_terminal or args.model != "transformer"
    ):
        print(
            "[error] --autoregressive-training requires "
            "--model transformer --output-contract full-terminal."
        )
        sys.exit(2)

    if (args.model, args.size) not in SIZE_PRESETS:
        print(f"[error] no preset for --model {args.model} "
              f"--size {args.size}")
        sys.exit(2)
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
        swa_mode=args.swa_mode, ema_decay=args.ema_decay,
        apply_filter=(args.apply_filter == "on"),
        class_weights=_parse_class_weights(args.class_weights),
        split_mode=args.split_mode,
        training_overlay_classes=_parse_class_names(
            args.training_overlay_classes),
    )

    # Phase 7 (V6.4.2) soft physics constraints — DirectNet only, opt-in.
    if args.monotonic and args.model != "direct":
        print("[error] --monotonic is a DirectNet-only Phase-7a flag.")
        sys.exit(2)
    if args.model == "direct":
        common["monotonic"] = args.monotonic
        common["ekv_core"] = args.ekv_core
        common["ekv_alpha"] = args.ekv_alpha
        common["ekv_hidden"] = args.ekv_hidden
        common["ekv_lam_lo"] = args.ekv_lam_lo
    if args.ekv_core and args.model != "direct":
        print("[error] --ekv-core is a DirectNet-only (S3) flag.")
        sys.exit(2)
    if args.ekv_core and args.loss_preset != "default":
        print("[error] --ekv-core composes on the id column; it is "
              "incompatible with loss presets that reshuffle/trim outputs.")
        sys.exit(2)

    print(f"\n=== Training {args.model} ({args.size}, "
          f"loss-preset={args.loss_preset}) → {save_prefix} ===")
    # V6.8: the Sobolev / subthreshold / charge-Sobolev aux terms now run for
    # BOTH models (the trainer permutes stats to BSIMAR column order for the
    # Transformer). The e2 output-subset preset is still incompatible.
    if args.sobolev and loss_preset["output_subset"] is not None:
        print("[error] --sobolev needs the id/gm/gds/gmb columns; it is "
              "incompatible with the e2 output-subset preset.")
        sys.exit(2)
    if args.subthresh and loss_preset["output_subset"] is not None:
        print("[error] --subthresh needs the id column; it is incompatible "
              "with the e2 output-subset preset.")
        sys.exit(2)
    if args.charge_sobolev and loss_preset["output_subset"] is not None:
        print("[error] --charge-sobolev needs the qg/qd + cgg/cgd/cdg/cdd "
              "columns; it is incompatible with the e2 output-subset preset.")
        sys.exit(2)
    if args.amp and (args.sobolev or args.charge_sobolev):
        print("[error] --amp is incompatible with the double-backward aux "
              "losses (sobolev / charge-sobolev).")
        sys.exit(2)
    if args.model == "direct":
        cfg = DirectNetConfig(**preset)
        train_directnet(
            str(data_path), config=cfg,
            output_columns=(
                list(FULL_TERMINAL_OUTPUT_COLUMN_ORDER)
                if full_terminal else None
            ),
            column_weights=loss_preset["column_weights"],
            output_subset=loss_preset["output_subset"],
            sobolev=args.sobolev, lam_sobolev=args.lam_sobolev,
            sobolev_floor=args.sobolev_floor,
            sobolev_strong_boost=args.sobolev_strong_boost,
            sobolev_corridor_only=args.sobolev_corridor_only,
            subthresh=args.subthresh, lam_subthresh=args.lam_subthresh,
            subthresh_s2=args.subthresh_s2,
            subthresh_upper=args.subthresh_upper,
            subthresh_floor=args.subthresh_floor,
            subthresh_off_floor=args.subthresh_off_floor,
            subthresh_ceiling_k=args.subthresh_ceiling_k,
            subthresh_ceiling_w=args.subthresh_ceiling_w,
            charge_sobolev=args.charge_sobolev,
            lam_charge_sobolev=args.lam_charge_sobolev,
            charge_sobolev_floor=args.charge_sobolev_floor,
            init_from=args.init_from,
            amp=args.amp,
            **common,
        )
    else:
        if (loss_preset["output_subset"] is not None
                or loss_preset["column_weights"] is not None):
            print("[warn] loss presets are DirectNet-only; "
                  "Transformer ignores them")
        cfg = TransformerConfig(**preset)
        train_transformer(
            str(data_path), config=cfg,
            output_columns=(
                list(FULL_TERMINAL_OUTPUT_COLUMN_ORDER)
                if full_terminal else None
            ),
            sobolev=args.sobolev, lam_sobolev=args.lam_sobolev,
            sobolev_floor=args.sobolev_floor,
            sobolev_strong_boost=args.sobolev_strong_boost,
            sobolev_corridor_only=args.sobolev_corridor_only,
            subthresh=args.subthresh, lam_subthresh=args.lam_subthresh,
            subthresh_s2=args.subthresh_s2,
            subthresh_upper=args.subthresh_upper,
            subthresh_floor=args.subthresh_floor,
            subthresh_off_floor=args.subthresh_off_floor,
            subthresh_ceiling_k=args.subthresh_ceiling_k,
            subthresh_ceiling_w=args.subthresh_ceiling_w,
            charge_sobolev=args.charge_sobolev,
            lam_charge_sobolev=args.lam_charge_sobolev,
            charge_sobolev_floor=args.charge_sobolev_floor,
            init_from=args.init_from,
            amp=args.amp,
            full_terminal_ar_target_dim=args.full_terminal_ar_targets,
            autoregressive_training=args.autoregressive_training,
            **common,
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Unified BSIMAR / DirectNet training CLI")
    p.add_argument("--model", choices=["direct", "transformer"],
                   default="direct")
    p.add_argument("--size", choices=["small", "medium", "large", "xl"],
                   default="medium",
                   help="Architecture-size preset (overridable below)")
    p.add_argument("--device-type", choices=["nmos", "pmos"], default="nmos")
    p.add_argument("--data", type=str, default=None,
                   help="Path to .npz dataset (auto-resolved if omitted)")
    p.add_argument(
        "--output-contract",
        choices=[REDUCED_OUTPUT_CONTRACT, FULL_TERMINAL_OUTPUT_CONTRACT],
        default=REDUCED_OUTPUT_CONTRACT,
        help="Train the legacy 13-head reduced model or the V7.6.0 "
             "six-surface full-terminal DirectNet/BSIM-AR families.",
    )
    p.add_argument(
        "--full-terminal-ar-targets",
        type=int,
        choices=[3, 6],
        default=None,
        help="Level-76 architecture arm: 6 keeps every surface "
             "autoregressive; 3 keeps qg/qb/qd autoregressive and emits "
             "i_d/i_g/i_b through the parallel tail. Unset preserves 6.",
    )
    p.add_argument(
        "--autoregressive-training",
        action="store_true",
        help="Fine-tune LEVEL=76 using the same predicted-prefix rollout "
             "used at inference. Full-terminal Transformer only; unset "
             "preserves teacher forcing.",
    )

    # Per-flag overrides (None means: use the size-preset default)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap dataset rows (after filter / exclude) for "
                        "fast smoke runs")
    p.add_argument(
        "--split-mode", choices=["combo", "random"], default="combo",
        help="Hold out complete (technology, VT, L, NFIN, temperature) "
             "combinations by default; 'random' is interpolation-only.",
    )

    p.add_argument("--cuda", action="store_true")
    p.add_argument("--amp", action="store_true",
                   help="bf16 autocast for train + teacher-forced val "
                        "(V6.8; Transformer wall-clock lever). Incompatible "
                        "with the double-backward aux losses. Final test "
                        "metrics stay fp32.")
    p.add_argument("--seed", type=int, default=42)

    # Tech-code embedding (shared by both models)
    p.add_argument("--exclude-techs", type=str, default=None)
    p.add_argument("--num-tech-codes", type=int, default=None,
                   help="Embedding vocabulary size. Unset (default) derives "
                        "it from --tech-scope: 18 universal (TSMC codes "
                        "0-16 + UNKNOWN 17), variants+1 per-tech. An "
                        "explicit value always wins (audit C6q).")
    p.add_argument("--p-unknown", type=float, default=0.1)

    p.add_argument("--tech-scope",
                   choices=list(VALID_TECH_SCOPES),
                   default="universal",
                   help="Per-tech dedicated training. 'tsmc5' / 'tsmc7' "
                        "auto-set --exclude-techs (all other techs), "
                        "--num-tech-codes (per-tech vocab + UNKNOWN), "
                        "default --data path, and the save_prefix "
                        "(tsmc{5,7}_dn_<size>_<dev>, or *_dnf_* for the "
                        "full DirectNet and *_tff_* for full BSIM-AR) "
                        "recognized by the parser "
                        "preempt cascade.")
    p.add_argument("--exp-name", type=str, default=None,
                   help="Override the auto-generated save_prefix")
    p.add_argument("--overwrite", action="store_true")

    # V6.4.7 S9 — within-run weight averaging (default flag on every
    # campaign arm; plan P6/S9). Default 'none' preserves legacy runs.
    p.add_argument("--swa-mode", choices=["none", "ema", "swa"],
                   default="none",
                   help="Weight averaging: 'ema' = per-step EMA "
                        "(--ema-decay), 'swa' = equal-weight averaging "
                        "from 75%% of max_epochs. Val selection and the "
                        "saved checkpoint use the averaged weights; "
                        "checkpoint key format is unchanged.")
    p.add_argument("--ema-decay", type=float, default=0.999)

    # V6.4.7 S9b (plan rev 3, rulings 3+5) — loader filter exposure +
    # sample_class loss weighting. Defaults preserve legacy behavior.
    p.add_argument("--apply-filter", choices=["on", "off"], default="on",
                   help="'on' (legacy default): drop rows with |id| <= "
                        "1e-15 before splitting. 'off': keep all "
                        "small-current rows (V6.4.7 arms).")
    p.add_argument("--class-weights", type=str, default=None,
                   help="Per-sample_class loss multipliers, e.g. "
                        "'subthresh=4.0,reverse_vds=2.0'. Folded into the "
                        "LDS tensor and renormalized to unit mean per "
                        "target. Requires a sample_class-tagged dataset.")
    p.add_argument(
        "--training-overlay-classes", type=str, default=None,
        help="Comma-separated circuit-derived sample classes that must be "
             "training evidence. With --split-mode combo, complete affected "
             "technology/geometry strata move to training and are reported.",
    )
    p.add_argument("--loss-preset",
                   choices=sorted(LOSS_PRESETS.keys()),
                   default="default",
                   help="DirectNet loss preset (per "
                        "2026-05-08-directnet-target-trim plan): "
                        "default=B0, e1=drop-qs, e2=4-output head, "
                        "e3=down-weight non-load-bearing targets")

    # V6.4.7 S10 (P4) — Sobolev id-derivative consistency, DirectNet only.
    p.add_argument("--sobolev", action="store_true",
                   help="Add the Sobolev id-derivative consistency term "
                        "(autograd ∂id/∂V vs OSDI gm/gds/gmb). DirectNet "
                        "only; requires asinh output norm.")
    p.add_argument("--lam-sobolev", type=float, default=0.1,
                   help="Sobolev term weight λ (default 0.1).")
    p.add_argument("--sobolev-floor", type=float, default=1e-12,
                   help="Trust-floor on |id_true| (A): rows below are "
                        "masked out of the Sobolev term (their OSDI "
                        "gm/gds/gmb are solve-tolerance noise). Default "
                        "1e-12 (regen-v2 data trustworthy to this).")
    p.add_argument("--sobolev-strong-boost", type=float, default=1.0,
                   help="Upweight rows with |id_true| > 1uA in the Sobolev "
                        "term (opamp-gain / conducting corridor). Default "
                        "1.0 = uniform.")
    p.add_argument("--sobolev-corridor-only", action="store_true",
                   help="Restrict the Sobolev term to the conducting "
                        "corridor (|id_true| > 1uA) — focuses op-point slope "
                        "supervision instead of diluting it across weak "
                        "rows. Overrides --sobolev-strong-boost.")
    p.add_argument("--init-from", type=str, default=None,
                   help="Warm-start from a checkpoint stem (under "
                        "CHECKPOINT_DIR) or an explicit *.pt path "
                        "(fine-tune; architecture must match --size / "
                        "--tech-scope).")

    # V6.4.7 S11 (P3) — subthreshold drain-current value+ceiling term.
    p.add_argument("--subthresh", action="store_true",
                   help="Add the subthreshold id value+ceiling term "
                        "(asinh-s2 sub-uA value MAE + sign-agnostic OFF "
                        "ceiling hinge). Targets weak-inversion circuit "
                        "states; requires asinh output norm.")
    p.add_argument("--lam-subthresh", type=float, default=0.05,
                   help="Subthreshold term weight λ (default 0.05). The "
                        "asinh-s2 term is O(1)/row vs the ~5e-3 base MAE, so "
                        "λ scales the subthreshold pull; >0.1 risks swamping "
                        "the strong-inversion fit (screen empirically).")
    p.add_argument("--subthresh-s2", type=float, default=1e-9,
                   help="Small asinh scale for the subthreshold band (A); "
                        "sets sub-uA resolution. Default 1e-9.")
    p.add_argument("--subthresh-upper", type=float, default=1e-6,
                   help="Upper |id_true| mask for the value term (A) — "
                        "above this, the strong-inversion / trip band is "
                        "left to the base loss. Default 1e-6.")
    p.add_argument("--subthresh-floor", type=float, default=1e-12,
                   help="Lower |id_true| trust floor for the value term (A); "
                        "below it only the ceiling hinge applies. Default "
                        "1e-12 (regen-v2 trustworthy).")
    p.add_argument("--subthresh-off-floor", type=float, default=1e-10,
                   help="|id_true| <= this selects hard-OFF rows for the "
                        "sign-agnostic ceiling hinge. Default 1e-10.")
    p.add_argument("--subthresh-ceiling-k", type=float, default=1.0,
                   help="Per-fin OFF ceiling = k·NFIN·1nA; |id_pred| above "
                        "is penalized (never injected). Default 1.0.")
    p.add_argument("--subthresh-ceiling-w", type=float, default=1.0,
                   help="Relative weight of the ceiling hinge vs the value "
                        "term within the subthreshold loss. Default 1.0.")

    # V6.5.2 — charge-derivative (cap) Sobolev consistency, DirectNet only.
    p.add_argument("--charge-sobolev", action="store_true",
                   help="Add the charge-derivative (cap) Sobolev term: couples "
                        "the autograd ∂qg/∂V, ∂qd/∂V the AC/transient solvers "
                        "consume to the supervised cgg/cgd/cdg/cdd columns "
                        "(+,−,−,+ sign map). DirectNet only; asinh output norm.")
    p.add_argument("--lam-charge-sobolev", type=float, default=0.05,
                   help="Charge-Sobolev term weight λ (default 0.05).")
    p.add_argument("--charge-sobolev-floor", type=float, default=1e-19,
                   help="Cap-target magnitude floor (F): rows below are masked "
                        "out of the charge-Sobolev term. Default 1e-19.")

    # Phase 7 (V6.4.2) — soft physics constraints, DirectNet only, default OFF.
    p.add_argument("--monotonic", action="store_true",
                   help="Phase 7a: add a residual sub-network monotone in "
                        "Vg to the DirectNet `id` output column. Shapes the "
                        "network, NOT the loss.")

    # V6.4.8 S3 — EKV analytic backbone + bounded residual, DirectNet only.
    p.add_argument("--ekv-core", action="store_true",
                   help="S3: compose the id column as an EKV/charge-sheet "
                        "analytic core + a tanh-bounded NN residual. Wires the "
                        "id V-shape to physics (self-limiting at Vds->0, "
                        "saturating). DirectNet only; shapes the network, NOT "
                        "the loss. Carries extra `core.*` checkpoint keys.")
    p.add_argument("--ekv-alpha", type=float, default=0.5,
                   help="EKV residual bound: Id = Id_core*(1+alpha*tanh(.)). "
                        "Default 0.5 -> residual within +/-50%% of the core.")
    p.add_argument("--ekv-lam-lo", type=float, default=0.05,
                   help="EKV CLM-band lower floor (min lambda = max r_o = max "
                        "opamp gain). V6.5.8: raise (e.g. 0.10-0.15) to cap the "
                        "vout-weighted-KCL over-flattened r_o and pull opamp "
                        "gain back toward the L72 target.")
    p.add_argument("--ekv-hidden", type=int, default=64,
                   help="Hidden width of the EKV coefficient head. Default 64.")

    args = p.parse_args()
    set_seed(args.seed)
    _run(args)


if __name__ == "__main__":
    main()
