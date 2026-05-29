"""B7 physics-skeleton training script (V6.4.5 Track-B).

Trains a DirectNet with the physics skeleton active on TSMC7 NMOS or PMOS.
Saves candidate checkpoint under a NON-canonical stem so the canonical
``tsmc7_dn_medium_*`` slots are never overwritten.

Usage:
    CUDA_VISIBLE_DEVICES=3 conda run -n pycircuitsim python \
        experiments/v6_4_5_track_b/B7_train.py \
        --device-type nmos --physics-skeleton --skeleton-eps 0.1 \
        --size medium --seed 42 --overwrite

Candidate stems: ``b7_skel_tsmc7_nmos`` / ``b7_skel_tsmc7_pmos``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ext = PROJECT_ROOT / "external_compact_models"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_ext))
sys.path.insert(0, str(_ext / "PyCMG"))


def main() -> None:
    ap = argparse.ArgumentParser(description="B7 physics-skeleton training")
    ap.add_argument("--device-type", choices=["nmos", "pmos"], default="nmos")
    ap.add_argument("--size", choices=["small", "medium", "large"],
                    default="medium")
    ap.add_argument("--physics-skeleton", action="store_true", default=True,
                    help="Enable physics skeleton (default True for B7)")
    ap.add_argument("--skeleton-eps", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from bsimar.utils.seed import set_seed
    set_seed(args.seed)

    from bsimar.config import (
        CHECKPOINT_DIR, DATA_DIR, DirectNetConfig,
        tech_scope_vocab_size,
    )
    from bsimar.data.dataset import load_and_split_bsimar
    from bsimar.data.normalize import OUTPUT_COLUMN_ORDER, NormStats
    from bsimar.training.trainer import _train_loop
    from bsimar.models.direct_net import DirectNet

    tech_scope = "tsmc7"
    device_type = args.device_type
    data_path = DATA_DIR / f"{tech_scope}_{device_type}.npz"
    if not data_path.exists():
        sys.exit(f"Dataset not found: {data_path}")

    # B7 non-canonical save stem (never overwrites canonical).
    save_prefix = f"b7_skel_tsmc7_{device_type}"

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"B7 skeleton training on {device}; device_type={device_type}")

    # Per-tech vocab (Rule 19).
    num_tech_codes = tech_scope_vocab_size(tech_scope)   # tsmc7 → 4
    p_unknown = 0.1
    exclude_techs = {"tsmc5", "tsmc12", "tsmc16", "asap7"}

    # Size presets (match CLI).
    SIZE_PRESETS = {
        "small":  dict(trunk_hidden=128, trunk_layers=3, batch_size=2048,
                       max_epochs=80,  patience=25, lr=1e-3),
        "medium": dict(trunk_hidden=256, trunk_layers=5, batch_size=2048,
                       max_epochs=200, patience=40, lr=1e-3),
        "large":  dict(trunk_hidden=384, trunk_layers=6, batch_size=2048,
                       max_epochs=800, patience=150, lr=1e-3),
    }
    preset = dict(SIZE_PRESETS[args.size])
    if args.epochs is not None:
        preset["max_epochs"] = args.epochs
    if args.patience is not None:
        preset["patience"] = args.patience
    if args.batch_size is not None:
        preset["batch_size"] = args.batch_size
    if args.lr is not None:
        preset["lr"] = args.lr

    cfg = DirectNetConfig(**preset)
    print(f"  Config: {cfg}")

    norm_mode = "asinh"
    train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
        str(data_path), OUTPUT_COLUMN_ORDER, device_type=device_type,
        train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio,
        apply_filter=True, exclude_techs=exclude_techs,
        norm_mode=norm_mode, max_rows=None,
        output_subset=None,
        tech_scope=tech_scope,
    )

    in_dim = train_ds.inputs.shape[1]   # 7
    out_dim = train_ds.outputs.shape[1] # 13

    # Load norm stats to get input_mean/std for the skeleton.
    stats: NormStats = normalizer.stats
    in_mean = torch.tensor(stats.input_mean, dtype=torch.float32)
    in_std  = torch.tensor(stats.input_std,  dtype=torch.float32)
    vdd_train = 0.75   # TSMC7 training VDD

    # nmos_sign: NMOS ON → id < 0 (PyCMG convention) → sign=-1.
    # PMOS training frame uses source-relative voltages so same sign for
    # the skeleton's Vgs direction; but id convention: PMOS id > 0 when on.
    nmos_sign = -1.0 if device_type == "nmos" else 1.0

    # Extract output normalization for the id column so the skeleton outputs
    # in normalized (asinh z-score) space, consistent with the base head.
    # OUTPUT_COLUMN_ORDER[0] = "id"
    id_col_idx = 0
    id_asinh_scale = float(stats.asinh_scale[id_col_idx]) if stats.asinh_scale is not None else None
    id_out_mean = float(stats.output_mean[id_col_idx])
    id_out_std = float(stats.output_std[id_col_idx])
    print(f"  id output norm: asinh_scale={id_asinh_scale:.4e} "
          f"out_mean={id_out_mean:.4f} out_std={id_out_std:.4f}")

    model = DirectNet(
        input_dim=in_dim, hidden_dim=cfg.trunk_hidden,
        n_layers=cfg.trunk_layers + 1, output_dim=out_dim,
        num_tech_codes=num_tech_codes,
        tech_embed_dim=32, tech_embed_dropout=p_unknown,
        unknown_code_id=num_tech_codes - 1,
        physics_skeleton=True,
        skeleton_in_mean=in_mean,
        skeleton_in_std=in_std,
        skeleton_nmos_sign=nmos_sign,
        skeleton_vdd_train=vdd_train,
        skeleton_eps=args.skeleton_eps,
        skeleton_id_asinh_scale=id_asinh_scale,
        skeleton_id_out_mean=id_out_mean,
        skeleton_id_out_std=id_out_std,
    ).to(device)

    n_params = model.count_parameters()
    print(f"  Skeleton model params: {n_params:,}")
    skel_params = sum(p.numel() for p in model.skeleton.parameters())
    print(f"  Skeleton-specific params: {skel_params} "
          f"(+ {n_params - skel_params} trunk)")

    _train_loop(
        model=model, is_transformer=False,
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        normalizer=normalizer,
        epochs=cfg.max_epochs, batch_size=cfg.batch_size,
        lr=cfg.lr, weight_decay=cfg.weight_decay,
        patience=cfg.patience, save_prefix=save_prefix,
        device=device, overwrite=args.overwrite,
        column_weights=None,
    )


if __name__ == "__main__":
    main()
