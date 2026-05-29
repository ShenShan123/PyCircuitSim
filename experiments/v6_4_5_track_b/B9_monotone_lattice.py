"""B9 hard-monotone id head training script (V6.4.5 Track-B).

Trains a DirectNet with the entire id column replaced by a globally
monotone-decreasing-in-Vg head (``_MonotoneIdHead``). The 12 other output
columns still come from the shared trunk (SiLU MLP).

Usage:
    CUDA_VISIBLE_DEVICES=3 conda run -n pycircuitsim python \\
        experiments/v6_4_5_track_b/B9_monotone_lattice.py \\
        --device-type nmos --size medium --seed 42 --overwrite

Candidate stems: ``b9_mono_tsmc7_nmos`` / ``b9_mono_tsmc7_pmos``.
NEVER overwrites canonical ``tsmc7_dn_medium_*`` checkpoints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ext = PROJECT_ROOT / "external_compact_models"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_ext))
sys.path.insert(0, str(_ext / "PyCMG"))


def main() -> None:
    ap = argparse.ArgumentParser(description="B9 hard-monotone id head training")
    ap.add_argument("--device-type", choices=["nmos", "pmos"], default="nmos")
    ap.add_argument("--size", choices=["small", "medium", "large"], default="medium")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--mono-id-hidden", type=int, default=128,
                    help="Hidden dim for the MonotoneIdHead (default 128)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from bsimar.utils.seed import set_seed
    set_seed(args.seed)

    from bsimar.config import (
        CHECKPOINT_DIR, DATA_DIR, DirectNetConfig,
        tech_scope_vocab_size,
    )
    from bsimar.data.dataset import load_and_split_bsimar
    from bsimar.data.normalize import OUTPUT_COLUMN_ORDER
    from bsimar.training.trainer import _train_loop
    from bsimar.models.direct_net import DirectNet

    tech_scope = "tsmc7"
    device_type = args.device_type
    data_path = DATA_DIR / f"{tech_scope}_{device_type}.npz"
    if not data_path.exists():
        sys.exit(f"Dataset not found: {data_path}")

    # Non-canonical save stem — never overwrites canonical.
    save_prefix = f"b9_mono_tsmc7_{device_type}"

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"B9 monotone-id-head training on {device}; device_type={device_type}")

    # Per-tech vocab (Rule 19: TSMC7 → 4 codes incl. UNKNOWN)
    num_tech_codes = tech_scope_vocab_size(tech_scope)   # 4
    p_unknown = 0.1
    exclude_techs = {"tsmc5", "tsmc12", "tsmc16", "asap7"}

    SIZE_PRESETS: dict[str, dict] = {
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

    in_dim = train_ds.inputs.shape[1]    # 7
    out_dim = train_ds.outputs.shape[1]  # 13

    # Sign convention: both NMOS and PMOS have id_norm monotone-DECREASING in Vg
    # (NMOS ON: id < 0 and |id| increases with Vg; PMOS source-relative frame: same).
    mono_id_sign = -1.0

    model = DirectNet(
        input_dim=in_dim, hidden_dim=cfg.trunk_hidden,
        n_layers=cfg.trunk_layers + 1, output_dim=out_dim,
        num_tech_codes=num_tech_codes,
        tech_embed_dim=32, tech_embed_dropout=p_unknown,
        unknown_code_id=num_tech_codes - 1,
        mono_full_id=True,
        mono_id_hidden=args.mono_id_hidden,
        mono_id_sign=mono_id_sign,
    ).to(device)

    n_params = model.count_parameters()
    id_head_params = sum(p.numel() for p in model.mono_id.parameters())
    print(f"  Total params: {n_params:,}")
    print(f"  MonotoneIdHead params: {id_head_params:,} "
          f"(+ {n_params - id_head_params:,} trunk)")
    print(f"  mono_id_sign={mono_id_sign:+.0f}, mono_id_hidden={args.mono_id_hidden}")

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
