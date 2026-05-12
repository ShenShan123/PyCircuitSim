"""Train a TSMC5-only residual head on top of a frozen V6 small-probe
DirectNet backbone.

Tier M2 of `docs/superpowers/plans/2026-05-11-tsmc5-inverter-tiered-fix.md`.

Usage::

    PYTHONPATH=external_compact_models PYTHONUNBUFFERED=1 \\
      /home/shenshan/.conda/envs/pycircuitsim/bin/python -u \\
      -m bsimar.training.tsmc5_residual_train \\
        --device-type nmos --epochs 30 --cuda

Output checkpoint::

    external_compact_models/bsimar/checkpoints/
        v6_dn_small_e2_asinh_<dev>_tsmc5res.pt

The norm.npz of the backbone is reused as-is — the residual lives in
the same normalised output space.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from bsimar.config import CHECKPOINT_DIR, DATA_DIR
from bsimar.data.normalize import NormStats, normalizer_from_stats
from bsimar.eval.loo_labels import get_or_build_tech_variant_labels
from bsimar.losses.bni_mae import MAELoss, compute_lds_weights_per_target
from bsimar.models.direct_net import DirectNet
from bsimar.models.tsmc5_residual import TSMC5ResidualHead
from bsimar.utils.seed import set_seed


# E2 4-output column set (matches the V6 small probe).
_OUTPUT_SUBSET = ["id", "qg", "qd", "qb"]
_BACKBONE_PREFIX = "v6_dn_small_e2_asinh"
_NUM_TSMC5_CODES = 4   # tech_codes 0..3
_NUM_TECH_CODES = 18
_TECH_EMBED_DIM = 32   # backbone's tech_embed_dim


def _build_backbone(state_dict: dict) -> DirectNet:
    """Match the V6 small probe architecture exactly from the saved state."""
    net_keys = [k for k in state_dict if k.startswith("net.") and k.endswith(".weight")]
    output_dim = state_dict[net_keys[-1]].shape[0]
    hidden_dim = state_dict[net_keys[-1]].shape[1]
    n_layers = len(net_keys) - 1
    num_tech_codes = state_dict["tech_embedding.weight"].shape[0]
    tech_embed_dim = state_dict["tech_embedding.weight"].shape[1]
    input_dim = state_dict[net_keys[0]].shape[1] - tech_embed_dim
    return DirectNet(
        input_dim=input_dim, hidden_dim=hidden_dim,
        n_layers=n_layers, output_dim=output_dim,
        num_tech_codes=num_tech_codes,
        tech_embed_dim=tech_embed_dim,
        tech_embed_dropout=0.0,   # frozen: no dropout
    )


def _load_tsmc5_data(
    data_path: Path, device_type: str, normalizer, train_ratio: float = 0.85,
    val_ratio: float = 0.10, seed: int = 42, apply_filter: bool = True,
) -> Tuple[TensorDataset, TensorDataset, np.ndarray]:
    """Load the universal_<dev>.npz, keep TSMC5 (codes 0..3) only,
    normalise with the backbone's normalizer, and split."""
    print(f"  Loading {data_path}")
    data = np.load(str(data_path), allow_pickle=True)
    inputs = data["inputs"]
    geometry = data["geometry"]
    outputs = data["outputs"]
    tech_codes = get_or_build_tech_variant_labels(
        str(data_path), device_type, verbose=False)

    if apply_filter:
        # Id-magnitude filter, same as the backbone training (1e-15 A).
        id_col = 0   # OUTPUT_COLUMN_ORDER[0] == "id"
        keep_filter = np.abs(outputs[:, id_col]) > 1e-15
        inputs = inputs[keep_filter]
        geometry = geometry[keep_filter]
        outputs = outputs[keep_filter]
        tech_codes = tech_codes[keep_filter]

    # TSMC5 only.
    is_tsmc5 = np.isin(tech_codes, np.arange(_NUM_TSMC5_CODES))
    inputs = inputs[is_tsmc5]
    geometry = geometry[is_tsmc5]
    outputs = outputs[is_tsmc5]
    tech_codes = tech_codes[is_tsmc5]
    print(f"  TSMC5 rows after filter: {len(outputs):,}")

    # E2 output subset.
    from bsimar.data.normalize import OUTPUT_COLUMN_ORDER
    col_idx = [OUTPUT_COLUMN_ORDER.index(c) for c in _OUTPUT_SUBSET]
    outputs = outputs[:, col_idx]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(outputs))
    n_train = int(len(perm) * train_ratio)
    n_val = int(len(perm) * val_ratio)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]

    def _to_tensors(idx):
        x_norm = normalizer.normalize_inputs(inputs[idx], geometry[idx])
        y_norm = normalizer.normalize_outputs(outputs[idx])
        return (
            torch.tensor(x_norm, dtype=torch.float32),
            torch.tensor(y_norm, dtype=torch.float32),
            torch.tensor(tech_codes[idx], dtype=torch.long),
        )

    train_x, train_y, train_tc = _to_tensors(train_idx)
    val_x, val_y, val_tc = _to_tensors(val_idx)
    print(f"  Split: train={len(train_x):,} val={len(val_x):,}")

    return (
        (train_x, train_y, train_tc),
        (val_x, val_y, val_tc),
        train_y.numpy(),
    )


def train(
    device_type: str = "nmos",
    epochs: int = 30,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    cuda: bool = False,
    overwrite: bool = True,
) -> None:
    set_seed(seed)
    device = (
        torch.device("cuda") if (cuda and torch.cuda.is_available())
        else torch.device("cpu"))

    backbone_path = (
        CHECKPOINT_DIR / f"{_BACKBONE_PREFIX}_{device_type}_best.pt")
    backbone_norm_path = (
        CHECKPOINT_DIR / f"{_BACKBONE_PREFIX}_{device_type}_norm.npz")
    if not backbone_path.exists():
        sys.exit(f"Backbone not found: {backbone_path}")
    if not backbone_norm_path.exists():
        sys.exit(f"Backbone norm.npz not found: {backbone_norm_path}")

    save_path = (
        CHECKPOINT_DIR / f"{_BACKBONE_PREFIX}_{device_type}_tsmc5res.pt")
    if save_path.exists() and not overwrite:
        sys.exit(f"Refusing to overwrite {save_path}; pass --overwrite.")

    print(f"=== Train TSMC5 residual for {device_type} on {device} ===")
    print(f"  Backbone: {backbone_path}")
    print(f"  Save to:  {save_path}")

    # ── Build + freeze backbone ────────────────────────────────────────
    state = torch.load(str(backbone_path), weights_only=True, map_location="cpu")
    backbone = _build_backbone(state)
    backbone.load_state_dict(state)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.to(device)
    print(f"  Backbone params (frozen): {sum(p.numel() for p in backbone.parameters()):,}")

    # ── Build residual head ────────────────────────────────────────────
    norm_stats = NormStats.load(str(backbone_norm_path))
    normalizer = normalizer_from_stats(norm_stats)
    in_dim = 7   # V6 small probe input
    out_dim = len(_OUTPUT_SUBSET)
    head = TSMC5ResidualHead(
        input_dim=in_dim, out_dim=out_dim,
        num_tech_codes=_NUM_TECH_CODES, tech_embed_dim=8,
        hidden=32, num_tsmc5_codes=_NUM_TSMC5_CODES,
    ).to(device)
    print(f"  Residual head params: {head.count_parameters():,}")

    # ── Data ───────────────────────────────────────────────────────────
    data_path = DATA_DIR / f"universal_{device_type}.npz"
    (train_x, train_y, train_tc), (val_x, val_y, val_tc), train_y_np = (
        _load_tsmc5_data(data_path, device_type, normalizer, seed=seed))

    # LDS weights computed on the TSMC5 train slice (same as production).
    print("  Computing LDS weights …")
    lds = compute_lds_weights_per_target(
        train_y_np, n_bins=100, lds_kernel="gaussian", lds_ks=5, lds_sigma=0.8)
    means = lds.mean(axis=0, keepdims=True)
    means[means < 1e-12] = 1.0
    lds = lds / means
    lds_t = torch.tensor(lds, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y, train_tc, lds_t),
        batch_size=batch_size, shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(
        TensorDataset(val_x, val_y, val_tc),
        batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True)

    optimizer = optim.AdamW(
        head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = MAELoss()

    # ── Sanity: epoch-0 val should match the bare backbone exactly ─────
    @torch.no_grad()
    def _eval(use_head: bool) -> float:
        backbone.eval()
        head.eval()
        total = 0.0
        n = 0
        for x, y, tc in val_loader:
            x, y, tc = x.to(device), y.to(device), tc.to(device)
            pred = backbone(x, tech_codes=tc)
            if use_head:
                pred = pred + head(x, tc)
            total += criterion(pred, y).item()
            n += 1
        return total / max(n, 1)

    base_val = _eval(use_head=False)
    init_val = _eval(use_head=True)
    print(f"  Initial val (backbone alone)        : {base_val:.6f}")
    print(f"  Initial val (backbone + zero-residual): {init_val:.6f}")
    assert abs(base_val - init_val) < 1e-6, (
        "Zero-init residual must produce bit-identical val loss")

    # ── Train ──────────────────────────────────────────────────────────
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        head.train()
        backbone.eval()   # frozen — eval mode disables dropout
        total = 0.0
        n = 0
        for x, y, tc, w in train_loader:
            x, y, tc, w = x.to(device), y.to(device), tc.to(device), w.to(device)
            with torch.no_grad():
                pred_back = backbone(x, tech_codes=tc)
            pred = pred_back + head(x, tc)
            loss = criterion(pred, y, weights=w)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        scheduler.step()
        train_loss = total / max(n, 1)
        val_loss = _eval(use_head=True)
        lr_now = scheduler.get_last_lr()[0]

        marker = ""
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            torch.save(head.state_dict(), str(save_path))
            marker = " *best*"
        print(f"  epoch {epoch:3d}/{epochs} | train={train_loss:.6f} "
              f"val={val_loss:.6f} lr={lr_now:.2e}{marker}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. "
          f"Best val={best_val:.6f}  (backbone-alone val={base_val:.6f}).")
    print(f"Saved: {save_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train a TSMC5-only residual head on the V6 small probe")
    p.add_argument("--device-type", choices=["nmos", "pmos"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--overwrite", action="store_true", default=True)
    args = p.parse_args()
    train(
        device_type=args.device_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        cuda=args.cuda,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
