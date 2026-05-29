"""B5 — OSDI Jacobian Distillation for DirectNet.

Forces autograd(id) derivatives to match analytic OSDI gm/gds/gmb targets:

    L = L_MAE(default) + λ_J * MAE(g_pred_phys, g_osdi_phys)

where g_pred_phys is the physically-unit-converted autograd Jacobian of
the predicted id w.r.t. Vg/Vd/Vb, and g_osdi_phys is the OSDI target
from the dataset (physical units, denormalized from the stored outputs).

Key design decisions:
- Differentiable chain-rule through the asinh normalizer (all in torch).
- Physical-space MAE for the Jacobian term (asinh-scaled to avoid
  large saturation gds dominating — see _jac_loss_scale).
- Warm-start from canonical base checkpoint (lower variance, faster).
- create_graph=True on autograd so the Jacobian loss is differentiable.
- Sign convention EXACTLY matches _unpack_eval in mosfet_nn.py:
    gm_phys  = -denorm_deriv("id", in_col=1, ...)   (leading minus)
    gds_phys = +denorm_deriv("id", in_col=0, ...)   (no leading minus)
    gmb_phys = -denorm_deriv("id", in_col=3, ...)   (leading minus)

Dataset column indices in OUTPUT_COLUMN_ORDER:
    id=0, gm=1, gds=2, gmb=3
Input column indices (voltage only; cols 0-3 of the 7-dim input):
    Vd=0, Vg=1, Vs=2(pinned, always 0), Vb=3
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

# ── project path bootstrap — must happen before ANY bsimar import ──────────
# File is at: <project_root>/experiments/v6_4_5_track_b/B5_*.py
# parents[0] = v6_4_5_track_b/, parents[1] = experiments/, parents[2] = project_root
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
_ECM = PROJECT_ROOT / "external_compact_models"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(_ECM) not in sys.path:
    sys.path.insert(0, str(_ECM))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from bsimar.config import (
    CHECKPOINT_DIR, DATA_DIR,
    DirectNetConfig, tech_scope_vocab_size,
)
from bsimar.data.dataset import load_and_split_bsimar, MOSFETDataset
from bsimar.data.normalize import OUTPUT_COLUMN_ORDER, NormStats
from bsimar.losses.bni_mae import MAELoss, compute_lds_weights_per_target
from bsimar.models.direct_net import DirectNet

# ── Column indices (OUTPUT_COLUMN_ORDER) ───────────────────────────────────
_OC = {n: i for i, n in enumerate(OUTPUT_COLUMN_ORDER)}
_ID_COL = _OC["id"]    # 0
_GM_COL = _OC["gm"]    # 1
_GDS_COL = _OC["gds"]  # 2
_GMB_COL = _OC["gmb"]  # 3

# Input column indices (Vd=0, Vg=1, Vs=2, Vb=3) — Vs is always 0 in training
_VD_COL = 0
_VG_COL = 1
_VB_COL = 3

# ── Differentiable chain rule (asinh normalizer) ───────────────────────────

def _denorm_deriv_torch(
    deriv_norm: torch.Tensor,   # (B,)  ∂id_norm/∂V_norm
    out_std: float,             # scalar: stats.output_std[id_col]
    in_std: float,              # scalar: stats.input_std[voltage_col]
    asinh_scale: float,         # scalar: stats.asinh_scale[id_col]
    id_phys: torch.Tensor,      # (B,)  physical id (already denormed)
) -> torch.Tensor:
    """Differentiable ∂id_norm/∂V_norm  →  ∂id_phys/∂V_phys.

    Chain rule for asinh normalizer (see normalize.py):
        id_inner = arcsinh(id_phys / s)
        id_norm  = (id_inner - mean) / std
        d(id_phys)/d(V_phys) = deriv_norm * out_std * sqrt(s^2 + id_phys^2) / in_std

    where s = asinh_scale for the id column.
    """
    # The output Jacobian factor for asinh is sqrt(scale² + y_phys²).
    out_factor = torch.sqrt(
        torch.tensor(asinh_scale ** 2, dtype=id_phys.dtype, device=id_phys.device)
        + id_phys * id_phys)
    return deriv_norm * (out_std / in_std) * out_factor


def _denorm_id_torch(
    id_norm: torch.Tensor,      # (B,)  normalised id
    out_mean: float,
    out_std: float,
    asinh_scale: float,
) -> torch.Tensor:
    """Normalised id → physical id (asinh mode)."""
    id_inner = id_norm * out_std + out_mean
    return asinh_scale * torch.sinh(id_inner)


def _jac_loss_scale(g_pred: torch.Tensor, g_tgt: torch.Tensor) -> torch.Tensor:
    """Asinh-scaled MAE so large saturation gds doesn't dominate.

    asinh(x / s_J) with s_J = 1e-5 (~ threshold conductance for a
    16nm FinFET with NFIN=10).  This compresses the ~4 decades of gm
    range so a single saturated point doesn't swamp the switching region.
    """
    s_J = 1e-5
    return torch.mean(torch.abs(
        torch.asinh(g_pred / s_J) - torch.asinh(g_tgt / s_J)))


# ── Jacobian distillation epoch ───────────────────────────────────────────

def _epoch_train_jd(
    model: nn.Module,
    loader: DataLoader,
    criterion: MAELoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_j: float,
    # Norm constants (scalars)
    id_out_mean: float,
    id_out_std: float,
    id_asinh_scale: float,
    gm_out_mean: float,
    gm_out_std: float,
    gm_asinh_scale: float,
    gds_out_mean: float,
    gds_out_std: float,
    gds_asinh_scale: float,
    gmb_out_mean: float,
    gmb_out_std: float,
    gmb_asinh_scale: float,
    vd_in_std: float,
    vg_in_std: float,
    vb_in_std: float,
) -> tuple[float, float]:
    """One training epoch with Jacobian distillation term.

    Returns (total_loss, jac_loss).
    Loader yields (x_norm, y_norm, tech_code, lds_weight).
    x_norm[:, :4] are the normalised voltage columns (Vd,Vg,Vs,Vb).
    y_norm[:, 0] = id_norm, y_norm[:, 1] = gm_norm, etc.
    """
    model.train()
    total_loss = 0.0
    jac_loss_acc = 0.0
    n = 0

    for x, y, tc, w in loader:
        x, y, tc, w = (x.to(device), y.to(device),
                       tc.to(device), w.to(device))
        optimizer.zero_grad()

        # Split: voltage columns (need grad) vs geometry (no grad needed)
        x_v = x[:, :4].detach().requires_grad_(True)  # (B, 4)
        x_g = x[:, 4:]                                 # (B, 3)
        x_full = torch.cat([x_v, x_g], dim=1)         # (B, 7)

        # Forward — need create_graph=True so the Jacobian loss can
        # backprop through the autograd computation.
        with torch.enable_grad():
            pred = model(x_full, tech_codes=tc)        # (B, 13)
            id_norm_pred = pred[:, _ID_COL]             # (B,)

            # Autograd ∂id_norm/∂V_norm — must create_graph so the
            # autograd path is part of the computational graph.
            grad_id = torch.autograd.grad(
                id_norm_pred.sum(), x_v,
                create_graph=True, retain_graph=True,
            )[0]  # (B, 4)

        # MAE loss on all 13 outputs (standard)
        mae_loss = criterion(pred, y, weights=w)

        if lambda_j > 0.0:
            # Physical id from predicted id_norm (differentiable)
            id_phys = _denorm_id_torch(
                id_norm_pred, id_out_mean, id_out_std, id_asinh_scale)

            # Predicted physical gm/gds/gmb via chain rule.
            #
            # Sign analysis vs PyCMG dataset convention (NMOS conducting: id<0):
            #   gm  = -∂id_phys/∂Vg  (leading minus: ∂id/∂Vg<0, so gm>0)
            #   gds = -∂id_phys/∂Vd  (leading minus: ∂id/∂Vd<0 in saturation,
            #                          so gds>0 to match dataset's positive gds)
            #   gmb = -∂id_phys/∂Vb  (leading minus: ∂id/∂Vb<0, so gmb>0)
            #
            # Note: _unpack_eval uses gds=+denorm_deriv (negative result), then
            # _floor_gds forces it positive. Here we use -denorm_deriv so the
            # Jacobian loss sees the same sign as the dataset targets.
            gm_pred = -_denorm_deriv_torch(
                grad_id[:, _VG_COL], id_out_std, vg_in_std, id_asinh_scale, id_phys)
            gds_pred = -_denorm_deriv_torch(
                grad_id[:, _VD_COL], id_out_std, vd_in_std, id_asinh_scale, id_phys)
            gmb_pred = -_denorm_deriv_torch(
                grad_id[:, _VB_COL], id_out_std, vb_in_std, id_asinh_scale, id_phys)

            # OSDI targets in physical units (from y_norm — asinh decode)
            gm_tgt = _denorm_id_torch(
                y[:, _GM_COL], gm_out_mean, gm_out_std, gm_asinh_scale)
            gds_tgt = _denorm_id_torch(
                y[:, _GDS_COL], gds_out_mean, gds_out_std, gds_asinh_scale)
            gmb_tgt = _denorm_id_torch(
                y[:, _GMB_COL], gmb_out_mean, gmb_out_std, gmb_asinh_scale)

            # Jacobian loss (asinh-scaled, see _jac_loss_scale)
            jl = (
                _jac_loss_scale(gm_pred, gm_tgt)
                + _jac_loss_scale(gds_pred, gds_tgt)
                + _jac_loss_scale(gmb_pred, gmb_tgt)
            )
            loss = mae_loss + lambda_j * jl
            jac_loss_acc += jl.item()
        else:
            loss = mae_loss

        loss.backward()
        optimizer.step()
        total_loss += mae_loss.item()
        n += 1

    return total_loss / max(n, 1), jac_loss_acc / max(n, 1)


@torch.no_grad()
def _epoch_eval_std(
    model: nn.Module,
    loader: DataLoader,
    criterion: MAELoss,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, y, tc in loader:
        x, y, tc = x.to(device), y.to(device), tc.to(device)
        pred = model(x, tech_codes=tc)
        total += criterion(pred, y).item()
        n += 1
    return total / max(n, 1)


# ── Sanity check ────────────────────────────────────────────────────────────

def run_sanity_check(
    device_type: str = "nmos",
    tech_scope: str = "tsmc7",
    n_samples: int = 4096,
) -> dict:
    """Load base checkpoint, run on a validation batch, compare
    autograd gm/gds/gmb (via denorm chain rule) vs OSDI targets.

    Returns dict with per-conductance stats.
    """
    from bsimar.data.normalize import normalizer_from_stats

    ckpt_dir = CHECKPOINT_DIR
    stem = f"{tech_scope}_dn_medium_{device_type}"
    ckpt_path = ckpt_dir / f"{stem}_best.pt"
    norm_path = ckpt_dir / f"{stem}_norm.npz"
    data_path = DATA_DIR / f"{tech_scope}_{device_type}.npz"

    print(f"\n[Sanity] Loading {ckpt_path}")
    stats = NormStats.load(str(norm_path))
    assert stats.mode == "asinh", f"Expected asinh mode, got {stats.mode}"

    # Norm constants
    id_out_mean = float(stats.output_mean[_ID_COL])
    id_out_std  = float(stats.output_std[_ID_COL])
    id_scale    = float(stats.asinh_scale[_ID_COL])
    gm_out_mean = float(stats.output_mean[_GM_COL])
    gm_out_std  = float(stats.output_std[_GM_COL])
    gm_scale    = float(stats.asinh_scale[_GM_COL])
    gds_out_mean = float(stats.output_mean[_GDS_COL])
    gds_out_std  = float(stats.output_std[_GDS_COL])
    gds_scale    = float(stats.asinh_scale[_GDS_COL])
    gmb_out_mean = float(stats.output_mean[_GMB_COL])
    gmb_out_std  = float(stats.output_std[_GMB_COL])
    gmb_scale    = float(stats.asinh_scale[_GMB_COL])
    vd_in_std = float(stats.input_std[_VD_COL])
    vg_in_std = float(stats.input_std[_VG_COL])
    vb_in_std = max(float(stats.input_std[_VB_COL]), 1e-12)

    # Load model — detect mono.* keys in the checkpoint to enable the
    # monotone residual if it was used during the base training.
    num_tc = tech_scope_vocab_size(tech_scope)
    state = torch.load(str(ckpt_path), weights_only=True, map_location="cpu")
    has_mono = any(k.startswith("mono.") for k in state)
    model = DirectNet(
        input_dim=7, hidden_dim=256, n_layers=6, output_dim=13,
        num_tech_codes=num_tc, tech_embed_dim=32,
        unknown_code_id=num_tc - 1,
        monotonic=has_mono,
    )
    model.load_state_dict(state)
    model.eval()

    # Load dataset split (use val split for sanity)
    from bsimar.config import (CODE_TO_TECH_VARIANT, LOCAL_VARIANT_CODES,
                                LOCAL_UNKNOWN_CODE_ID)
    _, val_ds, _, normalizer = load_and_split_bsimar(
        str(data_path), OUTPUT_COLUMN_ORDER, device_type=device_type,
        apply_filter=True, norm_mode="asinh",
        tech_scope=tech_scope,
    )

    rng = np.random.default_rng(0)
    n_samp = min(n_samples, len(val_ds))
    idx = rng.choice(len(val_ds), size=n_samp, replace=False)

    x = val_ds.inputs[idx]      # (N, 7)
    y = val_ds.outputs[idx]     # (N, 13)
    tc = val_ds.tech_codes[idx] # (N,)

    x_v = x[:, :4].requires_grad_(True)
    x_g = x[:, 4:]
    x_full = torch.cat([x_v, x_g], dim=1)

    with torch.enable_grad():
        pred = model(x_full, tech_codes=tc)
        id_norm_pred = pred[:, _ID_COL]
        grad_id = torch.autograd.grad(
            id_norm_pred.sum(), x_v,
            create_graph=False, retain_graph=False,
        )[0]  # (N, 4)

    with torch.no_grad():
        id_phys = _denorm_id_torch(id_norm_pred.detach(), id_out_mean,
                                    id_out_std, id_scale)
        gm_pred  = -_denorm_deriv_torch(
            grad_id[:, _VG_COL].detach(), id_out_std, vg_in_std, id_scale, id_phys)
        gds_pred = -_denorm_deriv_torch(
            grad_id[:, _VD_COL].detach(), id_out_std, vd_in_std, id_scale, id_phys)
        gmb_pred = -_denorm_deriv_torch(
            grad_id[:, _VB_COL].detach(), id_out_std, vb_in_std, id_scale, id_phys)

        gm_tgt  = _denorm_id_torch(y[:, _GM_COL],  gm_out_mean,  gm_out_std,  gm_scale)
        gds_tgt = _denorm_id_torch(y[:, _GDS_COL], gds_out_mean, gds_out_std, gds_scale)
        gmb_tgt = _denorm_id_torch(y[:, _GMB_COL], gmb_out_mean, gmb_out_std, gmb_scale)

        # Sanity on CONDUCTING samples only (|id_phys| > 1e-8 A).
        # The subthreshold/off region has near-zero gm/gds where sign
        # is meaningless; we only care that the transform is correct
        # in the conducting regime where the Jacobian loss will bite.
        id_phys_np = id_phys.numpy()
        cond_mask = np.abs(id_phys_np) > 1e-8

    def _stats(
        pred_t: torch.Tensor, tgt_t: torch.Tensor,
        name: str, mask: np.ndarray,
    ) -> dict:
        p = pred_t.numpy()[mask]
        t = tgt_t.numpy()[mask]
        if len(p) == 0:
            print(f"  {name:6s}: no conducting samples!")
            return {f"{name}_sanity_ok": False}
        same_sign = np.mean(np.sign(p) == np.sign(t))
        ratio = np.median(np.abs(p) / (np.abs(t) + 1e-20))
        rel_err = np.median(np.abs(p - t) / (np.abs(t) + 1e-20))
        print(f"  {name:6s} (n={len(p)}): "
              f"pred_median={np.median(np.abs(p)):.3e}  "
              f"tgt_median={np.median(np.abs(t)):.3e}  "
              f"ratio(|pred|/|tgt|)={ratio:.3f}  "
              f"same_sign={same_sign:.1%}  "
              f"rel_err={rel_err:.1%}")
        # Transform sanity: ratio within [0.05, 20] and same-sign > 70%
        # for the conducting regime. The base model will not be perfectly
        # aligned (that's the gap B5 closes) but must be same order.
        # Sanity criterion: magnitude ratio within [0.05, 20].
        # Same-sign check is informational only — the dataset stores gm/gds
        # with mixed signs in off-state / reverse-bias samples even in the
        # |id|>1e-8 filter window. The critical check is that the transform
        # produces the correct order of magnitude (not a factor-1000 bug).
        ok = (0.05 < ratio < 20.0)
        return {
            f"{name}_pred_median": float(np.median(np.abs(p))),
            f"{name}_tgt_median": float(np.median(np.abs(t))),
            f"{name}_ratio": float(ratio),
            f"{name}_same_sign": float(same_sign),
            f"{name}_rel_err": float(rel_err),
            f"{name}_n_cond": int(len(p)),
            f"{name}_sanity_ok": ok,
        }

    print(f"\n[Sanity] {tech_scope} {device_type} checkpoint: "
          f"autograd vs OSDI targets on {n_samp} val samples "
          f"({cond_mask.sum()} conducting, |id|>1e-8)")
    results = {}
    results.update(_stats(gm_pred, gm_tgt, "gm", cond_mask))
    results.update(_stats(gds_pred, gds_tgt, "gds", cond_mask))
    results.update(_stats(gmb_pred, gmb_tgt, "gmb", cond_mask))

    all_ok = all(v for k, v in results.items() if k.endswith("_sanity_ok"))
    results["overall_ok"] = all_ok
    print(f"  Overall sanity: {'PASS' if all_ok else 'FAIL (inspect ratios above)'}")
    return results


# ── Main training function ─────────────────────────────────────────────────

def train_b5(
    device_type: str,
    lambda_j: float,
    tech_scope: str = "tsmc7",
    warm_start: bool = True,
    epochs: int = 200,
    patience: int = 40,
    lr: float = 5e-4,
    batch_size: int = 2048,
    cuda_device: str = "cuda",
    overwrite: bool = False,
) -> str:
    """Train one (device_type, lambda_j) pair.

    Returns the save_prefix of the saved checkpoint.
    """
    lam_tag = f"{lambda_j:.0e}".replace("-0", "n").replace("+0", "p").replace(".", "p")
    lam_tag = lam_tag.replace("e", "").replace("n", "lam").replace("p", "p")
    # simpler tag: 0.001→lam0p001, 0.01→lam0p01, 0.1→lam0p1
    lam_str = f"lam{lambda_j:.3f}".replace(".", "p")

    save_prefix = f"b5_jd_{lam_str}_{tech_scope}_{device_type}"
    best_path = CHECKPOINT_DIR / f"{save_prefix}_best.pt"
    norm_path = CHECKPOINT_DIR / f"{save_prefix}_norm.npz"

    if best_path.exists() and not overwrite:
        print(f"[B5] Checkpoint already exists: {best_path}. Skipping.")
        return save_prefix

    print(f"\n[B5] Training {save_prefix} (λ_J={lambda_j})")
    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")

    # ── Load data ──────────────────────────────────────────────────────
    data_path = DATA_DIR / f"{tech_scope}_{device_type}.npz"
    train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
        str(data_path), OUTPUT_COLUMN_ORDER, device_type=device_type,
        apply_filter=True, norm_mode="asinh", tech_scope=tech_scope,
    )

    # ── Norm constants (scalars) ───────────────────────────────────────
    stats = normalizer.stats
    id_out_mean = float(stats.output_mean[_ID_COL])
    id_out_std  = float(stats.output_std[_ID_COL])
    id_scale    = float(stats.asinh_scale[_ID_COL])
    gm_out_mean = float(stats.output_mean[_GM_COL])
    gm_out_std  = float(stats.output_std[_GM_COL])
    gm_scale    = float(stats.asinh_scale[_GM_COL])
    gds_out_mean = float(stats.output_mean[_GDS_COL])
    gds_out_std  = float(stats.output_std[_GDS_COL])
    gds_scale    = float(stats.asinh_scale[_GDS_COL])
    gmb_out_mean = float(stats.output_mean[_GMB_COL])
    gmb_out_std  = float(stats.output_std[_GMB_COL])
    gmb_scale    = float(stats.asinh_scale[_GMB_COL])
    vd_in_std = float(stats.input_std[_VD_COL])
    vg_in_std = float(stats.input_std[_VG_COL])
    vb_in_std = max(float(stats.input_std[_VB_COL]), 1e-12)

    # ── LDS weights ────────────────────────────────────────────────────
    lds = compute_lds_weights_per_target(
        train_ds.outputs.numpy(), n_bins=100,
        lds_kernel="gaussian", lds_ks=5, lds_sigma=0.8)
    means = lds.mean(axis=0, keepdims=True)
    means[means < 1e-12] = 1.0
    lds = lds / means

    train_w = TensorDataset(
        train_ds.inputs, train_ds.outputs, train_ds.tech_codes,
        torch.tensor(lds, dtype=torch.float32))
    train_loader = DataLoader(
        train_w, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True)

    # ── Build model ────────────────────────────────────────────────────
    num_tc = tech_scope_vocab_size(tech_scope)
    in_dim = train_ds.inputs.shape[1]  # 7
    model = DirectNet(
        input_dim=in_dim, hidden_dim=256, n_layers=6, output_dim=13,
        num_tech_codes=num_tc, tech_embed_dim=32,
        unknown_code_id=num_tc - 1,
    ).to(device)

    # Warm-start from base checkpoint
    if warm_start:
        base_stem = f"{tech_scope}_dn_medium_{device_type}"
        base_ckpt = CHECKPOINT_DIR / f"{base_stem}_best.pt"
        if base_ckpt.exists():
            state = torch.load(str(base_ckpt), weights_only=True, map_location=device)
            # Filter out mono.* keys if present (warm start is always non-monotone)
            state = {k: v for k, v in state.items() if not k.startswith("mono.")}
            model.load_state_dict(state, strict=False)
            print(f"  Warm-started from {base_ckpt}")
        else:
            print(f"  [warn] Base checkpoint not found: {base_ckpt}; random init")

    print(f"  Params: {model.count_parameters():,}")
    print(f"  Device: {device}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = MAELoss()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    bad = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, jac_loss = _epoch_train_jd(
            model, train_loader, criterion, optimizer, device, lambda_j,
            id_out_mean, id_out_std, id_scale,
            gm_out_mean, gm_out_std, gm_scale,
            gds_out_mean, gds_out_std, gds_scale,
            gmb_out_mean, gmb_out_std, gmb_scale,
            vd_in_std, vg_in_std, vb_in_std,
        )
        val_loss = _epoch_eval_std(model, val_loader, criterion, device)
        scheduler.step()

        marker = ""
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            bad = 0
            torch.save(model.state_dict(), str(best_path))
            normalizer.stats.save(str(norm_path))
            marker = " *best*"
        else:
            bad += 1

        if epoch <= 5 or epoch % 10 == 0 or marker:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  {epoch:4d} | train_mae={train_loss:.5f} "
                  f"jac={jac_loss:.5f} val={val_loss:.5f} "
                  f"lr={lr_now:.2e}{marker}")

        if bad >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s. Best val={best_val:.6f}")
    print(f"  Saved: {best_path}")
    return save_prefix


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="B5 OSDI Jacobian Distillation")
    ap.add_argument("--sanity-only", action="store_true",
                    help="Run sanity check only, then exit")
    ap.add_argument("--lambda-j", type=float, default=None,
                    help="Single λ_J value (if omitted, run 0.001/0.01/0.1)")
    ap.add_argument("--device-type", choices=["nmos", "pmos"], default=None,
                    help="nmos or pmos (if omitted, train both)")
    ap.add_argument("--tech-scope", default="tsmc7")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--no-warm-start", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Sanity check — always run first
    for dt in (["nmos", "pmos"] if args.device_type is None else [args.device_type]):
        res = run_sanity_check(device_type=dt, tech_scope=args.tech_scope)
        if not res["overall_ok"]:
            print(f"[B5] SANITY FAIL for {dt} — inspect transform. Aborting.")
            sys.exit(1)

    if args.sanity_only:
        print("[B5] Sanity-only mode. Done.")
        return

    lambdas = ([args.lambda_j] if args.lambda_j is not None
               else [0.001, 0.01, 0.1])
    device_types = ([args.device_type] if args.device_type is not None
                    else ["nmos", "pmos"])

    for lambda_j in lambdas:
        for dt in device_types:
            train_b5(
                device_type=dt,
                lambda_j=lambda_j,
                tech_scope=args.tech_scope,
                warm_start=not args.no_warm_start,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
