"""MAE + LDS loss for BSIMAR / DirectNet training.

The production loss is plain MAE multiplied by per-target Label
Distribution Smoothing weights (Yang et al., ICML 2021). Computed
once on the train split and broadcast over batches.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal.windows import triang


def _kernel(kernel: str, ks: int, sigma: float) -> np.ndarray:
    assert kernel in ("gaussian", "triang", "laplace")
    half = (ks - 1) // 2
    if kernel == "gaussian":
        base = [0.0] * half + [1.0] + [0.0] * half
        w = gaussian_filter1d(base, sigma=sigma)
    elif kernel == "triang":
        w = triang(ks)
    else:
        f = lambda x: np.exp(-abs(x) / sigma) / (2.0 * sigma)
        w = np.array([f(x) for x in range(-half, half + 1)])
    return w / w.max()


def compute_lds_weights_per_target(
    y_train: np.ndarray,
    n_bins: int = 100,
    lds_kernel: str = "gaussian",
    lds_ks: int = 5,
    lds_sigma: float = 0.8,
    strategy: str = "uniform",
) -> np.ndarray:
    """Per-sample LDS weight, one column per target. Mean-normalised."""
    from sklearn.preprocessing import KBinsDiscretizer

    n, d = y_train.shape
    weights = np.ones((n, d), dtype=np.float32)
    kernel = _kernel(lds_kernel, lds_ks, lds_sigma)

    for k in range(d):
        col = y_train[:, k:k + 1]
        if col.max() == col.min():
            continue
        try:
            disc = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy=strategy)
            bin_idx = disc.fit_transform(col).flatten().astype(int)
        except Exception:
            continue
        counts = np.clip(
            np.bincount(bin_idx, minlength=n_bins).astype(np.float32),
            1e-8, None)
        smoothed = np.clip(
            convolve1d(counts, weights=kernel, mode="constant"), 1e-8, None)
        eff = np.clip(smoothed[bin_idx], 1e-4, None)
        w = np.clip(1.0 / eff, 0.01, 100.0)
        weights[:, k] = w / w.mean()
    return weights


class MAELoss(nn.Module):
    """MAE with optional per-sample-per-target weights."""

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ae = torch.abs(y_pred - y_true)
        if weights is not None:
            if weights.dim() == 1:
                weights = weights.unsqueeze(1)
            ae = ae * weights
        return ae.mean()


# ── Sobolev id-derivative consistency loss (V6.4.7 P4 / S10) ─────────────────
#
# Couples the autograd ∂id/∂V the SOLVER consumes (NN Rule 1) to the OSDI
# gm/gds/gmb columns the 13-head loss already supervises. The predicted gm/gds
# /gmb *heads* are accurate (~1 % NRMSE), but the autograd slope of the id
# head — the only Jacobian NR uses — drifts (S10 deriv-fidelity ref: NMOS gds
# 6-13 %, PMOS gds 20-70 %). This term supervises that slope.
#
# id-CHANNELS ONLY. The V5 Phase-C 8-channel form (id + 5 cap channels) was
# net-detrimental at S-scale (results/v5_phase_c_jac_loss_ab_2026_05_07.md);
# the chain-rule transform is recovered from `git show 930c274`, but restricted
# to the three id channels per the P4 design.
#
# SAME SPACE AS THE GATE: the comparison happens in the normalized-asinh
# derivative space the deriv-fidelity scorer measures
# (scripts/v6_4_7_deriv_fidelity.py), so the term supervises exactly the
# quantity the P4 promotion gate scores.
#
# SIGN CONVENTION (the P0-I §2 trap, settled empirically — see
# `_verify_sobolev_target` in scripts/v6_4_7_s10_sign_check.py and the scorer
# float64-FD selfcheck): the stored OSDI opvars are positive-magnitude
# conductances while the stored `id` keeps the PyCMG terminal sign, so
#     stored gm  = -d(id_stored)/dVg
#     stored gds = -d(id_stored)/dVd
#     stored gmb = -d(id_stored)/dVb
# i.e. a UNIFORM negation of all three channels maps autograd ∂id/∂V to the
# OSDI columns — for BOTH device types. This is NOT the 930c274 "gds is the
# diagonal so no flip" rule: that comment is wrong for this stored convention
# (it would compare ∂id/∂Vd against +gds, doubling the gds residual). The
# scorer's float64 central-FD selfcheck (<0.5 % median rel err) and a
# well-trained control-v2 net (autograd ∂id/∂Vd already ≈ -gds to ~7 % for
# NMOS) both confirm the uniform negation.

# (target_name, voltage-input column index) — ∂id/∂V_col vs the stored target.
SOBOLEV_ID_CHANNELS: list[tuple[str, int]] = [
    ("gm", 1),    # ∂id/∂Vg  vs gm
    ("gds", 0),   # ∂id/∂Vd  vs gds
    ("gmb", 3),   # ∂id/∂Vb  vs gmb
]


class SobolevIdLoss(nn.Module):
    """λ · mean_chan MAE(autograd ∂id/∂V, -target) in normalized-asinh space.

    DirectNet-only (asinh output norm). For each id channel the autograd
    derivative ``∂(id_norm)/∂(V_norm)`` is compared against the supervised
    gm/gds/gmb target transformed into the same normalized-derivative units
    via the asinh chain rule:

        d(id_norm)/d(V_norm) = (in_std_V / out_std_id)
            · d(id_phys)/d(V) / sqrt(s_id² + id_phys²)
        target_in_norm = -target_phys · in_std_V / out_std_id
                          / sqrt(s_id² + id_phys_pred²)

    Rows with ``|id_true| <= id_floor`` are masked out (their OSDI gm/gds/gmb
    are solve-tolerance noise with random sign — supervising slopes on noise
    is the V5 Phase-C failure mode; P4 rev-3 (iii) trust-floor mask). Rows
    with ``|id_true| > strong_floor`` may be up-weighted by ``strong_boost``
    to emphasise the conducting / opamp-gain corridor (gain ∝ gm/gds at
    strong inversion; P4 rev-2 corridor importance without P5's harvested
    classes).
    """

    def __init__(
        self,
        lam: float = 0.1,
        column_order: list[str] | None = None,
        id_floor: float = 1e-12,
        strong_boost: float = 1.0,
        strong_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lam = float(lam)
        self.id_floor = float(id_floor)
        self.strong_boost = float(strong_boost)
        self.strong_floor = float(strong_floor)
        from bsimar.data.normalize import OUTPUT_COLUMN_ORDER
        self.column_order = column_order or OUTPUT_COLUMN_ORDER
        col = {c: i for i, c in enumerate(self.column_order)}
        if "id" not in col:
            raise ValueError("SobolevIdLoss requires an 'id' output column")
        self.id_col = col["id"]
        # (target_col, voltage_input_col) for channels present in this order.
        self.channels: list[tuple[int, int]] = [
            (col[t], vidx) for (t, vidx) in SOBOLEV_ID_CHANNELS if t in col]

    def forward(
        self,
        x_norm: torch.Tensor,           # (B, in_dim) requires_grad=True
        y_pred_norm: torch.Tensor,      # (B, out_dim)
        y_true_norm: torch.Tensor,      # (B, out_dim)
        in_std: torch.Tensor,           # (in_dim,)
        out_std: torch.Tensor,          # (out_dim,)
        out_mean: torch.Tensor,         # (out_dim,)
        asinh_scale: torch.Tensor,      # (out_dim,)
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Caller must set ``x_norm.requires_grad_(True)`` before the forward
        pass that produced ``y_pred_norm``."""
        idc = self.id_col
        s_id = asinh_scale[idc]
        out_std_id = out_std[idc]

        # Predicted id physical (chain-rule denominator factor).
        u_id = y_pred_norm[:, idc] * out_std_id + out_mean[idc]
        id_phys_pred = s_id * torch.sinh(u_id)
        factor = torch.sqrt(s_id * s_id + id_phys_pred * id_phys_pred + 1e-30)

        # True id physical → trust-floor mask + strong-inversion boost.
        with torch.no_grad():
            u_id_t = y_true_norm[:, idc] * out_std_id + out_mean[idc]
            abs_id_true = torch.abs(s_id * torch.sinh(u_id_t))
            row_w = (abs_id_true > self.id_floor).to(y_pred_norm.dtype)
            if self.strong_boost != 1.0:
                row_w = row_w * torch.where(
                    abs_id_true > self.strong_floor,
                    torch.as_tensor(self.strong_boost, dtype=row_w.dtype,
                                    device=row_w.device),
                    torch.ones((), dtype=row_w.dtype, device=row_w.device))

        # One autograd.grad: all 3 channels differentiate the SAME id head,
        # and DirectNet is per-row independent, so grad of the column-sum is
        # the per-row gradient (same trick as the scorer / _MOSFETNNBase).
        g = torch.autograd.grad(
            y_pred_norm[:, idc].sum(), x_norm,
            create_graph=True, retain_graph=True)[0]   # (B, in_dim)

        norm_rows = row_w.sum().clamp_min(1.0)
        total = torch.zeros((), device=y_pred_norm.device,
                            dtype=y_pred_norm.dtype)
        n_chan = 0
        for tgt_col, vcol in self.channels:
            u_tgt = y_true_norm[:, tgt_col] * out_std[tgt_col] + out_mean[tgt_col]
            tgt_phys = asinh_scale[tgt_col] * torch.sinh(u_tgt)
            # UNIFORM negation (all three id channels) — see class docstring.
            tgt_in_norm = -tgt_phys * in_std[vcol] / out_std_id / factor
            ae = torch.abs(g[:, vcol] - tgt_in_norm) * row_w
            if weights is not None:
                w = weights[:, tgt_col] if weights.dim() == 2 else weights
                ae = ae * w
            total = total + ae.sum() / norm_rows
            n_chan += 1
        if n_chan == 0:
            return total
        return self.lam * total / n_chan
