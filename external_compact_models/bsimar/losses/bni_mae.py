"""MAE and LDS-weighted loss for BSIMAR training.

- ``MAELoss``: Plain / LDS-weighted MAE loss. Combined with per-target
  LDS weights this is the paper's MAE+LDS loss and the BSIMAR v3
  production loss.
- ``compute_lds_weights_per_target``: Label Distribution Smoothing
  weights (Yang et al., ICML 2021) computed on any 1D+ target
  distribution. BSIMAR v3 uses this on both per-target outputs and on
  ``Vg`` (as a proxy for ``Vov``) to form a two-axis sample weight.

The ``WeightedBNILoss`` (Batch-Normalized Interpolation) was removed in
the v3 sprint: it was never the winning loss and kept ``--loss bni``
alive as a stale CLI option.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal.windows import triang


def get_lds_kernel_window(kernel: str, ks: int, sigma: float) -> np.ndarray:
    """Get smoothing kernel for label distribution smoothing."""
    assert kernel in ["gaussian", "triang", "laplace"]
    half_ks = (ks - 1) // 2
    if kernel == "gaussian":
        base_kernel = [0.0] * half_ks + [1.0] + [0.0] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma)
        kernel_window = kernel_window / kernel_window.max()
    elif kernel == "triang":
        kernel_window = triang(ks)
    else:  # laplace
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2.0 * sigma)
        kernel_window = np.array([laplace(x) for x in range(-half_ks, half_ks + 1)])
        kernel_window = kernel_window / kernel_window.max()
    return kernel_window


def compute_lds_weights_per_target(
    y_train: np.ndarray,
    n_bins: int = 100,
    lds_kernel: str = "gaussian",
    lds_ks: int = 5,
    lds_sigma: float = 0.8,
    strategy: str = "uniform",
) -> np.ndarray:
    """Compute LDS weights for each target dimension.

    Args:
        y_train: (N, D) training targets.
        n_bins: Number of bins for discretization.
        lds_kernel: Kernel type for smoothing.
        lds_ks: Kernel size.
        lds_sigma: Kernel sigma.
        strategy: Binning strategy passed to KBinsDiscretizer.

    Returns:
        (N, D) per-sample weights, mean-normalized per target.
    """
    from sklearn.preprocessing import KBinsDiscretizer

    N, D = y_train.shape
    weights_all = np.ones((N, D), dtype=np.float32)

    for d in range(D):
        y_col = y_train[:, d : d + 1]
        if y_col.max() == y_col.min():
            weights_all[:, d] = 1.0
            continue

        disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
        try:
            discrete = disc.fit_transform(y_col).flatten().astype(int)
        except Exception:
            weights_all[:, d] = 1.0
            continue

        counts = np.bincount(discrete, minlength=n_bins).astype(np.float32)
        counts = np.clip(counts, 1e-8, None)

        kernel = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        smoothed = convolve1d(counts, weights=kernel, mode="constant")
        smoothed = np.clip(smoothed, 1e-8, None)

        eff_counts = smoothed[discrete]
        eff_counts = np.clip(eff_counts, 1e-4, None)
        weights = 1.0 / eff_counts
        weights = np.clip(weights, 0.01, 100.0)
        weights = weights / weights.mean()
        weights_all[:, d] = weights

    return weights_all


class MAELoss(nn.Module):
    """Simple MAE loss with optional per-sample weights.

    When used with pre-computed LDS weights, this becomes the paper's
    MAE+LDS composed loss. Without weights, it is plain MAE.
    """

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


# ── Jacobian-consistency auxiliary loss (V5 Phase C — C1) ────────────────────

# Eight Jacobian channels in (out_name, in_name, target_name) form.
# Inputs are in V-space (Vd, Vg, Vs, Vb at indices 0..3 of the 7-dim
# continuous input).  ∂/∂Vgs ≡ ∂/∂Vg − ∂/∂Vs, but in training we hold
# Vs constant (the data sampler always uses Vs=0), so ∂/∂Vgs = ∂/∂Vg
# and similarly for Vds/Vbs.  This matches the inference glue in
# ``mosfet_directnet.py``.
JAC_CHANNELS = [
    # (out_name, in_idx, target_name)
    ("id", 1, "gm"),     # ∂id/∂Vg
    ("id", 0, "gds"),    # ∂id/∂Vd
    ("id", 3, "gmb"),    # ∂id/∂Vb
    ("qg", 1, "cgg"),    # ∂qg/∂Vg
    ("qg", 0, "cgd"),    # ∂qg/∂Vd
    ("qg", 3, "cgs"),    # ∂qg/∂Vb  (note: paper uses "cgs" name for ∂qg/∂Vbs)
    ("qd", 1, "cdg"),    # ∂qd/∂Vg
    ("qd", 0, "cdd"),    # ∂qd/∂Vd
]


class JacobianConsistencyLoss(nn.Module):
    """Penalize disagreement between autograd Jacobian and supervised targets.

    For each channel ``(out_name, in_idx, target_name)``:
        autograd_in_norm = ∂(out_norm)/∂(in_norm)
        target_in_norm   = transformation of supervised target_name to
                           the "out_norm-per-in_norm" units
        L_chan           = MAE(autograd_in_norm, target_in_norm)
        L_jac            = λ_jac * sum_chan L_chan

    Target transformation depends on the normaliser:

    **zscore** (DirectNet, both inputs and outputs are z-score):
        target_phys = target_norm * out_std + out_mean
        target_in_norm_units = target_phys * (in_std / std_of_out_being_diffed)

        i.e. supervised gm_phys is converted into ``∂id_norm/∂Vg_norm`` units
        by multiplying by ``in_std_Vg / out_std_id``.

    **asinh** (Transformer, inputs zscore + outputs asinh+zscore):
        out_phys = scale * sinh(out_norm * std + mean)
        d(out_norm)/d(in_norm) = (in_std / out_std)
            * d(out_phys)/d(in_phys) / sqrt(scale^2 + out_phys^2)
        → target_in_norm_units = target_phys * (in_std / out_std)
                                 / sqrt(scale^2 + out_phys^2)

    The ``out_phys`` value comes from the ``y_pred`` normalised output via
    inverse asinh transform.
    """

    def __init__(
        self,
        lam: float = 0.1,
        column_order: list[str] | None = None,
        norm_mode: str = "zscore",
    ):
        super().__init__()
        self.lam = float(lam)
        self.norm_mode = norm_mode
        # Default to OUTPUT_COLUMN_ORDER (DirectNet); BSIMAR-Transformer
        # passes BSIMAR_COLUMN_ORDER explicitly.
        from bsimar.data.normalize import OUTPUT_COLUMN_ORDER
        self.column_order = column_order or OUTPUT_COLUMN_ORDER
        self._col_idx = {c: i for i, c in enumerate(self.column_order)}

    def forward(
        self,
        x_norm: torch.Tensor,           # (B, in_dim) requires_grad=True
        y_pred_norm: torch.Tensor,      # (B, out_dim)
        y_true_norm: torch.Tensor,      # (B, out_dim)
        in_std: torch.Tensor,           # (in_dim,)
        out_std: torch.Tensor,          # (out_dim,)
        out_mean: torch.Tensor,         # (out_dim,)
        asinh_scale: torch.Tensor | None = None,  # (out_dim,) — None for zscore
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute λ_jac * sum_channel MAE(autograd, transformed_target).

        Caller is responsible for setting ``x_norm.requires_grad_(True)``
        before the forward pass that produced y_pred_norm, and for using
        ``retain_graph=True`` on any earlier autograd.grad calls.
        """
        device = y_pred_norm.device
        total = torch.tensor(0.0, device=device)
        n_chan = 0

        for out_name, in_idx, tgt_name in JAC_CHANNELS:
            if out_name not in self._col_idx or tgt_name not in self._col_idx:
                continue
            out_idx = self._col_idx[out_name]
            tgt_idx = self._col_idx[tgt_name]

            # Autograd: ∂y_pred[:, out_idx] / ∂x_norm[:, in_idx]
            grad = torch.autograd.grad(
                y_pred_norm[:, out_idx].sum(), x_norm,
                create_graph=True, retain_graph=True,
            )[0][:, in_idx]   # (B,)

            # Convert supervised target_norm -> physical -> normalised-derivative
            # units.
            tgt_norm = y_true_norm[:, tgt_idx]
            if self.norm_mode == "asinh":
                # Recover physical values for the *output that is being
                # differentiated* (out_name), needed for the asinh chain rule
                # √(scale² + y_phys²).
                u_out = y_pred_norm[:, out_idx] * out_std[out_idx] + out_mean[out_idx]
                out_phys_pred = asinh_scale[out_idx] * torch.sinh(u_out)
                # Recover physical target (target_name has its own asinh+zscore).
                u_tgt = tgt_norm * out_std[tgt_idx] + out_mean[tgt_idx]
                tgt_phys = asinh_scale[tgt_idx] * torch.sinh(u_tgt)
                denom = torch.sqrt(
                    asinh_scale[out_idx] * asinh_scale[out_idx]
                    + out_phys_pred * out_phys_pred + 1e-30)
                tgt_in_norm_units = (
                    tgt_phys * in_std[in_idx] / out_std[out_idx] / denom)
            else:  # zscore
                tgt_phys = tgt_norm * out_std[tgt_idx] + out_mean[tgt_idx]
                # d(out_norm)/d(in_norm) = (in_std/out_std) * d(out_phys)/d(in_phys)
                tgt_in_norm_units = (
                    tgt_phys * in_std[in_idx] / out_std[out_idx])

            # Sign convention: in PyCMG, gm = -∂id/∂Vg (id is negative for
            # NMOS ON in terminal-current convention).  We compare in the
            # exact normalised-autograd space the model produces, so the
            # sign of the supervised target must match the normalised
            # derivative.  The supervised gm/gmb stored in the dataset are
            # PyCMG-positive — but the *autograd derivative* through the
            # network gives ∂(id_pred_norm)/∂(in_norm) which has the
            # PyCMG-id sign.  To keep both terms in PyCMG-physical sign
            # convention, flip sign for gm/gmb (where id is negative-going
            # in NMOS strong inversion).  PMOS ON has id positive but the
            # data also has a flipped Vg-Vs frame, so the same flip works.
            # gds is the diagonal so no flip; cgg/cgd/cgs/cdg/cdd are
            # capacitance derivatives of charges — no flip.
            if tgt_name in ("gm", "gmb"):
                tgt_in_norm_units = -tgt_in_norm_units

            ae = torch.abs(grad - tgt_in_norm_units)
            if weights is not None:
                # Use per-sample weight at the *target* column for this channel.
                if weights.dim() == 2:
                    w = weights[:, tgt_idx]
                else:
                    w = weights
                ae = ae * w
            total = total + ae.mean()
            n_chan += 1

        if n_chan == 0:
            return total
        return self.lam * total / n_chan
