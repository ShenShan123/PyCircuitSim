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


class SignConsistencyLoss(nn.Module):
    """Penalize wrong-sign drain current predictions in normalized space.

    NMOS: physical Id must be <= 0. In normalized space, correct-sign
    predictions have ``id_norm <= id_zero_norm``.

    PMOS: physical Id must be >= 0, so ``id_norm >= id_zero_norm``.

    The zero-current reference ``id_zero_norm`` is pre-computed from the
    normaliser stats: ``asinh(0) = 0`` then z-scored gives
    ``-mean / std``.

    Args:
        weight: Loss weight multiplier (default 5.0).
    """

    def __init__(self, weight: float = 5.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        y_pred: torch.Tensor,
        id_col: int,
        id_zero_norm: float,
        is_nmos: bool = True,
    ) -> torch.Tensor:
        id_pred = y_pred[:, id_col]
        if is_nmos:
            violation = torch.relu(id_pred - id_zero_norm)
        else:
            violation = torch.relu(id_zero_norm - id_pred)
        if violation.sum() == 0:
            return torch.tensor(0.0, device=y_pred.device)
        return self.weight * (violation ** 2).mean()


class BoundaryLoss(nn.Module):
    """Upweight accuracy at Vds ~ 0 where Id must vanish.

    Identifies near-Vds=0 samples via normalized input columns
    (Vd at index 0, Vs at index 2) and penalizes the deviation of
    the predicted Id from its target at those samples.

    Args:
        vds_threshold_norm: Normalized-space |Vds| below which a sample
            is considered "near zero" (default 0.15).
        weight: Loss weight multiplier (default 2.0).
    """

    def __init__(self, vds_threshold_norm: float = 0.15, weight: float = 2.0):
        super().__init__()
        self.vds_threshold_norm = vds_threshold_norm
        self.weight = weight

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        x: torch.Tensor,
        id_col: int,
    ) -> torch.Tensor:
        vds_norm = x[:, 0] - x[:, 2]  # Vd_norm - Vs_norm
        mask = torch.abs(vds_norm) < self.vds_threshold_norm
        if mask.sum() == 0:
            return torch.tensor(0.0, device=y_pred.device)
        return self.weight * (y_pred[mask, id_col] - y_true[mask, id_col]).abs().mean()
