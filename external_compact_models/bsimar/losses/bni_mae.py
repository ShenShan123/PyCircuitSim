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


# Back-compat: a no-op stand-in so any stray import does not break.
# The Jacobian-consistency loss was filed as a v5 Phase C dead-end
# (see plan: docs/2026-04-24-v5-inverter-accuracy.md).
class JacobianConsistencyLoss(nn.Module):  # pragma: no cover - deprecated
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.tensor(0.0)
