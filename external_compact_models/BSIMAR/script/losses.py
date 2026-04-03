"""Loss functions for BSIM-AR training.

WeightedBNILoss: Batch-Normalized Interpolation loss with optional LDS weights.
LDS (Label Distribution Smoothing) provides per-sample weights to address
imbalanced target distributions.
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
    lds_sigma: float = 2.0,
    strategy: str = "uniform",
) -> np.ndarray:
    """Compute LDS weights for each target dimension.

    Args:
        y_train: (N, D) training targets.
        n_bins: Number of bins for discretization.
        lds_kernel: Kernel type for smoothing.
        lds_ks: Kernel size.
        lds_sigma: Kernel sigma.
        strategy: Binning strategy.

    Returns:
        (N, D) per-sample weights, mean-normalized per target.
    """
    from sklearn.preprocessing import KBinsDiscretizer

    N, D = y_train.shape
    weights_all = np.ones((N, D), dtype=np.float32)

    for d in range(D):
        y_col = y_train[:, d : d + 1]
        # Skip constant columns
        if y_col.max() == y_col.min():
            weights_all[:, d] = 1.0
            continue

        disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
        try:
            discrete = disc.fit_transform(y_col).flatten().astype(int)
        except Exception:
            weights_all[:, d] = 1.0
            continue

        # Empirical count
        counts = np.bincount(discrete, minlength=n_bins).astype(np.float32)
        counts = np.clip(counts, 1e-8, None)

        # Smooth with kernel
        kernel = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        smoothed = convolve1d(counts, weights=kernel, mode="constant")
        smoothed = np.clip(smoothed, 1e-8, None)

        # Inverse frequency weights, clipped and mean-normalized
        eff_counts = smoothed[discrete]
        eff_counts = np.clip(eff_counts, 1e-4, None)
        weights = 1.0 / eff_counts
        weights = np.clip(weights, 0.01, 100.0)
        weights = weights / weights.mean()
        weights_all[:, d] = weights

    return weights_all


class WeightedBNILoss(nn.Module):
    """Batch-Normalized Interpolation loss with optional per-sample weights.

    Normalizes predictions and targets by batch statistics before computing
    a weighted combination of normalized absolute error (NAE) and
    normalized squared error (NSE).
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y_true.shape[0] <= 1:
            base_loss = torch.mean((y_pred - y_true) ** 2)
            if weights is not None:
                if weights.dim() == 1:
                    weights = weights.unsqueeze(1)
                base_loss = torch.mean((y_pred - y_true) ** 2 * weights)
            return base_loss

        # Batch normalization of targets
        mean_true = y_true.mean(dim=0, keepdim=True)
        std_true = y_true.std(dim=0, keepdim=True) + self.epsilon

        y_true_norm = (y_true - mean_true) / std_true
        y_pred_norm = (y_pred - mean_true) / std_true

        abs_err = torch.abs(y_pred_norm - y_true_norm)
        sq_err = (y_pred_norm - y_true_norm) ** 2

        if weights is not None:
            if weights.dim() == 1:
                weights = weights.unsqueeze(1)
            abs_err = abs_err * weights
            sq_err = sq_err * weights

        nae = torch.mean(abs_err)
        nse = torch.mean(sq_err)
        return 0.7 * nae + 0.3 * nse
