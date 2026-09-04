"""Losses for six-surface DirectNet-Full and BSIM-AR-Full training."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal.windows import triang
from torch import nn

from neural_network.data.contracts import FULL_TERMINAL_OUTPUT_COLUMN_ORDER


def _kernel(kernel: str, ks: int, sigma: float) -> np.ndarray:
    if kernel not in ("gaussian", "triang", "laplace"):
        raise ValueError(f"unsupported LDS kernel: {kernel}")
    half = (ks - 1) // 2
    if kernel == "gaussian":
        base = [0.0] * half + [1.0] + [0.0] * half
        weights = gaussian_filter1d(base, sigma=sigma)
    elif kernel == "triang":
        weights = triang(ks)
    else:
        weights = np.array([
            np.exp(-abs(value) / sigma) / (2.0 * sigma)
            for value in range(-half, half + 1)
        ])
    return weights / weights.max()


def compute_lds_weights_per_target(
    y_train: np.ndarray,
    n_bins: int = 100,
    lds_kernel: str = "gaussian",
    lds_ks: int = 5,
    lds_sigma: float = 0.8,
    strategy: str = "uniform",
) -> np.ndarray:
    """Return mean-normalized LDS weights for every sample and surface."""
    from sklearn.preprocessing import KBinsDiscretizer

    n_rows, n_targets = y_train.shape
    result = np.ones((n_rows, n_targets), dtype=np.float32)
    kernel = _kernel(lds_kernel, lds_ks, lds_sigma)
    for target in range(n_targets):
        column = y_train[:, target:target + 1]
        if column.max() == column.min():
            continue
        discretizer = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy=strategy,
        )
        bin_index = discretizer.fit_transform(column).ravel().astype(int)
        counts = np.clip(
            np.bincount(bin_index, minlength=n_bins).astype(np.float32),
            1e-8,
            None,
        )
        smoothed = np.clip(
            convolve1d(counts, weights=kernel, mode="constant"), 1e-8, None,
        )
        effective = np.clip(smoothed[bin_index], 1e-4, None)
        weights = np.clip(1.0 / effective, 0.01, 100.0)
        result[:, target] = weights / weights.mean()
    return result


class MAELoss(nn.Module):
    """MAE with optional per-sample, per-surface weights."""

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        absolute_error = torch.abs(y_pred - y_true)
        if weights is not None:
            if weights.dim() == 1:
                weights = weights.unsqueeze(1)
            absolute_error = absolute_error * weights
        return absolute_error.mean()


class SubthresholdIdLoss(nn.Module):
    """Resolve weak-inversion ``i_d`` values and suppress hard-OFF leakage."""

    def __init__(
        self,
        lam: float = 0.05,
        column_order: list[str] | None = None,
        s2: float = 1e-9,
        upper: float = 1e-6,
        id_floor: float = 1e-12,
        off_floor: float = 1e-10,
        ceiling_k: float = 1.0,
        ceiling_w: float = 1.0,
        huber_delta: float = 1.0,
        nfin_col: int = 4,
    ) -> None:
        super().__init__()
        self.lam = float(lam)
        self.s2 = float(s2)
        self.upper = float(upper)
        self.id_floor = float(id_floor)
        self.off_floor = float(off_floor)
        self.ceiling_k = float(ceiling_k)
        self.ceiling_w = float(ceiling_w)
        self.huber_delta = float(huber_delta)
        self.nfin_col = int(nfin_col)
        columns = column_order or list(FULL_TERMINAL_OUTPUT_COLUMN_ORDER)
        if "i_d" not in columns:
            raise ValueError("SubthresholdIdLoss requires an 'i_d' column")
        self.id_col = columns.index("i_d")

    @staticmethod
    def _huber(residual: torch.Tensor, delta: float) -> torch.Tensor:
        absolute = residual.abs()
        return torch.where(
            absolute <= delta,
            0.5 * residual * residual,
            delta * (absolute - 0.5 * delta),
        )

    def forward(
        self,
        x_norm: torch.Tensor,
        y_pred_norm: torch.Tensor,
        y_true_norm: torch.Tensor,
        in_mean: torch.Tensor,
        in_std: torch.Tensor,
        out_std: torch.Tensor,
        out_mean: torch.Tensor,
        asinh_scale: torch.Tensor,
    ) -> torch.Tensor:
        index = self.id_col
        output_scale = out_std[index]
        physical_scale = asinh_scale[index]
        predicted_inner = (
            y_pred_norm[:, index] * output_scale + out_mean[index]
        ).clamp(-30.0, 30.0)
        predicted_current = physical_scale * torch.sinh(predicted_inner)

        with torch.no_grad():
            true_inner = (
                y_true_norm[:, index] * output_scale + out_mean[index]
            ).clamp(-30.0, 30.0)
            true_current = physical_scale * torch.sinh(true_inner)
            absolute_true = true_current.abs()
            value_mask = (
                (absolute_true > self.id_floor) & (absolute_true < self.upper)
            ).to(y_pred_norm.dtype)
            off_mask = (absolute_true <= self.off_floor).to(y_pred_norm.dtype)
            nfin_log2 = (
                x_norm[:, self.nfin_col] * in_std[self.nfin_col]
                + in_mean[self.nfin_col]
            )
            nfin = torch.pow(
                torch.as_tensor(2.0, device=x_norm.device, dtype=x_norm.dtype),
                nfin_log2,
            )
            ceiling = (self.ceiling_k * nfin * 1e-9).clamp_min(1e-30)
            true_resolved = torch.asinh(true_current / self.s2)
            ceiling_resolved = torch.asinh(ceiling / self.s2)

        predicted_resolved = torch.asinh(predicted_current / self.s2)
        value_error = self._huber(
            predicted_resolved - true_resolved, self.huber_delta,
        ) * value_mask
        value_term = value_error.sum() / value_mask.sum().clamp_min(1.0)

        off_error = torch.relu(
            torch.asinh(predicted_current.abs() / self.s2) - ceiling_resolved,
        ) * off_mask
        off_term = off_error.sum() / off_mask.sum().clamp_min(1.0)
        return self.lam * (value_term + self.ceiling_w * off_term)


__all__ = [
    "MAELoss",
    "SubthresholdIdLoss",
    "compute_lds_weights_per_target",
]
