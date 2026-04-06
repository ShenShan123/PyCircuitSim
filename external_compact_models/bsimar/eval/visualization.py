"""Visualization for BSIMAR training results."""

import os
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from bsimar.data.normalize import OUTPUT_COLUMN_ORDER


def plot_scatter_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str,
    target_names: List[str] = OUTPUT_COLUMN_ORDER,
) -> None:
    """Scatter plots of predicted vs true values for each output."""
    n_targets = len(target_names)
    ncols = 4
    nrows = (n_targets + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=100)
    axes = axes.flatten()

    for i, target in enumerate(target_names):
        ax = axes[i]
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        non_zero = y_t != 0
        if non_zero.sum() > 0:
            y_t_nz, y_p_nz = y_t[non_zero], y_p[non_zero]
        else:
            y_t_nz, y_p_nz = y_t, y_p

        r2 = r2_score(y_t_nz, y_p_nz) if len(y_t_nz) > 1 else 0.0
        ax.scatter(y_t_nz, y_p_nz, alpha=0.3, s=2)
        lims = [min(y_t_nz.min(), y_p_nz.min()), max(y_t_nz.max(), y_p_nz.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_title(f"{target} (R2={r2:.4f})", fontsize=11)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)

    for j in range(n_targets, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "scatter_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot saved: {path}")


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_dir: str,
    title_prefix: str = "",
) -> None:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(train_losses, "b-", linewidth=1.5, label="Train")
    ax.plot(val_losses, "r-", linewidth=1.5, label="Validation")
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_title(f"{title_prefix}Loss Curves", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Loss curves saved: {path}")
