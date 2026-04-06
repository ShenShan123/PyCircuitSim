"""Evaluation: physical-units metrics and plotting."""

from bsimar.eval.metrics import compute_physical_metrics, print_metrics
from bsimar.eval.visualization import plot_scatter_comparison, plot_loss_curves

__all__ = [
    "compute_physical_metrics", "print_metrics",
    "plot_scatter_comparison", "plot_loss_curves",
]
