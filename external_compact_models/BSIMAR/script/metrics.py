"""Evaluation metrics for BSIM-AR predictions."""

import numpy as np
from typing import Dict

from nn_model.data.normalize import OUTPUT_COLUMN_ORDER


def compute_physical_metrics(
    pred_norm: np.ndarray,
    true_norm: np.ndarray,
    normalizer,
    mre_threshold_pct: float = 0.01,
) -> Dict[str, Dict[str, float]]:
    """Compute per-output metrics after denormalization.

    Works with both BSIMARNormalizer and nn_model Normalizer
    (duck-typing on denormalize_outputs).

    Args:
        pred_norm: (N, 13) normalized predictions.
        true_norm: (N, 13) normalized ground truth.
        normalizer: Fitted normalizer with denormalize_outputs() method.
        mre_threshold_pct: MRE filter as fraction of peak |y| per target.

    Returns:
        Dict mapping output name to metric dict.
    """
    metrics: Dict[str, Dict[str, float]] = {}

    pred_phys = normalizer.denormalize_outputs(pred_norm)
    true_phys = normalizer.denormalize_outputs(true_norm)

    for i, name in enumerate(OUTPUT_COLUMN_ORDER):
        y_t = true_phys[:, i]
        y_p = pred_phys[:, i]

        # MRE filter: 1% of peak absolute value (avoids near-zero inflation)
        max_abs = np.abs(y_t).max()
        threshold = max_abs * mre_threshold_pct if max_abs > 0 else 0
        valid = np.abs(y_t) > threshold

        # NRMSE (normalized to peak-to-peak range)
        data_range = y_t.max() - y_t.min()
        if data_range > 0:
            rmse = np.sqrt(np.mean((y_p - y_t) ** 2))
            nrmse_pct = rmse / data_range * 100
        else:
            nrmse_pct = 0.0

        # MRE on valid samples
        if valid.sum() > 0:
            mre_pct = np.mean(
                np.abs((y_t[valid] - y_p[valid]) / y_t[valid])) * 100
        else:
            mre_pct = float("nan")

        # R2 (physical space)
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - y_t.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # R2 in normalized space
        y_t_n = true_norm[:, i]
        y_p_n = pred_norm[:, i]
        ss_res_n = np.sum((y_t_n - y_p_n) ** 2)
        ss_tot_n = np.sum((y_t_n - y_t_n.mean()) ** 2)
        r2_norm = 1.0 - ss_res_n / ss_tot_n if ss_tot_n > 0 else 0.0

        # MAE in normalized space
        mae_norm = np.mean(np.abs(pred_norm[:, i] - true_norm[:, i]))

        metrics[name] = {
            "NRMSE(%)": nrmse_pct,
            "MRE(%)": mre_pct,
            "R2": r2,
            "R2_norm": r2_norm,
            "MAE_norm": mae_norm,
        }

    return metrics


def print_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print metrics table."""
    print(f"\n{'Target':>8s} | {'NRMSE%':>8s} | {'MRE%':>8s} | "
          f"{'R2':>8s} | {'R2_norm':>8s} | {'MAE_n':>8s}")
    print("-" * 62)
    for name in OUTPUT_COLUMN_ORDER:
        m = metrics[name]
        mre_str = (f"{m['MRE(%)']:8.2f}"
                   if not np.isnan(m["MRE(%)"]) else "     N/A")
        print(f"{name:>8s} | {m['NRMSE(%)']:8.3f} | {mre_str} | "
              f"{m['R2']:8.4f} | {m['R2_norm']:8.4f} | {m['MAE_norm']:8.4f}")

    avg_nrmse = np.mean([m["NRMSE(%)"] for m in metrics.values()])
    valid_mre = [m["MRE(%)"] for m in metrics.values()
                 if not np.isnan(m["MRE(%)"])]
    avg_mre = np.mean(valid_mre) if valid_mre else float("nan")
    avg_r2 = np.mean([m["R2"] for m in metrics.values()])
    avg_r2_norm = np.mean([m["R2_norm"] for m in metrics.values()])
    print("-" * 62)
    mre_avg_str = (f"{avg_mre:8.2f}"
                   if not np.isnan(avg_mre) else "     N/A")
    print(f"{'AVG':>8s} | {avg_nrmse:8.3f} | {mre_avg_str} | "
          f"{avg_r2:8.4f} | {avg_r2_norm:8.4f} |")
