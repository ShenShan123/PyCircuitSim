"""Physical-units evaluation metrics for BSIMAR predictions."""

from typing import Dict

import numpy as np

from bsimar.data.normalize import OUTPUT_COLUMN_ORDER


def compute_physical_metrics(
    pred_norm: np.ndarray,
    true_norm: np.ndarray,
    normalizer,
    mre_threshold_pct: float = 0.001,
) -> Dict[str, Dict[str, float]]:
    """Compute per-output metrics after denormalization.

    Works with both `BSIMARNormalizer` (z-score; physical-space metrics
    are well-conditioned) and legacy `Normalizer` (signed-log + z-score;
    physical-space metrics are sensitive to sub-floor outliers because
    `inv_signed_log` exponentiates log-space errors).

    The same per-target valid mask is applied to **NRMSE, MRE, and R²**.
    The mask drops samples where ``|y_true| < mre_threshold_pct * peak``
    (default 0.1% of peak |y|, the PyCMG numerical-noise floor). Without
    this mask the signed-log path produces astronomical NRMSE / negative
    R² because a tiny normalized-space error at a sub-floor sample
    becomes a multi-decade physical-space error after exponentiation.
    For the z-score path the mask is essentially a no-op.

    `NRMSE_norm` and `R2_norm` are also reported. They are computed in
    the trainer's normalized space (no denormalization, no mask) and are
    the safest cross-normalizer comparison metric: they are unaffected
    by the inverse-transform conditioning.

    Args:
        pred_norm: (N, 13) normalized predictions.
        true_norm: (N, 13) normalized ground truth.
        normalizer: Fitted normalizer with denormalize_outputs() method.
        mre_threshold_pct: Per-target valid threshold as a fraction of
            peak |y|. Default 0.001 = 0.1% of peak.

    Returns:
        Dict mapping output name to metric dict.
    """
    metrics: Dict[str, Dict[str, float]] = {}

    pred_phys = normalizer.denormalize_outputs(pred_norm)
    true_phys = normalizer.denormalize_outputs(true_norm)

    # Use the normalizer's own column list when available (E2 4-output
    # head); fall back to the canonical 13-column order.
    column_names = (
        normalizer.stats.output_columns
        if (normalizer.stats is not None
            and normalizer.stats.output_columns is not None)
        else OUTPUT_COLUMN_ORDER)

    for i, name in enumerate(column_names):
        y_t = true_phys[:, i]
        y_p = pred_phys[:, i]

        max_abs = np.abs(y_t).max()
        threshold = max_abs * mre_threshold_pct if max_abs > 0 else 0
        valid = np.abs(y_t) > threshold
        n_valid = int(valid.sum())

        # Physical-space NRMSE / MRE / R² are all computed on the same
        # valid mask. This is the only honest way to compare the two
        # normalizer modes — see the docstring above for why.
        if n_valid > 0:
            y_t_v = y_t[valid]
            y_p_v = y_p[valid]
            data_range = y_t_v.max() - y_t_v.min()
            if data_range > 0:
                rmse = np.sqrt(np.mean((y_p_v - y_t_v) ** 2))
                nrmse_pct = rmse / data_range * 100
            else:
                nrmse_pct = 0.0
            mre_pct = np.mean(np.abs((y_t_v - y_p_v) / y_t_v)) * 100
            ss_res = np.sum((y_t_v - y_p_v) ** 2)
            ss_tot = np.sum((y_t_v - y_t_v.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            nrmse_pct = float("nan")
            mre_pct = float("nan")
            r2 = float("nan")

        # Normalized-space metrics use the full set (no mask) — they are
        # well-defined regardless of the inverse-transform conditioning.
        y_t_n = true_norm[:, i]
        y_p_n = pred_norm[:, i]
        n_range = y_t_n.max() - y_t_n.min()
        if n_range > 0:
            rmse_n = np.sqrt(np.mean((y_p_n - y_t_n) ** 2))
            nrmse_norm_pct = rmse_n / n_range * 100
        else:
            nrmse_norm_pct = 0.0
        ss_res_n = np.sum((y_t_n - y_p_n) ** 2)
        ss_tot_n = np.sum((y_t_n - y_t_n.mean()) ** 2)
        r2_norm = 1.0 - ss_res_n / ss_tot_n if ss_tot_n > 0 else 0.0
        mae_norm = np.mean(np.abs(y_p_n - y_t_n))

        metrics[name] = {
            "NRMSE(%)": nrmse_pct,
            "MRE(%)": mre_pct,
            "R2": r2,
            "NRMSE_norm(%)": nrmse_norm_pct,
            "R2_norm": r2_norm,
            "MAE_norm": mae_norm,
            "n_valid": n_valid,
            "n_total": int(len(y_t)),
        }

    return metrics


def print_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print metrics table.

    Columns:
        NRMSE%/MRE%/R²    — physical-space, computed on the per-target
                            valid mask (|y| > 0.1% of peak).
        NRMSE_n%/R²_n     — normalized-space, full set, no mask.
        MAE_n             — normalized-space mean absolute error.
        n_val/n_tot       — sample counts after / before the mask.
    """
    def _f(v: float, fmt: str = "8.3f") -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "     N/A"
        return f"{v:{fmt}}"

    header = (f"\n{'Target':>8s} | {'NRMSE%':>8s} | {'MRE%':>8s} | "
              f"{'R2':>8s} | {'NRMSE_n%':>9s} | {'R2_n':>8s} | "
              f"{'MAE_n':>8s} | {'n_val/n_tot':>14s}")
    print(header)
    print("-" * len(header.lstrip("\n")))
    for name in metrics.keys():
        m = metrics[name]
        n_str = f"{m['n_valid']}/{m['n_total']}"
        print(
            f"{name:>8s} | {_f(m['NRMSE(%)'])} | {_f(m['MRE(%)'], '8.2f')} | "
            f"{_f(m['R2'], '8.4f')} | {_f(m['NRMSE_norm(%)'], '9.3f')} | "
            f"{_f(m['R2_norm'], '8.4f')} | {m['MAE_norm']:8.4f} | "
            f"{n_str:>14s}"
        )

    def _avg(key: str) -> float:
        vals = [m[key] for m in metrics.values()
                if not (m[key] is None or np.isnan(m[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    avg_nrmse = _avg("NRMSE(%)")
    avg_mre = _avg("MRE(%)")
    avg_r2 = _avg("R2")
    avg_nrmse_n = _avg("NRMSE_norm(%)")
    avg_r2_n = _avg("R2_norm")
    print("-" * len(header.lstrip("\n")))
    print(
        f"{'AVG':>8s} | {_f(avg_nrmse)} | {_f(avg_mre, '8.2f')} | "
        f"{_f(avg_r2, '8.4f')} | {_f(avg_nrmse_n, '9.3f')} | "
        f"{_f(avg_r2_n, '8.4f')} | {'':>8s} |"
    )
