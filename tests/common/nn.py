"""Shared helpers for NN compact model verification (LEVEL=73 + LEVEL=74).

Importing this module guarantees that:

1. The ``bsimar`` package is on ``sys.path`` (so ``import bsimar`` works).
2. The PyCMG submodule is on ``sys.path`` (so ``import pycmg`` works).
3. The reusable metrics (``nrmse`` / ``mre``) are computed identically in
   every verify script.

Individual verify scripts still own their orchestration (sweep
definitions, per-tech expectations, plot layouts).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ── Path bootstrap (runs on import) ────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_EXTERNAL_DIR = PROJECT_ROOT / "external_compact_models"
_PYCMG_DIR = _EXTERNAL_DIR / "PyCMG"
_PYCMG_TESTS = _PYCMG_DIR / "tests"

for _p in (PROJECT_ROOT, _EXTERNAL_DIR, _PYCMG_DIR, _PYCMG_TESTS):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)


# ── Metrics ────────────────────────────────────────────────────────────────

def nrmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Normalised RMSE as a percentage of the peak-to-peak range.

    Returns 0 when the ground truth has no dynamic range (all-constant
    signals are treated as a perfect match).
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    ptp = float(true.max() - true.min())
    if ptp < 1e-30:
        return 0.0
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    return rmse / ptp * 100.0


def mre(pred: np.ndarray, true: np.ndarray,
        threshold_rel: float = 0.01) -> float:
    """Mean relative error (percent), excluding near-zero samples.

    Samples with ``|true| < threshold_rel * peak |true|`` are dropped
    so a near-zero ground-truth point cannot blow up the average.
    Returns NaN if no samples survive the filter.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    max_abs = float(np.abs(true).max())
    if max_abs == 0:
        return 0.0
    mask = np.abs(true) > max_abs * threshold_rel
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(
        np.abs((true[mask] - pred[mask]) / true[mask]))) * 100.0


# ── ASAP7 tech-code guard ──────────────────────────────────────────────────

def tech_code_in_vocab(
    tech_key: str, vt_key: str, num_codes: int = 18,
) -> bool:
    """True if the (tech, vt) pair maps to a code inside the embedding.

    v4 universal models are trained with ``--num-tech-codes 18``
    (indices 0-17). ASAP7 codes (18-21) are out-of-range and will crash
    the embedding layer.
    """
    from bsimar.config import tech_variant_to_code
    return tech_variant_to_code(tech_key, vt_key) < num_codes


__all__ = ["PROJECT_ROOT", "nrmse", "mre", "tech_code_in_vocab"]
