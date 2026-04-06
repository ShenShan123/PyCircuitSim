"""Shared helpers for NN compact model verification (LEVEL=73 + LEVEL=74).

Consolidates the path bootstrap, the `nrmse` metric, and the test config
structure that were previously duplicated across:

- tests/verify_nn_multi_tech.py
- tests/verify_nn_universal.py
- tests/verify_nn_universal_v2.py
- tests/verify_nn_tran.py
- tests/verify_nn_leave_one_out.py

Importing from `tests.common.nn` guarantees that:
1. The `bsimar` package is on `sys.path` (so `import bsimar` works).
2. The PyCMG submodule is on `sys.path` (so `import pycmg` works).
3. The shared `nrmse` / `mre` metrics are computed identically everywhere.
4. Checkpoint and data paths are resolved via `bsimar.config`.

The individual verify scripts still own their test orchestration
(sweep definitions, per-technology expectations, plot layouts) — only
the reusable primitives live here.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np


# ── Path bootstrap (runs on import) ──────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_EXTERNAL_DIR = PROJECT_ROOT / "external_compact_models"
_PYCMG_DIR = _EXTERNAL_DIR / "PyCMG"
_PYCMG_TESTS = _PYCMG_DIR / "tests"

for _p in (PROJECT_ROOT, _EXTERNAL_DIR, _PYCMG_DIR, _PYCMG_TESTS):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)


# ── Public re-exports (for convenience) ──────────────────────────────────────

from bsimar.config import (  # noqa: E402
    TECH_CONFIGS, NNTechConfig, TechConfig,  # TechConfig = backward-compat alias
    CHECKPOINT_DIR, DATA_DIR, OSDI_PATH,
    PROCESS_PARAM_NAMES,
)


# ── Metrics ──────────────────────────────────────────────────────────────────

def nrmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Normalized RMSE as percentage of peak-to-peak range.

    Returns 0 when the ground-truth has no dynamic range (all-constant
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

    Args:
        pred: predictions
        true: ground truth
        threshold_rel: fraction of peak |true| below which samples are
            excluded from the MRE average to avoid near-zero blow-up.

    Returns `float('nan')` if no samples survive the filter.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    max_abs = float(np.abs(true).max())
    if max_abs == 0:
        return 0.0
    mask = np.abs(true) > max_abs * threshold_rel
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((true[mask] - pred[mask]) / true[mask]))) * 100.0


# ── Checkpoint path helpers ──────────────────────────────────────────────────

def directnet_checkpoint(device_type: str, tech_name: str | None = None) -> Path:
    """Resolve the DirectNet `_best.pt` checkpoint.

    Prefers the universal checkpoint if it exists; otherwise falls back to a
    per-tech checkpoint. Caller is responsible for checking `path.exists()`.
    """
    universal = CHECKPOINT_DIR / f"universal_{device_type}_best.pt"
    if universal.exists():
        return universal
    if tech_name and tech_name.lower() != "asap7":
        return CHECKPOINT_DIR / f"{tech_name.lower()}_{device_type}_best.pt"
    return CHECKPOINT_DIR / f"{device_type}_best.pt"


def transformer_checkpoint(device_type: str, tech_name: str | None = None) -> Path:
    """Resolve the BSIM-AR Transformer `_best.pt` checkpoint."""
    universal = CHECKPOINT_DIR / f"ar_universal_{device_type}_best.pt"
    if universal.exists():
        return universal
    if tech_name and tech_name.lower() != "asap7":
        return CHECKPOINT_DIR / f"ar_{tech_name.lower()}_{device_type}_best.pt"
    return CHECKPOINT_DIR / f"ar_{device_type}_best.pt"


__all__ = [
    "PROJECT_ROOT",
    "TECH_CONFIGS", "NNTechConfig", "TechConfig",
    "CHECKPOINT_DIR", "DATA_DIR", "OSDI_PATH", "PROCESS_PARAM_NAMES",
    "nrmse", "mre",
    "directnet_checkpoint", "transformer_checkpoint",
]
