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
    """Resolve the DirectNet v4 `_best.pt` checkpoint.

    Prefers the v4 universal checkpoint; otherwise falls back to a
    per-tech checkpoint. Caller is responsible for checking `path.exists()`.
    """
    v4_universal = CHECKPOINT_DIR / f"v4_dn_universal_{device_type}_best.pt"
    if v4_universal.exists():
        return v4_universal
    if tech_name and tech_name.lower() != "asap7":
        return CHECKPOINT_DIR / f"{tech_name.lower()}_{device_type}_best.pt"
    return CHECKPOINT_DIR / f"{device_type}_best.pt"


def transformer_checkpoint(device_type: str, tech_name: str | None = None) -> Path:
    """Resolve the BSIM-AR Transformer v4 `_best.pt` checkpoint.

    Prefers v4 universal phys-best > v4 universal plain > per-tech > bare.
    """
    # v4 universal (tech-code embedding)
    v4_phys = CHECKPOINT_DIR / f"v4_universal_{device_type}_best.phys.pt"
    if v4_phys.exists():
        return v4_phys
    v4_plain = CHECKPOINT_DIR / f"v4_universal_{device_type}_best.pt"
    if v4_plain.exists():
        return v4_plain
    if tech_name and tech_name.lower() != "asap7":
        phys = CHECKPOINT_DIR / f"v4_{tech_name.lower()}_{device_type}_best.phys.pt"
        if phys.exists():
            return phys
        return CHECKPOINT_DIR / f"v4_{tech_name.lower()}_{device_type}_best.pt"
    return CHECKPOINT_DIR / f"v4_{device_type}_best.pt"



# ── NN Checkpoint availability ─────────────────────────────────────────────

def get_available_nn_checkpoints(
    device_type: str = "nmos",
) -> dict[str, Path | None]:
    """Check which v4 NN checkpoints exist for *device_type*.

    Returns dict with keys ``'bsimar_v4'``, ``'directnet_v4'``.
    Value is the checkpoint ``Path`` if found, else ``None``.
    """
    from bsimar.config import CHECKPOINT_DIR as _CKPT

    result: dict[str, Path | None] = {}

    # BSIMAR v4 (phys-best > plain)
    for suffix in ("best.phys.pt", "best.pt"):
        p = _CKPT / f"v4_universal_{device_type}_{suffix}"
        if p.exists():
            result["bsimar_v4"] = p
            break
    else:
        result["bsimar_v4"] = None

    # DirectNet v4
    p = _CKPT / f"v4_dn_universal_{device_type}_best.pt"
    result["directnet_v4"] = p if p.exists() else None

    return result


# ── NGSPICE runner helpers ─────────────────────────────────────────────────

NGSPICE_BIN = "/usr/local/ngspice-45.2/bin/ngspice"
_OSDI_PATH_STR = str(OSDI_PATH)


def _parse_ngspice_wrdata(csv_path: Path) -> np.ndarray:
    """Parse NGSPICE ``wrdata`` ASCII output into a numpy array."""
    with csv_path.open() as f:
        lines = f.readlines()
    rows = []
    for line in lines[1:]:
        s = line.strip()
        if s:
            rows.append([float(x) for x in s.split()])
    data = np.array(rows)
    if not np.all(np.isfinite(data)):
        raise RuntimeError(f"NGSPICE output contains NaN/Inf in {csv_path}")
    return data


def run_ngspice_script(
    runner_path: Path,
    log_path: Path,
    csv_path: Path,
    label: str = "",
) -> np.ndarray:
    """Execute an NGSPICE runner script and return parsed wrdata."""
    import subprocess
    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
        capture_output=True, text=True,
    )
    if log_path.exists() and "Fatal:" in log_path.read_text():
        raise RuntimeError(f"NGSPICE OSDI fatal error ({label})")
    if not csv_path.exists():
        tail = log_path.read_text()[-500:] if log_path.exists() else "(no log)"
        raise RuntimeError(
            f"NGSPICE produced no output ({label}): RC={res.returncode}, "
            f"log tail: ...{tail}"
        )
    return _parse_ngspice_wrdata(csv_path)


# ── ASAP7 tech-code guard ─────────────────────────────────────────────────

def tech_code_in_vocab(tech_key: str, vt_key: str, num_codes: int = 18) -> bool:
    """Return True if the (tech, vt) pair maps to a code inside the embedding.

    v4 universal models are trained with ``--num-tech-codes 18`` (indices 0-17).
    ASAP7 codes (18-21) are out-of-range and will crash the embedding layer.
    """
    from bsimar.config import tech_variant_to_code
    code = tech_variant_to_code(tech_key, vt_key)
    return code < num_codes


__all__ = [
    "PROJECT_ROOT",
    "TECH_CONFIGS", "NNTechConfig", "TechConfig",
    "CHECKPOINT_DIR", "DATA_DIR", "OSDI_PATH",
    "nrmse", "mre",
    "directnet_checkpoint", "transformer_checkpoint",
    "get_available_nn_checkpoints",
    "NGSPICE_BIN", "run_ngspice_script",
    "tech_code_in_vocab",
]
