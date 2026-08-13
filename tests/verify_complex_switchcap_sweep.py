#!/usr/bin/env python3
"""Parametric sweep driver — switchcap (DirectNet LEVEL=73 vs NGSPICE BSIM-CMG).

Thin wrapper over tests.common.complex_sweep (docs/CHANGELOG.md §V6.4.8+).
Sweeps tech / VT
(sym + asym N/P) / geometry (L / NFIN / P-N ratio) / VDD / per-circuit stimulus,
baseline-gated, with a 3-state exit code (0=all-pass, 1=any-fail,
2=could-not-characterize). The single-point verify_complex_switchcap.py ship gate is
untouched.

Always CPU-pin:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

Usage:
    conda run -n pycircuitsim python tests/verify_complex_switchcap_sweep.py
    conda run -n pycircuitsim python tests/verify_complex_switchcap_sweep.py --tech TSMC16 --dimension vt_asym
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

print = functools.partial(print, flush=True)  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))

from tests.common.complex_sweep import driver_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(driver_main("switchcap"))
