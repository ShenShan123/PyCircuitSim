#!/usr/bin/env python3
"""Parametric sweep driver for all four circuit benchmarks (DirectNet vs NGSPICE).

Sweeps tech / VT (sym + asym N/P) / geometry (L / NFIN / P-N ratio) / VDD /
temperature / joint stress / per-circuit stimulus with a complete denominator
even when the baseline misses, and a
3-state exit code (0=all-pass,
1=any-fail, 2=could-not-characterize). The single-point qualification
definitions in ``verify_circuit_{opamp,ring_osc,switchcap,sram_snm}.py`` remain
unchanged;
``verify_circuit_sweep_canaries.py`` holds this driver's builders line-for-line
against the ship decks at the baseline stimulus.

Until V7.5.9 this was four files — ``verify_circuit_{opamp,ringosc,switchcap,
sram}_sweep.py`` — each 33 lines that differed only in the circuit name handed
to ``driver_main``. The circuit is an argument, so it is one now.

Always CPU-pin:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

Usage:
    python tests/simple_circuits/verify_circuit_sweep.py opamp
    python tests/simple_circuits/verify_circuit_sweep.py sram --tech TSMC16 \\
        --dimension vt_asym
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

print = functools.partial(print, flush=True)  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "bsim_cmg" / "tests"))

from tests.common.circuit_sweep import CIRCUITS, driver_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """``<circuit>`` first, everything else straight through to driver_main."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in CIRCUITS:
        print(f"usage: {Path(__file__).name} {{{'|'.join(CIRCUITS)}}} "
              f"[--tech ...] [--dimension ...] [--pin-strict]")
        return 2
    circuit, rest = argv[0], argv[1:]
    sys.argv = [f"{sys.argv[0]} {circuit}", *rest]
    return driver_main(circuit)


if __name__ == "__main__":
    sys.exit(main())
