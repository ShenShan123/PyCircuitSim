#!/usr/bin/env python3
"""V6.3.2 — NN single-device parametric DC verification.

Sweeps DirectNet (LEVEL=73) NMOS & PMOS Id-Vgs against NGSPICE BSIM-CMG ground
truth over device geometry:

  - Baseline: 1 Id-Vgs per tech/device (tech-default L/NFIN/VT)
  - L sweep:    per-tech modelcard L values (skip default)
  - NFIN sweep: symmetric NFIN [5, 10] (skip default 2)
  - VT sweep:   per-tech VT variants (skip default)

The parametric sweep runs only for tech/device pairs that pass baseline.
Off-bin L/NFIN points exercise NN extrapolation beyond the per-tech training
bins — elevated NRMSE/MRE there is expected model behaviour, not a fault.

ASAP7 is out of scope; DirectNet only. For
reproducible results invoke with ``OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`` — the
harness also pins torch to one thread (see tests/common/nn_sweep.py).

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      conda run -n pycircuitsim python tests/single_devices/verify_nn_multi_tech_dc.py \\
        [--tech TSMC5,TSMC7,TSMC12,TSMC16] [--device nmos,pmos]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.nn_sweep import (  # noqa: E402
    NN_TECHS,
    build_dc_parametric,
    make_dc_baseline,
    plot_nn_summary_bar,
    print_nn_summary_table,
    run_nn_multi_tech,
    run_single_nn_dc,
    save_nn_summary_csv,
)

# Env-overridable so parallel checkpoint bake-offs can isolate output dirs
# (same idiom as PYCIRCUITSIM_COMPLEX_RESULTS in tests/common/circuit_benchmarks.py).
import os as _os  # noqa: E402
RESULTS_DIR = Path(_os.environ.get(
    "PYCIRCUITSIM_NN_RESULTS",
    str(PROJECT_ROOT / "tests" / "verify_nn_multi_tech_dc_results")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tech", default=",".join(NN_TECHS),
        help="comma-separated techs (default: all four TSMC nodes)")
    parser.add_argument(
        "--device", default="nmos,pmos",
        help="comma-separated devices: nmos, pmos (default: both)")
    args = parser.parse_args()

    tech_keys = [t.strip() for t in args.tech.split(",") if t.strip()]
    devices = [d.strip().lower() for d in args.device.split(",") if d.strip()]

    for tk in tech_keys:
        if tk not in NN_TECHS:
            print(f"ERROR: tech '{tk}' not in scope {NN_TECHS} "
                  f"(ASAP7 excluded — out of scope)")
            return 2
    for dv in devices:
        if dv not in ("nmos", "pmos"):
            print(f"ERROR: device '{dv}' must be nmos or pmos")
            return 2

    print("=" * 70)
    print("  V6.3.2 — NN single-device parametric DC verification")
    print("=" * 70)
    print(f"  Techs:   {tech_keys}")
    print(f"  Devices: {devices}")
    print(f"  DC acceptance: NRMSE < 10%")

    results = []
    # audit B5d — get_available_checkpoints() now RAISES on a pinned-but-absent
    # stem instead of printing a SKIP with no result row. Catch it here so this
    # gate reports the same clean "ERROR: ... / rc 1" as tests/common/nn_gate.py
    # rather than a traceback.
    try:
        for device in devices:
            results.extend(run_nn_multi_tech(
                tech_keys, device, RESULTS_DIR,
                make_dc_baseline, build_dc_parametric, run_single_nn_dc,
            ))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    counts = print_nn_summary_table(results, kind="dc")
    save_nn_summary_csv(
        results, RESULTS_DIR / "verify_nn_multi_tech_dc_summary.csv", kind="dc")
    plot_nn_summary_bar(
        results, RESULTS_DIR / "verify_nn_multi_tech_dc_summary.png",
        "V6.3.2 NN single-device DC parametric sweep", kind="dc")

    print()
    if counts["fail"] == 0 and counts["error"] == 0:
        print(f"  RESULT: ALL {counts['pass']} configs PASSED")
        return 0
    print(f"  RESULT: {counts['fail']} FAIL, {counts['error']} ERROR "
          f"out of {len(results)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
