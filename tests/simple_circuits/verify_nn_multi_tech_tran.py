#!/usr/bin/env python3
"""V6.3.2 — NN inverter parametric verification (VTC + transient).

Sweeps a DirectNet (LEVEL=73) CMOS inverter against NGSPICE BSIM-CMG ground
truth over circuit-level parameters:

  - Baseline:   1 VTC + 1 transient per tech (tech defaults)
  - P/N ratio:  PMOS fin count vs default (bounded by the TSMC naive-modelcard
                NFIN-group rule, same as the BSIM-CMG harness — typically the
                single point nfin_p=3)
  - VDD:        nominal +/- 0.1 V                 (VTC + transient)
  - Cload:      5, 50, 100 fF                     (transient only)
  - Input slew: tr=tf 10, 500 ps                  (transient only)
  - Pulse width: 0.2, 0.5, 2.0 ns                 (transient only)

The parametric sweep runs only for tech/analysis pairs that pass baseline.

ASAP7 is out of scope; DirectNet only. Run against
a stable checkpoint set and with ``OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`` — the
inverter trip point has gain ~-15..-30 that amplifies any NN-weight change
(e.g. a concurrent retrain overwriting the checkpoints) ~20x into the VTC.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      conda run -n pycircuitsim python tests/simple_circuits/verify_nn_multi_tech_tran.py \\
        [--tech TSMC5,TSMC7,TSMC12,TSMC16] [--analysis vtc,tran]
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
    build_inv_parametric,
    make_inv_baseline,
    plot_nn_summary_bar,
    print_nn_summary_table,
    run_nn_multi_tech,
    run_single_nn_inv,
    save_nn_summary_csv,
)

# Env-overridable so parallel checkpoint bake-offs can isolate output dirs
# (same idiom as PYCIRCUITSIM_COMPLEX_RESULTS in tests/common/circuit_benchmarks.py).
import os as _os  # noqa: E402
RESULTS_DIR = Path(_os.environ.get(
    "PYCIRCUITSIM_NN_RESULTS",
    str(PROJECT_ROOT / "tests" / "verify_nn_multi_tech_tran_results")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tech", default=",".join(NN_TECHS),
        help="comma-separated techs (default: all four TSMC nodes)")
    parser.add_argument(
        "--analysis", default="vtc,tran",
        help="comma-separated analyses: vtc, tran (default: both)")
    args = parser.parse_args()

    tech_keys = [t.strip() for t in args.tech.split(",") if t.strip()]
    analyses = [a.strip().lower() for a in args.analysis.split(",") if a.strip()]

    for tk in tech_keys:
        if tk not in NN_TECHS:
            print(f"ERROR: tech '{tk}' not in scope {NN_TECHS} "
                  f"(ASAP7 excluded — out of scope)")
            return 2
    for an in analyses:
        if an not in ("vtc", "tran"):
            print(f"ERROR: analysis '{an}' must be vtc or tran")
            return 2

    print("=" * 70)
    print("  V6.3.2 — NN inverter parametric verification")
    print("=" * 70)
    print(f"  Techs:    {tech_keys}")
    print(f"  Analyses: {analyses}")
    print(f"  Inverter acceptance: NRMSE < 15%")

    results = []
    # audit B5d — get_available_checkpoints() now RAISES on a pinned-but-absent
    # stem instead of printing a SKIP with no result row. Catch it here so this
    # gate reports the same clean "ERROR: ... / rc 1" as tests/common/nn_gate.py
    # rather than a traceback.
    try:
        for analysis in analyses:
            results.extend(run_nn_multi_tech(
                tech_keys, analysis, RESULTS_DIR,
                make_inv_baseline, build_inv_parametric, run_single_nn_inv,
            ))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    counts = print_nn_summary_table(results, kind="inv")
    save_nn_summary_csv(
        results, RESULTS_DIR / "verify_nn_multi_tech_tran_summary.csv",
        kind="inv")
    plot_nn_summary_bar(
        results, RESULTS_DIR / "verify_nn_multi_tech_tran_summary.png",
        "V6.3.2 NN inverter parametric sweep", kind="inv")

    print()
    if counts["fail"] == 0 and counts["error"] == 0:
        print(f"  RESULT: ALL {counts['pass']} configs PASSED")
        return 0
    print(f"  RESULT: {counts['fail']} FAIL, {counts['error']} ERROR "
          f"out of {len(results)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
