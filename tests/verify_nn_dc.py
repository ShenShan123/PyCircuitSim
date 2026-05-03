#!/usr/bin/env python3
"""Level 1: Quick NN compact model DC verification.

Quick sanity check -- NMOS DC, PMOS DC, and inverter VTC for a single
tech (TSMC12 SVT by default) using both BSIMAR v4 (LEVEL=74) and
DirectNet v4 (LEVEL=73). Compares against NGSPICE BSIM-CMG ground truth.

~6 tests (2 models x 3 test types), runs in ~30 min on GPU.

Usage:
    conda run -n pycircuitsim python tests/verify_nn_dc.py
    conda run -n pycircuitsim python tests/verify_nn_dc.py --tech TSMC5
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the comprehensive test runner with restricted scope
from tests.verify_nn_dc_tran import (
    get_available_checkpoints,
    run_dc_tests,
    run_pmos_dc_tests,
    run_inverter_vtc_tests,
    print_summary,
)

RESULTS_DIR = PROJECT_ROOT / "tests" / "verify_nn_dc_results"

DEFAULT_TECH = "TSMC12"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="L1 NN DC verification")
    ap.add_argument("--tech", default=DEFAULT_TECH,
                    help=f"Technology to test (default: {DEFAULT_TECH})")
    args = ap.parse_args()

    tech_names = [t.strip() for t in args.tech.split(",")]
    checkpoints = get_available_checkpoints()

    print(f"  L1 NN DC Verification: {', '.join(tech_names)}")
    print(f"  NMOS ckpt: {checkpoints.get('bsimar_v4_nmos', 'N/A')}")
    print(f"  PMOS ckpt: {checkpoints.get('bsimar_v4_pmos', 'N/A')}")

    dc_results = run_dc_tests(tech_names, checkpoints)
    pmos_results = run_pmos_dc_tests(tech_names, checkpoints)
    vtc_results = run_inverter_vtc_tests(tech_names, checkpoints)

    n_pass, n_fail, n_error = print_summary(
        dc_results + pmos_results + vtc_results, [])

    total = n_pass + n_fail + n_error
    print(f"\n  RESULT: {n_pass} PASS, {n_fail} FAIL, {n_error} ERROR "
          f"out of {total}\n")
    return 0 if n_fail == 0 and n_error == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
