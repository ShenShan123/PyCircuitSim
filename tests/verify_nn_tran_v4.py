#!/usr/bin/env python3
"""Level 1: Quick NN compact model transient verification.

Quick sanity check -- NMOS pulse response and inverter transient for a
single tech (TSMC12 SVT by default) using both BSIMAR v4 (LEVEL=74) and
DirectNet v4 (LEVEL=73). Compares against NGSPICE BSIM-CMG ground truth.

~4 tests (2 models x 2 test types), runs in ~30 min on GPU.

Usage:
    conda run -n pycircuitsim python tests/verify_nn_tran_v4.py
    conda run -n pycircuitsim python tests/verify_nn_tran_v4.py --tech TSMC5
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.verify_nn_dc_tran import (
    get_available_checkpoints,
    run_tran_tests,
    run_inverter_tran_tests,
    print_summary,
)

RESULTS_DIR = PROJECT_ROOT / "tests" / "verify_nn_tran_v4_results"

DEFAULT_TECH = "TSMC12"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="L1 NN transient verification")
    ap.add_argument("--tech", default=DEFAULT_TECH,
                    help=f"Technology to test (default: {DEFAULT_TECH})")
    args = ap.parse_args()

    tech_names = [t.strip() for t in args.tech.split(",")]
    checkpoints = get_available_checkpoints()

    print(f"  L1 NN Transient Verification: {', '.join(tech_names)}")
    print(f"  NMOS ckpt: {checkpoints.get('bsimar_v4_nmos', 'N/A')}")
    print(f"  PMOS ckpt: {checkpoints.get('bsimar_v4_pmos', 'N/A')}")

    tran_results = run_tran_tests(tech_names, checkpoints)
    inv_tran_results = run_inverter_tran_tests(tech_names, checkpoints)

    n_pass, n_fail, n_error = print_summary(
        [], tran_results + inv_tran_results)

    total = n_pass + n_fail + n_error
    print(f"\n  RESULT: {n_pass} PASS, {n_fail} FAIL, {n_error} ERROR "
          f"out of {total}\n")
    return 0 if n_fail == 0 and n_error == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
