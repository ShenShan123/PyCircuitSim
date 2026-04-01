#!/usr/bin/env python3
"""Level 1: Simple BSIM-CMG transient verification.

Quick sanity check -- single CMOS inverter (ASAP7, RVT, default geometry).
Compares PyCircuitSim vs NGSPICE using the same OSDI binary.
~1 test, runs in seconds.

Usage:
    python tests/verify_bsimcmg_tran.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.bsimcmg_tran_common import (
    ALL_TECHS,
    make_baseline,
    run_test_suite,
)

RESULTS_DIR = PROJECT_ROOT / "tests" / "verify_bsimcmg_tran_results"


def main() -> int:
    tech = ALL_TECHS["ASAP7"]
    configs = [make_baseline(tech)]
    return run_test_suite(configs, RESULTS_DIR,
                          title="Level 1: Simple Transient Verification (ASAP7)")


if __name__ == "__main__":
    sys.exit(main())
