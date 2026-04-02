#!/usr/bin/env python3
"""Level 1: Simple BSIM-CMG DC sweep verification.

Quick sanity check -- NMOS and PMOS Id-Vgs (ASAP7, RVT, default geometry).
Compares PyCircuitSim vs NGSPICE using the same OSDI binary.
~2 tests, runs in seconds.

Usage:
    python tests/verify_bsimcmg_dc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.bsimcmg_dc_common import (
    ALL_TECHS,
    NMOS_IDVGS,
    PMOS_IDVGS,
    make_dc_config,
    run_dc_test_suite,
)

RESULTS_DIR = PROJECT_ROOT / "tests" / "verify_bsimcmg_dc_results"


def main() -> int:
    tech = ALL_TECHS["ASAP7"]
    configs = [
        make_dc_config(tech, NMOS_IDVGS),
        make_dc_config(tech, PMOS_IDVGS),
    ]
    return run_dc_test_suite(configs, RESULTS_DIR,
                             title="Level 1: Simple DC Verification (ASAP7)")


if __name__ == "__main__":
    sys.exit(main())
