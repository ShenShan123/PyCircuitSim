#!/usr/bin/env python3
"""Level 2: Comprehensive BSIM-CMG transient verification.

Sweeps device parameters across all technologies:
  - VT variants: all available threshold voltage flavors per tech
  - L sweep: all available gate lengths (symmetric NMOS=PMOS)
  - NFIN sweep: [1, 2, 5, 10, 20] at default VT and L

Each test compares PyCircuitSim vs NGSPICE using the same OSDI binary.
Skips combos where modelcard files are missing.

Usage:
    python tests/verify_bsimcmg_tran_comprehensive.py [--tech ASAP7,TSMC5,...] [--sweep vt,l,nfin]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.bsimcmg_tran_common import (
    ALL_TECHS,
    TECH_ORDER,
    TechProfile,
    TestConfig,
    VtPair,
    make_baseline,
    run_test_suite,
)

RESULTS_DIR = PROJECT_ROOT / "tests" / "verify_bsimcmg_tran_results" / "comprehensive"

NFIN_SWEEP_VALUES = [2, 5, 10]


def build_vt_sweep(tech: TechProfile) -> List[TestConfig]:
    """One baseline test per VT variant at default L and NFIN."""
    configs: List[TestConfig] = []
    for vt in tech.vt_pairs:
        if not tech.is_combo_available(vt, tech.default_l_nmos, tech.default_l_pmos):
            print(f"  [skip] {tech.name}/{vt.vt_name}: modelcard missing at default L")
            continue
        configs.append(make_baseline(
            tech, vt=vt,
            config_name=f"vt_{vt.vt_name}",
            sweep_type="vt",
        ))
    return configs


def build_l_sweep(tech: TechProfile) -> List[TestConfig]:
    """Sweep gate length (symmetric L) at default VT and NFIN.

    Skips the default L combination (already covered by VT sweep baseline).
    """
    vt = tech.default_vt_pair
    configs: List[TestConfig] = []
    available_ls = tech.get_available_l_values(vt)
    for l_val in available_ls:
        # Skip if this is the default L combo
        if l_val == tech.default_l_nmos and l_val == tech.default_l_pmos:
            continue
        l_nm = round(l_val * 1e9)
        configs.append(make_baseline(
            tech, vt=vt,
            config_name=f"l_{l_nm}nm",
            sweep_type="l",
            l_nmos=l_val,
            l_pmos=l_val,
        ))
    return configs


def build_nfin_sweep(tech: TechProfile) -> List[TestConfig]:
    """Sweep NFIN (equal N and P) at default VT and L.

    Skips the default NFIN (already covered by VT sweep baseline).
    """
    vt = tech.default_vt_pair
    configs: List[TestConfig] = []
    for nfin in NFIN_SWEEP_VALUES:
        if nfin == tech.default_nfin:
            continue
        configs.append(make_baseline(
            tech, vt=vt,
            config_name=f"nfin_{nfin}",
            sweep_type="nfin",
            nfin_n=nfin,
            nfin_p=nfin,
        ))
    return configs


def build_configs(
    tech_names: List[str],
    sweep_types: List[str],
) -> List[TestConfig]:
    """Build the full test matrix."""
    configs: List[TestConfig] = []
    for name in tech_names:
        tech = ALL_TECHS[name]
        if "vt" in sweep_types:
            configs.extend(build_vt_sweep(tech))
        if "l" in sweep_types:
            configs.extend(build_l_sweep(tech))
        if "nfin" in sweep_types:
            configs.extend(build_nfin_sweep(tech))
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Level 2: Comprehensive transient verification (VT/L/NFIN)")
    parser.add_argument("--tech", type=str, default=",".join(TECH_ORDER),
                        help="Comma-separated tech names (default: all)")
    parser.add_argument("--sweep", type=str, default="vt,l,nfin",
                        help="Comma-separated sweep types: vt,l,nfin (default: all)")
    args = parser.parse_args()

    tech_names = [t.strip() for t in args.tech.split(",")]
    sweep_types = [s.strip() for s in args.sweep.split(",")]

    for t in tech_names:
        if t not in ALL_TECHS:
            print(f"ERROR: Unknown tech '{t}'. Available: {list(ALL_TECHS.keys())}")
            return 1

    configs = build_configs(tech_names, sweep_types)
    if not configs:
        print("No test configs generated. Check --tech and --sweep arguments.")
        return 1

    return run_test_suite(
        configs, RESULTS_DIR,
        title=f"Level 2: Comprehensive Transient ({', '.join(tech_names)}; "
              f"sweeps: {', '.join(sweep_types)})",
    )


if __name__ == "__main__":
    sys.exit(main())
