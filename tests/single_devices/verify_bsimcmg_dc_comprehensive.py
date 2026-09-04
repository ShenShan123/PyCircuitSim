#!/usr/bin/env python3
"""Level 2: Comprehensive BSIM-CMG DC sweep verification.

Sweeps device parameters across all technologies for single NMOS and PMOS:
  - VT variants: all available threshold voltage flavors per tech
  - L sweep: all available gate lengths (per device type)
  - NFIN sweep: [1, 2, 5, 10, 20] at default VT and L

Each test compares PyCircuitSim vs NGSPICE using the same OSDI binary.
Skips combos where modelcard files are missing.

Usage:
    python tests/single_devices/verify_bsimcmg_dc_comprehensive.py [--tech ASAP7,TSMC5,...] \\
                                                     [--sweep vt,l,nfin] \\
                                                     [--device nmos,pmos]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.bsimcmg_dc import (
    ALL_TECHS,
    DCTestConfig,
    NFIN_SWEEP_VALUES,
    NMOS_IDVGS,
    PMOS_IDVGS,
    TECH_ORDER,
    TechProfile,
    make_dc_config,
    run_dc_test_suite,
)
from tests.common.base import parse_csv_choices  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "tests" / "bsimcmg_dc" / "comprehensive"


def build_vt_sweep(tech: TechProfile, devices: List[str]) -> List[DCTestConfig]:
    """One test per VT variant at default L and NFIN, for each device type."""
    configs: List[DCTestConfig] = []
    for vt in tech.vt_pairs:
        if "nmos" in devices:
            if tech.single_file or tech.get_nmos_modelcard(vt, tech.default_l_nmos).exists():
                configs.append(make_dc_config(
                    tech, NMOS_IDVGS, vt=vt,
                    config_name=f"vt_{vt.vt_name}", sweep_type="vt",
                ))
            else:
                print(f"  [skip] {tech.name}/{vt.vt_name} NMOS: modelcard missing")
        if "pmos" in devices:
            if tech.single_file or tech.get_pmos_modelcard(vt, tech.default_l_pmos).exists():
                configs.append(make_dc_config(
                    tech, PMOS_IDVGS, vt=vt,
                    config_name=f"vt_{vt.vt_name}", sweep_type="vt",
                ))
            else:
                print(f"  [skip] {tech.name}/{vt.vt_name} PMOS: modelcard missing")
    return configs


def build_l_sweep(tech: TechProfile, devices: List[str]) -> List[DCTestConfig]:
    """Sweep gate length at default VT and NFIN. Skips default L per device."""
    vt = tech.default_vt_pair
    configs: List[DCTestConfig] = []
    for l_val in tech.l_values:
        l_nm = round(l_val * 1e9)
        if "nmos" in devices and l_val != tech.default_l_nmos:
            if tech.single_file or tech.get_nmos_modelcard(vt, l_val).exists():
                configs.append(make_dc_config(
                    tech, NMOS_IDVGS, vt=vt,
                    config_name=f"l_{l_nm}nm", sweep_type="l",
                    l_nmos=l_val,
                ))
            else:
                print(f"  [skip] {tech.name}/{vt.vt_name} NMOS L={l_nm}nm: modelcard missing")
        if "pmos" in devices and l_val != tech.default_l_pmos:
            if tech.single_file or tech.get_pmos_modelcard(vt, l_val).exists():
                configs.append(make_dc_config(
                    tech, PMOS_IDVGS, vt=vt,
                    config_name=f"l_{l_nm}nm", sweep_type="l",
                    l_pmos=l_val,
                ))
            else:
                print(f"  [skip] {tech.name}/{vt.vt_name} PMOS L={l_nm}nm: modelcard missing")
    return configs


def build_nfin_sweep(tech: TechProfile, devices: List[str]) -> List[DCTestConfig]:
    """Sweep NFIN at default VT and L. Skips default NFIN."""
    vt = tech.default_vt_pair
    configs: List[DCTestConfig] = []
    for nfin in NFIN_SWEEP_VALUES:
        if nfin == tech.default_nfin:
            continue
        if "nmos" in devices:
            configs.append(make_dc_config(
                tech, NMOS_IDVGS, vt=vt,
                config_name=f"nfin_{nfin}", sweep_type="nfin",
                nfin_n=nfin,
            ))
        if "pmos" in devices:
            configs.append(make_dc_config(
                tech, PMOS_IDVGS, vt=vt,
                config_name=f"nfin_{nfin}", sweep_type="nfin",
                nfin_p=nfin,
            ))
    return configs


def build_configs(
    tech_names: List[str],
    sweep_types: List[str],
    devices: List[str],
) -> List[DCTestConfig]:
    """Build the full test matrix."""
    configs: List[DCTestConfig] = []
    for name in tech_names:
        tech = ALL_TECHS[name]
        if "vt" in sweep_types:
            configs.extend(build_vt_sweep(tech, devices))
        if "l" in sweep_types:
            configs.extend(build_l_sweep(tech, devices))
        if "nfin" in sweep_types:
            configs.extend(build_nfin_sweep(tech, devices))
    return configs


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Level 2: Comprehensive DC verification (VT/L/NFIN)")
    parser.add_argument("--tech", type=str, default=",".join(TECH_ORDER),
                        help="Comma-separated tech names (default: all)")
    parser.add_argument("--sweep", type=str, default="vt,l,nfin",
                        help="Comma-separated sweep types: vt,l,nfin (default: all)")
    parser.add_argument("--device", type=str, default="nmos,pmos",
                        help="Comma-separated device types: nmos,pmos (default: both)")
    args = parser.parse_args(argv)

    tech_names = parse_csv_choices(
        parser, args.tech, flag="--tech", choices=TECH_ORDER,
        normalize=str.upper,
    )
    sweep_types = parse_csv_choices(
        parser, args.sweep, flag="--sweep", choices=("vt", "l", "nfin"),
        normalize=str.lower,
    )
    devices = parse_csv_choices(
        parser, args.device, flag="--device", choices=("nmos", "pmos"),
        normalize=str.lower,
    )

    configs = build_configs(tech_names, sweep_types, devices)
    if not configs:
        parser.error("selected matrix produced no test configurations")

    return run_dc_test_suite(
        configs, RESULTS_DIR,
        title=f"Level 2: Comprehensive DC ({', '.join(tech_names)}; "
              f"sweeps: {', '.join(sweep_types)}; devices: {', '.join(devices)})",
    )


if __name__ == "__main__":
    sys.exit(main())
