#!/usr/bin/env python3
"""Level 3: Multi-technology DC verification with inverter VTC.

Tests all 5 FinFET technologies with inverter VTC parametric variations:
  - Baseline: 1 inverter VTC per tech (nominal VDD, default VT/L/NFIN)
  - VT sweep: all threshold voltage flavors (skip default)
  - L sweep: symmetric L (same NMOS=PMOS) for all available values
  - NFIN sweep: symmetric NFIN [1, 2, 5, 10, 20] (skip default)
  - P/N ratio: NFIN_P/NFIN_N = 0.5, 1.5, 2.0

Parametric sweep only runs for techs that pass baseline.

Usage:
    python tests/verify_multi_tech_dc.py [--tech ASAP7,TSMC5,...]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.bsimcmg_dc import (
    DCTestConfig,
    INVERTER_VTC,
    NFIN_SWEEP_VALUES,
    NRMSE_THRESHOLD,
    MAX_REL_ERR_THRESHOLD,
    TechProfile,
    make_dc_config,
    print_dc_summary_table,
    plot_dc_summary_bar,
    run_single_dc_test,
    save_dc_summary_csv,
)

RESULTS_DIR = PROJECT_ROOT / "tests" / "verify_multi_tech_dc_results"


def build_parametric_configs(tech: TechProfile) -> List[DCTestConfig]:
    """Build inverter VTC parametric sweep configs for a technology."""
    vt = tech.default_vt_pair
    configs: List[DCTestConfig] = []

    # VT sweep (skip default — already covered by baseline)
    for vt_other in tech.vt_pairs:
        if vt_other.vt_name == tech.default_vt:
            continue
        if not tech.is_combo_available(vt_other, tech.default_l_nmos, tech.default_l_pmos):
            print(f"  [skip] {tech.name}/{vt_other.vt_name}: modelcard missing")
            continue
        configs.append(make_dc_config(
            tech, INVERTER_VTC, vt=vt_other,
            config_name=f"vt_{vt_other.vt_name}", sweep_type="vt",
        ))

    # L sweep (symmetric, skip if matches default asymmetric combo)
    for l_val in tech.l_values:
        if l_val == tech.default_l_nmos and l_val == tech.default_l_pmos:
            continue
        if not tech.is_combo_available(vt, l_val, l_val):
            l_nm = round(l_val * 1e9)
            print(f"  [skip] {tech.name} L={l_nm}nm: modelcard missing")
            continue
        l_nm = round(l_val * 1e9)
        configs.append(make_dc_config(
            tech, INVERTER_VTC, config_name=f"l_{l_nm}nm", sweep_type="l",
            l_nmos=l_val, l_pmos=l_val,
        ))

    # NFIN sweep (symmetric, skip default)
    for nfin in NFIN_SWEEP_VALUES:
        if nfin == tech.default_nfin:
            continue
        configs.append(make_dc_config(
            tech, INVERTER_VTC, config_name=f"nfin_{nfin}", sweep_type="nfin",
            nfin_n=nfin, nfin_p=nfin,
        ))

    # P/N ratio sweep (NFIN_P varies, NFIN_N = default)
    for ratio in [0.5, 1.5, 2.0]:
        nfin_p = max(2, round(tech.default_nfin * ratio))
        if nfin_p == tech.default_nfin:
            continue  # same as baseline
        # TSMC naive modelcards are NFIN-group-specific; skip if NFIN_P
        # leaves the default group [N, N+1] (ASAP7 single-file covers all)
        if not tech.single_file and nfin_p > tech.default_nfin + 1:
            continue
        tag = f"pn_{ratio:.1f}".replace(".", "p")
        configs.append(make_dc_config(
            tech, INVERTER_VTC, config_name=tag, sweep_type="pn_ratio",
            nfin_p=nfin_p,
        ))

    return configs


def main() -> int:
    from tests.common.base import run_multi_tech_main, parse_tech_args
    tech_names = parse_tech_args("Level 3: Multi-tech DC verification (Inverter VTC)")
    if isinstance(tech_names, int):
        return tech_names

    return run_multi_tech_main(
        tech_names=tech_names,
        results_dir=RESULTS_DIR,
        title="Level 3: Multi-Technology DC Verification (Inverter VTC)",
        acceptance_msg=f"NRMSE < {NRMSE_THRESHOLD*100:.0f}%, MaxRelErr < {MAX_REL_ERR_THRESHOLD*100:.0f}%",
        make_baseline_fn=lambda tech: make_dc_config(tech, INVERTER_VTC),
        build_parametric_fn=build_parametric_configs,
        run_single_fn=run_single_dc_test,
        print_summary_fn=print_dc_summary_table,
        save_csv_fn=save_dc_summary_csv,
        plot_bar_fn=plot_dc_summary_bar,
    )


if __name__ == "__main__":
    sys.exit(main())
