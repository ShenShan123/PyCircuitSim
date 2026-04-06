#!/usr/bin/env python3
"""Level 3: Multi-technology transient verification with circuit-level sweeps.

Tests all 5 FinFET technologies with circuit-level parameter variations:
  - Baseline: 1 test per tech (nominal VDD, default VT/L/NFIN)
  - P/N ratio: NFIN_P/NFIN_N = 0.5, 1.5, 2.0
  - VDD sweep: nominal +/- 0.1V
  - Cload sweep: 1, 5, 50, 100 fF
  - Input slew: 10, 50, 500 ps (baseline=100ps)
  - Pulse width: 0.2, 0.5, 2.0 ns (baseline=0.8ns)

Parametric sweep only runs for techs that pass baseline.

Usage:
    python tests/verify_multi_tech_tran.py [--tech ASAP7,TSMC5,...]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.bsimcmg_tran import (
    NRMSE_THRESHOLD,
    TechProfile,
    TestConfig,
    make_baseline,
    print_summary_table,
    run_single_test,
    save_summary_csv,
    plot_summary_bar,
)

RESULTS_DIR = PROJECT_ROOT / "tests" / "verify_multi_tech_tran_results"


def build_parametric_configs(tech: TechProfile) -> List[TestConfig]:
    """Build circuit-level parametric sweep configs for a technology."""
    vt = tech.default_vt_pair
    configs: List[TestConfig] = []

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
        configs.append(make_baseline(
            tech, vt=vt, config_name=tag, sweep_type="pn_ratio",
            nfin_p=nfin_p,
        ))

    # VDD sweep (+/- 0.1V)
    for delta in [-0.1, 0.1]:
        vdd_val = tech.vdd + delta
        if vdd_val <= 0:
            continue
        tag = f"vdd_{vdd_val:.1f}".replace(".", "p")
        configs.append(make_baseline(
            tech, vt=vt, config_name=tag, sweep_type="vdd",
            vdd=vdd_val,
        ))

    # Cload sweep
    for cload_fF in [1, 5, 50, 100]:
        configs.append(make_baseline(
            tech, vt=vt,
            config_name=f"cload_{cload_fF}fF", sweep_type="cload",
            cload=cload_fF * 1e-15,
        ))

    # Input slew sweep (tr=tf)
    for slew_ps in [10, 50, 500]:
        configs.append(make_baseline(
            tech, vt=vt,
            config_name=f"slew_{slew_ps}ps", sweep_type="slew",
            tr=slew_ps * 1e-12, tf=slew_ps * 1e-12,
        ))

    # Pulse width sweep
    for pw_ns in [0.2, 0.5, 2.0]:
        tag = f"pw_{pw_ns:.1f}ns".replace(".", "p")
        configs.append(make_baseline(
            tech, vt=vt, config_name=tag, sweep_type="pw",
            pw=pw_ns * 1e-9,
        ))

    return configs


def main() -> int:
    from tests.common.base import run_multi_tech_main, parse_tech_args
    tech_names = parse_tech_args("Level 3: Multi-tech transient verification")
    if isinstance(tech_names, int):
        return tech_names

    return run_multi_tech_main(
        tech_names=tech_names,
        results_dir=RESULTS_DIR,
        title="Level 3: Multi-Technology Transient Verification",
        acceptance_msg=f"NRMSE < {NRMSE_THRESHOLD*100:.0f}% of Vdd (post-settling)",
        make_baseline_fn=make_baseline,
        build_parametric_fn=build_parametric_configs,
        run_single_fn=run_single_test,
        print_summary_fn=print_summary_table,
        save_csv_fn=save_summary_csv,
        plot_bar_fn=plot_summary_bar,
    )


if __name__ == "__main__":
    sys.exit(main())
