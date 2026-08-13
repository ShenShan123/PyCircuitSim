#!/usr/bin/env python3
"""Level 3: Multi-technology BSIM-CMG verification on the CMOS inverter.

One circuit, five FinFET technologies, two analyses — each with its own
parametric fan-out, both baseline-gated (the sweep runs only for techs whose
baseline passes).

``dc`` — inverter VTC:
  - Baseline: 1 VTC per tech (nominal VDD, default VT/L/NFIN)
  - VT sweep: all threshold-voltage flavors (skip default)
  - L sweep: symmetric L (NMOS=PMOS) for every available value
  - NFIN sweep: symmetric NFIN [1, 2, 5, 10, 20] (skip default)
  - P/N ratio: NFIN_P/NFIN_N = 0.5, 1.5, 2.0

``tran`` — inverter pulse response:
  - Baseline: 1 transient per tech (nominal VDD, default VT/L/NFIN)
  - P/N ratio: NFIN_P/NFIN_N = 0.5, 1.5, 2.0
  - VDD sweep: nominal +/- 0.1 V
  - Cload sweep: 1, 5, 50, 100 fF
  - Input slew: 10, 50, 500 ps (baseline 100 ps)
  - Pulse width: 0.2, 0.5, 2.0 ns (baseline 0.8 ns)

Until V7.5.9 these were two files (``verify_multi_tech_dc.py`` /
``verify_multi_tech_tran.py``) whose only difference was the config builder
below — same circuit, same techs, same ``run_multi_tech_main`` call, same
results convention. Both builders are here verbatim; nothing was re-tuned, and
each analysis keeps its own results directory so existing baselines still
resolve.

Usage:
    python tests/simple_circuits/verify_multi_tech.py            # dc + tran
    python tests/simple_circuits/verify_multi_tech.py --analysis dc
    python tests/simple_circuits/verify_multi_tech.py --analysis tran \\
        --tech ASAP7,TSMC5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.base import ALL_TECHS, TECH_ORDER, run_multi_tech_main  # noqa: E402
from tests.common.bsimcmg_dc import (  # noqa: E402
    DCTestConfig,
    INVERTER_VTC,
    MAX_REL_ERR_THRESHOLD,
    NFIN_SWEEP_VALUES,
    NRMSE_THRESHOLD as DC_NRMSE_THRESHOLD,
    TechProfile,
    make_dc_config,
    plot_dc_summary_bar,
    print_dc_summary_table,
    run_single_dc_test,
    save_dc_summary_csv,
)
from tests.common.bsimcmg_tran import (  # noqa: E402
    NRMSE_THRESHOLD as TRAN_NRMSE_THRESHOLD,
    TestConfig,
    make_baseline,
    plot_summary_bar,
    print_summary_table,
    run_single_test,
    save_summary_csv,
)

DC_RESULTS_DIR = PROJECT_ROOT / "tests" / "verify_multi_tech_dc_results"
TRAN_RESULTS_DIR = PROJECT_ROOT / "tests" / "verify_multi_tech_tran_results"


def build_dc_parametric(tech: TechProfile) -> List[DCTestConfig]:
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


def build_tran_parametric(tech: TechProfile) -> List[TestConfig]:
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


def run_dc(tech_names: List[str]) -> int:
    return run_multi_tech_main(
        tech_names=tech_names,
        results_dir=DC_RESULTS_DIR,
        title="Level 3: Multi-Technology DC Verification (Inverter VTC)",
        acceptance_msg=(f"NRMSE < {DC_NRMSE_THRESHOLD*100:.0f}%, "
                        f"MaxRelErr < {MAX_REL_ERR_THRESHOLD*100:.0f}%"),
        make_baseline_fn=lambda tech: make_dc_config(tech, INVERTER_VTC),
        build_parametric_fn=build_dc_parametric,
        run_single_fn=run_single_dc_test,
        print_summary_fn=print_dc_summary_table,
        save_csv_fn=save_dc_summary_csv,
        plot_bar_fn=plot_dc_summary_bar,
    )


def run_tran(tech_names: List[str]) -> int:
    return run_multi_tech_main(
        tech_names=tech_names,
        results_dir=TRAN_RESULTS_DIR,
        title="Level 3: Multi-Technology Transient Verification",
        acceptance_msg=(f"NRMSE < {TRAN_NRMSE_THRESHOLD*100:.0f}% of Vdd "
                        f"(post-settling)"),
        make_baseline_fn=make_baseline,
        build_parametric_fn=build_tran_parametric,
        run_single_fn=run_single_test,
        print_summary_fn=print_summary_table,
        save_csv_fn=save_summary_csv,
        plot_bar_fn=plot_summary_bar,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--analysis", default="dc,tran",
                        help="comma list of dc,tran (default: both)")
    parser.add_argument("--tech", default=",".join(TECH_ORDER),
                        help="comma-separated tech names (default: all)")
    args = parser.parse_args()

    analyses = [a.strip().lower() for a in args.analysis.split(",") if a.strip()]
    for a in analyses:
        if a not in ("dc", "tran"):
            print(f"ERROR: unknown analysis '{a}' (expected dc or tran)")
            return 2
    tech_names = [t.strip() for t in args.tech.split(",") if t.strip()]
    for t in tech_names:
        if t not in ALL_TECHS:
            print(f"ERROR: Unknown tech '{t}'. Available: {TECH_ORDER}")
            return 2

    rc = 0
    for a in analyses:
        rc |= (run_dc if a == "dc" else run_tran)(tech_names)
    return rc


if __name__ == "__main__":
    sys.exit(main())
