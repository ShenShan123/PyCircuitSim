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

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.bsimcmg_tran_common import (
    ALL_TECHS,
    NRMSE_THRESHOLD,
    TECH_ORDER,
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
        nfin_p = max(1, round(tech.default_nfin * ratio))
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
    parser = argparse.ArgumentParser(
        description="Level 3: Multi-tech transient verification")
    parser.add_argument("--tech", type=str, default=",".join(TECH_ORDER),
                        help="Comma-separated tech names (default: all)")
    args = parser.parse_args()

    tech_names = [t.strip() for t in args.tech.split(",")]
    for t in tech_names:
        if t not in ALL_TECHS:
            print(f"ERROR: Unknown tech '{t}'. Available: {list(ALL_TECHS.keys())}")
            return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    tech_status = {}

    print("=" * 78)
    print("Level 3: Multi-Technology Transient Verification")
    print(f"  Technologies: {', '.join(tech_names)}")
    print(f"  Acceptance: NRMSE < {NRMSE_THRESHOLD*100:.0f}% of Vdd (post-settling)")
    print("=" * 78)

    # Print tech parameters
    print(f"\n  {'Tech':8s} | {'VDD':>5s} | {'L_n':>5s} | {'L_p':>5s} | "
          f"{'NFIN':>4s} | {'VT':5s} | {'NMOS':15s} | {'PMOS':15s}")
    print("  " + "-" * 80)
    for name in tech_names:
        tech = ALL_TECHS[name]
        vt = tech.default_vt_pair
        print(f"  {tech.name:8s} | {tech.vdd:5.2f} | "
              f"{tech.default_l_nmos*1e9:4.0f}n | {tech.default_l_pmos*1e9:4.0f}n | "
              f"{tech.default_nfin:4d} | {tech.default_vt:5s} | "
              f"{vt.nmos_model:15s} | {vt.pmos_model:15s}")

    for name in tech_names:
        tech = ALL_TECHS[name]
        work_dir = RESULTS_DIR / tech.name

        # Phase 1: Baseline
        print(f"\n{'='*78}")
        print(f"  {tech.name}: Phase 1 — Baseline")
        print(f"{'='*78}")

        baseline_cfg = make_baseline(tech)
        try:
            result = run_single_test(baseline_cfg, work_dir)
            all_results.append(result)
            if result["passed"]:
                tech_status[tech.name] = "PASS"
            else:
                print(f"  => BASELINE FAIL — skipping parametric sweep")
                tech_status[tech.name] = "BASELINE_FAIL"
                continue
        except Exception as exc:
            print(f"  => BASELINE ERROR: {exc}")
            all_results.append({"config": baseline_cfg, "error": str(exc), "passed": False})
            tech_status[tech.name] = "BASELINE_ERROR"
            continue

        # Phase 2: Parametric sweep
        print(f"\n  {tech.name}: Phase 2 — Parametric sweep")
        sweep_configs = build_parametric_configs(tech)
        for cfg in sweep_configs:
            try:
                result = run_single_test(cfg, work_dir)
                all_results.append(result)
            except Exception as exc:
                print(f"    ERROR ({cfg.label}): {exc}")
                all_results.append({"config": cfg, "error": str(exc), "passed": False})

    # Summary
    print(f"\n{'='*78}")
    print("SUMMARY TABLE")
    print(f"{'='*78}")
    n_pass, n_fail, n_error = print_summary_table(all_results)
    total = len(all_results)

    print(f"\n  Total: {total}  Pass: {n_pass}  Fail: {n_fail}  Error: {n_error}")

    print(f"\n  Technology status:")
    for name, status in tech_status.items():
        print(f"    {name:8s}: {status}")

    save_summary_csv(all_results, RESULTS_DIR / "multi_tech_summary.csv")
    plot_summary_bar(all_results, RESULTS_DIR / "multi_tech_summary.png",
                     title="Level 3: Multi-Tech Transient Verification")

    print(f"\n{'='*78}")
    if n_fail > 0:
        print(f"RESULT: {n_fail} FAIL, {n_error} ERROR out of {total}")
        return 1
    if n_error > 0:
        print(f"RESULT: {n_pass} PASS, {n_error} ERROR (modelcard issues) out of {total}")
    else:
        print(f"RESULT: ALL {n_pass} tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
