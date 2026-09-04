#!/usr/bin/env python3
"""V6.3.2 — NN-family inverter parametric verification (VTC + transient).

Sweeps the selected LEVEL=75/76 NN-family CMOS inverter against NGSPICE
BSIM-CMG ground truth over circuit-level parameters:

  - Baseline:   1 VTC + 1 transient per tech (tech defaults)
  - P/N ratio:  PMOS fin count vs default (bounded by the TSMC naive-modelcard
                NFIN-group rule, same as the BSIM-CMG harness — typically the
                single point nfin_p=3)
  - VDD:        nominal +/- 0.1 V                 (VTC + transient)
  - Cload:      5, 50, 100 fF                     (transient only)
  - Input slew: tr=tf 10, 500 ps                  (transient only)
  - Pulse width: 0.2, 0.5, 2.0 ns                 (transient only)

Every declared baseline and parametric cell keeps its denominator slot.

ASAP7 is out of scope. Run against a stable checkpoint set and with
``OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`` — the
inverter trip point has gain ~-15..-30 that amplifies any NN-weight change
(e.g. a concurrent retrain overwriting the checkpoints) ~20x into the VTC.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      conda run -n pycircuitsim python tests/simple_circuits/verify_nn_multi_tech_tran.py \\
        [--tech TSMC5,TSMC7,TSMC12,TSMC16] [--analysis vtc,tran]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.nn_sweep import (  # noqa: E402
    NN_TECHS,
    build_inv_parametric,
    make_inv_baseline,
    plot_nn_summary_bar,
    print_nn_summary_table,
    run_nn_multi_tech,
    run_single_nn_inv,
    save_nn_summary_csv,
    sweep_gate_results,
)
from tests.common.circuit_benchmarks import active_model_label  # noqa: E402
from tests.common.gate_result import result_exit_code  # noqa: E402
from tests.common.simple_circuit_harness import RunSpec  # noqa: E402

# Env-overridable so parallel checkpoint bake-offs can isolate output dirs
# (same idiom as the shared simple-circuit results root).
import os as _os  # noqa: E402
RESULTS_DIR = Path(_os.environ.get(
    "PYCIRCUITSIM_NN_RESULTS",
    str(PROJECT_ROOT / "results" / "tests" / "nn_multi_tech_tran")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tech", default=",".join(NN_TECHS),
        help="comma-separated techs (default: all four TSMC nodes)")
    parser.add_argument(
        "--analysis", default="vtc,tran",
        help="comma-separated analyses: vtc, tran (default: both)")
    args = parser.parse_args(argv)

    tech_keys = [t.strip() for t in args.tech.split(",") if t.strip()]
    analyses = [a.strip().lower() for a in args.analysis.split(",") if a.strip()]

    if not tech_keys or len(tech_keys) != len(set(tech_keys)):
        parser.error("--tech must select one or more unique technologies")
    if not analyses or len(analyses) != len(set(analyses)):
        parser.error("--analysis must select one or more unique analyses")
    for tk in tech_keys:
        if tk not in NN_TECHS:
            print(f"ERROR: tech '{tk}' not in scope {NN_TECHS} "
                  f"(ASAP7 excluded — out of scope)")
            return 2
    for an in analyses:
        if an not in ("vtc", "tran"):
            print(f"ERROR: analysis '{an}' must be vtc or tran")
            return 2

    print("=" * 70)
    model_label = active_model_label()
    print(f"  V6.3.2 — {model_label} inverter parametric verification")
    print("=" * 70)
    print(f"  Techs:    {tech_keys}")
    print(f"  Analyses: {analyses}")
    print("  Inverter acceptance: NRMSE < 15%")

    try:
        run_spec = RunSpec.from_environment()
        run_spec.validate_checkpoint_pins(Path(_os.environ.get(
            "BSIMAR_CHECKPOINT_DIR",
            PROJECT_ROOT / "external_compact_models" / "neural_network"
            / "checkpoints",
        )))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    results = []
    # Runtime resolution failures remain explicit infrastructure outcomes;
    # the complete declared config list is otherwise always executed.
    try:
        for analysis in analyses:
            results.extend(run_nn_multi_tech(
                tech_keys, analysis, RESULTS_DIR,
                make_inv_baseline, build_inv_parametric, run_single_nn_inv,
            ))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    counts = print_nn_summary_table(results, kind="inv")
    save_nn_summary_csv(
        results, RESULTS_DIR / "verify_nn_multi_tech_tran_summary.csv",
        kind="inv")
    plot_nn_summary_bar(
        results, RESULTS_DIR / "verify_nn_multi_tech_tran_summary.png",
        f"V6.3.2 {model_label} inverter parametric sweep", kind="inv")
    gate_rows = sweep_gate_results(
        results,
        run_spec,
        case_id="nn_parametric_inverter",
        max_error_unit="mV",
    )
    for row in gate_rows:
        print(row.marker())

    print()
    exit_code = result_exit_code(gate_rows)
    if exit_code == 0:
        print(f"  RESULT: ALL {counts['pass']} configs PASSED")
    else:
        print(f"  RESULT: {counts['fail']} FAIL, {counts['error']} ERROR "
              f"out of {len(results)}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
