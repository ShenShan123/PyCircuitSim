#!/usr/bin/env python3
"""V6.3.2 — NN-family single-device parametric DC verification.

Sweeps the selected LEVEL=73--76 NN family for NMOS & PMOS Id-Vgs against
NGSPICE BSIM-CMG ground truth over device geometry:

  - Baseline: 1 Id-Vgs per tech/device (tech-default L/NFIN/VT)
  - L sweep:    per-tech modelcard L values (skip default)
  - NFIN sweep: symmetric NFIN [5, 10] (skip default 2)
  - VT sweep:   per-tech VT variants (skip default)

Every declared baseline and parametric cell keeps its denominator slot.
Off-bin L/NFIN points exercise NN extrapolation beyond the per-tech training
bins — elevated NRMSE/MRE there is expected model behaviour, not a fault.

ASAP7 is out of scope. For reproducible results invoke with
``OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`` — the harness also pins torch to one
thread (see tests/common/nn_sweep.py).

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      conda run -n pycircuitsim python tests/single_devices/verify_nn_multi_tech_dc.py \\
        [--tech TSMC5,TSMC7,TSMC12,TSMC16] [--device nmos,pmos]
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
    build_dc_parametric,
    make_dc_baseline,
    plot_nn_summary_bar,
    print_nn_summary_table,
    run_nn_multi_tech,
    run_single_nn_dc,
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
    str(PROJECT_ROOT / "results" / "tests" / "nn_multi_tech_dc")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tech", default=",".join(NN_TECHS),
        help="comma-separated techs (default: all four TSMC nodes)")
    parser.add_argument(
        "--device", default="nmos,pmos",
        help="comma-separated devices: nmos, pmos (default: both)")
    args = parser.parse_args(argv)

    tech_keys = [t.strip() for t in args.tech.split(",") if t.strip()]
    devices = [d.strip().lower() for d in args.device.split(",") if d.strip()]

    if not tech_keys or len(tech_keys) != len(set(tech_keys)):
        parser.error("--tech must select one or more unique technologies")
    if not devices or len(devices) != len(set(devices)):
        parser.error("--device must select one or more unique devices")
    for tk in tech_keys:
        if tk not in NN_TECHS:
            print(f"ERROR: tech '{tk}' not in scope {NN_TECHS} "
                  f"(ASAP7 excluded — out of scope)")
            return 2
    for dv in devices:
        if dv not in ("nmos", "pmos"):
            print(f"ERROR: device '{dv}' must be nmos or pmos")
            return 2

    print("=" * 70)
    model_label = active_model_label()
    print(f"  V6.3.2 — {model_label} single-device parametric DC verification")
    print("=" * 70)
    print(f"  Techs:   {tech_keys}")
    print(f"  Devices: {devices}")
    print("  DC acceptance: NRMSE < 10%")

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
        for device in devices:
            results.extend(run_nn_multi_tech(
                tech_keys, device, RESULTS_DIR,
                make_dc_baseline, build_dc_parametric, run_single_nn_dc,
            ))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    counts = print_nn_summary_table(results, kind="dc")
    save_nn_summary_csv(
        results, RESULTS_DIR / "verify_nn_multi_tech_dc_summary.csv", kind="dc")
    plot_nn_summary_bar(
        results, RESULTS_DIR / "verify_nn_multi_tech_dc_summary.png",
        f"V6.3.2 {model_label} single-device DC parametric sweep", kind="dc")
    gate_rows = sweep_gate_results(
        results,
        run_spec,
        case_id="nn_parametric_dc",
        max_error_unit="A",
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
