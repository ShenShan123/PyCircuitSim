#!/usr/bin/env python3
"""Score the single-device surfaces the parametric DC gate never reaches.

``verify_nn_multi_tech_dc`` sweeps Id-Vgs on a linear axis at one fixed
``Vds = 0.5*VDD``.  This gate adds the output characteristic (``gds``), the
subthreshold decades (``Ioff``, subthreshold slope), the triode region
(``Ron``, origin symmetry) and ``gm``/``gds``/``gmb`` measured against ground
truth rather than against the network's own finite differences.  The contract
is in ``tests/common/device_integrity.py``.

These are **diagnostics**.  They report metrics and convergence; no threshold
is frozen and no published score changes.  The process exits nonzero only when
a requested cell could not be characterized at all, so a large but measured
error stays visible instead of collapsing the run.

Convergence is reported separately from error throughout.  A sweep that never
reached a physical fixed point is an ``ERROR`` row that keeps its slot in the
denominator; it is never averaged into an accuracy number.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# The scored contract is one pinned CPU thread.  Campaign jobs may set these
# explicitly before launch; setdefault preserves such deliberate probes.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYCIRCUITSIM_TORCH_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.circuit_benchmarks import (  # noqa: E402
    BENCH, BENCH_TECHS, RESULTS_BASE, active_model_label, active_model_level,
)
from tests.common.device_integrity import (  # noqa: E402
    DEVICE_KINDS, SUITES, build_sweeps, device_corner_applies,
    run_device_suites,
)
from tests.common.gate_result import GateResult, result_exit_code  # noqa: E402
from tests.common.simple_circuit_harness import CORNERS  # noqa: E402
from tests.common.simple_circuit_harness import RunSpec  # noqa: E402


#: One headline number per suite, chosen so a reader can localize a failure
#: without opening the JSON.
HEADLINE: Dict[str, str] = {
    "output": "gds_sat_error_pct",
    "subthreshold": "max_decade_error",
    "linear": "ron_error_pct",
    "derivative": "deriv_nrmse_pct",
}


def _comma_values(raw: str) -> List[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def _selection(
    parser: argparse.ArgumentParser, raw: str, available: List[str], flag: str,
) -> List[str]:
    """Resolve one comma-separated selection, failing closed on any typo."""
    selected = list(available) if raw == "all" else _comma_values(raw)
    if not selected:
        parser.error(f"{flag} must select at least one value")
    unknown = [name for name in selected if name not in available]
    if unknown:
        parser.error(f"unknown {flag} values {unknown}; available: {available}")
    if len(set(selected)) != len(selected):
        parser.error(f"{flag} contains duplicates: {selected}")
    return selected


def _write_results(path: Path, results: List[GateResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([result.payload() for result in results],
                   indent=2, sort_keys=True) + "\n")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tech", default=",".join(BENCH_TECHS),
                        help="comma-separated technologies, or all")
    parser.add_argument("--suite", default="all",
                        help=f"comma-separated suites {list(SUITES)}, or all")
    parser.add_argument("--device", default="all",
                        help="nmos, pmos, or all")
    parser.add_argument("--corner", default="nominal",
                        help="comma-separated corners, or all")
    parser.add_argument("--reference-repeats", type=int, default=1,
                        help="repeat LEVEL=72 before comparison (>=3 for promotion)")
    parser.add_argument("--list", action="store_true",
                        help="list the declared sweep matrix without running")
    args = parser.parse_args(argv)

    techs = _selection(parser, args.tech, list(BENCH_TECHS), "--tech")
    suites = _selection(parser, args.suite, list(SUITES), "--suite")
    devices = _selection(parser, args.device, list(DEVICE_KINDS), "--device")
    corners = _selection(parser, args.corner, list(CORNERS), "--corner")
    if args.reference_repeats < 1:
        parser.error("--reference-repeats must be >= 1")

    if args.list:
        print("device-integrity sweeps (per technology and corner):")
        for device in devices:
            for spec in build_sweeps(BENCH[techs[0]], device):
                if spec.suite not in suites:
                    continue
                print(f"  {spec.suite:13s} {spec.device:5s} {spec.label:10s} "
                      f"sweep {spec.sweep_source} {spec.start:+.4g} -> "
                      f"{spec.stop:+.4g} step {spec.step:+.4g}")
        print("corners: " + ",".join(CORNERS))
        return 0

    level = active_model_level()
    try:
        RunSpec.from_environment().validate_checkpoint_pins(Path(os.environ.get(
            "BSIMAR_CHECKPOINT_DIR",
            PROJECT_ROOT / "external_compact_models" / "neural_network"
            / "checkpoints",
        )))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print("=" * 88)
    print(f"Single-device integrity — {active_model_label()} vs NGSPICE "
          "BSIM-CMG LEVEL=72")
    print(f"Suites: {', '.join(suites)}; devices: {', '.join(devices)}")
    print(f"Techs: {', '.join(techs)}; corners: {', '.join(corners)}")
    print("Role: diagnostic (no frozen threshold; no published score changes)")
    print("=" * 88)

    all_results: List[GateResult] = []
    for tech in techs:
        for corner_name in corners:
            print(f"\n--- {tech} / {corner_name} ---")
            skipped = [
                device for device in devices
                if not device_corner_applies(
                    BENCH[tech], device, CORNERS[corner_name],
                )
            ]
            if skipped:
                print("  NOT-APPLICABLE: " + ", ".join(skipped))
            work_dir = (
                RESULTS_BASE / "device-integrity" / f"level-{level}"
                / tech / corner_name
            )
            results = run_device_suites(
                BENCH[tech], CORNERS[corner_name], work_dir,
                level=level, suites=suites, devices=devices,
                reference_repeats=args.reference_repeats,
            )
            for result in results:
                all_results.append(result)
                name = f"{result.case_id.removeprefix('device_')}/{result.analysis}"
                if result.status == "error":
                    print(f"  {name:34s} ERROR {result.error}")
                else:
                    suite = result.case_id.removeprefix("device_")
                    headline = result.domain.get(HEADLINE[suite])
                    shown = (f"{headline:.3f}" if isinstance(headline, float)
                             else "—")
                    print(f"  {name:34s} "
                          f"NRMSE={result.metrics['nrmse_pct']:8.3f}%  "
                          f"{HEADLINE[suite]}={shown}")
                print(result.marker())

    output = (
        RESULTS_BASE / "device-integrity" / f"level-{level}"
        / "latest_results.json"
    )
    _write_results(output, all_results)

    errors = [result for result in all_results if result.status == "error"]
    converged = sum(result.reference_converged and result.candidate_converged
                    for result in all_results)
    print("\n" + "=" * 88)
    # Convergence and accuracy are reported as two independent facts.  A gate
    # that folds them together cannot distinguish "wrong" from "never solved",
    # which is exactly how a 0/10 AC score came to mean an unconverged DC
    # operating point.
    print(f"CHARACTERIZED : {len(all_results) - len(errors)}/{len(all_results)}")
    print(f"CONVERGED     : {converged}/{len(all_results)}")
    for result in errors:
        print(f"  ERROR {result.tech}/{result.corner}/{result.case_id}/"
              f"{result.analysis}: {result.error}")
    print(f"JSON={output}")
    return result_exit_code(all_results)


if __name__ == "__main__":
    raise SystemExit(main())
