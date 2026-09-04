#!/usr/bin/env python3
"""Run the versioned simple-v2 topology diagnostics against LEVEL=72.

The new cases remain diagnostics until their LEVEL=72 repeat stability and
promotion thresholds are frozen.  A diagnostic process exits nonzero only
when a requested cell could not be characterized; every numerical mismatch is
still emitted in its structured result marker and remains available to the
campaign collector.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

# The scored/default contract is one pinned CPU thread.  Campaign jobs may set
# these explicitly before launch; setdefault preserves such deliberate probes.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYCIRCUITSIM_TORCH_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.circuit_benchmarks import (  # noqa: E402
    BENCH, BENCH_TECHS, RESULTS_BASE, active_model_label,
)
from tests.common.gate_result import GateResult, result_exit_code  # noqa: E402
from tests.common.simple_circuit_catalog import (  # noqa: E402
    SIMPLE_V2, CircuitCase, cases, get_case,
)
from tests.common.simple_circuit_harness import (  # noqa: E402
    CORNERS, RunSpec, applicable_analyses, run_case,
)


def _comma_values(raw: str) -> List[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def _select_cases(raw: str) -> Tuple[CircuitCase, ...]:
    available = cases(score_version=SIMPLE_V2)
    if raw == "all":
        return available
    selected = []
    for case_id in _comma_values(raw):
        case = get_case(case_id)
        if case.score_version != SIMPLE_V2:
            raise ValueError(
                f"{case_id!r} belongs to {case.score_version}, not {SIMPLE_V2}")
        selected.append(case)
    return tuple(selected)


def _write_results(path: Path, results: Iterable[GateResult]) -> None:
    payload = [result.payload() for result in results]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", default="all",
        help="comma-separated simple-v2 case IDs, or all",
    )
    parser.add_argument(
        "--tech", default=",".join(BENCH_TECHS),
        help="comma-separated technologies",
    )
    parser.add_argument(
        "--corner", default="nominal",
        help="comma-separated corners, or all",
    )
    parser.add_argument(
        "--reference-repeats", type=int, default=1,
        help="repeat LEVEL=72 before candidate comparison (promotion uses >=3)",
    )
    parser.add_argument(
        "--no-support-diagnostic", action="store_true",
        help="skip accepted-LEVEL=72 trajectory support inspection",
    )
    parser.add_argument(
        "--level72-control", action="store_true",
        help="also run PyCircuitSim LEVEL=72 to isolate solver-owned failures",
    )
    parser.add_argument("--list", action="store_true",
                        help="list cases/corners without running")
    args = parser.parse_args(argv)

    if args.list:
        print("simple-v2 cases:")
        for case in cases(score_version=SIMPLE_V2):
            analyses = ",".join(spec.name for spec in case.analyses)
            print(f"  {case.case_id:26s} {analyses:34s} {case.label}")
        print("corners: " + ",".join(CORNERS))
        return 0

    try:
        selected_cases = _select_cases(args.case)
    except ValueError as exc:
        parser.error(str(exc))
    if not selected_cases:
        parser.error("--case must select at least one simple-v2 case")
    selected_ids = [case.case_id for case in selected_cases]
    if len(set(selected_ids)) != len(selected_ids):
        parser.error(f"--case contains duplicates: {selected_ids}")
    techs = _comma_values(args.tech)
    if not techs:
        parser.error("--tech must select at least one technology")
    unknown_techs = [tech for tech in techs if tech not in BENCH]
    if unknown_techs:
        parser.error(
            f"unknown technologies {unknown_techs}; available: {list(BENCH)}")
    if len(set(techs)) != len(techs):
        parser.error(f"--tech contains duplicates: {techs}")
    corner_names = (list(CORNERS) if args.corner == "all"
                    else _comma_values(args.corner))
    if not corner_names:
        parser.error("--corner must select at least one corner")
    unknown_corners = [name for name in corner_names if name not in CORNERS]
    if unknown_corners:
        parser.error(
            f"unknown corners {unknown_corners}; available: {list(CORNERS)}")
    if len(set(corner_names)) != len(corner_names):
        parser.error(f"--corner contains duplicates: {corner_names}")
    if args.reference_repeats < 1:
        parser.error("--reference-repeats must be >= 1")

    print("=" * 88)
    print(f"Simple-v2 topology diagnostics — {active_model_label()} vs "
          "NGSPICE BSIM-CMG LEVEL=72")
    print(f"Cases: {', '.join(case.case_id for case in selected_cases)}")
    print(f"Techs: {', '.join(techs)}; corners: {', '.join(corner_names)}")
    print("Role: diagnostic (not part of the historical simple-v1 /20 score)")
    print("=" * 88)

    try:
        run_spec = RunSpec.from_environment()
        run_spec.validate_checkpoint_pins(Path(os.environ.get(
            "BSIMAR_CHECKPOINT_DIR",
            PROJECT_ROOT / "external_compact_models" / "neural_network"
            / "checkpoints",
        )))
    except ValueError as exc:
        parser.error(str(exc))
    except FileNotFoundError as exc:
        parser.error(str(exc))
    explicit_corner_selection = args.corner != "all"
    no_op_cells = [
        f"{case.case_id}/{tech}/{corner_name}"
        for case in selected_cases
        for tech in techs
        for corner_name in corner_names
        if not applicable_analyses(case, BENCH[tech], CORNERS[corner_name])
    ]
    if explicit_corner_selection and no_op_cells:
        parser.error(
            "selected corner is unsupported or changes no observed physical "
            f"field for: {no_op_cells}"
        )

    all_results: List[GateResult] = []
    for case in selected_cases:
        for tech in techs:
            for corner_name in corner_names:
                if not applicable_analyses(
                    case, BENCH[tech], CORNERS[corner_name],
                ):
                    print(
                        f"\n--- {case.case_id} / {tech} / {corner_name} "
                        "NOT-APPLICABLE ---"
                    )
                    continue
                print(f"\n--- {case.case_id} / {tech} / {corner_name} ---")
                work_dir = (
                    RESULTS_BASE / "simple-v2"
                    / f"level-{run_spec.model_level}" / case.case_id
                    / tech / corner_name
                )
                results = run_case(
                    case, BENCH[tech], CORNERS[corner_name], work_dir,
                    reference_repeats=args.reference_repeats,
                    diagnose_support=not args.no_support_diagnostic,
                    run_spec=run_spec,
                    run_level72_control=args.level72_control,
                )
                for result in results:
                    all_results.append(result)
                    if result.status == "error":
                        print(f"  {result.analysis:22s} ERROR {result.error}")
                    else:
                        nrmse = result.metrics.get("nrmse_pct")
                        max_err = result.metrics.get("max_err")
                        # Convergence is printed beside the error, never
                        # folded into it: a partial transient and an accurate
                        # one must not read the same on a terminal.
                        state = ("converged" if result.candidate_converged
                                 else ("partial" if result.partial
                                       else "NOT-CONVERGED"))
                        print(f"  {result.analysis:22s} "
                              f"NRMSE={nrmse:.3f}% MaxErr={max_err:.6g} "
                              f"{result.status.upper()} [{state}]")
                    print(result.marker())

    output = (
        RESULTS_BASE / "simple-v2" / f"level-{run_spec.model_level}"
        / "latest_results.json"
    )
    _write_results(output, all_results)
    errors = sum(result.status == "error" for result in all_results)
    failures = sum(result.status == "fail" for result in all_results)
    diagnostics = sum(result.status == "diagnostic" for result in all_results)
    unconverged = sum(not result.candidate_converged for result in all_results)
    print("\n" + "=" * 88)
    print(f"RESULT: {diagnostics} characterized, {failures} qualification "
          f"failures, {errors} errors; JSON={output}")
    print(f"CONVERGED: {len(all_results) - unconverged}/{len(all_results)} "
          "candidate solves reached a physical fixed point")
    return result_exit_code(all_results)


if __name__ == "__main__":
    raise SystemExit(main())
