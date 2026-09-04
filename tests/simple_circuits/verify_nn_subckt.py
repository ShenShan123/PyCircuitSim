#!/usr/bin/env python3
"""Verify flat and nested NN buffers against one LEVEL=72 reference deck."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYCIRCUITSIM_TORCH_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.base import SUBCIRCUIT_DECKS, render_template  # noqa: E402
from tests.common.circuit_benchmarks import (  # noqa: E402
    BENCH,
    BENCH_TECHS,
    RESULTS_BASE,
    BenchTech,
    full_metrics,
    get_baked_modelcard,
    run_directnet_transient,
    run_ngspice_wrdata,
)
from tests.common.gate_result import GateResult, result_exit_code  # noqa: E402
from tests.common.simple_circuit_harness import RunSpec, topology_mismatch  # noqa: E402


def _family_parameter(level: int) -> str:
    return {
        73: "", 74: "", 75: " FAMILY=directnet-full",
        76: " FAMILY=bsimar-full",
    }[level]


def render_candidate_pair(bt: BenchTech, run_spec: RunSpec) -> Tuple[str, str]:
    """Render flat and nested buffers with the selected NN family."""
    family = _family_parameter(run_spec.model_level)
    setup = (
        f".model nmos_nn NMOS (LEVEL={run_spec.model_level}{family} "
        f"TECH={bt.nn_tech} VT={bt.effective_nmos_vt})\n"
        f".model pmos_nn PMOS (LEVEL={run_spec.model_level}{family} "
        f"TECH={bt.nn_tech} VT={bt.effective_pmos_vt})"
    )
    pulse = f"PULSE 0 {bt.vdd:g} 0.5n 50p 50p 1n 2n"
    common = {
        "MODEL_SETUP": setup,
        "TEMP": f"{bt.temperature_c:g}",
        "VDD": f"{bt.vdd:g}",
        "INPUT_SPEC": pulse,
        "P_PREFIX": "M",
        "N_PREFIX": "M",
        "OUTPUT_LOAD": "Cload out 0 5f",
        "ANALYSIS": ".tran 2p 4n uic",
    }
    flat = render_template(
        SUBCIRCUIT_DECKS / "inverter_buffer_flat.spice.tmpl",
        {
            **common,
            "P_DEVICE": (
                f"pmos_nn L={bt.l_pmos * 1e9:g}n "
                f"NFIN={bt.effective_nfin_p}"
            ),
            "N_DEVICE": f"nmos_nn L={bt.l_nmos * 1e9:g}n NFIN={bt.nfin}",
            "INITIAL_CONDITION": f".ic V(mid)={bt.vdd:g} V(out)=0",
        },
    )
    hierarchical = render_template(
        SUBCIRCUIT_DECKS / "inverter_buffer_hierarchical.spice.tmpl",
        {
            **common,
            "P_DEVICE": f"pmos_nn L={bt.l_pmos * 1e9:g}n",
            "N_DEVICE": f"nmos_nn L={bt.l_nmos * 1e9:g}n",
            "NFN": str(bt.nfin),
            "NFP": str(bt.effective_nfin_p),
            "OUT_IC": "0",
        },
    )
    return flat, hierarchical


def render_reference(bt: BenchTech, baked: Path) -> Tuple[str, str]:
    """Render the flat LEVEL=72 ground-truth buffer."""
    card = "tran 2p 4n uic"
    deck = render_template(
        SUBCIRCUIT_DECKS / "inverter_buffer_flat.spice.tmpl",
        {
            "MODEL_SETUP": f'.include "{baked}"',
            "TEMP": f"{bt.temperature_c:g}",
            "VDD": f"{bt.vdd:g}",
            "INPUT_SPEC": f"PULSE(0 {bt.vdd:g} 0.5n 50p 50p 1n 2n)",
            "P_PREFIX": "N",
            "N_PREFIX": "N",
            "P_DEVICE": bt.pmos_model,
            "N_DEVICE": bt.nmos_model,
            "OUTPUT_LOAD": "Cload out 0 5f",
            "INITIAL_CONDITION": f".ic V(mid)={bt.vdd:g} V(out)=0",
            "ANALYSIS": "",
        },
    )
    return deck, card


def _body(deck: str) -> str:
    return "\n".join(
        line for line in deck.splitlines()
        if line.strip().lower() != ".end" and not line.lstrip().startswith("*")
    )


def _candidate_trace(deck: str, path: Path) -> Tuple[np.ndarray, np.ndarray, bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(deck)
    results, partial, _error = run_directnet_transient(path)
    return (
        np.asarray(results["time"], dtype=float),
        np.asarray(results["out"], dtype=float),
        partial,
    )


def run_nn_subckt_case(
    bt: BenchTech,
    work_dir: Path,
    run_spec: RunSpec,
) -> GateResult:
    """Compare flat/hierarchical NN execution and LEVEL=72 ground truth."""
    provenance = run_spec.result_fields()
    reference_converged = False
    stage = "setup"
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        baked = get_baked_modelcard(
            bt, bt.nfin, work_dir, nfin_p=bt.effective_nfin_p,
        )
        flat, hierarchical = render_candidate_pair(bt, run_spec)
        reference, card = render_reference(bt, baked)
        mismatch = topology_mismatch(flat, reference)
        if mismatch:
            raise ValueError(mismatch)
        stage = "reference"
        reference_data = run_ngspice_wrdata(
            _body(reference), "v(out)", work_dir, "reference", card,
        )
        ref_time = np.asarray(reference_data[:, 0], dtype=float)
        ref_out = np.asarray(reference_data[:, 1], dtype=float)
        reference_converged = True
        stage = "candidate"
        flat_time, flat_out, flat_partial = _candidate_trace(
            flat, work_dir / "candidate_flat.sp",
        )
        hier_time, hier_out, hier_partial = _candidate_trace(
            hierarchical, work_dir / "candidate_hierarchical.sp",
        )
        if flat_partial or hier_partial:
            raise RuntimeError("flat or hierarchical transient ended early")
        lo = max(ref_time[0], flat_time[0], hier_time[0])
        hi = min(ref_time[-1], flat_time[-1], hier_time[-1])
        if hi < 4e-9 * (1.0 - 1e-9):
            raise RuntimeError(f"hierarchy trace is incomplete at {hi:g} s")
        stage = "metrics"
        grid = np.linspace(lo, hi, 600)
        truth = np.interp(grid, ref_time, ref_out)
        flat_values = np.interp(grid, flat_time, flat_out)
        hier_values = np.interp(grid, hier_time, hier_out)
        flat_metrics = full_metrics(flat_values, truth)
        hier_metrics = full_metrics(hier_values, truth)
        equivalence = full_metrics(hier_values, flat_values)
        metrics = {
            name: max(float(flat_metrics[name]), float(hier_metrics[name]))
            if name != "r2" else min(
                float(flat_metrics[name]), float(hier_metrics[name]),
            )
            for name in ("mre_pct", "r2", "nrmse_pct", "max_err")
        }
        domain: Dict[str, float] = {
            "flat_nrmse_pct": float(flat_metrics["nrmse_pct"]),
            "hierarchical_nrmse_pct": float(hier_metrics["nrmse_pct"]),
            "flat_hierarchical_max_error_v": float(equivalence["max_err"]),
        }
        return GateResult(
            case_id="nn_subckt",
            tech=bt.name,
            corner="nominal",
            analysis="buffer",
            role="diagnostic",
            status="diagnostic",
            metrics=metrics,
            domain=domain,
            **provenance,
        )
    except Exception as exc:  # noqa: BLE001 - one explicit denominator row
        return GateResult(
            case_id="nn_subckt",
            tech=bt.name,
            corner="nominal",
            analysis="buffer",
            role="diagnostic",
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            reference_converged=reference_converged,
            candidate_converged=False,
            execution_state=(
                "reference_error" if stage == "reference"
                else "nonconverged" if stage == "candidate"
                else "infrastructure_error" if stage == "setup" else "error"
            ),
            error_kind=(
                "reference" if stage == "reference"
                else "candidate" if stage == "candidate"
                else "result_schema" if stage == "metrics"
                else "infrastructure"
            ),
            **provenance,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tech", default=",".join(BENCH_TECHS))
    args = parser.parse_args(argv)
    techs = [value.strip().upper() for value in args.tech.split(",")
             if value.strip()]
    unknown = [tech for tech in techs if tech not in BENCH]
    if not techs or unknown or len(set(techs)) != len(techs):
        parser.error(
            f"invalid technologies {unknown or techs}; available: {list(BENCH)}"
        )
    try:
        run_spec = RunSpec.from_environment()
        run_spec.validate_checkpoint_pins(Path(os.environ.get(
            "BSIMAR_CHECKPOINT_DIR",
            PROJECT_ROOT / "external_compact_models" / "neural_network"
            / "checkpoints",
        )))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    root = RESULTS_BASE / "nn-subckt" / f"level-{run_spec.model_level}"
    results = [
        run_nn_subckt_case(BENCH[tech], root / tech, run_spec)
        for tech in techs
    ]
    for result in results:
        print(result.marker())
    root.mkdir(parents=True, exist_ok=True)
    (root / "latest_results.json").write_text(json.dumps(
        [result.payload() for result in results], indent=2, sort_keys=True,
    ) + "\n")
    return result_exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
