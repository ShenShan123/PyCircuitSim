#!/usr/bin/env python3
"""Verify flat and nested NN buffers against LEVEL=72 in DC, transient, and AC."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYCIRCUITSIM_TORCH_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.base import (  # noqa: E402
    SUBCIRCUIT_DECKS,
    parse_csv_choices,
    render_template,
)
from tests.common.circuit_benchmarks import (  # noqa: E402
    BENCH,
    BENCH_TECHS,
    RESULTS_BASE,
    BenchTech,
    get_baked_modelcard,
    parse_netlist,
)
from tests.common.gate_result import GateResult, result_exit_code  # noqa: E402
from tests.common.simple_circuit_catalog import AnalysisSpec  # noqa: E402
from tests.common.subcircuit_catalog import SUBCKT_ANALYSES  # noqa: E402
from tests.common.simple_circuit_harness import (  # noqa: E402
    CandidateConvergenceError,
    CandidateSupportError,
    RunSpec,
    Trace,
    compare_traces,
    physical_deck_mismatch,
    run_candidate_trace,
    run_reference_trace,
    validate_analysis_metrics,
)


FLAT_HIERARCHICAL_TOLERANCE_V = 1e-9


def _family_parameter(level: int) -> str:
    return {
        73: "", 74: "", 75: " FAMILY=directnet-full",
        76: " FAMILY=bsimar-full",
    }[level]


def _resolved_analysis(bt: BenchTech, analysis: AnalysisSpec) -> AnalysisSpec:
    return replace(analysis, card=analysis.card.replace("<VDD>", f"{bt.vdd:g}"))


def _input_spec(
    bt: BenchTech,
    analysis: AnalysisSpec,
    *,
    reference: bool = False,
) -> str:
    if analysis.kind == "tran":
        pulse = f"0 {bt.vdd:g} 0.5n 50p 50p 1n 2n"
        return f"PULSE({pulse})" if reference else f"PULSE {pulse}"
    if analysis.kind == "ac":
        return "DC=0 AC=1 0"
    return "0"


def render_candidate_pair(
    bt: BenchTech,
    run_spec: RunSpec,
    analysis: AnalysisSpec = SUBCKT_ANALYSES[1],
) -> Tuple[str, str]:
    """Render flat and nested buffers with the selected NN family."""
    analysis = _resolved_analysis(bt, analysis)
    family = _family_parameter(run_spec.model_level)
    setup = (
        f".model nmos_nn NMOS (LEVEL={run_spec.model_level}{family} "
        f"TECH={bt.nn_tech} VT={bt.effective_nmos_vt})\n"
        f".model pmos_nn PMOS (LEVEL={run_spec.model_level}{family} "
        f"TECH={bt.nn_tech} VT={bt.effective_pmos_vt})"
    )
    common = {
        "MODEL_SETUP": setup,
        "TEMP": f"{bt.temperature_c:g}",
        "VDD": f"{bt.vdd:g}",
        "INPUT_SPEC": _input_spec(bt, analysis),
        "P_PREFIX": "M",
        "N_PREFIX": "M",
        "OUTPUT_LOAD": "Cload out 0 5f",
        "ANALYSIS": f".{analysis.card}",
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


def render_reference(
    bt: BenchTech,
    baked: Path,
    analysis: AnalysisSpec = SUBCKT_ANALYSES[1],
) -> str:
    """Render the flat LEVEL=72 ground-truth buffer."""
    analysis = _resolved_analysis(bt, analysis)
    deck = render_template(
        SUBCIRCUIT_DECKS / "inverter_buffer_flat.spice.tmpl",
        {
            "MODEL_SETUP": f'.include "{baked}"',
            "TEMP": f"{bt.temperature_c:g}",
            "VDD": f"{bt.vdd:g}",
            "INPUT_SPEC": _input_spec(bt, analysis, reference=True),
            "P_PREFIX": "N",
            "N_PREFIX": "N",
            "P_DEVICE": bt.pmos_model,
            "N_DEVICE": bt.nmos_model,
            "OUTPUT_LOAD": "Cload out 0 5f",
            "INITIAL_CONDITION": f".ic V(mid)={bt.vdd:g} V(out)=0",
            "ANALYSIS": "",
        },
    )
    return deck


_PHYSICAL_ATTRIBUTES = (
    "value", "resistance", "capacitance", "inductance",
    "L", "NFIN", "m", "temperature", "ac_magnitude", "ac_phase",
    "v1", "v2", "i1", "i2", "td", "tr", "tf", "pw", "per",
)


def _frozen_value(value: Any) -> Any:
    """Convert parsed metadata into a deterministic, hashable value."""
    if isinstance(value, Mapping):
        return tuple(sorted(
            (str(key), _frozen_value(item)) for key, item in value.items()
        ))
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_value(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    return value


def _resolved_candidate_signature(
    parsed: Any,
    *,
    node_aliases: Mapping[str, str],
) -> Tuple[Any, ...]:
    aliases = {key.lower(): value.lower() for key, value in node_aliases.items()}

    def node_name(value: str) -> str:
        lowered = value.lower()
        if lowered == "gnd":
            lowered = "0"
        return aliases.get(lowered, lowered)

    components: Counter[Tuple[Any, ...]] = Counter()
    for component in parsed.circuit.components:
        attributes = tuple(
            (name, _frozen_value(getattr(component, name)))
            for name in _PHYSICAL_ATTRIBUTES
            if hasattr(component, name)
        )
        components[(
            type(component).__name__,
            tuple(node_name(node) for node in component.nodes),
            attributes,
        )] += 1
    initial_conditions = tuple(sorted(
        (node_name(node), _frozen_value(value))
        for node, value in parsed.circuit.initial_conditions.items()
    ))
    return (
        tuple(sorted(components.items(), key=repr)),
        initial_conditions,
        parsed.analysis_type,
        _frozen_value(parsed.analysis_params),
        _frozen_value(parsed.models),
        _frozen_value(parsed._temperature_kelvin),
    )


def flattened_candidate_mismatch(flat: Any, hierarchical: Any) -> str:
    """Compare parsed flat and hierarchy-expanded physical experiments."""
    flat_signature = _resolved_candidate_signature(flat, node_aliases={})
    hierarchical_signature = _resolved_candidate_signature(
        hierarchical,
        node_aliases={"Xbuf.m": "mid"},
    )
    if flat_signature == hierarchical_signature:
        return ""
    return (
        "resolved hierarchy mismatch: "
        f"flat={flat_signature!r}; hierarchical={hierarchical_signature!r}"
    )


def _hierarchical_analysis(analysis: AnalysisSpec) -> AnalysisSpec:
    return replace(
        analysis,
        signals=tuple(
            "v(Xbuf.m)" if signal == "v(mid)" else signal
            for signal in analysis.signals
        ),
    )


def _normalize_hierarchical_signals(trace: Trace) -> None:
    internal = trace.signals.pop("v(Xbuf.m)", None)
    if internal is not None:
        trace.signals["v(mid)"] = internal


def _flat_hierarchical_max_error(
    flat: Trace,
    hierarchical: Trace,
    signals: Tuple[str, ...],
) -> float:
    """Compare representation-equivalent traces on one shared dense grid."""
    low = max(float(np.min(flat.axis)), float(np.min(hierarchical.axis)))
    high = min(float(np.max(flat.axis)), float(np.max(hierarchical.axis)))
    if high < low:
        raise ValueError("flat and hierarchical traces do not overlap")
    count = max(min(flat.axis.size, hierarchical.axis.size, 800), 1)
    grid = np.linspace(low, high, count)
    worst = 0.0
    for signal in signals:
        flat_values = np.interp(grid, flat.axis, flat.signals[signal])
        hierarchical_values = np.interp(
            grid, hierarchical.axis, hierarchical.signals[signal],
        )
        worst = max(
            worst,
            float(np.max(np.abs(flat_values - hierarchical_values))),
        )
    return worst


def run_nn_subckt_analysis(
    bt: BenchTech,
    analysis: AnalysisSpec,
    work_dir: Path,
    run_spec: RunSpec,
) -> GateResult:
    """Compare one flat/hierarchical NN analysis with LEVEL=72 truth."""
    analysis = _resolved_analysis(bt, analysis)
    provenance = run_spec.result_fields()
    reference_converged = False
    stage = "setup"
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        baked = get_baked_modelcard(
            bt, bt.nfin, work_dir, nfin_p=bt.effective_nfin_p,
        )
        flat, hierarchical = render_candidate_pair(bt, run_spec, analysis)
        reference = render_reference(bt, baked, analysis)
        mismatch = physical_deck_mismatch(
            flat,
            reference,
            analysis,
            bt,
            baked_lib=baked,
            model_level=run_spec.model_level,
        )
        if mismatch:
            raise ValueError(mismatch)
        flat_path = work_dir / "candidate_flat.sp"
        hierarchical_path = work_dir / "candidate_hierarchical.sp"
        flat_path.write_text(flat)
        hierarchical_path.write_text(hierarchical)
        flat_parser = parse_netlist(flat_path)
        hierarchical_parser = parse_netlist(hierarchical_path)
        mismatch = flattened_candidate_mismatch(flat_parser, hierarchical_parser)
        if mismatch:
            raise ValueError(mismatch)
        stage = "reference"
        reference_trace = run_reference_trace(
            reference, analysis, work_dir, f"reference_{analysis.name}",
        )
        reference_converged = True
        stage = "candidate"
        flat_trace, _flat_path = run_candidate_trace(
            flat, analysis, work_dir, f"flat_{analysis.name}",
        )
        hierarchical_trace, _hierarchical_path = run_candidate_trace(
            hierarchical,
            _hierarchical_analysis(analysis),
            work_dir,
            f"hierarchical_{analysis.name}",
        )
        _normalize_hierarchical_signals(hierarchical_trace)
        if flat_trace.partial or hierarchical_trace.partial:
            raise CandidateConvergenceError(
                "flat or hierarchical transient ended early"
            )
        stage = "metrics"
        flat_metrics, _flat_domain = compare_traces(
            flat_trace, reference_trace, analysis, vdd=bt.vdd,
        )
        hierarchical_metrics, _hierarchical_domain = compare_traces(
            hierarchical_trace, reference_trace, analysis, vdd=bt.vdd,
        )
        equivalence_error = _flat_hierarchical_max_error(
            flat_trace, hierarchical_trace, analysis.signals,
        )
        if equivalence_error > FLAT_HIERARCHICAL_TOLERANCE_V:
            raise RuntimeError(
                "flat/hierarchical execution differs by "
                f"{equivalence_error:.6g} V"
            )
        metrics = {
            name: max(
                float(flat_metrics[name]),
                float(hierarchical_metrics[name]),
            )
            if name != "r2" else min(
                float(flat_metrics[name]),
                float(hierarchical_metrics[name]),
            )
            for name in ("mre_pct", "r2", "nrmse_pct", "max_err")
        }
        domain: Dict[str, float] = {
            "flat_nrmse_pct": float(flat_metrics["nrmse_pct"]),
            "hierarchical_nrmse_pct": float(
                hierarchical_metrics["nrmse_pct"]
            ),
            "flat_hierarchical_max_error_v": equivalence_error,
        }
        validate_analysis_metrics(analysis, metrics, domain)
        return GateResult(
            case_id="nn_subckt",
            tech=bt.name,
            corner="nominal",
            analysis=analysis.name,
            role="diagnostic",
            status="diagnostic",
            metrics=metrics,
            domain=domain,
            **provenance,
        )
    except Exception as exc:  # noqa: BLE001 - one explicit denominator row
        if stage == "reference":
            execution_state, error_kind = "reference_error", "reference"
        elif isinstance(exc, CandidateConvergenceError):
            execution_state, error_kind = "nonconverged", "candidate"
        elif isinstance(exc, CandidateSupportError):
            execution_state, error_kind = "error", "candidate"
        elif stage == "metrics":
            execution_state, error_kind = "error", "result_schema"
        else:
            execution_state, error_kind = (
                "infrastructure_error", "infrastructure",
            )
        return GateResult(
            case_id="nn_subckt",
            tech=bt.name,
            corner="nominal",
            analysis=analysis.name,
            role="diagnostic",
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            reference_converged=reference_converged,
            candidate_converged=False,
            execution_state=execution_state,
            error_kind=error_kind,
            **provenance,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tech", default=",".join(BENCH_TECHS))
    parser.add_argument(
        "--analysis",
        default="dc,tran,ac",
        help="comma-separated analyses: dc,tran,ac",
    )
    args = parser.parse_args(argv)
    techs = parse_csv_choices(
        parser, args.tech, flag="--tech", choices=BENCH_TECHS,
        normalize=str.upper,
    )
    available_analyses = {analysis.name: analysis for analysis in SUBCKT_ANALYSES}
    requested_analyses = parse_csv_choices(
        parser,
        args.analysis,
        flag="--analysis",
        choices=tuple(available_analyses),
        normalize=str.lower,
    )
    selected_analyses = [available_analyses[name] for name in requested_analyses]
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
        run_nn_subckt_analysis(
            BENCH[tech],
            analysis,
            root / tech / analysis.name,
            run_spec,
        )
        for tech in techs
        for analysis in selected_analyses
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
