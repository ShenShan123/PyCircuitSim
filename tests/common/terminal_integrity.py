"""Four-terminal physical diagnostics for NN compact models.

Currents and the 4x4 transcapacitance matrix are the surfaces the full-terminal
LEVEL=75/76 families add, and until V7.6.9 both were measured at one nominal
geometry and one nominal temperature — while ``device_integrity`` already swept
the same device across all fourteen declared corners. A charge model can be
exact at 27 C / NFIN=2 and wrong at 125 C or NFIN=5, so the corner axis is now
the same one, opt-in through ``--corner`` and defaulting to ``nominal`` so no
existing campaign denominator moves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from tests.common.circuit_benchmarks import full_metrics
from tests.common.base import DEVICE_DECKS, render_template
from tests.common.circuit_benchmarks import (
    BenchTech,
    get_baked_modelcard,
    parse_netlist,
    run_directnet_dc_sweep,
    run_ngspice_wrdata,
)
from tests.common.device_integrity import device_corner_applies
from tests.common.gate_result import GateResult
from tests.common.simple_circuit_catalog import AnalysisSpec
from tests.common.simple_circuit_harness import (
    CORNERS,
    Corner,
    RunSpec,
    Trace,
    apply_corner,
    analysis_endpoint_tolerance,
    analysis_minimum_points,
    analysis_max_step,
    analysis_axis_limits,
    physical_deck_mismatch,
    validate_analysis_metrics,
)


TERMINALS: Tuple[str, ...] = ("d", "g", "s", "b")
TEMPLATE = DEVICE_DECKS / "mosfet.spice.tmpl"
AC_FREQUENCY_HZ = 1e6


@dataclass(frozen=True)
class TerminalSweep:
    """One source-relative four-terminal DC sweep."""

    name: str
    axis: str
    start: float
    stop: float
    step: float
    vds: float
    vgs: float
    vbs: float
    source: float


@dataclass(frozen=True)
class TerminalBias:
    """One source-relative bias used for four AC terminal excitations."""

    name: str
    vds: float
    vgs: float
    vbs: float
    source: float


def terminal_current_metrics(
    candidate: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compare terminal-source currents and report interface KCL closure."""
    missing = [
        terminal for terminal in TERMINALS
        if terminal not in candidate or terminal not in reference
    ]
    if missing:
        raise ValueError(f"terminal current traces missing {missing}")
    test_rows = [np.asarray(candidate[name], dtype=float) for name in TERMINALS]
    ref_rows = [np.asarray(reference[name], dtype=float) for name in TERMINALS]
    shapes = {row.shape for row in (*test_rows, *ref_rows)}
    if len(shapes) != 1 or next(iter(shapes), ()) == ():
        raise ValueError("terminal current traces must share one non-scalar shape")
    test = np.vstack(test_rows)
    truth = np.vstack(ref_rows)
    if not np.all(np.isfinite(test)) or not np.all(np.isfinite(truth)):
        raise ValueError("terminal current traces contain NaN/Inf")
    metrics = dict(full_metrics(test.ravel(), truth.ravel()))
    domain: Dict[str, float] = {
        "candidate_kcl_max_a": float(np.max(np.abs(test.sum(axis=0)))),
        "reference_kcl_max_a": float(np.max(np.abs(truth.sum(axis=0)))),
    }
    names = {"d": "drain", "g": "gate", "s": "source", "b": "bulk"}
    for index, terminal in enumerate(TERMINALS):
        domain[f"{names[terminal]}_current_max_error_a"] = float(
            np.max(np.abs(test[index] - truth[index]))
        )
    return metrics, domain


def validate_sweep_lengths(
    candidate: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    *,
    expected_points: int,
) -> None:
    """Require aligned sweeps, allowing one floating-step endpoint omission."""
    candidate_lengths = {np.asarray(value).size for value in candidate.values()}
    reference_lengths = {np.asarray(value).size for value in reference.values()}
    if len(candidate_lengths) != 1 or len(reference_lengths) != 1:
        raise ValueError(
            f"terminal sweep lengths differ: candidate={candidate_lengths}, "
            f"reference={reference_lengths}"
        )
    candidate_count = next(iter(candidate_lengths))
    reference_count = next(iter(reference_lengths))
    if abs(candidate_count - reference_count) > 1:
        raise ValueError(
            f"terminal sweep lengths differ: candidate={candidate_count}, "
            f"reference={reference_count}"
        )
    if min(candidate_count, reference_count) < expected_points - 1:
        raise ValueError(
            f"terminal sweep incomplete: expected {expected_points} points, "
            f"observed candidate={candidate_count}, reference={reference_count}"
        )


def capacitance_from_admittance(
    admittance: np.ndarray,
    frequency_hz: float,
) -> np.ndarray:
    """Extract the quasi-static transcapacitance matrix from complex Y."""
    matrix = np.asarray(admittance, dtype=complex)
    if matrix.shape != (4, 4):
        raise ValueError(f"terminal admittance must be 4x4, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("terminal admittance contains NaN/Inf")
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("AC frequency must be finite and positive")
    return np.imag(matrix) / (2.0 * math.pi * frequency_hz)


def device_admittance_from_source_currents(
    source_currents: np.ndarray,
) -> np.ndarray:
    """Convert MNA voltage-source branch currents to device terminal current."""
    currents = np.asarray(source_currents, dtype=complex)
    if not np.all(np.isfinite(currents)):
        raise ValueError("terminal source currents contain NaN/Inf")
    return -currents


def _polarity(device: str) -> float:
    if device == "nmos":
        return 1.0
    if device == "pmos":
        return -1.0
    raise ValueError(f"unknown device kind {device!r}")


def terminal_sweeps(bt: BenchTech, device: str) -> Tuple[TerminalSweep, ...]:
    """Return DC sweeps spanning leakage, reversal, body, and source lift."""
    sign = _polarity(device)
    vdd = bt.vdd
    source = 0.0 if device == "nmos" else vdd
    lifted = 0.2 * vdd if device == "nmos" else 0.8 * vdd
    return (
        TerminalSweep(
            "gate", "g", 0.0, sign * vdd, sign * 0.01 * vdd,
            sign * 0.5 * vdd, 0.0, 0.0, source,
        ),
        TerminalSweep(
            "drain_reversal", "d", -sign * 0.2 * vdd,
            sign * vdd, sign * 0.01 * vdd,
            0.0, sign * 0.8 * vdd, 0.0, source,
        ),
        TerminalSweep(
            "body", "b", -sign * 0.3 * vdd, 0.0, sign * 0.01 * vdd,
            sign * 0.5 * vdd, sign * 0.8 * vdd, 0.0, source,
        ),
        TerminalSweep(
            "lifted_gate", "g", 0.0, sign * vdd, sign * 0.01 * vdd,
            sign * 0.5 * vdd, 0.0, 0.0, lifted,
        ),
    )


def terminal_biases(bt: BenchTech, device: str) -> Tuple[TerminalBias, ...]:
    """Return representative biases for the physical terminal C matrix."""
    sign = _polarity(device)
    vdd = bt.vdd
    source = 0.0 if device == "nmos" else vdd
    lifted = 0.2 * vdd if device == "nmos" else 0.8 * vdd
    return (
        TerminalBias("off", sign * 0.5 * vdd, 0.0, 0.0, source),
        TerminalBias(
            "linear", sign * 0.05 * vdd, sign * vdd, 0.0, source,
        ),
        TerminalBias(
            "saturation", sign * 0.5 * vdd, sign * 0.8 * vdd, 0.0, source,
        ),
        TerminalBias(
            "body", sign * 0.5 * vdd, sign * 0.8 * vdd,
            -sign * 0.2 * vdd, source,
        ),
        TerminalBias(
            "lifted", sign * 0.5 * vdd, sign * 0.8 * vdd, 0.0, lifted,
        ),
    )


def _device_values(
    bt: BenchTech,
    device: str,
    *,
    reference: bool,
    baked_lib: Path,
) -> Dict[str, str]:
    is_pmos = device == "pmos"
    kind = "PMOS" if is_pmos else "NMOS"
    model = bt.pmos_model if is_pmos else bt.nmos_model
    length = bt.l_pmos if is_pmos else bt.l_nmos
    nfin = bt.effective_nfin_p if is_pmos else bt.nfin
    vt = bt.effective_pmos_vt if is_pmos else bt.effective_nmos_vt
    if reference:
        setup = f'.include "{baked_lib}"'
        name = "Npdut" if is_pmos else "Nndut"
        model_spec = model
    else:
        nn_model = f"{device}_nn"
        setup = (
            f".model {nn_model} {kind} (LEVEL={{LEVEL}}{{FAMILY}} "
            f"TECH={bt.nn_tech} VT={vt})"
        )
        name = "Mpdut" if is_pmos else "Mndut"
        model_spec = f"{nn_model} L={length * 1e9:g}n NFIN={nfin}"
    return {
        "MODEL_SETUP": setup,
        "TEMP": f"{bt.temperature_c:.12g}",
        "DEVICE_NAME": name,
        "DRAIN_NODE": "d",
        "GATE_NODE": "g",
        "SOURCE_NODE": "s",
        "BULK_NODE": "b",
        "DEVICE": model_spec,
        "EXTRA_DEVICES": "",
        "LOAD": "",
    }


def _render_terminal_deck(
    bt: BenchTech,
    device: str,
    biases: Mapping[str, str],
    analysis: str,
    *,
    reference: bool,
    baked_lib: Path,
    level: int,
) -> str:
    values = {
        **_device_values(bt, device, reference=reference, baked_lib=baked_lib),
        **biases,
        "ANALYSIS": "" if reference else f".{analysis}",
    }
    values["MODEL_SETUP"] = values["MODEL_SETUP"].replace("{LEVEL}", str(level))
    values["MODEL_SETUP"] = values["MODEL_SETUP"].replace(
        "{FAMILY}",
        {75: " FAMILY=directnet-full", 76: " FAMILY=bsimar-full"}.get(
            level, "",
        ),
    )
    return render_template(TEMPLATE, values)


def render_terminal_sweep_decks(
    bt: BenchTech,
    device: str,
    sweep: TerminalSweep,
    *,
    baked_lib: Path,
    level: int,
) -> Tuple[str, str, str]:
    """Render both adapters for one four-terminal DC sweep."""
    source_name = {"d": "Vd", "g": "Vg", "b": "Vb"}[sweep.axis]
    absolute = {
        "d": sweep.source + sweep.vds,
        "g": sweep.source + sweep.vgs,
        "s": sweep.source,
        "b": sweep.source + sweep.vbs,
    }
    absolute[sweep.axis] = sweep.source + sweep.start
    biases = {
        "DRAIN_BIAS": f"Vd d 0 {absolute['d']:.12g}",
        "GATE_BIAS": f"Vg g 0 {absolute['g']:.12g}",
        "SOURCE_BIAS": f"Vs s 0 {absolute['s']:.12g}",
        "BULK_BIAS": f"Vb b 0 {absolute['b']:.12g}",
    }
    start = sweep.source + sweep.start
    stop = sweep.source + sweep.stop
    card = f"dc {source_name} {start:.12g} {stop:.12g} {sweep.step:.12g}"
    return (
        _render_terminal_deck(
            bt, device, biases, card, reference=False,
            baked_lib=baked_lib, level=level,
        ),
        _render_terminal_deck(
            bt, device, biases, card, reference=True,
            baked_lib=baked_lib, level=level,
        ),
        card,
    )


def _body(deck: str) -> str:
    return "\n".join(
        line for line in deck.splitlines()
        if line.strip().lower() != ".end" and not line.lstrip().startswith("*")
    )


def _terminal_columns(data: np.ndarray) -> Dict[str, np.ndarray]:
    if data.ndim != 2 or data.shape[1] != 8:
        raise RuntimeError(f"four-terminal wrdata has invalid shape {data.shape}")
    return {
        terminal: np.asarray(data[:, 2 * index + 1], dtype=float)
        for index, terminal in enumerate(TERMINALS)
    }


def run_terminal_current_sweep(
    bt: BenchTech,
    device: str,
    sweep: TerminalSweep,
    work_dir: Path,
    run_spec: RunSpec,
    corner: Corner = CORNERS["nominal"],
) -> GateResult:
    """Run and compare one physical four-terminal current sweep."""
    provenance = run_spec.result_fields()
    reference_converged = False
    stage = "setup"
    try:
        baked = get_baked_modelcard(
            bt, bt.nfin, work_dir, nfin_p=bt.effective_nfin_p,
        )
        candidate, reference, card = render_terminal_sweep_decks(
            bt, device, sweep, baked_lib=baked, level=run_spec.model_level,
        )
        analysis = AnalysisSpec(
            sweep.name,
            "dc",
            card,
            tuple(f"i(V{name.upper()})" for name in TERMINALS),
            device_kinds=(device,),
        )
        mismatch = physical_deck_mismatch(
            candidate,
            reference,
            analysis,
            bt,
            baked_lib=baked,
            model_level=run_spec.model_level,
            device_kinds=(device,),
        )
        if mismatch:
            raise ValueError(mismatch)
        signals = " ".join(f"i(V{name.upper()})" for name in TERMINALS)
        stage = "reference"
        reference_data = run_ngspice_wrdata(
            _body(reference), signals, work_dir, f"{device}_{sweep.name}_ref", card,
        )
        reference_currents = _terminal_columns(reference_data)
        reference_converged = True
        stage = "candidate"
        path = work_dir / f"candidate_{device}_{sweep.name}.sp"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(candidate)
        parser = parse_netlist(path)
        candidate_data = run_directnet_dc_sweep(
            path, work_dir, f"{device}_{sweep.name}", require_convergence=True,
        )
        candidate_currents = {
            terminal: np.asarray(candidate_data[f"i(V{terminal})"], dtype=float)
            for terminal in TERMINALS
        }
        expected = int(round(abs((sweep.stop - sweep.start) / sweep.step))) + 1
        validate_sweep_lengths(
            candidate_currents,
            reference_currents,
            expected_points=expected,
        )
        candidate_count = next(iter(candidate_currents.values())).size
        candidate_axis = (
            float(parser.analysis_params["start"])
            + float(parser.analysis_params["step"])
            * np.arange(candidate_count, dtype=float)
        )
        reference_axis = np.asarray(reference_data[:, 0], dtype=float)
        start, stop = analysis_axis_limits(analysis)
        for trace in (
            Trace("sweep", candidate_axis, dict(candidate_currents)),
            Trace(
                "sweep",
                reference_axis,
                dict(reference_currents),
                reference=True,
            ),
        ):
            trace.validate(
                expected_start=start,
                expected_stop=stop,
                endpoint_tolerance=analysis_endpoint_tolerance(analysis),
                max_step=analysis_max_step(analysis),
                minimum_points=analysis_minimum_points(analysis),
            )
        lo = max(float(np.min(candidate_axis)), float(np.min(reference_axis)))
        hi = min(float(np.max(candidate_axis)), float(np.max(reference_axis)))
        grid = np.linspace(lo, hi, min(candidate_axis.size, reference_axis.size))
        candidate_currents = {
            terminal: np.interp(
                grid,
                candidate_axis if candidate_axis[0] < candidate_axis[-1]
                else candidate_axis[::-1],
                values if candidate_axis[0] < candidate_axis[-1]
                else values[::-1],
            )
            for terminal, values in candidate_currents.items()
        }
        reference_currents = {
            terminal: np.interp(
                grid,
                reference_axis if reference_axis[0] < reference_axis[-1]
                else reference_axis[::-1],
                values if reference_axis[0] < reference_axis[-1]
                else values[::-1],
            )
            for terminal, values in reference_currents.items()
        }
        stage = "metrics"
        metrics, domain = terminal_current_metrics(
            candidate_currents, reference_currents,
        )
        validate_analysis_metrics(analysis, metrics, {})
        domain.update(
            terminal_current_capability=(
                "full" if run_spec.model_level in {75, 76} else "reduced"
            ),
            gate_bulk_accuracy_supported=run_spec.model_level in {75, 76},
            sweep_source=parser.analysis_params["source"],
        )
        return GateResult(
            case_id="terminal_currents",
            tech=bt.name,
            corner=corner.name,
            analysis=f"{device}_{sweep.name}",
            role="diagnostic",
            status="diagnostic",
            metrics=metrics,
            domain=domain,
            **provenance,
        )
    except Exception as exc:  # noqa: BLE001 - preserve one denominator row
        return GateResult(
            case_id="terminal_currents",
            tech=bt.name,
            corner=corner.name,
            analysis=f"{device}_{sweep.name}",
            role="diagnostic",
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            reference_converged=reference_converged,
            candidate_converged=False,
            execution_state=(
                "reference_error" if stage == "reference"
                else "nonconverged" if "converg" in str(exc).lower()
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


def _ac_bias_sources(bias: TerminalBias, excited: str) -> Dict[str, str]:
    absolute = {
        "d": bias.source + bias.vds,
        "g": bias.source + bias.vgs,
        "s": bias.source,
        "b": bias.source + bias.vbs,
    }
    return {
        "DRAIN_BIAS": (
            f"Vd d 0 DC={absolute['d']:.12g} AC={int(excited == 'd')} 0"
        ),
        "GATE_BIAS": (
            f"Vg g 0 DC={absolute['g']:.12g} AC={int(excited == 'g')} 0"
        ),
        "SOURCE_BIAS": (
            f"Vs s 0 DC={absolute['s']:.12g} AC={int(excited == 's')} 0"
        ),
        "BULK_BIAS": (
            f"Vb b 0 DC={absolute['b']:.12g} AC={int(excited == 'b')} 0"
        ),
    }


def render_terminal_ac_decks(
    bt: BenchTech,
    device: str,
    bias: TerminalBias,
    excited: str,
    *,
    baked_lib: Path,
    level: int,
) -> Tuple[str, str, str]:
    """Render one column of the four-terminal small-signal Y matrix."""
    if excited not in TERMINALS:
        raise ValueError(f"unknown excited terminal {excited!r}")
    card = f"ac lin 1 {AC_FREQUENCY_HZ:g} {AC_FREQUENCY_HZ:g}"
    sources = _ac_bias_sources(bias, excited)
    return (
        _render_terminal_deck(
            bt, device, sources, card, reference=False,
            baked_lib=baked_lib, level=level,
        ),
        _render_terminal_deck(
            bt, device, sources, card, reference=True,
            baked_lib=baked_lib, level=level,
        ),
        card,
    )


def _candidate_ac_currents(deck: str, path: Path) -> np.ndarray:
    from pycircuitsim.simulation import _circuit_has_nn, _solve_dc_with_retry
    from pycircuitsim.solver import ACSolver, DCSolver

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(deck)
    parser = parse_netlist(path)
    circuit = parser.circuit

    def _solve(use_gmin: bool) -> Tuple[DCSolver, Dict[str, float]]:
        solver = DCSolver(
            circuit,
            use_source_stepping=True,
            use_gmin_stepping=use_gmin,
        )
        return solver, solver.solve()

    solver, dc_solution = _solve_dc_with_retry(
        circuit, _circuit_has_nn(circuit), _solve,
    )
    if not solver._last_solve_converged:
        raise RuntimeError("terminal AC operating point did not converge")
    result = ACSolver(circuit, dc_solution=dc_solution).solve(
        np.asarray([AC_FREQUENCY_HZ]),
    )
    return device_admittance_from_source_currents(np.asarray(
        [result[f"i(V{terminal})"][0] for terminal in TERMINALS],
        dtype=complex,
    ))


def _reference_ac_currents(
    deck: str,
    card: str,
    work_dir: Path,
    tag: str,
) -> np.ndarray:
    from tests.common.simple_circuit_harness import _run_ngspice_ac_trace

    signals = tuple(f"i(V{terminal.upper()})" for terminal in TERMINALS)
    trace = _run_ngspice_ac_trace(deck, signals, card, work_dir, tag)
    trace.validate(
        expected_start=AC_FREQUENCY_HZ,
        expected_stop=AC_FREQUENCY_HZ,
        minimum_points=1,
    )
    if trace.axis.size != 1:
        raise ValueError(
            f"terminal AC expected one frequency, got {trace.axis.size}"
        )
    return device_admittance_from_source_currents(np.asarray(
        [trace.signals[signal][0] for signal in signals], dtype=complex,
    ))


def run_terminal_capacitance_bias(
    bt: BenchTech,
    device: str,
    bias: TerminalBias,
    work_dir: Path,
    run_spec: RunSpec,
    corner: Corner = CORNERS["nominal"],
) -> GateResult:
    """Build and compare the complete four-terminal quasi-static C matrix."""
    provenance = run_spec.result_fields()
    reference_converged = False
    stage = "setup"
    try:
        baked = get_baked_modelcard(
            bt, bt.nfin, work_dir, nfin_p=bt.effective_nfin_p,
        )
        candidate_y = np.zeros((4, 4), dtype=complex)
        reference_y = np.zeros((4, 4), dtype=complex)
        for column, excited in enumerate(TERMINALS):
            candidate, reference, card = render_terminal_ac_decks(
                bt, device, bias, excited,
                baked_lib=baked, level=run_spec.model_level,
            )
            analysis = AnalysisSpec(
                f"{bias.name}_{excited}",
                "ac",
                card,
                tuple(f"i(V{name.upper()})" for name in TERMINALS),
                device_kinds=(device,),
            )
            mismatch = physical_deck_mismatch(
                candidate,
                reference,
                analysis,
                bt,
                baked_lib=baked,
                model_level=run_spec.model_level,
                device_kinds=(device,),
            )
            if mismatch:
                raise ValueError(mismatch)
            stage = "reference"
            reference_y[:, column] = _reference_ac_currents(
                reference, card, work_dir,
                f"{device}_{bias.name}_{excited}_ref",
            )
            stage = "candidate"
            candidate_y[:, column] = _candidate_ac_currents(
                candidate, work_dir / f"candidate_{device}_{bias.name}_{excited}.sp",
            )
        reference_converged = True
        stage = "metrics"
        candidate_c = capacitance_from_admittance(
            candidate_y, AC_FREQUENCY_HZ,
        )
        reference_c = capacitance_from_admittance(
            reference_y, AC_FREQUENCY_HZ,
        )
        metrics = dict(full_metrics(
            candidate_c.ravel() * 1e15,
            reference_c.ravel() * 1e15,
        ))
        metrics["max_err"] *= 1e-15
        validate_analysis_metrics(analysis, metrics, {})
        domain: Dict[str, Any] = {
            "capacitance_max_error_f": float(
                np.max(np.abs(candidate_c - reference_c))
            ),
            "candidate_charge_column_closure_f": float(
                np.max(np.abs(candidate_c.sum(axis=0)))
            ),
            "reference_charge_column_closure_f": float(
                np.max(np.abs(reference_c.sum(axis=0)))
            ),
            "candidate_voltage_row_closure_f": float(
                np.max(np.abs(candidate_c.sum(axis=1)))
            ),
            "reference_voltage_row_closure_f": float(
                np.max(np.abs(reference_c.sum(axis=1)))
            ),
            "candidate_capacitance_f": candidate_c.tolist(),
            "reference_capacitance_f": reference_c.tolist(),
        }
        return GateResult(
            case_id="terminal_capacitance",
            tech=bt.name,
            corner=corner.name,
            analysis=f"{device}_{bias.name}",
            role="diagnostic",
            status="diagnostic",
            metrics=metrics,
            domain=domain,
            **provenance,
        )
    except Exception as exc:  # noqa: BLE001 - preserve one denominator row
        return GateResult(
            case_id="terminal_capacitance",
            tech=bt.name,
            corner=corner.name,
            analysis=f"{device}_{bias.name}",
            role="diagnostic",
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            reference_converged=reference_converged,
            candidate_converged=False,
            execution_state=(
                "reference_error" if stage == "reference"
                else "nonconverged" if "converg" in str(exc).lower()
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


def terminal_corner_applies(
    base_bt: BenchTech,
    device: str,
    corner: Corner,
) -> bool:
    """Whether a corner changes a field these source-relative decks observe.

    Delegates to the device-integrity rule so the two single-device gates
    cannot disagree about which corners are no-ops for a polarity — a corner
    that changes nothing must not create a denominator row.
    """
    return device_corner_applies(base_bt, device, corner)


def run_terminal_integrity(
    bt: BenchTech,
    devices: Sequence[str],
    work_dir: Path,
    run_spec: RunSpec,
    corner: Corner = CORNERS["nominal"],
) -> List[GateResult]:
    """Run all declared terminal-current and transcapacitance diagnostics.

    ``bt`` is the *base* technology; the corner is applied here so that the
    sweep and bias grids are built from the stressed supply, and so the caller
    cannot pass a stressed profile and an unrelated corner label.
    """
    selected = [device for device in devices
                if terminal_corner_applies(bt, device, corner)]
    if not selected:
        # ``apply_corner`` raises for a stress this technology cannot express
        # (TSMC7 has no alternate trained VT). Filtering first keeps that an
        # empty result rather than an infrastructure error, exactly as
        # ``run_device_suites`` does.
        return []
    stressed = apply_corner(bt, corner)
    results: List[GateResult] = []
    for device in selected:
        for sweep in terminal_sweeps(stressed, device):
            results.append(run_terminal_current_sweep(
                stressed, device, sweep,
                work_dir / device / "dc" / sweep.name, run_spec, corner,
            ))
        for bias in terminal_biases(stressed, device):
            results.append(run_terminal_capacitance_bias(
                stressed, device, bias,
                work_dir / device / "ac" / bias.name, run_spec, corner,
            ))
    return results
