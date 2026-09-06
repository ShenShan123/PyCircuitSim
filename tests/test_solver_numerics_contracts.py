"""Hermetic contracts for the solver numerics AGENTS.md states.

Each of these rules was stated as a solver contract and exercised only
indirectly, through circuit gates that need NGSPICE and a checkpoint, where a
regression shows up as a slightly worse NRMSE instead of a named failure
(V7.7.2 audit B6, B7, B9).  They come back here the way
``test_core_device_contracts.py`` answers the same class of question: in
process, against closed forms, with no simulator.

The nonlinear cases use ``ClosedFormDevice``: a LEVEL=75-shaped four-terminal
element whose drain current is a closed-form function of one controlling
voltage, stamped through the same ``get_terminal_stamp`` seam the trained
models use.  Nothing here compares a compact model against itself.
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

from pycircuitsim.circuit import Circuit
from pycircuitsim.models.mosfet_cmg import MOSFET_CMG, NMOS_CMG
from pycircuitsim.models.mosfet_directnet_full import NMOS_DNF
from pycircuitsim.models.passive import (
    Capacitor, PulseVoltageSource, Resistor, VoltageSource,
)
import pycircuitsim.solver as solver_module
from pycircuitsim.solver import DCSolver, TransientSolver, _nr_step_converged

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from neural_network.data.contracts import FULL_TERMINAL_OUTPUT_COLUMN_ORDER
from neural_network.data.normalize import NormStats
from tests.common.base import MODELCARDS_DIR, OSDI_PATH

Law = Callable[[float], Tuple[float, float]]

_UNBOUNDED_STATS = NormStats(
    mode="zscore",
    input_mean=np.zeros(7), input_std=np.ones(7),
    input_min=np.full(7, -1e9), input_max=np.full(7, 1e9),
    output_mean=np.zeros(6), output_std=np.ones(6),
    output_columns=list(FULL_TERMINAL_OUTPUT_COLUMN_ORDER),
)


class ClosedFormDevice(NMOS_DNF):
    """Four-terminal element with a closed-form drain current.

    ``law(v)`` returns ``(i_d, di_d/dv)`` for the controlling voltage
    ``v = V(control) - V(source)``; the source row closes KCL and the source
    column follows from translation invariance, exactly as the trained
    full-terminal families are stamped.  An optional drain-source capacitance
    exercises charge history alongside explicit capacitors.
    """

    _artifact_loader = staticmethod(lambda _path: (
        None, _UNBOUNDED_STATS, 2, tuple(FULL_TERMINAL_OUTPUT_COLUMN_ORDER),
    ))

    def __init__(
        self, name: str, nodes: List[str], law: Law, *, control: str = "d",
        capacitance: float = 0.0,
    ) -> None:
        super().__init__(
            name, nodes, "/synthetic/closed_form_best.pt",
            L=16e-9, NFIN=2.0, tech_code=0,
        )
        self._law = law
        self._control = {"d": 0, "g": 1}[control]
        self._capacitance = capacitance

    def _eval(
        self, voltages: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        terminal = [float(voltages.get(node, 0.0)) for node in self.nodes]
        current, slope = self._law(terminal[self._control] - terminal[2])
        currents = np.zeros(4)
        currents[0], currents[2] = current, -current
        jacobian = np.zeros((4, 4))
        jacobian[0, self._control] += slope
        jacobian[0, 2] -= slope
        jacobian[2] = -jacobian[0]
        capacitance = np.zeros((4, 4)) if self._caps_required else None
        charges = np.zeros(4)
        charges[0] = self._capacitance * (terminal[0] - terminal[2])
        charges[2] = -charges[0]
        if capacitance is not None:
            capacitance[0, 0] = capacitance[2, 2] = self._capacitance
            capacitance[0, 2] = capacitance[2, 0] = -self._capacitance
        return currents, jacobian, charges, capacitance


def _linear_law(conductance: float) -> Law:
    return lambda v: (conductance * v, conductance)


def _series_circuit(
    law: Law, resistance: float = 1_000.0, *, capacitance: float = 0.0,
) -> Circuit:
    circuit = Circuit()
    circuit.add_component(VoltageSource("V1", ["in", "0"], 1.0))
    circuit.add_component(Resistor("R1", ["in", "out"], resistance))
    circuit.add_component(ClosedFormDevice(
        "M1", ["out", "0", "0", "0"], law, capacitance=capacitance,
    ))
    return circuit


# ---------------------------------------------------------------------------
# Terminal-current convention and pinned tolerances
# ---------------------------------------------------------------------------
def test_full_terminal_stamp_takes_positive_current_into_each_terminal() -> None:
    """A 1 kOhm device in series with 1 kOhm from 1 V must sit at 0.5 V.

    The device/runtime boundary is positive current leaving the node into
    the terminal; a flipped convention would put the node at 1.5 V or fail.
    """
    circuit = _series_circuit(_linear_law(1e-3))
    solver = DCSolver(circuit)
    solution = solver.solve(skip_header=True)
    assert solver._last_solve_converged
    assert solution["out"] == pytest.approx(0.5, abs=1e-9)


def test_default_tolerances_and_physical_gmin_are_pinned() -> None:
    circuit = _series_circuit(_linear_law(1e-3))
    dc = DCSolver(circuit)
    tran = TransientSolver(circuit, t_stop=1e-9, dt=1e-11)
    assert (dc.reltol, dc.vntol, dc.gmin) == (1e-4, 1e-7, 1e-12)
    assert (tran.reltol, tran.vntol, tran.gmin) == (1e-4, 1e-7, 1e-12)
    assert tran.gmin_final == 1e-12


def test_convergence_test_is_the_spice_form_on_both_scales() -> None:
    """``|dV| < VNTOL + RELTOL * max(|V_old|, |V_new|)`` with strict ``<``."""
    reltol, vntol = 1e-4, 1e-7
    new = np.asarray([1.0, 0.0])
    old = np.asarray([0.5, 2.0])
    threshold = vntol + reltol * np.asarray([1.0, 2.0])
    below = threshold * (1.0 - 1e-9)
    assert _nr_step_converged(below, new, old, reltol, vntol)
    assert not _nr_step_converged(threshold, new, old, reltol, vntol)
    one_node_over = below.copy()
    one_node_over[1] = threshold[1]
    assert not _nr_step_converged(one_node_over, new, old, reltol, vntol)
    # The larger of the two voltages sets the scale, whichever side it is on.
    assert _nr_step_converged(
        np.asarray([1.5e-4]), np.asarray([0.0]), np.asarray([2.0]), reltol, vntol,
    )
    assert not _nr_step_converged(
        np.asarray([1.5e-4]), np.asarray([0.0]), np.asarray([1.0]), reltol, vntol,
    )


# ---------------------------------------------------------------------------
# GMIN ladder
# ---------------------------------------------------------------------------
def _recorded_gmin_levels(
    circuit: Circuit, monkeypatch: pytest.MonkeyPatch,
) -> Tuple[List[float], DCSolver]:
    levels: List[float] = []
    original = DCSolver._apply_gmin_stepping

    def record(self: DCSolver, matrix, node_map, gmin: float) -> None:
        levels.append(gmin)
        return original(self, matrix, node_map, gmin)

    monkeypatch.setattr(DCSolver, "_apply_gmin_stepping", record)
    solver = DCSolver(circuit, use_gmin_stepping=True)
    solver.solve(skip_header=True)
    return sorted(set(levels), reverse=True), solver


def test_wide_gmin_ladder_walks_a_decade_at_a_time_down_to_physical_gmin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-NN full-stamp circuits use the eleven-level ladder.

    Only levels above physical GMIN are stamped; the final level is the
    plain circuit, which is where the verdict is taken.
    """
    monkeypatch.setattr(solver_module, "_has_nn_device", lambda _circuit: False)
    levels, solver = _recorded_gmin_levels(
        _series_circuit(_linear_law(1e-3)), monkeypatch,
    )
    assert levels == [10.0 ** -k for k in range(2, 12)]
    assert solver.gmin == 1e-12 and solver.gmin not in levels
    assert solver._last_solve_converged


def test_nn_circuits_keep_the_measured_two_level_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    levels, solver = _recorded_gmin_levels(
        _series_circuit(_linear_law(1e-3)), monkeypatch,
    )
    assert levels == [1e-8]
    assert solver._last_solve_converged


# ---------------------------------------------------------------------------
# Limiting and oscillation acceptance
# ---------------------------------------------------------------------------
class _AlwaysLimited(ClosedFormDevice):
    """Reports a live limiter on every evaluation without moving anything."""

    def nr_limit_voltages(self, voltages: Dict[str, float]) -> Dict[str, float]:
        self._nr_limited = True
        return voltages


def test_an_iteration_that_still_applied_a_limiter_is_not_converged() -> None:
    circuit = Circuit()
    circuit.add_component(VoltageSource("V1", ["in", "0"], 1.0))
    circuit.add_component(Resistor("R1", ["in", "out"], 1_000.0))
    circuit.add_component(
        _AlwaysLimited("M1", ["out", "0", "0", "0"], _linear_law(1e-3)),
    )
    solver = DCSolver(circuit)
    solution = solver.solve(skip_header=True)
    # The iterate is the right answer, and it is still refused: a stamp about
    # a limited bias is an extrapolation, so an unlimited follow-up is required.
    assert solution["out"] == pytest.approx(0.5, abs=1e-6)
    assert solver._last_solve_converged is False


def _two_cycle_law(amplitude_a: float, level: float = 0.4) -> Law:
    """No fixed point: the current flips sign either side of ``level``.

    With ``R`` to ground the iterate alternates ``level +/- R*amplitude``.
    """
    bias = -level / 1_000.0
    return lambda v: (bias + (amplitude_a if v >= level else -amplitude_a), 0.0)


@pytest.mark.parametrize(
    ("amplitude_a", "accepted"),
    (
        (1e-12, True),   # 1 nV two-cycle: within tolerance, KCL residual tiny
        (1e-3, False),   # 1 V two-cycle: no physical fixed point exists
    ),
)
def test_oscillation_average_is_accepted_only_within_tolerance_and_kcl(
    amplitude_a: float, accepted: bool,
) -> None:
    circuit = Circuit()
    circuit.add_component(Resistor("R1", ["out", "0"], 1_000.0))
    circuit.add_component(
        ClosedFormDevice("M1", ["out", "0", "0", "0"], _two_cycle_law(amplitude_a)),
    )
    solver = DCSolver(circuit)
    solution = solver.solve(skip_header=True)
    assert solver._last_solve_converged is accepted
    if accepted:
        assert solution["out"] == pytest.approx(0.4, abs=1e-6)
    assert math.isfinite(solution["out"])


# ---------------------------------------------------------------------------
# Transient integration ladder and breakpoints
# ---------------------------------------------------------------------------
_RC_R, _RC_C, _RC_H, _RC_V = 1_000.0, 1e-12, 2e-13, 1.0


def _rc_response(method: str, steps: int = 6) -> np.ndarray:
    circuit = Circuit()
    circuit.add_component(VoltageSource("V1", ["in", "0"], _RC_V))
    circuit.add_component(Resistor("R1", ["in", "out"], _RC_R))
    circuit.add_component(Capacitor("C1", ["out", "0"], _RC_C))
    solver = TransientSolver(
        circuit, t_stop=steps * _RC_H, dt=_RC_H, integration_method=method,
        initial_guess={"in": _RC_V, "out": 0.0},
    )
    return np.asarray(solver.solve()["out"])


def _closed_form(first: str, rest: str, steps: int = 6) -> np.ndarray:
    """Discrete RC step response for a first-step method then a steady one."""
    a = _RC_H / (_RC_R * _RC_C)
    values = [0.0]

    def advance(method: str) -> float:
        if method == "be":
            return (values[-1] + a * _RC_V) / (1.0 + a)
        if method == "trap":
            return (values[-1] * (1.0 - a / 2.0) + a * _RC_V) / (1.0 + a / 2.0)
        if method == "bdf2":
            return (4.0 * values[-1] - values[-2] + 2.0 * a * _RC_V) / (3.0 + 2.0 * a)
        raise ValueError(method)

    values.append(advance(first))
    for _ in range(steps - 1):
        values.append(advance(rest))
    return np.asarray(values)


@pytest.mark.parametrize("method", ("auto", "trap", "gear2"))
def test_first_accepted_step_is_backward_euler_in_every_mode(method: str) -> None:
    response = _rc_response(method)
    assert response[1] == pytest.approx(_closed_form("be", "be")[1], rel=1e-12)
    assert response[1] != pytest.approx(_closed_form("trap", "trap")[1], rel=1e-6)


def test_gear2_ladder_is_backward_euler_then_bdf2_exactly() -> None:
    np.testing.assert_allclose(
        _rc_response("gear2"), _closed_form("be", "bdf2"), rtol=1e-12, atol=1e-18,
    )


@pytest.mark.parametrize("method", ("auto", "trap"))
def test_auto_ladder_is_backward_euler_then_trapezoidal_exactly(method: str) -> None:
    np.testing.assert_allclose(
        _rc_response(method), _closed_form("be", "trap"), rtol=1e-9, atol=1e-15,
    )


@pytest.mark.parametrize("method", ("auto", "trap", "gear2"))
def test_stiffness_promotes_auto_once_and_respects_pinned_methods(
    monkeypatch: pytest.MonkeyPatch, method: str,
) -> None:
    """A hard converged step promotes auto; later easy steps cannot undo it."""
    circuit = _series_circuit(_linear_law(1e-3))
    circuit.add_component(Capacitor("C1", ["out", "0"], _RC_C))
    solver = TransientSolver(
        circuit, t_stop=6 * _RC_H, dt=_RC_H, integration_method=method,
        initial_guess={"in": 1.0, "out": 0.0},
    )
    solve_step = solver._solve_timestep_newton
    methods: List[str] = []

    def record_step(**kwargs: object) -> Dict[str, float]:
        methods.append(solver._integration_method)
        result = solve_step(**kwargs)
        # Drive the policy input after a real, converged nonlinear solve.
        solver._last_nr_iterations = 21 if kwargs["step_index"] == 2 else 2
        return result

    monkeypatch.setattr(solver, "_solve_timestep_newton", record_step)
    solver.solve()
    expected = {
        "auto": ["be", "trap", "trap", "bdf2", "bdf2", "bdf2"],
        "trap": ["be"] + ["trap"] * 5,
        "gear2": ["be"] + ["bdf2"] * 5,
    }
    assert methods == expected[method]


@pytest.mark.parametrize("refine", (False, True))
@pytest.mark.parametrize("method", ("trap", "gear2"))
def test_failed_newton_piece_retries_earlier_without_committing_history(
    monkeypatch: pytest.MonkeyPatch, refine: bool, method: str,
) -> None:
    """Halving must move the target time and keep both device histories intact."""
    circuit = _series_circuit(_linear_law(1e-3), capacitance=_RC_C)
    capacitor = Capacitor("C1", ["out", "0"], _RC_C)
    circuit.add_component(capacitor)
    solver = TransientSolver(
        circuit, t_stop=3 * _RC_H, dt=_RC_H, refine_output=refine,
        initial_guess={"in": 1.0, "out": 0.0}, integration_method=method,
    )
    solve_step = solver._solve_timestep_newton
    attempts: List[tuple] = []
    rejected_attempt: List[int] = []

    def fail_once(**kwargs: object) -> Dict[str, float]:
        attempts.append((kwargs["time"], solver._current_dt,
                         (solver._snapshot_tran_state(), solver._previous_dt)))
        result = solve_step(**kwargs)
        if kwargs["step_index"] == 1 and not rejected_attempt:
            rejected_attempt.append(len(attempts) - 1)
            raise RuntimeError("controlled nonlinear-step rejection")
        return result

    monkeypatch.setattr(solver, "_solve_timestep_newton", fail_once)
    result = solver.solve()
    index, = rejected_attempt
    failed_time, failed_dt, failed_state = attempts[index]
    retry_time, retry_dt, retry_state = attempts[index + 1]
    assert retry_dt == pytest.approx(failed_dt / 2, rel=1e-12, abs=0)
    assert retry_time == pytest.approx(failed_time - failed_dt / 2, rel=1e-12, abs=0)
    assert retry_state == failed_state
    assert len(solver._dt_halve_events) == 1
    assert capacitor.v_prev == pytest.approx(result["out"][-1])
    assert result["time"][-1] == pytest.approx(solver.t_stop, rel=1e-12, abs=0)
    assert np.all(np.diff(result["time"]) > 0)


@pytest.mark.parametrize("method", ("trap", "gear2"))
def test_lte_rejection_restores_device_history_before_retry(
    monkeypatch: pytest.MonkeyPatch, method: str,
) -> None:
    """A converged candidate rejected by LTE must not seed the next companion."""
    circuit = _series_circuit(_linear_law(1e-3), capacitance=_RC_C)
    circuit.add_component(Capacitor("C1", ["out", "0"], _RC_C))
    solver = TransientSolver(
        circuit, t_stop=5 * _RC_H, dt=_RC_H, refine_output=True,
        integration_method=method,
        initial_guess={"in": 1.0, "out": 0.0},
    )
    solve_step = solver._solve_timestep_newton
    attempts: List[tuple] = []
    rejected_attempt: List[int] = []

    def record_step(**kwargs: object) -> Dict[str, float]:
        attempts.append((kwargs["time"], solver._current_dt,
                         (solver._snapshot_tran_state(), solver._previous_dt)))
        return solve_step(**kwargs)

    def reject_once(hist: List[tuple], time: float, voltage: np.ndarray) -> float:
        if not rejected_attempt:
            rejected_attempt.append(len(attempts) - 1)
            return 8.0
        return 0.0

    monkeypatch.setattr(solver, "_solve_timestep_newton", record_step)
    monkeypatch.setattr(solver, "_refine_lte_ratio", reject_once)
    result = solver.solve()
    index, = rejected_attempt
    failed_time, failed_dt, failed_state = attempts[index]
    retry_time, retry_dt, retry_state = attempts[index + 1]
    assert retry_dt == pytest.approx(failed_dt / 2, rel=1e-12, abs=0)
    assert retry_time < failed_time
    assert retry_state == failed_state
    assert np.all(np.diff(result["time"]) > 0)
    assert len(result["time"]) == len(attempts)  # initial point replaces rejection


@pytest.mark.parametrize("refine", (False, True))
@pytest.mark.parametrize("stop_steps", (0.5, 2.5))
def test_partial_final_interval_evaluates_sources_at_the_reported_time(
    refine: bool, stop_steps: float,
) -> None:
    """An endpoint clipped in the output must also clip the physical solve."""
    circuit = Circuit()
    circuit.add_component(PulseVoltageSource(
        "V1", ["in", "0"], 0.0, 1.0, 0.0,
        10 * _RC_H, _RC_H, 10 * _RC_H, 30 * _RC_H,
    ))
    circuit.add_component(Resistor("R1", ["in", "0"], _RC_R))
    solver = TransientSolver(
        circuit, t_stop=stop_steps * _RC_H, dt=_RC_H,
        initial_guess={"in": 0.0}, refine_output=refine,
    )
    result = solver.solve()
    assert result["time"][-1] == pytest.approx(solver.t_stop, rel=1e-12, abs=0)
    assert np.all(result["time"] <= solver.t_stop)
    np.testing.assert_allclose(result["in"], result["time"] / (10 * _RC_H))
    np.testing.assert_allclose(solver.source_currents["V1"], -result["in"] / _RC_R)


@pytest.mark.parametrize("full_terminal", (False, True))
def test_gear2_current_on_a_ramp_uses_the_previous_accepted_piece_length(
    full_terminal: bool,
) -> None:
    """d(CV)/dt stays C*slope when the final BDF-2 interval is halved."""
    circuit = Circuit()
    circuit.add_component(PulseVoltageSource(
        "V1", ["in", "0"], 0.0, 1.0, 0.0,
        10 * _RC_H, _RC_H, 10 * _RC_H, 30 * _RC_H,
    ))
    if full_terminal:
        device = ClosedFormDevice(
            "M1", ["in", "0", "0", "0"], _linear_law(0.0), capacitance=_RC_C,
        )
    else:
        device = Capacitor("C1", ["in", "0"], _RC_C)
    circuit.add_component(device)
    solver = TransientSolver(
        circuit, t_stop=2.5 * _RC_H, dt=_RC_H, integration_method="gear2",
        initial_guess={"in": 0.0},
    )
    solver.solve()
    expected_current = _RC_C / (10 * _RC_H)
    current = device._i_prev_drain if full_terminal else device._i_prev
    assert current == pytest.approx(expected_current, rel=1e-10)
    np.testing.assert_allclose(
        solver.source_currents["V1"][1:], -expected_current, rtol=1e-10,
    )


def test_pulse_breakpoints_are_coalesced_with_minbreak_tolerance() -> None:
    """Two sources naming the same corner one ulp apart yield one breakpoint."""
    circuit = Circuit()
    delay, rise, fall, width, period = 1e-9, 1e-11, 1e-11, 2e-9, 5e-9
    circuit.add_component(PulseVoltageSource(
        "Vp", ["a", "0"], 0.0, 1.0, delay, rise, fall, width, period))
    circuit.add_component(PulseVoltageSource(
        "Vq", ["b", "0"], 0.0, 1.0, delay * (1.0 + 1e-15), rise, fall, width, period))
    circuit.add_component(Resistor("Ra", ["a", "0"], 1e3))
    circuit.add_component(Resistor("Rb", ["b", "0"], 1e3))
    solver = TransientSolver(circuit, t_stop=6e-9, dt=1e-11)
    assert solver._breakpoint_min_break() == pytest.approx(5e-5 * 1e-11)
    breakpoints = solver._collect_breakpoints()
    expected = [delay, delay + rise, delay + rise + width, delay + rise + width + fall]
    np.testing.assert_allclose(breakpoints, expected, rtol=1e-9)
    assert all(0.0 < t < 6e-9 for t in breakpoints)


# ---------------------------------------------------------------------------
# .nodeset: clamp, converge, release (B7)
# ---------------------------------------------------------------------------
def _cubic_law(v: float) -> Tuple[float, float]:
    """i = a(v^3 - v): with 2 kOhm to ground the node has roots 0, +/-1/sqrt2."""
    return 1e-3 * (v ** 3 - v), 1e-3 * (3.0 * v ** 2 - 1.0)


def _bistable_node() -> Circuit:
    circuit = Circuit()
    circuit.add_component(Resistor("R1", ["out", "0"], 2_000.0))
    circuit.add_component(ClosedFormDevice("M1", ["out", "0", "0", "0"], _cubic_law))
    return circuit


@pytest.mark.parametrize(
    ("nodesets", "root"),
    (
        (None, 0.0),
        ({"out": 0.5}, math.sqrt(0.5)),
        ({"out": -0.5}, -math.sqrt(0.5)),
        ({"OUT": 0.5}, math.sqrt(0.5)),          # names resolve case-insensitively
        ({"no_such_node": 0.5}, 0.0),           # ignored, as NGSPICE warns and continues
        ({"out": float("nan")}, 0.0),           # a failed clamp keeps the plain solve
    ),
)
def test_nodeset_selects_a_branch_then_releases_the_node(
    nodesets: Optional[Dict[str, float]], root: float,
) -> None:
    solver = DCSolver(_bistable_node(), nodesets=nodesets)
    with warnings.catch_warnings():
        # A NaN hint makes the clamped pre-solve singular; that pre-solve is
        # the thing being discarded, so its warning is part of the contract.
        warnings.simplefilter("ignore")
        solution = solver.solve(skip_header=True)
    assert solver._last_solve_converged
    assert solution["out"] == pytest.approx(root, abs=1e-6)
    # The clamp is temporary: the released circuit is the original one.
    assert [c.name for c in solver.circuit.components] == ["R1", "M1"]


# ---------------------------------------------------------------------------
# Latch basin (B9)
# ---------------------------------------------------------------------------
_LATCH_VDD, _LATCH_MID, _LATCH_I0, _LATCH_W, _LATCH_R = 0.8, 0.4, 3.5e-4, 0.1, 1_000.0


def _tanh_transconductor(v_gate: float) -> Tuple[float, float]:
    x = (v_gate - _LATCH_MID) / _LATCH_W
    return _LATCH_I0 * math.tanh(x), _LATCH_I0 / (_LATCH_W * math.cosh(x) ** 2)


def _latch() -> Circuit:
    """Two cross-coupled transconductors: loop gain R*I0/W = 3.5 > 1.

    Stable states are ``V_mid +/- R*I0*tanh(3.5)``; the symmetric point is
    the unstable saddle.
    """
    circuit = Circuit()
    circuit.add_component(VoltageSource("Vmid", ["mid", "0"], _LATCH_MID))
    circuit.add_component(Resistor("Rq", ["q", "mid"], _LATCH_R))
    circuit.add_component(Resistor("Rqb", ["qb", "mid"], _LATCH_R))
    circuit.add_component(
        ClosedFormDevice("Mq", ["q", "qb", "0", "0"], _tanh_transconductor, control="g"))
    circuit.add_component(
        ClosedFormDevice("Mqb", ["qb", "q", "0", "0"], _tanh_transconductor, control="g"))
    circuit.add_component(Capacitor("Cq", ["q", "0"], 2e-15))
    circuit.add_component(Capacitor("Cqb", ["qb", "0"], 2e-15))
    return circuit


def _latch_state(q: float, qb: float) -> str:
    assert abs(q - qb) > 0.3, f"latch left its basin: q={q:.4f} qb={qb:.4f}"
    return "q1" if q > qb else "q0"


_STORED_STATES = (("q1", 0.75, 0.05), ("q0", 0.05, 0.75))


@pytest.mark.parametrize(("state", "q0", "qb0"), _STORED_STATES)
def test_hard_ic_operating_point_lands_in_the_requested_basin(
    state: str, q0: float, qb0: float,
) -> None:
    """``.ic`` in hard mode pins, releases, and stays in the pinned basin."""
    swing = _LATCH_R * _LATCH_I0 * math.tanh(3.5)
    solver = DCSolver(_latch(), force_ic=True, initial_guess={"q": q0, "qb": qb0})
    solution = solver.solve(skip_header=True)
    assert solver._last_solve_converged
    assert _latch_state(solution["q"], solution["qb"]) == state
    high, low = max(solution["q"], solution["qb"]), min(solution["q"], solution["qb"])
    assert high == pytest.approx(_LATCH_MID + swing, abs=2e-3)
    assert low == pytest.approx(_LATCH_MID - swing, abs=2e-3)


@pytest.mark.parametrize(("state", "q0", "qb0"), _STORED_STATES)
@pytest.mark.parametrize("perturbation", ("reference", "refine_output"))
def test_latch_retains_both_stored_states_under_every_opt_in_march(
    state: str, q0: float, qb0: float, perturbation: str,
) -> None:
    """A change that can alter a nonlinear basin must keep both states.

    The reference march is the scored fixed-grid path; ``refine_output`` is
    the one surviving opt-in fidelity control that changes how the march is
    taken.  Any new opt-in knob that can perturb a basin belongs in this
    parametrization before it is used.
    """
    solver = TransientSolver(
        _latch(), t_stop=2e-9, dt=1e-11,
        initial_guess={"q": q0, "qb": qb0, "mid": _LATCH_MID},
        refine_output=(perturbation == "refine_output"),
    )
    result = solver.solve()
    assert _latch_state(float(result["q"][0]), float(result["qb"][0])) == state
    assert _latch_state(float(result["q"][-1]), float(result["qb"][-1])) == state


# ---------------------------------------------------------------------------
# Outer device bounds (B8)
# ---------------------------------------------------------------------------
def test_level72_evaluation_window_is_the_named_constant() -> None:
    """LEVEL=72 evaluates inside +/-_NR_LIM_WINDOW on every normalized pair."""
    assert MOSFET_CMG._NR_LIM_WINDOW == 2.5
    device = NMOS_CMG(
        "M1", ["d", "g", "s", "b"], str(OSDI_PATH),
        str(MODELCARDS_DIR / "ASAP7" / "7nm_TT_160803.pm"), "nmos_rvt",
        L=30e-9, NFIN=10.0,
    )
    device.reset_nr_limits()
    limited = device.nr_limit_voltages({"d": 10.0, "g": 10.0, "s": 0.0, "b": -10.0})
    assert limited == {"s": 0.0, "d": 2.5, "g": 2.5, "b": -2.5}
    assert device._nr_limited is True
    device.reset_nr_limits()
    inside = {"d": 0.7, "g": 0.7, "s": 0.0, "b": 0.0}
    assert device.nr_limit_voltages(inside) is inside
    assert device._nr_limited is False
