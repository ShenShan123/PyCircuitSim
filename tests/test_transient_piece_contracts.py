"""Integration order and endpoint contracts follow accepted internal pieces."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pycircuitsim.circuit import Circuit
from pycircuitsim.models.passive import (
    Capacitor, PulseVoltageSource, Resistor, VoltageSource,
)
from pycircuitsim.solver import TransientSolver


@pytest.mark.parametrize("method", ("auto", "trap", "gear2"))
@pytest.mark.parametrize("split", ("refine", "retry"))
def test_integration_promotes_after_first_accepted_piece(
    monkeypatch: pytest.MonkeyPatch, method: str, split: str,
) -> None:
    """Splitting the first output interval must not extend the BE startup."""
    resistance, capacitance, step = 1e3, 1e-12, 1e-10
    circuit = Circuit()
    circuit.add_component(VoltageSource("V1", ["in", "0"], 1.0))
    circuit.add_component(Resistor("R1", ["in", "out"], resistance))
    circuit.add_component(Capacitor("C1", ["out", "0"], capacitance))
    solver = TransientSolver(
        circuit, t_stop=step, dt=step, integration_method=method,
        initial_guess={"in": 1.0, "out": 0.0},
        refine_output=split == "refine", refine_max_dt=step / 2,
    )
    # Exercise the real Newton retry path on a network with a closed-form
    # response; inject one rejected solve after its numerical work succeeds.
    monkeypatch.setattr(solver, "_has_non_linear_components", lambda: True)
    solve_piece = solver._solve_timestep_newton
    attempts: list[str] = []
    accepted: list[str] = []

    def record_piece(**kwargs: Any) -> dict[str, float]:
        attempts.append(solver._integration_method)
        result = solve_piece(**kwargs)
        if split == "retry" and len(attempts) == 1:
            raise RuntimeError("controlled first-piece rejection")
        accepted.append(solver._integration_method)
        return result

    monkeypatch.setattr(solver, "_solve_timestep_newton", record_piece)
    result = solver.solve()

    next_method = "bdf2" if method == "gear2" else "trap"
    assert accepted == ["be", next_method]
    if split == "retry":
        assert attempts[:2] == ["be", "be"]
    a = step / (2 * resistance * capacitance)
    first = a / (1 + a)
    expected = ((4 * first + 2 * a) / (3 + 2 * a) if method == "gear2"
                else ((1 - a / 2) * first + a) / (1 + a / 2))
    assert result["out"][-1] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("method", ("auto", "trap", "gear2"))
def test_breakpoint_restart_still_uses_one_backward_euler_piece(
    monkeypatch: pytest.MonkeyPatch, method: str,
) -> None:
    """Order promotion inside an output interval must preserve PULSE restarts."""
    step = 1e-10
    circuit = Circuit()
    circuit.add_component(PulseVoltageSource(
        "V1", ["in", "0"], 0.0, 1.0, step / 4,
        step / 4, step / 4, 2 * step, 4 * step,
    ))
    circuit.add_component(Resistor("R1", ["in", "0"], 1e3))
    solver = TransientSolver(
        circuit, t_stop=0.9 * step, dt=step, integration_method=method,
        initial_guess={"in": 0.0}, refine_output=True,
    )
    monkeypatch.setattr(solver, "_has_non_linear_components", lambda: True)
    solve_piece = solver._solve_timestep_newton
    attempted_methods: dict[float, str] = {}

    def record_piece(**kwargs: Any) -> dict[str, float]:
        result = solve_piece(**kwargs)
        attempted_methods[kwargs["time"]] = solver._integration_method
        return result

    monkeypatch.setattr(solver, "_solve_timestep_newton", record_piece)
    result = solver.solve()
    # Successful Newton attempts can still be rejected by LTE. The dense
    # output identifies the pieces that actually committed at each corner.
    accepted = [(time, attempted_methods[time]) for time in result["time"][1:]]

    next_method = "bdf2" if method == "gear2" else "trap"
    for breakpoint in (step / 4, step / 2):
        index = next(i for i, (time, _) in enumerate(accepted)
                     if np.isclose(time, breakpoint, rtol=1e-12, atol=0))
        assert accepted[index + 1][1] == "be"
        assert accepted[index + 2][1] == next_method


@pytest.mark.parametrize("refine", (False, True))
def test_positive_span_smaller_than_grid_rounding_tolerance_is_solved(
    refine: bool,
) -> None:
    """A valid positive duration cannot collapse to an initial-condition row."""
    circuit = Circuit()
    circuit.add_component(PulseVoltageSource(
        "V1", ["in", "0"], 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 5.0,
    ))
    circuit.add_component(Resistor("R1", ["in", "0"], 1e3))
    solver = TransientSolver(
        circuit, t_stop=1e-10, dt=1.0,
        initial_guess={"in": 0.0}, refine_output=refine,
    )
    result = solver.solve()
    assert result["time"][-1] == solver.t_stop
    assert result["in"][-1] == pytest.approx(solver.t_stop, rel=1e-12, abs=0)


@pytest.mark.parametrize("refine", (False, True))
@pytest.mark.parametrize("stop", (2.0 - 5e-10, 2.0 + 5e-10))
def test_near_integer_span_still_reaches_the_exact_requested_endpoint(
    refine: bool, stop: float,
) -> None:
    """Rounding the interval count cannot round away the requested stop time."""
    circuit = Circuit()
    circuit.add_component(PulseVoltageSource(
        "V1", ["in", "0"], 0.0, 1.0, 0.0, 10.0, 1.0, 10.0, 30.0,
    ))
    circuit.add_component(Resistor("R1", ["in", "0"], 1e3))
    solver = TransientSolver(
        circuit, t_stop=stop, dt=1.0,
        initial_guess={"in": 0.0}, refine_output=refine,
    )
    result = solver.solve()
    assert result["time"][-1] == stop
    assert np.all(np.diff(result["time"]) > 0)
    assert np.all(result["time"] <= stop)
    if not refine:
        np.testing.assert_array_equal(result["time"], [0.0, 1.0, stop])
    np.testing.assert_allclose(result["in"], result["time"] / 10.0,
                               rtol=1e-12, atol=0)
