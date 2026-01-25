"""
Unit tests for the Transient Solver using Backward Euler integration.

This module tests the TransientSolver class functionality:
- DC solution for initial conditions
- Capacitor companion model updates
- Time-stepping with Backward Euler
- RC circuit charging behavior
"""
import pytest
import numpy as np
from pycircuitsim.circuit import Circuit
from pycircuitsim.solver import TransientSolver
from pycircuitsim.models.passive import Resistor, VoltageSource, Capacitor


def test_rc_charging():
    """
    Test RC charging circuit: Vdd-R-C-GND, verify V_c at t=tau.

    Circuit:
        Vdd = 5V
        R = 1kΩ
        C = 1μF
        τ = R*C = 1ms

    Expected behavior:
        V_c(t) = Vdd * (1 - exp(-t/τ))
        At t = τ: V_c = 5 * (1 - exp(-1)) ≈ 3.16V

    Test verifies:
    - DC solution at t=0 (V_c = 0V)
    - Transient analysis from t=0 to t=τ
    - Capacitor voltage at t=τ is approximately 3.16V (within 1%)
    """
    # Create circuit
    circuit = Circuit()

    # Vdd = 5V connected to node 'n1'
    vdd = VoltageSource("Vdd", ["n1", "0"], 5.0)
    circuit.add_component(vdd)

    # R = 1k between n1 and n2
    r1 = Resistor("R1", ["n1", "n2"], 1000.0)
    circuit.add_component(r1)

    # C = 1uF between n2 and ground
    c1 = Capacitor("C1", ["n2", "0"], 1e-6)
    circuit.add_component(c1)

    # Calculate time constant
    tau = r1.resistance * c1.capacitance  # τ = RC = 1ms

    # Create transient solver
    # Simulate from t=0 to t=tau with timestep dt = tau/100
    dt = tau / 100.0
    t_stop = tau

    # Set initial capacitor voltage to 0V (discharged)
    # This is the initial condition for transient analysis
    c1.v_prev = 0.0

    solver = TransientSolver(circuit, t_stop=t_stop, dt=dt)

    # Solve transient analysis
    results = solver.solve()

    # Verify structure of results
    assert "time" in results, "Results should contain 'time' array"
    assert "n2" in results, "Results should contain node 'n2' voltages"

    # Get time array and capacitor voltage
    time = results["time"]
    v_c = results["n2"]

    # Check time array
    assert len(time) > 0, "Time array should not be empty"
    assert time[0] == 0.0, "Simulation should start at t=0"
    assert abs(time[-1] - t_stop) < dt, "Simulation should end at t_stop"

    # Check initial condition (t=0)
    # At t=0, capacitor is uncharged: V_c(0) = 0V
    assert abs(v_c[0]) < 0.01, f"Initial V_c should be ~0V, got {v_c[0]}V"

    # Check final condition (t=τ)
    # At t=τ, V_c = Vdd * (1 - exp(-1)) ≈ 3.16V
    expected_final = 5.0 * (1.0 - np.exp(-1.0))
    actual_final = v_c[-1]

    # Allow 1% tolerance (Backward Euler has some numerical error)
    rel_error = abs(actual_final - expected_final) / expected_final
    assert rel_error < 0.01, (
        f"At t=τ, expected V_c ≈ {expected_final:.3f}V, "
        f"got {actual_final:.3f}V (rel_error={rel_error:.3%})"
    )

    # Check monotonic charging (voltage should always increase)
    for i in range(1, len(v_c)):
        assert v_c[i] >= v_c[i-1], (
            f"Capacitor voltage should monotonically increase: "
            f"V_c[{i-1}]={v_c[i-1]:.3f}V, V_c[{i}]={v_c[i]:.3f}V"
        )

    # Check voltage never exceeds source voltage
    assert max(v_c) <= 5.0 + 0.01, "Capacitor voltage should not exceed source voltage"


def test_rc_multiple_time_constants():
    """
    Test RC circuit over multiple time constants.

    Circuit:
        Vdd = 10V
        R = 2kΩ
        C = 0.5μF
        τ = R*C = 1ms

    Expected behavior:
        After 5τ, capacitor should be charged to >99% of Vdd
    """
    # Create circuit
    circuit = Circuit()

    # Vdd = 10V
    vdd = VoltageSource("Vdd", ["n1", "0"], 10.0)
    circuit.add_component(vdd)

    # R = 2k
    r1 = Resistor("R1", ["n1", "n2"], 2000.0)
    circuit.add_component(r1)

    # C = 0.5uF
    c1 = Capacitor("C1", ["n2", "0"], 0.5e-6)
    circuit.add_component(c1)

    # Calculate time constant
    tau = r1.resistance * c1.capacitance  # τ = 1ms

    # Simulate for 5 time constants
    dt = tau / 100.0
    t_stop = 5.0 * tau

    solver = TransientSolver(circuit, t_stop=t_stop, dt=dt)
    results = solver.solve()

    # Get final capacitor voltage
    v_c = results["n2"]
    final_voltage = v_c[-1]

    # After 5τ, V_c should be > 99% of Vdd
    expected_final = 10.0 * (1.0 - np.exp(-5.0))  # ≈ 9.93V
    rel_error = abs(final_voltage - expected_final) / expected_final

    assert rel_error < 0.02, (
        f"After 5τ, expected V_c ≈ {expected_final:.3f}V, "
        f"got {final_voltage:.3f}V"
    )


def test_rc_discharging():
    """
    Test RC discharging circuit with initial voltage.

    Circuit:
        Vdd = 0V (short to ground for discharge)
        R = 1kΩ
        C = 1μF
        τ = R*C = 1ms

    Expected behavior:
        V_c(t) = V_initial * exp(-t/τ)
        At t = τ: V_c = 5 * exp(-1) ≈ 1.84V
    """
    # Create circuit
    circuit = Circuit()

    # Vdd = 0V (short to ground to discharge capacitor)
    vdd = VoltageSource("Vdd", ["n1", "0"], 0.0)
    circuit.add_component(vdd)

    # R = 1k between n1 and n2
    r1 = Resistor("R1", ["n1", "n2"], 1000.0)
    circuit.add_component(r1)

    # C = 1uF between n2 and ground
    c1 = Capacitor("C1", ["n2", "0"], 1e-6)
    circuit.add_component(c1)

    # Set initial capacitor voltage to 5V
    c1.v_prev = 5.0

    # Calculate time constant
    tau = r1.resistance * c1.capacitance  # τ = 1ms

    # Simulate for 1 time constant
    dt = tau / 100.0
    t_stop = tau

    solver = TransientSolver(circuit, t_stop=t_stop, dt=dt)
    results = solver.solve()

    # Get final capacitor voltage
    v_c = results["n2"]
    final_voltage = v_c[-1]

    # At t=τ, V_c should be ≈ 1.84V
    expected_final = 5.0 * np.exp(-1.0)
    rel_error = abs(final_voltage - expected_final) / expected_final

    assert rel_error < 0.02, (
        f"At t=τ, expected V_c ≈ {expected_final:.3f}V, "
        f"got {final_voltage:.3f}V"
    )

    # Check monotonic discharging (voltage should always decrease)
    for i in range(1, len(v_c)):
        assert v_c[i] <= v_c[i-1], (
            f"Capacitor voltage should monotonically decrease: "
            f"V_c[{i-1}]={v_c[i-1]:.3f}V, V_c[{i}]={v_c[i]:.3f}V"
        )
