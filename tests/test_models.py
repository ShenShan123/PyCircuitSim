"""
Tests for concrete device models.

This module tests the implementation of specific circuit components
starting with the Resistor model.
"""
import pytest
import numpy as np
from pycircuitsim.models.passive import Resistor, VoltageSource, CurrentSource


def test_resistor_creation():
    """Test that a Resistor can be created with proper parameters."""
    # Valid resistor
    r1 = Resistor("R1", ["n1", "n2"], 1000.0)
    assert r1.name == "R1"
    assert r1.get_nodes() == ["n1", "n2"]
    assert r1.resistance == 1000.0
    assert r1.conductance == 1.0 / 1000.0

    # Resistor connected to ground
    r2 = Resistor("R2", ["n1", "0"], 100.0)
    assert r2.get_nodes() == ["n1", "0"]
    assert r2.resistance == 100.0


def test_resistor_creation_invalid():
    """Test that invalid resistor parameters raise errors."""
    # Negative resistance
    with pytest.raises(ValueError, match="Resistance must be positive"):
        Resistor("R1", ["n1", "n2"], -100.0)

    # Zero resistance
    with pytest.raises(ValueError, match="Resistance must be positive"):
        Resistor("R1", ["n1", "n2"], 0.0)

    # Wrong number of nodes
    with pytest.raises(ValueError, match="Resistor must have exactly 2 nodes"):
        Resistor("R1", ["n1"], 100.0)

    with pytest.raises(ValueError, match="Resistor must have exactly 2 nodes"):
        Resistor("R1", ["n1", "n2", "n3"], 100.0)


def test_resistor_stamp_conductance():
    """Test that resistor stamps conductance correctly to MNA matrix."""
    # Create a 3x3 matrix (2 non-ground nodes + 1 extra)
    matrix = np.zeros((3, 3))
    node_map = {"n1": 0, "n2": 1, "n3": 2}

    # Add resistor between n1 and n2
    r1 = Resistor("R1", ["n1", "n2"], 1000.0)  # G = 0.001 S
    r1.stamp_conductance(matrix, node_map)

    # Check diagonal terms (conductance added)
    assert np.isclose(matrix[0, 0], 0.001)  # G[n1, n1]
    assert np.isclose(matrix[1, 1], 0.001)  # G[n2, n2]

    # Check off-diagonal terms (conductance subtracted)
    assert np.isclose(matrix[0, 1], -0.001)  # G[n1, n2]
    assert np.isclose(matrix[1, 0], -0.001)  # G[n2, n1]

    # Other terms should remain zero
    assert matrix[2, 2] == 0.0
    assert matrix[0, 2] == 0.0


def test_resistor_stamp_conductance_with_ground():
    """Test that resistor stamps correctly when connected to ground."""
    # Create a 2x2 matrix (1 non-ground node + 1 extra)
    matrix = np.zeros((2, 2))
    node_map = {"n1": 0, "n2": 1}

    # Add resistor between n1 and ground (0)
    r1 = Resistor("R1", ["n1", "0"], 100.0)  # G = 0.01 S
    r1.stamp_conductance(matrix, node_map)

    # Only diagonal term should be stamped (ground is not in matrix)
    assert np.isclose(matrix[0, 0], 0.01)  # G[n1, n1]

    # All other terms should be zero
    assert matrix[1, 1] == 0.0
    assert matrix[0, 1] == 0.0
    assert matrix[1, 0] == 0.0


def test_resistor_stamp_rhs():
    """Test that resistor doesn't contribute to RHS vector."""
    rhs = np.zeros(3)
    node_map = {"n1": 0, "n2": 1, "n3": 2}

    r1 = Resistor("R1", ["n1", "n2"], 1000.0)
    r1.stamp_rhs(rhs, node_map)

    # RHS should remain all zeros (resistors don't contribute)
    assert np.allclose(rhs, 0.0)


def test_resistor_current():
    """Test that resistor current is calculated correctly using Ohm's law."""
    r1 = Resistor("R1", ["n1", "n2"], 1000.0)

    # Test with voltage difference: V_n1 = 5V, V_n2 = 2V
    # Current should flow from n1 to n2: I = (5 - 2) / 1000 = 0.003 A
    voltages = {"n1": 5.0, "n2": 2.0}
    current = r1.calculate_current(voltages)
    assert np.isclose(current, 0.003)

    # Test with zero voltage difference
    voltages = {"n1": 3.0, "n2": 3.0}
    current = r1.calculate_current(voltages)
    assert np.isclose(current, 0.0)

    # Test with reversed voltage: V_n1 = 1V, V_n2 = 4V
    # Current should flow from n2 to n1 (negative from n1 to n2)
    voltages = {"n1": 1.0, "n2": 4.0}
    current = r1.calculate_current(voltages)
    assert np.isclose(current, -0.003)

    # Test with ground node: V_n1 = 10V, V_0 = 0V
    r2 = Resistor("R2", ["n1", "0"], 100.0)
    voltages = {"n1": 10.0, "0": 0.0}
    current = r2.calculate_current(voltages)
    assert np.isclose(current, 0.1)  # (10 - 0) / 100 = 0.1 A


def test_voltage_source_creation():
    """Test that a VoltageSource can be created with proper parameters."""
    # Valid voltage source
    v1 = VoltageSource("V1", ["n1", "n2"], 5.0)
    assert v1.name == "V1"
    assert v1.get_nodes() == ["n1", "n2"]
    assert v1.voltage == 5.0

    # Voltage source connected to ground
    v2 = VoltageSource("V2", ["n1", "0"], 3.3)
    assert v2.get_nodes() == ["n1", "0"]
    assert v2.voltage == 3.3


def test_voltage_source_creation_invalid():
    """Test that invalid voltage source parameters raise errors."""
    # Wrong number of nodes
    with pytest.raises(ValueError, match="VoltageSource must have exactly 2 nodes"):
        VoltageSource("V1", ["n1"], 5.0)

    with pytest.raises(ValueError, match="VoltageSource must have exactly 2 nodes"):
        VoltageSource("V1", ["n1", "n2", "n3"], 5.0)


def test_voltage_source_get_nodes():
    """Test that VoltageSource returns correct nodes."""
    v1 = VoltageSource("V1", ["n1", "n2"], 5.0)
    assert v1.get_nodes() == ["n1", "n2"]

    # Test with ground
    v2 = VoltageSource("V2", ["n1", "0"], 3.3)
    assert v2.get_nodes() == ["n1", "0"]


def test_voltage_source_stamp_rhs():
    """Test that voltage source interface is available (solver handles MNA)."""
    # Voltage sources don't stamp to RHS in the traditional sense
    # The solver will handle the B/C matrix augmentation
    rhs = np.zeros(3)
    node_map = {"n1": 0, "n2": 1, "n3": 2}

    v1 = VoltageSource("V1", ["n1", "n2"], 5.0)
    # This should not raise an error, but also not modify RHS
    # (the actual voltage value will be used by the solver)
    v1.stamp_rhs(rhs, node_map)

    # For now, voltage sources don't directly stamp to RHS
    # The solver will handle this when it builds the augmented MNA matrix
    assert np.allclose(rhs, 0.0)


def test_voltage_source_stamp_conductance():
    """Test that voltage source interface is available for conductance stamping."""
    # Voltage sources require special MNA handling (adds rows/cols)
    # The solver will handle the B/C matrix augmentation
    matrix = np.zeros((3, 3))
    node_map = {"n1": 0, "n2": 1, "n3": 2}

    v1 = VoltageSource("V1", ["n1", "n2"], 5.0)
    # This should not raise an error
    # (the actual stamping will be done by the solver)
    v1.stamp_conductance(matrix, node_map)

    # For now, voltage sources don't directly stamp to conductance matrix
    # The solver will handle this when it builds the augmented MNA matrix
    assert np.allclose(matrix, 0.0)


def test_voltage_source_current():
    """Test that voltage source current calculation interface exists."""
    v1 = VoltageSource("V1", ["n1", "n2"], 5.0)

    # Current through a voltage source is determined by the circuit
    # For now, we just test that the interface exists
    # (actual current will be calculated by the solver)
    voltages = {"n1": 5.0, "n2": 0.0}
    current = v1.calculate_current(voltages)

    # Current calculation will be handled by the solver
    # For now, just verify the interface works
    assert isinstance(current, float)


def test_current_source_creation():
    """Test that a CurrentSource can be created with proper parameters."""
    # Valid current source
    i1 = CurrentSource("I1", ["n1", "n2"], 0.005)
    assert i1.name == "I1"
    assert i1.get_nodes() == ["n1", "n2"]
    assert i1.current == 0.005

    # Current source connected to ground
    i2 = CurrentSource("I2", ["n1", "0"], 0.001)
    assert i2.get_nodes() == ["n1", "0"]
    assert i2.current == 0.001


def test_current_source_creation_invalid():
    """Test that invalid current source parameters raise errors."""
    # Wrong number of nodes
    with pytest.raises(ValueError, match="CurrentSource must have exactly 2 nodes"):
        CurrentSource("I1", ["n1"], 0.005)

    with pytest.raises(ValueError, match="CurrentSource must have exactly 2 nodes"):
        CurrentSource("I1", ["n1", "n2", "n3"], 0.005)


def test_current_source_stamp_rhs():
    """Test that current source stamps RHS correctly (+I to source, -I to sink)."""
    rhs = np.zeros(3)
    node_map = {"n1": 0, "n2": 1, "n3": 2}

    # Current source from n1 to n2 with 0.01 A
    i1 = CurrentSource("I1", ["n1", "n2"], 0.01)
    i1.stamp_rhs(rhs, node_map)

    # Should add +I to source node (n1) and -I to sink node (n2)
    assert np.isclose(rhs[0], 0.01)   # +I at n1
    assert np.isclose(rhs[1], -0.01)  # -I at n2
    assert np.isclose(rhs[2], 0.0)    # No change at n3


def test_current_source_stamp_rhs_with_ground():
    """Test that current source stamps correctly when connected to ground."""
    rhs = np.zeros(2)
    node_map = {"n1": 0, "n2": 1}

    # Current source from n1 to ground with 0.005 A
    i1 = CurrentSource("I1", ["n1", "0"], 0.005)
    i1.stamp_rhs(rhs, node_map)

    # Should add +I to source node (n1), ground is not in matrix
    assert np.isclose(rhs[0], 0.005)   # +I at n1
    assert np.isclose(rhs[1], 0.0)     # No change at n2


def test_current_source_current():
    """Test that current source returns its current value."""
    i1 = CurrentSource("I1", ["n1", "n2"], 0.005)

    # Current source should return its current value regardless of voltages
    voltages = {"n1": 5.0, "n2": 0.0}
    current = i1.calculate_current(voltages)
    assert np.isclose(current, 0.005)

    # Test with different voltage (should still return same current)
    voltages = {"n1": 10.0, "n2": 2.0}
    current = i1.calculate_current(voltages)
    assert np.isclose(current, 0.005)


def test_current_source_stamp_conductance():
    """Test that current source doesn't stamp to conductance matrix."""
    matrix = np.zeros((3, 3))
    node_map = {"n1": 0, "n2": 1, "n3": 2}

    i1 = CurrentSource("I1", ["n1", "n2"], 0.01)
    i1.stamp_conductance(matrix, node_map)

    # Current sources don't contribute to conductance matrix
    assert np.allclose(matrix, 0.0)
