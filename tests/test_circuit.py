"""
Unit tests for the Circuit container class.

This module tests the Circuit class functionality:
- Creating empty circuits
- Adding components
- Auto-discovering unique nodes
- Creating node mappings (excluding ground)
- Counting voltage sources for MNA matrix sizing
"""
import pytest
from pycircuitsim.circuit import Circuit
from pycircuitsim.models.passive import Resistor, VoltageSource, CurrentSource


def test_circuit_creation():
    """Test creating an empty circuit."""
    circuit = Circuit()

    assert circuit is not None
    assert len(circuit.components) == 0
    assert len(circuit.nodes) == 0
    assert circuit.get_nodes() == []


def test_circuit_add_component():
    """Test adding a component to the circuit."""
    circuit = Circuit()
    resistor = Resistor("R1", ["n1", "n2"], 1000.0)

    circuit.add_component(resistor)

    assert len(circuit.components) == 1
    assert circuit.components[0] == resistor


def test_circuit_auto_discover_nodes():
    """Test that nodes are auto-discovered when adding components."""
    circuit = Circuit()

    # Add resistor with nodes n1, n2
    r1 = Resistor("R1", ["n1", "n2"], 1000.0)
    circuit.add_component(r1)

    # Add resistor with nodes n2, n3
    r2 = Resistor("R2", ["n2", "n3"], 2000.0)
    circuit.add_component(r2)

    # Add voltage source with nodes n1, 0
    v1 = VoltageSource("V1", ["n1", "0"], 5.0)
    circuit.add_component(v1)

    # Check that all unique nodes are discovered (including ground)
    assert "n1" in circuit.nodes
    assert "n2" in circuit.nodes
    assert "n3" in circuit.nodes
    assert "0" in circuit.nodes

    # Total unique nodes should be 4
    assert len(circuit.nodes) == 4


def test_circuit_node_mapping():
    """Test creating node map excluding ground nodes."""
    circuit = Circuit()

    # Add components with various nodes
    r1 = Resistor("R1", ["n1", "n2"], 1000.0)
    circuit.add_component(r1)

    r2 = Resistor("R2", ["n2", "n3"], 2000.0)
    circuit.add_component(r2)

    v1 = VoltageSource("V1", ["n1", "0"], 5.0)
    circuit.add_component(v1)

    # Add component with GND (alternative ground notation)
    v2 = VoltageSource("V2", ["n3", "GND"], 3.3)
    circuit.add_component(v2)

    # Get node map (should exclude ground nodes)
    node_map = circuit.get_node_map()

    # Ground nodes should not be in the map
    assert "0" not in node_map
    assert "GND" not in node_map

    # Other nodes should be in the map
    assert "n1" in node_map
    assert "n2" in node_map
    assert "n3" in node_map

    # Node mapping should assign sequential indices starting from 0
    assert node_map["n1"] == 0
    assert node_map["n2"] == 1
    assert node_map["n3"] == 2


def test_circuit_count_voltage_sources():
    """Test counting voltage sources for MNA matrix sizing."""
    circuit = Circuit()

    # Initially no voltage sources
    assert circuit.count_voltage_sources() == 0

    # Add a resistor (not a voltage source)
    r1 = Resistor("R1", ["n1", "n2"], 1000.0)
    circuit.add_component(r1)
    assert circuit.count_voltage_sources() == 0

    # Add a current source (not a voltage source)
    i1 = CurrentSource("I1", ["n2", "0"], 0.001)
    circuit.add_component(i1)
    assert circuit.count_voltage_sources() == 0

    # Add first voltage source
    v1 = VoltageSource("V1", ["n1", "0"], 5.0)
    circuit.add_component(v1)
    assert circuit.count_voltage_sources() == 1

    # Add second voltage source
    v2 = VoltageSource("V2", ["n2", "0"], 3.3)
    circuit.add_component(v2)
    assert circuit.count_voltage_sources() == 2


def test_circuit_repr():
    """Test string representation of circuit for debugging."""
    circuit = Circuit()

    r1 = Resistor("R1", ["n1", "n2"], 1000.0)
    circuit.add_component(r1)

    v1 = VoltageSource("V1", ["n1", "0"], 5.0)
    circuit.add_component(v1)

    # Check that repr contains useful information
    repr_str = repr(circuit)
    assert "Circuit" in repr_str
    assert "2" in repr_str  # Number of components
    assert "2" in repr_str  # Number of non-ground nodes (n1, n2)


def test_circuit_get_nodes_excludes_ground():
    """Test that get_nodes() excludes ground nodes."""
    circuit = Circuit()

    # Add components with ground nodes
    r1 = Resistor("R1", ["n1", "0"], 1000.0)
    circuit.add_component(r1)

    v1 = VoltageSource("V1", ["n2", "GND"], 5.0)
    circuit.add_component(v1)

    # get_nodes() should exclude both "0" and "GND"
    nodes = circuit.get_nodes()
    assert "0" not in nodes
    assert "GND" not in nodes
    assert "n1" in nodes
    assert "n2" in nodes
    assert len(nodes) == 2


def test_circuit_multiple_ground_notations():
    """Test that circuit handles multiple ground notations correctly."""
    circuit = Circuit()

    # Add components using different ground notations
    v1 = VoltageSource("V1", ["n1", "0"], 5.0)
    circuit.add_component(v1)

    v2 = VoltageSource("V2", ["n2", "GND"], 3.3)
    circuit.add_component(v2)

    r1 = Resistor("R1", ["0", "GND"], 1000.0)
    circuit.add_component(r1)

    # Both ground notations should be in circuit.nodes
    assert "0" in circuit.nodes
    assert "GND" in circuit.nodes

    # But both should be excluded from get_nodes() and get_node_map()
    nodes = circuit.get_nodes()
    assert "0" not in nodes
    assert "GND" not in nodes

    node_map = circuit.get_node_map()
    assert "0" not in node_map
    assert "GND" not in node_map
