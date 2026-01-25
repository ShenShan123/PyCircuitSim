"""
Passive component models.

This module implements linear passive components like resistors, capacitors,
and inductors. These devices follow well-defined linear relationships between
voltage and current.
"""
from typing import List, Dict, Any
import numpy as np

from pycircuitsim.models.base import Component


class Resistor(Component):
    """
    Linear resistor following Ohm's Law: V = I * R.

    The resistor stamps conductance (G = 1/R) to the MNA matrix.
    For a resistor between nodes i and j:
        G[i,i] += g
        G[j,j] += g
        G[i,j] -= g
        G[j,i] -= g

    Ground nodes (node "0") are not stamped in the MNA matrix.

    Attributes:
        name: Component identifier (e.g., 'R1')
        nodes: List of two node names
        resistance: Resistance value in ohms
    """

    def __init__(self, name: str, nodes: List[str], value: float):
        """
        Initialize a resistor.

        Args:
            name: Component identifier (e.g., 'R1', 'R_load')
            nodes: List of exactly two node names (e.g., ['n1', 'n2'])
            value: Resistance value in ohms (must be positive)

        Raises:
            ValueError: If resistance is not positive or nodes count is not 2
        """
        super().__init__(name, nodes, value)

        # Validate number of nodes
        if len(nodes) != 2:
            raise ValueError(f"Resistor must have exactly 2 nodes, got {len(nodes)}")

        # Validate resistance value
        if value is None or value <= 0:
            raise ValueError(f"Resistance must be positive, got {value}")

        self.resistance = float(value)

    @property
    def conductance(self) -> float:
        """
        Get the conductance of the resistor.

        Returns:
            Conductance in siemens (G = 1/R)
        """
        return 1.0 / self.resistance

    def get_nodes(self) -> List[str]:
        """
        Return list of node names this resistor connects to.

        Returns:
            List of two node names
        """
        return self.nodes

    def stamp_conductance(self, matrix: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Add conductance terms to the MNA matrix.

        For a resistor between nodes i and j with conductance g:
        - Add g to diagonal entries G[i,i] and G[j,j]
        - Subtract g from off-diagonal entries G[i,j] and G[j,i]

        Ground node (node "0") is not in the node_map and is skipped.

        Args:
            matrix: The MNA matrix to modify (in-place)
            node_map: Mapping from node names to matrix indices
        """
        node_i, node_j = self.nodes[0], self.nodes[1]
        g = self.conductance

        # Stamp node i (skip if ground)
        if node_i != "0" and node_i in node_map:
            idx_i = node_map[node_i]
            matrix[idx_i, idx_i] += g

            # Stamp connection to node j
            if node_j != "0" and node_j in node_map:
                idx_j = node_map[node_j]
                matrix[idx_i, idx_j] -= g
                matrix[idx_j, idx_i] -= g

        # Stamp node j (skip if ground or already handled)
        if node_j != "0" and node_j in node_map:
            idx_j = node_map[node_j]
            matrix[idx_j, idx_j] += g

    def stamp_rhs(self, rhs: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Add current/source terms to the RHS vector.

        Resistors do not contribute to the RHS vector in MNA formulation.
        They only affect the conductance matrix.

        Args:
            rhs: The RHS vector to modify (in-place)
            node_map: Mapping from node names to matrix indices
        """
        # Resistors don't contribute to RHS
        pass

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """
        Calculate current flowing through the resistor.

        Uses Ohm's Law: I = (V_i - V_j) / R
        Current direction is from node_i to node_j (conventional current).

        Args:
            voltages: Dictionary mapping node names to voltage values

        Returns:
            Current flowing from first node to second node (in amperes)
        """
        node_i, node_j = self.nodes[0], self.nodes[1]

        # Get voltages (default to 0 if node not found)
        v_i = voltages.get(node_i, 0.0)
        v_j = voltages.get(node_j, 0.0)

        # Calculate current: I = (V_i - V_j) / R
        current = (v_i - v_j) / self.resistance

        return current

    def __repr__(self) -> str:
        """String representation of the resistor."""
        return f"Resistor({self.name}, nodes={self.nodes}, R={self.resistance}Ω)"


class VoltageSource(Component):
    """
    Ideal DC voltage source.

    A voltage source maintains a fixed voltage difference between its terminals.
    In MNA formulation, voltage sources require special handling:
    - They add a row and column to the MNA matrix (the current through the source)
    - The voltage constraint is added to the RHS vector

    For a voltage source between nodes i and j:
    - Adds equation: V_i - V_j = V_source
    - Adds unknown: I_source (current flowing from positive to negative terminal)

    The actual matrix stamping is handled by the solver, which builds the
    augmented MNA matrix with B and C blocks for voltage sources.

    Attributes:
        name: Component identifier (e.g., 'V1', 'V_dd')
        nodes: List of two node names [positive, negative]
        voltage: Voltage value in volts
    """

    def __init__(self, name: str, nodes: List[str], value: float):
        """
        Initialize a voltage source.

        Args:
            name: Component identifier (e.g., 'V1', 'V_dd')
            nodes: List of exactly two node names [positive, negative]
            value: Voltage value in volts

        Raises:
            ValueError: If nodes count is not 2
        """
        super().__init__(name, nodes, value)

        # Validate number of nodes
        if len(nodes) != 2:
            raise ValueError(f"VoltageSource must have exactly 2 nodes, got {len(nodes)}")

        # Store voltage value
        self.voltage = float(value)

    def get_nodes(self) -> List[str]:
        """
        Return list of node names this voltage source connects to.

        Returns:
            List of two node names [positive, negative]
        """
        return self.nodes

    def stamp_conductance(self, matrix: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Interface for conductance stamping (pass-through for voltage sources).

        Voltage sources require special MNA handling with augmented matrix.
        The actual stamping of B and C matrix blocks is handled by the solver.

        This method exists to satisfy the Component interface but does nothing,
        as the solver will handle the matrix augmentation when it detects
        voltage sources in the circuit.

        Args:
            matrix: The MNA matrix (not modified by voltage sources directly)
            node_map: Mapping from node names to matrix indices
        """
        # Voltage sources don't stamp to conductance matrix directly
        # The solver will handle B/C matrix augmentation
        pass

    def stamp_rhs(self, rhs: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Interface for RHS stamping (pass-through for voltage sources).

        The voltage constraint equation (V_pos - V_neg = V_source)
        is added by the solver when building the augmented MNA system.

        This method exists to satisfy the Component interface but does nothing,
        as the solver will handle the RHS modification for voltage constraints.

        Args:
            rhs: The RHS vector (not modified by voltage sources directly)
            node_map: Mapping from node names to matrix indices
        """
        # Voltage sources don't stamp to RHS directly
        # The solver will handle this when building augmented system
        pass

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """
        Interface for current calculation (placeholder for voltage sources).

        The current through a voltage source is determined by the circuit
        topology and is calculated by the solver during MNA solving.

        This method exists to satisfy the Component interface but returns 0.0,
        as the actual current will be extracted from the solution vector.

        Args:
            voltages: Dictionary mapping node names to voltage values

        Returns:
            Current placeholder (0.0, actual current calculated by solver)
        """
        # Current through voltage source is calculated by solver
        # This is a placeholder to satisfy the interface
        return 0.0

    def __repr__(self) -> str:
        """String representation of the voltage source."""
        return f"VoltageSource({self.name}, nodes={self.nodes}, V={self.voltage}V)"
