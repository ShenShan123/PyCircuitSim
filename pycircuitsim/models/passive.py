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
