"""
DC Solver for linear circuits using Modified Nodal Analysis (MNA).

This module implements the DCSolver class, which solves for the DC operating
point of linear circuits. The solver uses MNA formulation to construct and
solve the circuit equations:

    [G  B] [v]     [i]
    [    ] [ ] =   [ ]
    [C  D] [j]     [e]

Where:
    - G: Conductance matrix from passive components
    - B/C: Voltage source connection matrices
    - v: Node voltages (unknown)
    - j: Voltage source currents (unknown)
    - i: Current source vector (known)
    - e: Voltage source values (known)

The solver handles:
- Linear resistors (conductance stamping)
- Voltage sources (augmented matrix with B and C blocks)
- Current sources (RHS vector stamping)
"""
from typing import Dict, List
import numpy as np
from pycircuitsim.circuit import Circuit
from pycircuitsim.models.passive import VoltageSource


class DCSolver:
    """
    DC Solver for linear circuits using Modified Nodal Analysis.

    The DCSolver constructs the MNA matrix and solves for the DC operating
    point of a circuit. It handles linear components (resistors, voltage
    sources, current sources) and returns the node voltages.

    Attributes:
        circuit: Circuit object containing components and topology
        tolerance: Convergence tolerance (not used for linear circuits)
        max_iterations: Maximum iterations (not used for linear circuits)
    """

    def __init__(self, circuit: Circuit, tolerance: float = 1e-9, max_iterations: int = 50):
        """
        Initialize the DC Solver.

        Args:
            circuit: Circuit object to solve
            tolerance: Convergence tolerance (for consistency with non-linear)
            max_iterations: Maximum iterations (for consistency with non-linear)
        """
        self.circuit = circuit
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def solve(self) -> Dict[str, float]:
        """
        Solve the circuit for DC operating point.

        This method constructs the MNA matrix and RHS vector, then solves
        the linear system to find node voltages and voltage source currents.

        The MNA matrix has size (num_nodes + num_voltage_sources) x (num_nodes + num_voltage_sources).

        Returns:
            Dictionary mapping node names to voltage values (including ground at 0V)

        Raises:
            np.linalg.LinAlgError: If the circuit matrix is singular (unsolvable)
        """
        # Get circuit topology
        nodes = self.circuit.get_nodes()
        node_map = self.circuit.get_node_map()
        num_nodes = len(nodes)
        num_voltage_sources = self.circuit.count_voltage_sources()

        # Matrix size: num_nodes + num_voltage_sources
        matrix_size = num_nodes + num_voltage_sources

        # Initialize MNA matrix and RHS vector
        mna_matrix = np.zeros((matrix_size, matrix_size))
        rhs = np.zeros(matrix_size)

        # Stamp conductances (G matrix) and current sources (RHS)
        for component in self.circuit.components:
            component.stamp_conductance(mna_matrix, node_map)
            component.stamp_rhs(rhs, node_map)

        # Handle voltage sources (B and C matrices)
        self._stamp_voltage_sources(mna_matrix, rhs, node_map, num_nodes)

        # Solve the linear system
        try:
            solution = np.linalg.solve(mna_matrix, rhs)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"Circuit is singular or unsolvable. Check for floating nodes or short circuits."
            ) from e

        # Extract node voltages from solution
        voltages = self._extract_voltages(solution, nodes)

        return voltages

    def _stamp_voltage_sources(
        self,
        mna_matrix: np.ndarray,
        rhs: np.ndarray,
        node_map: Dict[str, int],
        num_nodes: int,
    ) -> None:
        """
        Stamp voltage source equations to MNA matrix.

        For each voltage source, we add:
        - B matrix column: connection to node voltages
        - C matrix row: voltage constraint equation
        - RHS entry: voltage source value

        The voltage source equation is: V_pos - V_neg = V_source

        Args:
            mna_matrix: MNA matrix to modify (in-place)
            rhs: RHS vector to modify (in-place)
            node_map: Mapping from node names to matrix indices
            num_nodes: Number of non-ground nodes
        """
        voltage_source_index = 0

        for component in self.circuit.components:
            if isinstance(component, VoltageSource):
                # Get voltage source nodes
                pos_node = component.nodes[0]  # Positive terminal
                neg_node = component.nodes[1]  # Negative terminal
                voltage = component.voltage

                # The row index for this voltage source's equation
                vs_row = num_nodes + voltage_source_index

                # Stamp B matrix (voltage source current flows into nodes)
                if pos_node != "0" and pos_node in node_map:
                    pos_idx = node_map[pos_node]
                    mna_matrix[vs_row, pos_idx] += 1.0
                    mna_matrix[pos_idx, vs_row] += 1.0

                if neg_node != "0" and neg_node in node_map:
                    neg_idx = node_map[neg_node]
                    mna_matrix[vs_row, neg_idx] -= 1.0
                    mna_matrix[neg_idx, vs_row] -= 1.0

                # Stamp voltage source value to RHS
                rhs[vs_row] = voltage

                # Move to next voltage source
                voltage_source_index += 1

    def _extract_voltages(self, solution: np.ndarray, nodes: List[str]) -> Dict[str, float]:
        """
        Extract node voltages from solution vector.

        The solution vector contains:
        - First num_nodes entries: node voltages
        - Remaining entries: voltage source currents

        Args:
            solution: Solution vector from np.linalg.solve
            nodes: List of non-ground node names

        Returns:
            Dictionary mapping node names to voltages (including ground)
        """
        voltages = {}

        # Extract node voltages (first num_nodes entries)
        for idx, node in enumerate(nodes):
            voltages[node] = float(solution[idx])

        # Add ground node (reference voltage)
        voltages["0"] = 0.0
        voltages["GND"] = 0.0

        return voltages

    def __repr__(self) -> str:
        """String representation of the solver."""
        return (
            f"DCSolver(circuit={self.circuit}, "
            f"tolerance={self.tolerance}, "
            f"max_iterations={self.max_iterations})"
        )
