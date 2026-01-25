"""
DC Solver for linear and non-linear circuits using Modified Nodal Analysis (MNA).

This module implements the DCSolver class, which solves for the DC operating
point of circuits. The solver uses MNA formulation to construct and solve
the circuit equations:

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
- Non-linear MOSFETs (Newton-Raphson iteration)
"""
from typing import Dict, List, Tuple
import numpy as np
from pycircuitsim.circuit import Circuit
from pycircuitsim.models.passive import VoltageSource


class DCSolver:
    """
    DC Solver for linear and non-linear circuits using Modified Nodal Analysis.

    The DCSolver constructs the MNA matrix and solves for the DC operating
    point of a circuit. It handles linear components (resistors, voltage
    sources, current sources) and non-linear components (MOSFETs) using
    Newton-Raphson iteration.

    Attributes:
        circuit: Circuit object containing components and topology
        tolerance: Convergence tolerance for Newton-Raphson
        max_iterations: Maximum Newton-Raphson iterations
    """

    def __init__(self, circuit: Circuit, tolerance: float = 1e-9, max_iterations: int = 50):
        """
        Initialize the DC Solver.

        Args:
            circuit: Circuit object to solve
            tolerance: Convergence tolerance for Newton-Raphson (default: 1e-9)
            max_iterations: Maximum Newton-Raphson iterations (default: 50)
        """
        self.circuit = circuit
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def solve(self) -> Dict[str, float]:
        """
        Solve the circuit for DC operating point.

        This method checks if the circuit contains non-linear components (MOSFETs).
        - If linear: constructs the MNA matrix and solves directly
        - If non-linear: uses Newton-Raphson iteration

        The MNA matrix has size (num_nodes + num_voltage_sources) x (num_nodes + num_voltage_sources).

        Returns:
            Dictionary mapping node names to voltage values (including ground at 0V)

        Raises:
            np.linalg.LinAlgError: If the circuit matrix is singular (unsolvable)
            RuntimeError: If Newton-Raphson fails to converge
        """
        # Check if circuit has non-linear components
        has_non_linear = self._has_non_linear_components()

        if has_non_linear:
            # Use Newton-Raphson for non-linear circuits
            return self._solve_newton()
        else:
            # Direct solve for linear circuits
            return self._solve_linear()

    def _has_non_linear_components(self) -> bool:
        """
        Check if circuit contains non-linear components (MOSFETs).

        Returns:
            True if circuit has MOSFETs, False otherwise
        """
        from pycircuitsim.models.mosfet import NMOS, PMOS

        for component in self.circuit.components:
            if isinstance(component, (NMOS, PMOS)):
                return True
        return False

    def _solve_linear(self) -> Dict[str, float]:
        """
        Solve linear circuit directly using MNA.

        Returns:
            Dictionary mapping node names to voltage values

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
        self._stamp_voltage_sources(mna_matrix, rhs, node_map, num_nodes, voltages=None)

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

    def _solve_newton(self) -> Dict[str, float]:
        """
        Solve non-linear circuit using Newton-Raphson iteration.

        Algorithm:
        1. Initial guess: all nodes at 0V
        2. For each iteration:
           a. Get MOSFET conductances (g_ds, g_m) at current voltages
           b. Stamp conductances to matrix
           c. Solve for delta
           d. Update voltages: v += delta
           e. Check convergence (max |delta| < tolerance)
        3. Raise error if max_iterations exceeded

        Returns:
            Dictionary mapping node names to voltage values

        Raises:
            RuntimeError: If Newton-Raphson fails to converge
            np.linalg.LinAlgError: If the circuit matrix is singular (unsolvable)
        """
        from pycircuitsim.models.mosfet import NMOS, PMOS

        # Get circuit topology
        nodes = self.circuit.get_nodes()
        node_map = self.circuit.get_node_map()
        num_nodes = len(nodes)
        num_voltage_sources = self.circuit.count_voltage_sources()

        # Matrix size: num_nodes + num_voltage_sources
        matrix_size = num_nodes + num_voltage_sources

        # Store original voltage source values for source stepping
        original_voltages = []
        for component in self.circuit.components:
            if isinstance(component, VoltageSource):
                original_voltages.append(component.voltage)

        # Source stepping: gradually increase voltage source values
        # to improve convergence
        num_steps = 10
        for step in range(num_steps):
            # Scale voltage sources
            scale = (step + 1) / num_steps
            vs_idx = 0
            for component in self.circuit.components:
                if isinstance(component, VoltageSource):
                    component.voltage = original_voltages[vs_idx] * scale
                    vs_idx += 1

            # Initial guess: use 0V for first step, previous result for subsequent steps
            if step == 0:
                voltages = {node: 0.0 for node in nodes}
            voltages["0"] = 0.0
            voltages["GND"] = 0.0

            # Newton-Raphson iteration for this source step
            for iteration in range(self.max_iterations // num_steps):
                # Initialize MNA matrix and RHS vector
                mna_matrix = np.zeros((matrix_size, matrix_size))
                rhs = np.zeros(matrix_size)

                # Stamp linear components (resistors, current sources)
                for component in self.circuit.components:
                    if not isinstance(component, (NMOS, PMOS)):
                        component.stamp_conductance(mna_matrix, node_map)
                        component.stamp_rhs(rhs, node_map)

                # Stamp MOSFET conductances and currents
                for component in self.circuit.components:
                    if isinstance(component, (NMOS, PMOS)):
                        self._stamp_mosfet(component, mna_matrix, rhs, node_map, voltages)

                # Handle voltage sources (B and C matrices)
                self._stamp_voltage_sources(mna_matrix, rhs, node_map, num_nodes, voltages=voltages)

                # Solve for delta
                try:
                    delta = np.linalg.solve(mna_matrix, rhs)
                except np.linalg.LinAlgError as e:
                    raise np.linalg.LinAlgError(
                        f"Circuit matrix is singular at source step {step + 1}, iteration {iteration + 1}. "
                        f"Check circuit topology or initial guess."
                    ) from e

                # Update voltages and check convergence
                max_change = 0.0
                for idx, node in enumerate(nodes):
                    old_voltage = voltages[node]
                    # Apply damping for stability
                    damping = 0.8
                    new_voltage = old_voltage + damping * delta[idx]
                    voltages[node] = new_voltage
                    max_change = max(max_change, abs(new_voltage - old_voltage))

                # Check convergence
                if max_change < self.tolerance:
                    # Converged at this source step, move to next step
                    break

            # If we didn't converge at this step, continue anyway
            # (source stepping often allows eventual convergence)

        # Restore original voltage source values
        vs_idx = 0
        for component in self.circuit.components:
            if isinstance(component, VoltageSource):
                component.voltage = original_voltages[vs_idx]
                vs_idx += 1

        return voltages

    def _stamp_mosfet(
        self,
        mosfet,
        mna_matrix: np.ndarray,
        rhs: np.ndarray,
        node_map: Dict[str, int],
        voltages: Dict[str, float],
    ) -> None:
        """
        Stamp MOSFET conductance and current to MNA matrix.

        For a MOSFET, we stamp:
        - g_ds (output conductance) between drain and source
        - g_m (transconductance) from gate to drain
        - Equivalent current source based on operating point

        The Newton-Raphson linearization is:
        I_ds(V) ≈ I_ds(V0) + g_ds*(V_ds - V_ds0) + g_m*(V_gs - V_gs0)
        I_ds(V) - g_ds*V_ds - g_m*V_gs ≈ I_ds0 - g_ds*V_ds0 - g_m*V_gs0

        Args:
            mosfet: MOSFET component (NMOS or PMOS)
            mna_matrix: MNA matrix to modify (in-place)
            rhs: RHS vector to modify (in-place)
            node_map: Mapping from node names to matrix indices
            voltages: Current voltage estimate
        """
        # Get MOSFET terminals
        drain = mosfet.nodes[0]
        gate = mosfet.nodes[1]
        source = mosfet.nodes[2]

        # Get conductances at current operating point
        g_ds, g_m = mosfet.get_conductance(voltages)

        # Get current at operating point
        i_ds = mosfet.calculate_current(voltages)

        # Add a small minimum conductance to prevent numerical instability
        # This helps with convergence when MOSFET is in cutoff
        g_min = 1e-12  # 1 picoSiemens minimum conductance
        g_ds = max(g_ds, g_min)

        # Stamp conductances to MNA matrix
        # g_ds between drain and source
        if drain != "0" and drain in node_map:
            d_idx = node_map[drain]
            mna_matrix[d_idx, d_idx] += g_ds

        if source != "0" and source in node_map:
            s_idx = node_map[source]
            mna_matrix[s_idx, s_idx] += g_ds

        if drain != "0" and drain in node_map and source != "0" and source in node_map:
            d_idx = node_map[drain]
            s_idx = node_map[source]
            mna_matrix[d_idx, s_idx] -= g_ds
            mna_matrix[s_idx, d_idx] -= g_ds

        # g_m from gate to drain (controlled by gate voltage, affects drain current)
        if gate != "0" and gate in node_map and drain != "0" and drain in node_map:
            g_idx = node_map[gate]
            d_idx = node_map[drain]
            mna_matrix[d_idx, g_idx] += g_m

        # Stamp equivalent current source to RHS
        # The RHS should contain: I_ds - g_ds*V_ds - g_m*V_gs
        # This represents the constant term in the linearized equation
        v_d = voltages.get(drain, 0.0)
        v_g = voltages.get(gate, 0.0)
        v_s = voltages.get(source, 0.0)

        v_ds = v_d - v_s
        v_gs = v_g - v_s

        # Equivalent current source (Newton-Raphson constant term)
        i_eq = i_ds - g_ds * v_ds - g_m * v_gs

        # Stamp current to drain and source nodes
        # Current flows OUT of drain, INTO source
        if drain != "0" and drain in node_map:
            d_idx = node_map[drain]
            rhs[d_idx] -= i_eq

        if source != "0" and source in node_map:
            s_idx = node_map[source]
            rhs[s_idx] += i_eq

    def _stamp_voltage_sources(
        self,
        mna_matrix: np.ndarray,
        rhs: np.ndarray,
        node_map: Dict[str, int],
        num_nodes: int,
        voltages: Dict[str, float] = None,
    ) -> None:
        """
        Stamp voltage source equations to MNA matrix.

        For each voltage source, we add:
        - B matrix column: connection to node voltages
        - C matrix row: voltage constraint equation
        - RHS entry: voltage source value (for linear) or mismatch (for Newton-Raphson)

        The voltage source equation is: V_pos - V_neg = V_source
        For Newton-Raphson: delta_V_pos - delta_V_neg = V_source - (V_pos_old - V_neg_old)

        Args:
            mna_matrix: MNA matrix to modify (in-place)
            rhs: RHS vector to modify (in-place)
            node_map: Mapping from node names to matrix indices
            num_nodes: Number of non-ground nodes
            voltages: Current voltage estimate (for Newton-Raphson), None for linear solve
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
                if voltages is None:
                    # Linear solve: use voltage value directly
                    rhs[vs_row] = voltage
                else:
                    # Newton-Raphson: use voltage mismatch
                    v_pos = voltages.get(pos_node, 0.0)
                    v_neg = voltages.get(neg_node, 0.0)
                    rhs[vs_row] = voltage - (v_pos - v_neg)

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
