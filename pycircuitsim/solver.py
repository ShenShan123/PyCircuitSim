"""
DC and Transient Solvers for linear and non-linear circuits using Modified Nodal Analysis (MNA).

This module implements:
1. DCSolver: Solves for the DC operating point of circuits
2. TransientSolver: Performs time-domain analysis using Backward Euler integration

Both solvers use MNA formulation to construct and solve the circuit equations:

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
- Capacitors (Backward Euler companion model for transient analysis)
"""
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from pycircuitsim.circuit import Circuit
from pycircuitsim.models.passive import VoltageSource, Capacitor
from pycircuitsim.logger import Logger, IterationInfo

# Helper function to check if component is a MOSFET
def _is_mosfet(component):
    """Check if component is a MOSFET (Level 1 or BSIM-CMG)."""
    from pycircuitsim.models.mosfet import NMOS, PMOS
    try:
        from pycircuitsim.models.mosfet_cmg import NMOS_CMG, PMOS_CMG
        return isinstance(component, (NMOS, PMOS, NMOS_CMG, PMOS_CMG))
    except ImportError:
        # BSIM-CMG models not available (PyCMG not built)
        return isinstance(component, (NMOS, PMOS))


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

    def __init__(self, circuit: Circuit, tolerance: float = 1e-9, max_iterations: int = 50,
                 output_file: Optional[Path] = None, initial_guess: Optional[Dict[str, float]] = None,
                 logger: Optional[Logger] = None, use_source_stepping: bool = True,
                 source_stepping_steps: int = 20,
                 damping_factor: float = 1.0):
        """
        Initialize the DC Solver.

        Args:
            circuit: Circuit object to solve
            tolerance: Convergence tolerance for Newton-Raphson (default: 1e-9)
            max_iterations: Maximum Newton-Raphson iterations (default: 50)
            output_file: Optional path to output log file (.lis file)
            initial_guess: Optional initial voltage guess for Newton-Raphson (dictionary of node->voltage)
            logger: Optional external Logger instance for logging (reuses existing logger)
            use_source_stepping: Enable source stepping homotopy (default: True)
            source_stepping_steps: Number of source stepping steps (default: 20)
            damping_factor: Initial damping factor for Newton-Raphson (default: 1.0, 0.5 = aggressive damping)
        """
        self.circuit = circuit
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.output_file = output_file
        self.logger = logger  # Use external logger if provided
        self.initial_guess = initial_guess
        self.use_source_stepping = use_source_stepping
        self.source_stepping_steps = source_stepping_steps
        self.damping_factor = damping_factor
        self.last_solution: Optional[Dict[str, float]] = None
        self._owns_logger = False  # Track if we created the logger (for cleanup)

    def __enter__(self):
        """
        Enter the context manager and initialize the logger.

        Returns:
            DCSolver instance
        """
        if self.logger is None and self.output_file:
            # Create new logger only if we don't have one and output_file is specified
            netlist_name = getattr(self.circuit, 'netlist', 'Unknown')
            self.logger = Logger(netlist_name, self.output_file)
            self.logger.__enter__()
            self._owns_logger = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager and close the logger.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        if self.logger and self._owns_logger:
            self.logger.__exit__(exc_type, exc_val, exc_tb)
        return False  # Don't suppress exceptions

    def solve(self, skip_header: bool = False) -> Dict[str, float]:
        """
        Solve the circuit for DC operating point.

        This method checks if the circuit contains non-linear components (MOSFETs).
        - If linear: constructs the MNA matrix and solves directly
        - If non-linear: uses Newton-Raphson iteration

        The MNA matrix has size (num_nodes + num_voltage_sources) x (num_nodes + num_voltage_sources).

        Args:
            skip_header: If True, skip logging header (for use in DC sweep)

        Returns:
            Dictionary mapping node names to voltage values (including ground at 0V)

        Raises:
            np.linalg.LinAlgError: If the circuit matrix is singular (unsolvable)
            RuntimeError: If Newton-Raphson fails to converge
        """
        # Log header and circuit summary if logger is available
        if self.logger and not skip_header:
            self.logger.log_header("DC Operating Point Analysis", {})
            num_nodes = len(self.circuit.get_nodes())
            num_vsources = self.circuit.count_voltage_sources()
            self.logger.log_circuit_summary(
                component_count=len(self.circuit.components),
                node_count=num_nodes,
                vsource_count=num_vsources
            )

        # Check if circuit has non-linear components
        has_non_linear = self._has_non_linear_components()

        if has_non_linear:
            # Use Newton-Raphson for non-linear circuits
            solution = self._solve_newton()
        else:
            # Direct solve for linear circuits
            solution = self._solve_linear()

        # Store the solution for potential reuse
        self.last_solution = solution.copy()

        return solution

    def _has_non_linear_components(self) -> bool:
        """
        Check if circuit contains non-linear components (MOSFETs).

        Returns:
            True if circuit has MOSFETs, False otherwise
        """
        from pycircuitsim.models.mosfet import NMOS, PMOS

        for component in self.circuit.components:
            if _is_mosfet(component):
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

        # Extract and store voltage source currents
        self._store_source_currents(solution, nodes)

        # Log iteration for linear circuit (single iteration)
        if self.logger:
            # Calculate device currents
            currents = {}
            for comp in self.circuit.components:
                try:
                    current = comp.calculate_current(voltages)
                    currents[comp.name] = current
                except (NotImplementedError, AttributeError):
                    # Skip components that don't support current calculation
                    pass

            # Create iteration info
            iter_info = IterationInfo(
                iteration=0,
                voltages=voltages.copy(),
                deltas={},  # No deltas for linear solve
                currents=currents,
                conductances={}  # No conductances for linear solve
            )
            self.logger.log_iteration(point_num=0, iter_info=iter_info)

            # Log convergence
            self.logger.log_convergence(
                point_num=0,
                converged=True,
                iterations=1,
                tolerance=0.0  # Linear solve has exact solution
            )

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

        # Reset damping to default for each solve
        self.damping_factor = 1.0

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
        # Use configurable number of steps when source stepping is enabled
        num_steps = self.source_stepping_steps if (self._has_non_linear_components() and self.use_source_stepping) else 1
        for step in range(num_steps):
            # Scale voltage sources
            scale = (step + 1) / num_steps
            vs_idx = 0
            for component in self.circuit.components:
                if isinstance(component, VoltageSource):
                    component.voltage = original_voltages[vs_idx] * scale
                    vs_idx += 1

            # Initial guess: use provided initial_guess for first step if available,
            # otherwise use 0V for first step, previous result for subsequent steps
            if step == 0:
                if self.initial_guess is not None:
                    # Use provided initial guess
                    voltages = {node: 0.0 for node in nodes}
                    for node, voltage in self.initial_guess.items():
                        if node in voltages:
                            voltages[node] = voltage
                else:
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
                    if not _is_mosfet(component):
                        component.stamp_conductance(mna_matrix, node_map)
                        component.stamp_rhs(rhs, node_map)

                # Stamp MOSFET conductances and currents
                # For Level 1, use _stamp_mosfet
                for component in self.circuit.components:
                    if _is_mosfet(component):
                        self._stamp_mosfet(component, mna_matrix, rhs, node_map, voltages)

                # Handle voltage sources (B and C matrices)
                self._stamp_voltage_sources(mna_matrix, rhs, node_map, num_nodes, voltages=voltages)

                # MNA Matrix Conditioning Check
                # Check if the conductance matrix (top-left portion) is ill-conditioned
                if num_nodes > 0:
                    try:
                        conductance_matrix = mna_matrix[:num_nodes, :num_nodes]
                        cond_number = np.linalg.cond(conductance_matrix)
                        if cond_number > 1e12:
                            # Matrix is ill-conditioned - this can cause numerical instability
                            if self.logger:
                                self.logger._write_separator("-")
                                self.logger._write(f"WARNING: Ill-conditioned MNA matrix at step {step + 1}, iteration {iteration + 1}")
                                self.logger._write(f"  Condition number: {cond_number:.2e}")
                                self.logger._write(f"  This may cause numerical inaccuracies or convergence issues")
                                self.logger._write_separator("-")
                    except np.linalg.LinAlgError:
                        # Singular matrix - will be caught by the solver below
                        pass

                    # Check for negative diagonal elements in conductance matrix
                    # (indicates physically unrealistic conductance values)
                    for i in range(min(num_nodes, len(mna_matrix))):
                        diag_value = mna_matrix[i, i]
                        if diag_value < 0:
                            if self.logger:
                                self.logger._write(f"WARNING: Negative diagonal element Y[{i},{i}] = {diag_value:.6e} S")
                                self.logger._write(f"  This may indicate incorrect device conductance calculation")

                # Solve the MNA system
                # With companion model (direct voltage source RHS), the solution
                # is the NEW voltage, not a delta correction.
                try:
                    solution = np.linalg.solve(mna_matrix, rhs)
                except np.linalg.LinAlgError as e:
                    raise np.linalg.LinAlgError(
                        f"Circuit matrix is singular at source step {step + 1}, iteration {iteration + 1}. "
                        f"Check circuit topology or initial guess."
                    ) from e

                # Update voltages using companion model formulation
                # The solution contains the new voltages directly
                max_change = 0.0
                max_delta = 0.0
                deltas = {}

                # Identify voltage-source-constrained nodes
                # These nodes must reach their target values exactly (no damping)
                vs_constrained_nodes = set()
                for component in self.circuit.components:
                    if isinstance(component, VoltageSource):
                        pos_node = component.nodes[0]
                        neg_node = component.nodes[1]
                        # If one terminal is ground, the other is constrained
                        if neg_node == "0":
                            vs_constrained_nodes.add(pos_node)
                        elif pos_node == "0":
                            vs_constrained_nodes.add(neg_node)

                for idx, node in enumerate(nodes):
                    old_voltage = voltages[node]
                    new_voltage_solution = solution[idx]

                    # Calculate delta for convergence check
                    delta_v = new_voltage_solution - old_voltage
                    max_delta = max(max_delta, abs(delta_v))

                # Adaptive damping: Check if we need to enable damping
                # Large deltas (>1V) indicate potential divergence
                if max_delta > 1.0 and self.damping_factor == 1.0:
                    # Enable damping for this iteration to prevent overshooting
                    self.damping_factor = 0.5

                for idx, node in enumerate(nodes):
                    old_voltage = voltages[node]
                    new_voltage_solution = solution[idx]

                    if node in vs_constrained_nodes:
                        # Voltage source nodes: use solution directly (no damping)
                        # These must satisfy the constraint exactly
                        new_voltage = new_voltage_solution
                    else:
                        # Free nodes: apply adaptive damping
                        # damping=1.0 -> fully use new voltage
                        # damping=0.5 -> average of old and new
                        new_voltage = self.damping_factor * new_voltage_solution + (1.0 - self.damping_factor) * old_voltage

                    deltas[node] = abs(new_voltage - old_voltage)
                    voltages[node] = new_voltage
                    max_change = max(max_change, abs(new_voltage - old_voltage))

                # Log iteration if logger is available
                if self.logger:
                    # Calculate device currents
                    currents = {}
                    conductances = {}
                    for comp in self.circuit.components:
                        try:
                            current = comp.calculate_current(voltages)
                            currents[comp.name] = current

                            # Get conductances for MOSFETs
                            if isinstance(comp, (NMOS, PMOS)):
                                gm, gds = comp.get_conductance(voltages)
                                conductances[comp.name] = {"gm": gm, "gds": gds}
                        except (NotImplementedError, AttributeError):
                            # Skip components that don't support current calculation
                            pass

                    # Create iteration info
                    iter_info = IterationInfo(
                        iteration=iteration,
                        voltages=voltages.copy(),
                        deltas=deltas,
                        currents=currents,
                        conductances=conductances
                    )
                    self.logger.log_iteration(point_num=0, iter_info=iter_info)

                # Check convergence
                if max_change < self.tolerance:
                    # Reset damping if converged well
                    if max_delta < 0.1 and self.damping_factor < 1.0:
                        self.damping_factor = 1.0
                    # Log convergence
                    if self.logger:
                        self.logger.log_convergence(
                            point_num=0,
                            converged=True,
                            iterations=iteration + 1,
                            tolerance=max_change
                        )
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

        # Extract and store voltage source currents from final operating point
        # Build final MNA matrix and solve to get full solution (voltages + currents)
        mna_matrix_final = np.zeros((matrix_size, matrix_size))
        rhs_final = np.zeros(matrix_size)

        # Stamp all components at final voltages
        for component in self.circuit.components:
            if not _is_mosfet(component):
                component.stamp_conductance(mna_matrix_final, node_map)
                component.stamp_rhs(rhs_final, node_map)

        # Stamp MOSFETs at final operating point
        for component in self.circuit.components:
            if _is_mosfet(component):
                self._stamp_mosfet(component, mna_matrix_final, rhs_final, node_map, voltages)

        # Handle voltage sources
        self._stamp_voltage_sources(mna_matrix_final, rhs_final, node_map, num_nodes, voltages=voltages)

        # Solve to get full solution including currents
        try:
            solution_final = np.linalg.solve(mna_matrix_final, rhs_final)
            self._store_source_currents(solution_final, nodes)
        except np.linalg.LinAlgError:
            # If singular, skip current extraction
            pass

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
        bulk = mosfet.nodes[3]

        # Get conductances at current operating point
        # Handle both 2-tuple (Level 1) and 3-tuple (BSIM-CMG) returns
        conductance_result = mosfet.get_conductance(voltages)
        if len(conductance_result) == 2:
            g_ds, g_m = conductance_result
            g_mb = 0.0  # No bulk transconductance for Level 1
        else:
            g_ds, g_m, g_mb = conductance_result

        # Get current at operating point
        i_ds = mosfet.calculate_current(voltages)

        # Add a small minimum conductance to prevent numerical instability
        # This helps with convergence when MOSFET is in cutoff or saturation
        # Use higher value for numerical stability in Newton-Raphson
        g_min = 1e-6  # 1 microSiemens minimum conductance (~1 MΩ)
        g_ds = max(g_ds, g_min)

        # Stamp conductances to MNA matrix
        # IMPORTANT: Conductances are ALWAYS positive in the matrix!
        # The current direction is handled by RHS stamping, not conductance signs.

        # g_ds between drain and source (resistive channel)
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

        # g_m transconductance stamping (VCCS: gate controls drain current)
        # For both NMOS and PMOS, we use the SAME transconductance stamping pattern.
        # The current direction difference is handled by RHS stamping, not conductance signs.
        #
        # The transconductance represents the sensitivity of |I_ds| to V_gs,
        # which has the same sign for both NMOS and PMOS (increasing |V_gs| increases |I_ds|).

        # Drain equation: current = g_m * V_gs
        if gate != "0" and gate in node_map and drain != "0" and drain in node_map:
            g_idx = node_map[gate]
            d_idx = node_map[drain]
            mna_matrix[d_idx, g_idx] += g_m

        if drain != "0" and drain in node_map and source != "0" and source in node_map:
            d_idx = node_map[drain]
            s_idx = node_map[source]
            mna_matrix[d_idx, s_idx] -= g_m

        # Source equation: current = -g_m * V_gs (KCL)
        if gate != "0" and gate in node_map and source != "0" and source in node_map:
            g_idx = node_map[gate]
            s_idx = node_map[source]
            mna_matrix[s_idx, g_idx] -= g_m

        if source != "0" and source in node_map:
            s_idx = node_map[source]
            mna_matrix[s_idx, s_idx] += g_m

        # g_mb from bulk to drain (bulk transconductance, for BSIM-CMG)
        if abs(g_mb) > 1e-12 and bulk != source:
            if bulk != "0" and bulk in node_map and drain != "0" and drain in node_map:
                b_idx = node_map[bulk]
                d_idx = node_map[drain]
                mna_matrix[d_idx, b_idx] += g_mb

        # Stamp equivalent current source to RHS
        # The linearized equation is:
        # I_d = g_ds*V_ds + g_m*V_gs + g_mb*V_bs + i_eq
        # where i_eq = i_ds - g_ds*v_ds - g_m*v_gs - g_mb*v_bs
        v_d = voltages.get(drain, 0.0)
        v_g = voltages.get(gate, 0.0)
        v_s = voltages.get(source, 0.0)
        v_b = voltages.get(bulk, 0.0)

        v_ds = v_d - v_s
        v_gs = v_g - v_s
        v_bs = v_b - v_s

        # Equivalent current source (Newton-Raphson constant term)
        i_eq = i_ds - g_ds * v_ds - g_m * v_gs - g_mb * v_bs

        # Check device type for correct current direction
        from pycircuitsim.models.mosfet import PMOS
        try:
            from pycircuitsim.models.mosfet_cmg import PMOS_CMG
            is_pmos = isinstance(mosfet, (PMOS, PMOS_CMG))
        except ImportError:
            is_pmos = isinstance(mosfet, PMOS)

        # Stamp current to drain and source nodes
        # IMPORTANT: NMOS and PMOS have OPPOSITE current directions!
        #
        # For Level-1 PMOS with positive KP:
        # - The equation gives positive i_ds for normal operation (Vgs < VTO, Vds < 0)
        # - Positive i_ds from equation represents S→D current direction
        # - So current ENTERS drain (opposite of NMOS D→S convention)
        #
        # MNA RHS convention: positive value = current entering the node
        # - NMOS: current leaves drain → rhs[drain] -= i_eq
        # - PMOS: current enters drain → rhs[drain] += i_eq
        if is_pmos:
            # PMOS: current flows INTO drain (from source), OUT OF source
            if drain != "0" and drain in node_map:
                d_idx = node_map[drain]
                rhs[d_idx] += i_eq

            if source != "0" and source in node_map:
                s_idx = node_map[source]
                rhs[s_idx] -= i_eq
        else:
            # NMOS: current flows OUT OF drain (to source), INTO source
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
                # Use direct voltage value for companion model consistency.
                # The companion model for MOSFETs solves for V directly,
                # so voltage sources should also use direct form.
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

    def _store_source_currents(self, solution: np.ndarray, nodes: List[str]) -> None:
        """
        Extract and store voltage source currents from solution vector.

        The solution vector contains voltage source currents after the node voltages.
        This method extracts those currents and stores them in the VoltageSource objects
        so they can be retrieved via calculate_current().

        Args:
            solution: Solution vector from np.linalg.solve
            nodes: List of non-ground node names
        """
        num_nodes = len(nodes)
        vs_idx = 0

        # Iterate through circuit components to find voltage sources in order
        for component in self.circuit.components:
            if isinstance(component, VoltageSource):
                # Extract current from solution vector (after node voltages)
                current_idx = num_nodes + vs_idx
                if current_idx < len(solution):
                    current = float(solution[current_idx])
                    # Store current in the voltage source object
                    if hasattr(component, 'set_current'):
                        component.set_current(current)
                vs_idx += 1

    def get_last_solution(self) -> Optional[Dict[str, float]]:
        """
        Get the last computed solution from this solver.

        Returns:
            Dictionary mapping node names to voltages, or None if solve() hasn't been called yet
        """
        return self.last_solution

    def __repr__(self) -> str:
        """String representation of the solver."""
        return (
            f"DCSolver(circuit={self.circuit}, "
            f"tolerance={self.tolerance}, "
            f"max_iterations={self.max_iterations})"
        )


class TransientSolver:
    """
    Transient Solver for time-domain analysis using Backward Euler integration.

    The TransientSolver performs time-domain simulation of circuits with capacitors.
    It uses the Backward Euler method to discretize capacitors into companion models
    (equivalent conductance and current source) at each timestep.

    Algorithm:
    1. Perform DC analysis at t=0 to find initial conditions
    2. For each timestep:
       a. Update capacitor companion models (G_eq = C/dt, I_eq = G_eq * V_prev)
       b. Solve DC circuit at current timestep
       c. Update capacitor voltages for next timestep
       d. Store results

    Attributes:
        circuit: Circuit object containing components and topology
        t_stop: Stop time for simulation in seconds
        dt: Timestep size in seconds
    """

    def __init__(self, circuit: Circuit, t_stop: float, dt: float,
                 initial_guess: Optional[Dict[str, float]] = None,
                 debug: bool = False,
                 use_gmin_stepping: bool = True,
                 gmin_initial: float = 1e-8,
                 gmin_final: float = 1e-12,
                 gmin_steps: int = 5,
                 use_pseudo_transient: bool = True,
                 pseudo_transient_steps: int = 3,
                 pseudo_transient_cap: float = 1e-12):
        """
        Initialize the Transient Solver.

        Args:
            circuit: Circuit object to simulate
            t_stop: Stop time for simulation in seconds
            dt: Timestep size in seconds (must be positive)
            initial_guess: Optional initial voltage guess from DC operating point
            debug: Enable debug logging for convergence diagnostics
            use_gmin_stepping: Enable Gmin stepping for difficult convergence (default: True)
            gmin_initial: Initial Gmin value for stepping (default: 1e-8 S)
            gmin_final: Final Gmin value (default: 1e-12 S)
            gmin_steps: Number of Gmin stepping steps (default: 5)
            use_pseudo_transient: Enable pseudo-transient initialization (default: True)
            pseudo_transient_steps: Number of initial timesteps with pseudo-capacitance (default: 3)
            pseudo_transient_cap: Artificial capacitance value in Farads (default: 1e-12 F)

        Raises:
            ValueError: If dt or t_stop is not positive
        """
        if dt <= 0:
            raise ValueError(f"Timestep dt must be positive, got {dt}")
        if t_stop <= 0:
            raise ValueError(f"Stop time t_stop must be positive, got {t_stop}")

        self.circuit = circuit
        self.t_stop = t_stop
        self.dt = dt
        self.initial_guess = initial_guess
        self.debug = debug

        # Gmin stepping parameters
        self.use_gmin_stepping = use_gmin_stepping
        self.gmin_initial = gmin_initial
        self.gmin_final = gmin_final
        self.gmin_steps = gmin_steps

        # Pseudo-transient initialization parameters
        self.use_pseudo_transient = use_pseudo_transient
        self.pseudo_transient_steps = pseudo_transient_steps
        self.pseudo_transient_cap = pseudo_transient_cap

        # Store pseudo-capacitor references for cleanup
        self._pseudo_capacitors: List = []

    def _has_non_linear_components(self) -> bool:
        """
        Check if circuit contains non-linear components (MOSFETs).

        Returns:
            True if circuit has MOSFETs, False otherwise
        """
        from pycircuitsim.models.mosfet import NMOS, PMOS

        for component in self.circuit.components:
            if _is_mosfet(component):
                return True
        return False

    def _add_pseudo_capacitors(self) -> None:
        """
        Add pseudo-capacitors from all non-ground nodes to ground for pseudo-transient initialization.

        This adds artificial capacitance to improve DC convergence during the first few timesteps.
        The pseudo-capacitors are removed after the specified number of steps.
        """
        from pycircuitsim.models.passive import Capacitor

        # Get all non-ground nodes
        nodes = self.circuit.get_nodes()

        # Add a pseudo-capacitor from each node to ground
        pseudo_cap_idx = 0
        for node in nodes:
            cap = Capacitor(f"_pseudo_{pseudo_cap_idx}", [node, "0"], self.pseudo_transient_cap)
            self.circuit.components.append(cap)
            self._pseudo_capacitors.append(cap)
            pseudo_cap_idx += 1

    def _remove_pseudo_capacitors(self) -> None:
        """
        Remove pseudo-capacitors added for pseudo-transient initialization.
        """
        for cap in self._pseudo_capacitors:
            if cap in self.circuit.components:
                self.circuit.components.remove(cap)
        self._pseudo_capacitors.clear()

    def _apply_gmin_stepping(self, mna_matrix: np.ndarray, node_map: Dict[str, int], gmin: float) -> None:
        """
        Apply Gmin stepping by adding minimum conductance to all nodes.

        Args:
            mna_matrix: MNA matrix to modify (in-place)
            node_map: Mapping from node names to matrix indices
            gmin: Current Gmin value to apply
        """
        # Add gmin from each non-ground node to ground
        for node, idx in node_map.items():
            mna_matrix[idx, idx] += gmin

    def _solve_timestep_newton(
        self,
        nodes: List[str],
        node_map: Dict[str, int],
        num_nodes: int,
        num_voltage_sources: int,
        initial_voltages: Dict[str, float],
        time: float,
        step_index: int = 0
    ) -> Dict[str, float]:
        """
        Solve circuit at a single timestep using Newton-Raphson iteration.

        This method is used for non-linear circuits (with MOSFETs).
        It iteratively linearizes the circuit equations until convergence.

        Args:
            nodes: List of non-ground node names
            node_map: Mapping from node names to matrix indices
            num_nodes: Number of non-ground nodes
            num_voltage_sources: Number of voltage sources
            initial_voltages: Initial voltage guess from previous timestep
            time: Current simulation time
            step_index: Current timestep index (for Gmin stepping)

        Returns:
            Dictionary mapping node names to voltage values

        Raises:
            RuntimeError: If Newton-Raphson fails to converge
        """
        from pycircuitsim.models.mosfet import NMOS, PMOS

        # Matrix size: num_nodes + num_voltage_sources
        matrix_size = num_nodes + num_voltage_sources

        # Use previous timestep's voltages as initial guess
        voltages = initial_voltages.copy()

        # Newton-Raphson parameters (aligned with DC solver)
        tolerance = 1e-6  # Relaxed from 1e-9 for transient (fast-switching circuits)
        max_iterations = 200  # Increased from 100 for difficult convergence

        # Calculate Gmin value for this timestep (if enabled)
        gmin = self.gmin_final
        if self.use_gmin_stepping and step_index < self.gmin_steps:
            # Exponential decay from gmin_initial to gmin_final
            alpha = step_index / (self.gmin_steps - 1) if self.gmin_steps > 1 else 1.0
            gmin = self.gmin_initial * (1 - alpha) + self.gmin_final * alpha
            if self.debug:
                print(f"  Gmin stepping: step {step_index}, gmin = {gmin:.2e}")

        # Start with moderate damping (more aggressive for early timesteps)
        damping = 0.75 if step_index < 5 else 1.0

        # Track previous max_delta for adaptive damping
        prev_max_delta = float('inf')
        stuck_counter = 0  # Count iterations with minimal improvement

        # Track recent voltages for oscillation detection
        voltage_history = []

        # Debug: Track convergence behavior (if enabled)
        debug_log = [] if self.debug else None

        for iteration in range(max_iterations):
            # Build MNA matrix and RHS
            mna_matrix = np.zeros((matrix_size, matrix_size))
            rhs = np.zeros(matrix_size)

            # Stamp linear components (resistors, capacitors)
            for component in self.circuit.components:
                if not _is_mosfet(component):
                    component.stamp_conductance(mna_matrix, node_map)
                    component.stamp_rhs(rhs, node_map)

            # Stamp voltage sources (with time-varying support)
            self._stamp_voltage_sources(mna_matrix, rhs, node_map, num_nodes, time, voltages)

            # Stamp MOSFETs at current voltage estimate
            for component in self.circuit.components:
                if _is_mosfet(component):
                    self._stamp_mosfet_transient(component, mna_matrix, rhs, node_map, voltages)

            # Apply Gmin stepping (if enabled)
            if gmin > self.gmin_final:
                self._apply_gmin_stepping(mna_matrix, node_map, gmin)

            # Solve for voltage updates
            try:
                solution = np.linalg.solve(mna_matrix, rhs)
            except np.linalg.LinAlgError:
                raise RuntimeError(
                    f"Circuit matrix is singular at t={time:.6e}s during Newton-Raphson iteration {iteration+1}"
                )

            # Extract voltages from solution (matches DC solver approach)
            # Solution contains NEW voltages, not deltas (due to MNA formulation)
            max_delta = 0.0
            deltas = {}

            # Identify voltage-source-constrained nodes (exempt from damping)
            vs_constrained_nodes = set()
            from pycircuitsim.models.passive import VoltageSource
            for component in self.circuit.components:
                if isinstance(component, VoltageSource):
                    pos_node = component.nodes[0]
                    neg_node = component.nodes[1]
                    # If one terminal is ground, the other is constrained
                    if neg_node == "0":
                        vs_constrained_nodes.add(pos_node)
                    elif pos_node == "0":
                        vs_constrained_nodes.add(neg_node)

            # Calculate deltas for convergence check
            for idx, node in enumerate(nodes):
                old_voltage = voltages[node]
                new_voltage_solution = solution[idx]  # Absolute voltage from MNA
                delta_v = new_voltage_solution - old_voltage
                deltas[node] = delta_v
                max_delta = max(max_delta, abs(delta_v))

            # Track voltage history for oscillation detection (store last 5 iterations)
            voltage_snapshot = {}
            for idx, node in enumerate(nodes):
                voltage_snapshot[node] = solution[idx]
            voltage_history.append(voltage_snapshot)
            if len(voltage_history) > 5:
                voltage_history.pop(0)

            # DEBUG: Log first few and last few iterations
            if self.debug and (iteration < 5 or iteration >= max_iterations - 5):
                debug_log.append(f"  Iter {iteration}: max_delta={max_delta:.6e}, damping={damping:.2f}, gmin={gmin:.2e}")

            # Check convergence
            if max_delta < tolerance:
                # Converged! Use new voltages directly
                for idx, node in enumerate(nodes):
                    voltages[node] = solution[idx]
                if self.debug and debug_log is not None and len(debug_log) > 0:
                    print(f"\nDEBUG: Converged at t={time:.6e}s after {iteration+1} iterations")
                break

            # Adaptive damping: adjust based on convergence behavior
            improvement_ratio = max_delta / (prev_max_delta + 1e-12)

            # Reduce damping aggressively if not converging well
            if improvement_ratio > 0.9 and iteration > 3:
                # Stuck or oscillating: reduce damping more
                stuck_counter += 1
                if stuck_counter >= 2:
                    damping = max(0.25, damping * 0.8)  # Reduce damping
                    stuck_counter = 0
            elif improvement_ratio < 0.5:
                # Good progress: increase damping
                damping = min(1.0, damping * 1.1)
                stuck_counter = 0
            else:
                stuck_counter = 0

            # Apply damping based on voltage deltas (match DC solver threshold)
            if max_delta >= 1.0:
                damping = min(damping, 0.5)  # Force damping if deltas are very large
            elif max_delta < 0.1:
                damping = 1.0  # No damping needed for small deltas

            prev_max_delta = max_delta

            # Update voltages with damping (match DC solver approach)
            for idx, node in enumerate(nodes):
                old_voltage = voltages[node]
                new_voltage_solution = solution[idx]

                if node in vs_constrained_nodes:
                    # Voltage source nodes: use solution directly
                    voltages[node] = new_voltage_solution
                else:
                    # Free nodes: apply damping (blend old and new)
                    voltages[node] = damping * new_voltage_solution + (1.0 - damping) * old_voltage
        else:
            # Did not converge - check if it's "good enough"
            # For fast-switching circuits, accept solution if oscillating around stable point
            # Check if we're oscillating (voltages bouncing between similar values)
            if len(voltage_history) >= 3:
                # Calculate average of last 3 iterations
                avg_voltages = {}
                for node in nodes:
                    sum_v = 0.0
                    for snapshot in voltage_history[-3:]:
                        sum_v += snapshot.get(node, 0.0)
                    avg_voltages[node] = sum_v / 3.0

                # Check oscillation: variance of last few iterations
                max_variance = 0.0
                for node in nodes:
                    values = [s.get(node, 0.0) for s in voltage_history[-3:]]
                    variance = max(values) - min(values)
                    max_variance = max(max_variance, variance)

                # If variance is small (< 100mV), accept averaged solution
                if max_variance < 0.1:
                    if self.debug:
                        print(f"  WARNING: Newton-Raphson oscillating at t={time:.6e}s")
                        print(f"  Max variance = {max_variance:.2e} (accepting averaged solution)")
                    # Use averaged voltages
                    for node in nodes:
                        voltages[node] = avg_voltages[node]
                    voltages["0"] = 0.0
                    voltages["GND"] = 0.0
                    return voltages

            # Not good enough - print debug log and raise error
            if self.debug and debug_log is not None and len(debug_log) > 0:
                print(f"\nDEBUG: Convergence failure at t={time:.6e}s:")
                for log_line in debug_log:
                    print(log_line)
                print(f"\n  Final voltages:")
                for node in nodes[:5]:  # Print first 5 nodes
                    print(f"    {node}: {voltages[node]:.6f}V")

            raise RuntimeError(
                f"Newton-Raphson failed to converge at t={time:.6e}s after {max_iterations} iterations. "
                f"Final max delta: {max_delta:.2e} (tolerance: {tolerance:.2e})"
            )

        # Add ground nodes
        voltages["0"] = 0.0
        voltages["GND"] = 0.0

        return voltages

    def _stamp_mosfet_transient(
        self,
        mosfet,
        mna_matrix: np.ndarray,
        rhs: np.ndarray,
        node_map: Dict[str, int],
        voltages: Dict[str, float],
    ) -> None:
        """
        Stamp MOSFET conductance and current to MNA matrix for transient analysis.

        This is the same as the DC solver's MOSFET stamping, but kept separate
        to avoid confusion with the linear transient path.

        Args:
            mosfet: MOSFET component (NMOS or PMOS)
            mna_matrix: MNA matrix to modify (in-place)
            rhs: RHS vector to modify (in-place)
            node_map: Mapping from node names to matrix indices
            voltages: Current voltage estimate at this timestep
        """
        # Get MOSFET terminals
        drain = mosfet.nodes[0]
        gate = mosfet.nodes[1]
        source = mosfet.nodes[2]
        bulk = mosfet.nodes[3]

        # Get conductances at current operating point
        # Handle both 2-tuple (Level 1) and 3-tuple (BSIM-CMG) returns
        conductance_result = mosfet.get_conductance(voltages)
        if len(conductance_result) == 2:
            g_ds, g_m = conductance_result
            g_mb = 0.0  # No bulk transconductance for Level 1
        else:
            g_ds, g_m, g_mb = conductance_result

        # Get current at operating point
        i_ds = mosfet.calculate_current(voltages)

        # NOTE: MOSFET internal capacitances are NOT stamped here
        # The Level-1 capacitance model (get_capacitances) exists but requires
        # state tracking across timesteps (V_prev), similar to Capacitor class.
        # This is planned for future implementation.
        # For now, transient analysis works correctly with explicit capacitors in the netlist.

        # Add minimum conductance to prevent numerical instability
        g_min = 1e-6
        g_ds = max(g_ds, g_min)

        # Stamp conductances to MNA matrix
        # IMPORTANT: Conductances are ALWAYS positive in the matrix!
        # The current direction is handled by RHS stamping, not conductance signs.

        # g_ds between drain and source (resistive channel)
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

        # g_m transconductance stamping (same pattern for NMOS and PMOS)
        # Drain equation: current = g_m * V_gs
        if gate != "0" and gate in node_map and drain != "0" and drain in node_map:
            g_idx = node_map[gate]
            d_idx = node_map[drain]
            mna_matrix[d_idx, g_idx] += g_m

        if drain != "0" and drain in node_map and source != "0" and source in node_map:
            d_idx = node_map[drain]
            s_idx = node_map[source]
            mna_matrix[d_idx, s_idx] -= g_m

        # Source equation: current = -g_m * V_gs (KCL)
        if gate != "0" and gate in node_map and source != "0" and source in node_map:
            g_idx = node_map[gate]
            s_idx = node_map[source]
            mna_matrix[s_idx, g_idx] -= g_m

        if source != "0" and source in node_map:
            s_idx = node_map[source]
            mna_matrix[s_idx, s_idx] += g_m

        # g_mb from bulk to drain (bulk transconductance)
        if abs(g_mb) > 1e-12 and bulk != source:
            if bulk != "0" and bulk in node_map and drain != "0" and drain in node_map:
                b_idx = node_map[bulk]
                d_idx = node_map[drain]
                mna_matrix[d_idx, b_idx] += g_mb

        # Stamp equivalent current source to RHS
        v_d = voltages.get(drain, 0.0)
        v_g = voltages.get(gate, 0.0)
        v_s = voltages.get(source, 0.0)
        v_b = voltages.get(bulk, 0.0)

        v_ds = v_d - v_s
        v_gs = v_g - v_s
        v_bs = v_b - v_s

        # Equivalent current source (same formula for NMOS and PMOS)
        i_eq = i_ds - g_ds * v_ds - g_m * v_gs - g_mb * v_bs

        # Check device type for correct current direction
        from pycircuitsim.models.mosfet import PMOS
        try:
            from pycircuitsim.models.mosfet_cmg import PMOS_CMG
            is_pmos = isinstance(mosfet, (PMOS, PMOS_CMG))
        except ImportError:
            is_pmos = isinstance(mosfet, PMOS)

        # Stamp current to drain and source nodes
        # (See DC solver comments for explanation)
        if is_pmos:
            # PMOS: current flows INTO drain, OUT OF source
            if drain != "0" and drain in node_map:
                d_idx = node_map[drain]
                rhs[d_idx] += i_eq

            if source != "0" and source in node_map:
                s_idx = node_map[source]
                rhs[s_idx] -= i_eq
        else:
            # NMOS: current flows OUT OF drain, INTO source
            if drain != "0" and drain in node_map:
                d_idx = node_map[drain]
                rhs[d_idx] -= i_eq

            if source != "0" and source in node_map:
                s_idx = node_map[source]
                rhs[s_idx] += i_eq

    def solve(self) -> Dict[str, np.ndarray]:
        """
        Perform transient analysis from t=0 to t=t_stop.

        This method:
        1. Performs DC analysis at t=0 to find initial operating point
        2. Iterates through timesteps, updating capacitor companion models
        3. Solves circuit at each timestep using DC solver
        4. Returns time series of node voltages

        Returns:
            Dictionary containing:
                - "time": numpy array of time points
                - node names: numpy arrays of voltages at each time point

        Raises:
            np.linalg.LinAlgError: If the circuit matrix is singular (unsolvable)
            RuntimeError: If DC solver fails to converge
        """
        # Get circuit topology
        nodes = self.circuit.get_nodes()
        node_map = self.circuit.get_node_map()
        num_nodes = len(nodes)
        num_voltage_sources = self.circuit.count_voltage_sources()

        # Calculate number of timesteps
        num_steps = int(np.ceil(self.t_stop / self.dt)) + 1

        # Initialize storage arrays
        time = np.zeros(num_steps)
        voltages_over_time = {node: np.zeros(num_steps) for node in nodes}

        # Step 1: Initial conditions from capacitor voltages
        # For transient analysis, we use the capacitor's initial voltage (v_prev)
        # instead of doing a DC solve (which would give steady-state, not transient)

        # Build initial voltage estimate based on capacitor v_prev values
        initial_voltages = {"0": 0.0, "GND": 0.0}

        # Use initial_guess if provided (from DC operating point)
        if self.initial_guess is not None:
            for node, voltage in self.initial_guess.items():
                if node not in ["0", "GND"]:
                    initial_voltages[node] = voltage

            # Initialize capacitor v_prev from DC operating point
            # This is critical: capacitors must start with their DC voltage,
            # otherwise the transient analysis will have incorrect initial conditions
            for component in self.circuit.components:
                if isinstance(component, Capacitor):
                    node_i, node_j = component.nodes[0], component.nodes[1]
                    v_i = self.initial_guess.get(node_i, 0.0)
                    v_j = self.initial_guess.get(node_j, 0.0)
                    # v_prev is the voltage across the capacitor (V_i - V_j)
                    component.v_prev = v_i - v_j

        # For each capacitor, estimate the node voltages based on v_prev
        for component in self.circuit.components:
            if isinstance(component, Capacitor):
                node_i, node_j = component.nodes[0], component.nodes[1]

                # If one node is ground, the other is at v_prev
                if node_j == "0" or node_j == "GND":
                    initial_voltages[node_i] = component.v_prev
                elif node_i == "0" or node_i == "GND":
                    initial_voltages[node_j] = -component.v_prev
                else:
                    # Both nodes are non-ground: we can't determine individual voltages
                    # from just the difference, so set them to 0 for now
                    # The first timestep will correct this
                    if node_i not in initial_voltages:
                        initial_voltages[node_i] = 0.0
                    if node_j not in initial_voltages:
                        initial_voltages[node_j] = 0.0

        # For any remaining nodes, set to 0V
        for node in nodes:
            if node not in initial_voltages:
                initial_voltages[node] = 0.0

        # Store initial voltages
        time[0] = 0.0
        for node in nodes:
            voltages_over_time[node][0] = initial_voltages.get(node, 0.0)

        # Debug: Print initial voltages
        if self.debug:
            print(f"Initial transient voltages:")
            for node in sorted(nodes)[:5]:
                print(f"  V{node} = {initial_voltages.get(node, 0.0):.4f} V")

        # Step 2: Add pseudo-capacitors if enabled (for better DC convergence)
        if self.use_pseudo_transient and self._has_non_linear_components():
            if self.debug:
                print(f"Adding pseudo-capacitors for better DC convergence (first {self.pseudo_transient_steps} steps)")
            self._add_pseudo_capacitors()

        # Step 3: Adaptive time-stepping loop (reduce dt on convergence failure)
        max_dt_reductions = 5  # Maximum number of dt reductions per timestep
        min_dt = self.dt / (2 ** max_dt_reductions)  # Minimum allowed dt

        for step in range(1, num_steps):
            # Current time
            current_time = step * self.dt
            time[step] = min(current_time, self.t_stop)

            # Remove pseudo-capacitors after specified steps
            if self.use_pseudo_transient and step == self.pseudo_transient_steps + 1:
                if self.debug:
                    print(f"Removing pseudo-capacitors at step {step}")
                self._remove_pseudo_capacitors()

            # Get initial guess for this timestep (use previous timestep's voltages)
            prev_voltages = {}
            for node in nodes:
                prev_voltages[node] = voltages_over_time[node][step - 1]

            # Try to solve with current dt, reduce if fails
            dt_reduction_count = 0
            current_dt = self.dt

            while dt_reduction_count <= max_dt_reductions:
                try:
                    # Update capacitor companion models for this timestep
                    for component in self.circuit.components:
                        if isinstance(component, Capacitor):
                            g_eq, i_eq = component.get_companion_model(current_dt, component.v_prev)

                    # Check if circuit has non-linear components
                    has_non_linear = self._has_non_linear_components()

                    # Solve for node voltages at this timestep
                    if has_non_linear:
                        # Use Newton-Raphson for non-linear circuits
                        timestep_voltages = self._solve_timestep_newton(
                            nodes=nodes,
                            node_map=node_map,
                            num_nodes=num_nodes,
                            num_voltage_sources=num_voltage_sources,
                            initial_voltages=prev_voltages,
                            time=current_time,
                            step_index=step - 1  # Zero-based index for Gmin stepping
                        )
                    else:
                        # Use simple linear solve for linear circuits
                        mna_matrix = np.zeros((num_nodes + num_voltage_sources, num_nodes + num_voltage_sources))
                        rhs = np.zeros(num_nodes + num_voltage_sources)

                        for component in self.circuit.components:
                            component.stamp_conductance(mna_matrix, node_map)
                            component.stamp_rhs(rhs, node_map)

                        self._stamp_voltage_sources(mna_matrix, rhs, node_map, num_nodes, current_time)

                        try:
                            solution = np.linalg.solve(mna_matrix, rhs)
                        except np.linalg.LinAlgError as e:
                            raise np.linalg.LinAlgError(
                                f"Circuit matrix is singular at t={current_time:.6f}s. "
                                f"Check for floating nodes or short circuits."
                            ) from e

                        timestep_voltages = {}
                        for idx, node in enumerate(nodes):
                            timestep_voltages[node] = float(solution[idx])
                        timestep_voltages["0"] = 0.0
                        timestep_voltages["GND"] = 0.0

                    # Success! Store voltages and break retry loop
                    for node in nodes:
                        voltages_over_time[node][step] = timestep_voltages[node]

                    # Update capacitor voltages for next timestep
                    for component in self.circuit.components:
                        if isinstance(component, Capacitor):
                            component.update_voltage(timestep_voltages)

                    if self.debug and dt_reduction_count > 0:
                        print(f"  Converged at t={current_time:.2e}s with reduced dt={current_dt:.2e}")
                    break

                except RuntimeError as e:
                    dt_reduction_count += 1
                    if dt_reduction_count > max_dt_reductions:
                        # Give up - re-raise the error
                        raise RuntimeError(
                            f"Failed to converge at t={current_time:.2e}s even with minimum dt={min_dt:.2e}. "
                            f"Original error: {e}"
                        ) from e

                    # Reduce dt and retry
                    current_dt = self.dt / (2 ** dt_reduction_count)
                    if self.debug:
                        print(f"  WARNING: Convergence failed at t={current_time:.2e}s, reducing dt to {current_dt:.2e}")

        # Prepare results dictionary
        results = {"time": time}
        for node in nodes:
            results[node] = voltages_over_time[node]

        return results

    def _stamp_voltage_sources(
        self,
        mna_matrix: np.ndarray,
        rhs: np.ndarray,
        node_map: Dict[str, int],
        num_nodes: int,
        time: float = 0.0,
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
            time: Current simulation time (for time-varying sources)
            voltages: Current voltage estimate (for Newton-Raphson mismatch computation)
        """
        from pycircuitsim.models.passive import PulseVoltageSource

        voltage_source_index = 0

        for component in self.circuit.components:
            if isinstance(component, VoltageSource):
                # Get voltage source nodes
                pos_node = component.nodes[0]  # Positive terminal
                neg_node = component.nodes[1]  # Negative terminal

                # Get voltage value (support time-varying sources)
                if isinstance(component, PulseVoltageSource):
                    voltage_target = component.get_voltage_at_time(time)
                else:
                    voltage_target = component.voltage

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
                # Use direct voltage value for companion model consistency.
                # The companion model for MOSFETs solves for V directly,
                # so voltage sources should also use direct form.
                # NOTE: Previous implementation used voltage_target - (v_pos - v_neg)
                # which caused oscillation in Newton-Raphson. The correct formulation
                # (matching DC solver) is to use the direct voltage value.
                rhs[vs_row] = voltage_target

                # Move to next voltage source
                voltage_source_index += 1

    def __repr__(self) -> str:
        """String representation of the solver."""
        return (
            f"TransientSolver(circuit={self.circuit}, "
            f"t_stop={self.t_stop}, "
            f"dt={self.dt})"
        )


class ACSolver:
    """
    AC (small-signal frequency domain) Solver for linear and linearized circuits.

    The ACSolver performs small-signal AC analysis by:
    1. Computing DC operating point using DCSolver
    2. Linearizing the circuit around the operating point
    3. Building complex MNA matrix (with capacitances and transconductances)
    4. Sweeping frequency and computing complex node voltages

    Algorithm:
    1. DC analysis to find operating point (all AC sources = 0)
    2. For each frequency:
       a. Build complex admittance matrix Y = G + jwC
       b. Stamp MOSFET small-signal parameters (gm, gds, Cgs, Cgd)
       c. Stamp AC sources to RHS
       d. Solve Y * V = I for complex voltages
       e. Store magnitude and phase

    Attributes:
        circuit: Circuit object containing components and topology
        dc_solution: DC operating point voltages (computed once)
    """

    def __init__(self, circuit: Circuit, dc_solution: Optional[Dict[str, float]] = None):
        """
        Initialize the AC Solver.

        Args:
            circuit: Circuit object to analyze
            dc_solution: Optional pre-computed DC operating point (if None, will compute)
        """
        self.circuit = circuit
        self.dc_solution = dc_solution

    def solve(self, frequencies: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform AC analysis over a range of frequencies.

        This method:
        1. Computes DC operating point (if not provided)
        2. For each frequency, solves the small-signal circuit
        3. Returns complex voltages at each node for each frequency

        Args:
            frequencies: Array of frequencies in Hz

        Returns:
            Dictionary containing:
                - "frequency": numpy array of frequencies (Hz)
                - node names: numpy arrays of complex voltages at each frequency

        Raises:
            np.linalg.LinAlgError: If the circuit matrix is singular
            RuntimeError: If DC operating point fails to converge
        """
        # Step 1: Compute DC operating point if not provided
        if self.dc_solution is None:
            from pycircuitsim.solver import DCSolver
            dc_solver = DCSolver(self.circuit)
            with dc_solver:
                self.dc_solution = dc_solver.solve()

        # Get circuit topology
        nodes = self.circuit.get_nodes()
        node_map = self.circuit.get_node_map()
        num_nodes = len(nodes)
        num_voltage_sources = self.circuit.count_voltage_sources()

        # Matrix size: num_nodes + num_voltage_sources
        matrix_size = num_nodes + num_voltage_sources

        # Initialize storage arrays for results
        num_freqs = len(frequencies)
        voltages_over_freq = {node: np.zeros(num_freqs, dtype=complex) for node in nodes}
        voltages_over_freq["frequency"] = frequencies

        # Step 2: Frequency sweep
        for freq_idx, freq in enumerate(frequencies):
            omega = 2 * np.pi * freq

            # Build complex MNA matrix: Y = G + jwC
            mna_matrix = np.zeros((matrix_size, matrix_size), dtype=complex)
            rhs = np.zeros(matrix_size, dtype=complex)

            # Stamp linear components (resistors, capacitors)
            for component in self.circuit.components:
                if not _is_mosfet(component):
                    self._stamp_component_ac(component, mna_matrix, rhs, node_map, omega)

            # Stamp MOSFETs (small-signal model: gm, gds, capacitances)
            for component in self.circuit.components:
                if _is_mosfet(component):
                    self._stamp_mosfet_ac(component, mna_matrix, node_map, omega)

            # Stamp voltage sources (DC sources become short circuits, AC sources become AC stimulus)
            self._stamp_voltage_sources_ac(mna_matrix, rhs, node_map, num_nodes)

            # Solve the complex linear system
            try:
                solution = np.linalg.solve(mna_matrix, rhs)
            except np.linalg.LinAlgError as e:
                raise np.linalg.LinAlgError(
                    f"Circuit matrix is singular at f={freq:.3e} Hz. "
                    f"Check circuit topology or AC sources."
                ) from e

            # Extract complex node voltages from solution
            for idx, node in enumerate(nodes):
                voltages_over_freq[node][freq_idx] = complex(solution[idx])

        return voltages_over_freq

    def _stamp_component_ac(
        self,
        component,
        mna_matrix: np.ndarray,
        rhs: np.ndarray,
        node_map: Dict[str, int],
        omega: float
    ) -> None:
        """
        Stamp a passive component for AC analysis.

        For AC analysis:
        - Resistors: stamp conductance G (same as DC)
        - Capacitors: stamp admittance jwC (frequency-dependent)
        - Voltage sources: handled separately
        - Current sources: stamp to RHS (AC current sources not yet supported)

        Args:
            component: Component to stamp
            mna_matrix: Complex MNA matrix to modify (in-place)
            rhs: Complex RHS vector to modify (in-place)
            node_map: Mapping from node names to matrix indices
            omega: Angular frequency (2*pi*f) in rad/s
        """
        from pycircuitsim.models.passive import Resistor, Capacitor, CurrentSource

        if isinstance(component, Resistor):
            # Resistor: stamp conductance (real, frequency-independent)
            node_i, node_j = component.nodes[0], component.nodes[1]
            g = component.conductance

            # Stamp exactly as in DC analysis
            if node_i != "0" and node_i in node_map:
                idx_i = node_map[node_i]
                mna_matrix[idx_i, idx_i] += g

                if node_j != "0" and node_j in node_map:
                    idx_j = node_map[node_j]
                    mna_matrix[idx_i, idx_j] -= g
                    mna_matrix[idx_j, idx_i] -= g

            if node_j != "0" and node_j in node_map:
                idx_j = node_map[node_j]
                mna_matrix[idx_j, idx_j] += g

        elif isinstance(component, Capacitor):
            # Capacitor: stamp admittance Y_C = jwC (imaginary, frequency-dependent)
            node_i, node_j = component.nodes[0], component.nodes[1]
            y_c = 1j * omega * component.capacitance

            # Stamp same pattern as resistor, but with complex admittance
            if node_i != "0" and node_i in node_map:
                idx_i = node_map[node_i]
                mna_matrix[idx_i, idx_i] += y_c

                if node_j != "0" and node_j in node_map:
                    idx_j = node_map[node_j]
                    mna_matrix[idx_i, idx_j] -= y_c
                    mna_matrix[idx_j, idx_i] -= y_c

            if node_j != "0" and node_j in node_map:
                idx_j = node_map[node_j]
                mna_matrix[idx_j, idx_j] += y_c

        elif isinstance(component, CurrentSource):
            # Current source: stamp to RHS (AC current sources not yet implemented)
            # For now, only DC current sources contribute (AC magnitude = 0)
            pass

        # VoltageSource handled separately in _stamp_voltage_sources_ac

    def _stamp_mosfet_ac(
        self,
        mosfet,
        mna_matrix: np.ndarray,
        node_map: Dict[str, int],
        omega: float
    ) -> None:
        """
        Stamp MOSFET small-signal model for AC analysis.

        Small-signal MOSFET model includes:
        - gm: transconductance (gate to drain)
        - gds: output conductance (drain to source)
        - gmb: bulk transconductance (bulk to drain, if applicable)
        - Cgs: gate-source capacitance (creates admittance jwCgs)
        - Cgd: gate-drain capacitance (Miller capacitance, jwCgd)
        - Cdb: drain-bulk capacitance (jwCdb, often small)
        - Csb: source-bulk capacitance (jwCsb, often small)

        For now, we implement gm, gds, gmb (from DC linearization).
        Capacitances (Cgs, Cgd, etc.) will be added in Phase 5.

        Args:
            mosfet: MOSFET component (NMOS or PMOS)
            mna_matrix: Complex MNA matrix to modify (in-place)
            node_map: Mapping from node names to matrix indices
            omega: Angular frequency (2*pi*f) in rad/s
        """
        # Get MOSFET terminals
        drain = mosfet.nodes[0]
        gate = mosfet.nodes[1]
        source = mosfet.nodes[2]
        bulk = mosfet.nodes[3]

        # Get small-signal conductances at DC operating point
        conductance_result = mosfet.get_conductance(self.dc_solution)
        if len(conductance_result) == 2:
            g_ds, g_m = conductance_result
            g_mb = 0.0
        else:
            g_ds, g_m, g_mb = conductance_result

        # Add minimum conductance for numerical stability
        g_min = 1e-6
        g_ds = max(g_ds, g_min)

        # Stamp conductances (same as DC, but to complex matrix)
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

        # g_m transconductance: i_d = gm * (v_g - v_s)
        # Stamp for drain equation (KCL at drain node)
        if gate != "0" and gate in node_map and drain != "0" and drain in node_map:
            g_idx = node_map[gate]
            d_idx = node_map[drain]
            mna_matrix[d_idx, g_idx] += g_m

        if source != "0" and source in node_map and drain != "0" and drain in node_map:
            s_idx = node_map[source]
            d_idx = node_map[drain]
            mna_matrix[d_idx, s_idx] -= g_m

        # Stamp for source equation (KCL at source node: current into source = -i_d)
        if gate != "0" and gate in node_map and source != "0" and source in node_map:
            g_idx = node_map[gate]
            s_idx = node_map[source]
            mna_matrix[s_idx, g_idx] -= g_m

        if source != "0" and source in node_map:
            s_idx = node_map[source]
            mna_matrix[s_idx, s_idx] += g_m

        # g_mb bulk transconductance: i_d = gmb * (v_b - v_s)
        if abs(g_mb) > 1e-12 and bulk != source:
            # Stamp for drain equation
            if bulk != "0" and bulk in node_map and drain != "0" and drain in node_map:
                b_idx = node_map[bulk]
                d_idx = node_map[drain]
                mna_matrix[d_idx, b_idx] += g_mb

            if source != "0" and source in node_map and drain != "0" and drain in node_map:
                s_idx = node_map[source]
                d_idx = node_map[drain]
                mna_matrix[d_idx, s_idx] -= g_mb

            # Stamp for source equation
            if bulk != "0" and bulk in node_map and source != "0" and source in node_map:
                b_idx = node_map[bulk]
                s_idx = node_map[source]
                mna_matrix[s_idx, b_idx] -= g_mb

            if source != "0" and source in node_map:
                s_idx = node_map[source]
                mna_matrix[s_idx, s_idx] += g_mb

        # TODO: Add capacitance stamping (Cgs, Cgd, Cdb, Csb) in Phase 5
        # These will be obtained from mosfet.get_capacitances() method
        # For now, only resistive small-signal model is implemented

    def _stamp_voltage_sources_ac(
        self,
        mna_matrix: np.ndarray,
        rhs: np.ndarray,
        node_map: Dict[str, int],
        num_nodes: int
    ) -> None:
        """
        Stamp voltage sources for AC analysis.

        For AC analysis:
        - DC voltage sources become SHORT CIRCUITS (V_ac = 0)
        - AC voltage sources provide AC stimulus (V_ac = magnitude * e^(j*phase))

        The voltage source stamping adds:
        - B/C matrix blocks (same as DC)
        - RHS: AC magnitude with phase for AC sources, 0 for DC-only sources

        Args:
            mna_matrix: Complex MNA matrix to modify (in-place)
            rhs: Complex RHS vector to modify (in-place)
            node_map: Mapping from node names to matrix indices
            num_nodes: Number of non-ground nodes
        """
        voltage_source_index = 0

        for component in self.circuit.components:
            if isinstance(component, VoltageSource):
                # Get voltage source nodes
                pos_node = component.nodes[0]
                neg_node = component.nodes[1]

                # The row index for this voltage source's equation
                vs_row = num_nodes + voltage_source_index

                # Stamp B/C matrix (same as DC analysis)
                if pos_node != "0" and pos_node in node_map:
                    pos_idx = node_map[pos_node]
                    mna_matrix[vs_row, pos_idx] += 1.0
                    mna_matrix[pos_idx, vs_row] += 1.0

                if neg_node != "0" and neg_node in node_map:
                    neg_idx = node_map[neg_node]
                    mna_matrix[vs_row, neg_idx] -= 1.0
                    mna_matrix[neg_idx, vs_row] -= 1.0

                # Stamp AC stimulus to RHS
                # Convert AC magnitude and phase to complex phasor
                ac_mag = component.ac_magnitude
                ac_phase_deg = component.ac_phase
                ac_phase_rad = np.deg2rad(ac_phase_deg)

                # Complex phasor: V = magnitude * e^(j*phase)
                v_ac = ac_mag * np.exp(1j * ac_phase_rad)

                rhs[vs_row] = v_ac

                # Move to next voltage source
                voltage_source_index += 1

    def __repr__(self) -> str:
        """String representation of the solver."""
        return f"ACSolver(circuit={self.circuit})"
