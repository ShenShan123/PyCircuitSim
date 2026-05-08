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
from scipy.sparse import lil_matrix, issparse
from scipy.sparse.linalg import spsolve
from pycircuitsim.circuit import Circuit
from pycircuitsim.models.passive import VoltageSource, Capacitor
from pycircuitsim.logger import Logger, IterationInfo


def _create_mna_matrix(size: int) -> lil_matrix:
    """Create a sparse MNA matrix (LIL format for efficient element-by-element assembly)."""
    return lil_matrix((size, size), dtype=np.float64)


def _solve_mna(mna_matrix, rhs: np.ndarray) -> np.ndarray:
    """Solve MNA system Ax=b, using sparse solver if matrix is sparse."""
    if issparse(mna_matrix):
        return spsolve(mna_matrix.tocsr(), rhs)
    return np.linalg.solve(mna_matrix, rhs)

# --- Module-level MOSFET helpers (used by both DCSolver and TransientSolver) ---

def _mosfet_types() -> tuple:
    """Return tuple of all MOSFET classes (BSIM-CMG, NN, BSIM-AR)."""
    types = []
    try:
        from pycircuitsim.models.mosfet_cmg import NMOS_CMG, PMOS_CMG
        types.extend([NMOS_CMG, PMOS_CMG])
    except ImportError:
        pass
    try:
        from pycircuitsim.models.mosfet_directnet import NMOS_NN, PMOS_NN
        types.extend([NMOS_NN, PMOS_NN])
    except ImportError:
        pass
    try:
        from pycircuitsim.models.mosfet_bsimar import NMOS_BSIMAR, PMOS_BSIMAR
        types.extend([NMOS_BSIMAR, PMOS_BSIMAR])
    except ImportError:
        pass
    return tuple(types)


def _pmos_types() -> tuple:
    """Return tuple of all PMOS classes (BSIM-CMG, DirectNet, BSIM-AR)."""
    types = []
    try:
        from pycircuitsim.models.mosfet_cmg import PMOS_CMG
        types.append(PMOS_CMG)
    except ImportError:
        pass
    try:
        from pycircuitsim.models.mosfet_directnet import PMOS_NN
        types.append(PMOS_NN)
    except ImportError:
        pass
    try:
        from pycircuitsim.models.mosfet_bsimar import PMOS_BSIMAR
        types.append(PMOS_BSIMAR)
    except ImportError:
        pass
    return tuple(types)


def _is_mosfet(component) -> bool:
    """Check if component is any MOSFET variant."""
    return isinstance(component, _mosfet_types())


def _is_pmos(component) -> bool:
    """Check if component is any PMOS variant."""
    return isinstance(component, _pmos_types())


def _has_non_linear(circuit: Circuit) -> bool:
    """Check if circuit contains non-linear components (MOSFETs)."""
    return any(_is_mosfet(c) for c in circuit.components)


def _has_nn_device(circuit: Circuit) -> bool:
    """Check if circuit contains any NN compact-model device (LEVEL>=73)."""
    from pycircuitsim.models.mosfet_directnet import _MOSFETNNBase
    return any(isinstance(c, _MOSFETNNBase) for c in circuit.components)


def _stamp_mosfet_dc(
    mosfet,
    mna_matrix: np.ndarray,
    rhs: np.ndarray,
    node_map: Dict[str, int],
    voltages: Dict[str, float],
    gmin: float,
) -> None:
    """Stamp MOSFET conductance and NR current source to MNA matrix.

    Shared by DCSolver._stamp_mosfet and TransientSolver._stamp_mosfet_transient.
    The NR linearization stamps g_ds, g_m, g_mb conductances and the equivalent
    current source i_eq = I_leaving(V0) - g_ds*V_ds0 - g_m*V_gs0 - g_mb*V_bs0.
    """
    drain, gate, source, bulk = mosfet.nodes

    # Get conductances (3-tuple: g_ds, g_m, g_mb)
    g_ds, g_m, g_mb = mosfet.get_conductance(voltages)

    i_ds = mosfet.calculate_current(voltages)
    g_ds = max(g_ds, gmin)  # SPICE GMIN floor

    # --- Stamp conductances ---
    # g_ds between drain and source
    if drain != "0" and drain in node_map:
        d_idx = node_map[drain]
        mna_matrix[d_idx, d_idx] += g_ds
    if source != "0" and source in node_map:
        s_idx = node_map[source]
        mna_matrix[s_idx, s_idx] += g_ds
    if drain != "0" and drain in node_map and source != "0" and source in node_map:
        d_idx, s_idx = node_map[drain], node_map[source]
        mna_matrix[d_idx, s_idx] -= g_ds
        mna_matrix[s_idx, d_idx] -= g_ds

    # g_m transconductance (VCCS: gate controls drain current)
    if gate != "0" and gate in node_map and drain != "0" and drain in node_map:
        mna_matrix[node_map[drain], node_map[gate]] += g_m
    if drain != "0" and drain in node_map and source != "0" and source in node_map:
        mna_matrix[node_map[drain], node_map[source]] -= g_m
    if gate != "0" and gate in node_map and source != "0" and source in node_map:
        mna_matrix[node_map[source], node_map[gate]] -= g_m
    if source != "0" and source in node_map:
        mna_matrix[node_map[source], node_map[source]] += g_m

    # g_mb bulk transconductance: i_d = gmb * (v_b - v_s)
    # Full 4-entry VCCS stamp (matching AC solver pattern at lines 2002-2023).
    if abs(g_mb) > 1e-12 and bulk != source:
        # Stamp for drain equation
        if bulk != "0" and bulk in node_map and drain != "0" and drain in node_map:
            mna_matrix[node_map[drain], node_map[bulk]] += g_mb
        if source != "0" and source in node_map and drain != "0" and drain in node_map:
            mna_matrix[node_map[drain], node_map[source]] -= g_mb
        # Stamp for source equation
        if bulk != "0" and bulk in node_map and source != "0" and source in node_map:
            mna_matrix[node_map[source], node_map[bulk]] -= g_mb
        if source != "0" and source in node_map:
            mna_matrix[node_map[source], node_map[source]] += g_mb

    # --- Stamp NR equivalent current source to RHS ---
    v_d, v_g = voltages.get(drain, 0.0), voltages.get(gate, 0.0)
    v_s, v_b = voltages.get(source, 0.0), voltages.get(bulk, 0.0)
    v_ds, v_gs, v_bs = v_d - v_s, v_g - v_s, v_b - v_s

    # Convert to "leaving drain" convention:
    # NMOS: i_ds positive = leaving drain; PMOS: i_ds positive = INTO drain
    i_leaving = -i_ds if _is_pmos(mosfet) else i_ds
    i_eq = i_leaving - g_ds * v_ds - g_m * v_gs - g_mb * v_bs

    if drain != "0" and drain in node_map:
        rhs[node_map[drain]] -= i_eq
    if source != "0" and source in node_map:
        rhs[node_map[source]] += i_eq


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
                 damping_factor: float = 1.0,
                 reltol: float = 1e-4, vntol: float = 1e-7, gmin: float = 1e-12,
                 use_gmin_stepping: bool = False, force_ic: bool = False):
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
            reltol: Relative convergence tolerance (default: 1e-4, tighter than SPICE 1e-3)
            vntol: Absolute voltage tolerance (default: 1e-7 V, tighter than SPICE 1e-6)
            gmin: Minimum MOSFET channel conductance (SPICE GMIN, default: 1e-12 S)
            use_gmin_stepping: Enable DC GMIN stepping for bistable convergence (default: False)
            force_ic: Enforce .ic as voltage constraints, not just initial guess (default: False)
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
        self.reltol = reltol
        self.vntol = vntol
        self.gmin = gmin
        self.use_gmin_stepping = use_gmin_stepping
        self.force_ic = force_ic
        self.last_solution: Optional[Dict[str, float]] = None
        self._owns_logger = False  # Track if we created the logger (for cleanup)
        # V5 Phase A retry-design: True if the last `solve()` reached
        # SPICE convergence AND the returned voltage vector is finite.
        # The simulation orchestrator inspects this to decide whether
        # to retry with GMIN stepping enabled.
        self._last_solve_converged: bool = False

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
            # Linear solve either succeeds or raises — if we got here,
            # converged.
            self._last_solve_converged = True

        # Store the solution for potential reuse
        self.last_solution = solution.copy()

        return solution

    def _has_non_linear_components(self) -> bool:
        """Check if circuit contains non-linear components (MOSFETs)."""
        return _has_non_linear(self.circuit)

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
        mna_matrix = _create_mna_matrix(matrix_size)
        rhs = np.zeros(matrix_size)

        # Stamp conductances (G matrix) and current sources (RHS)
        for component in self.circuit.components:
            component.stamp_conductance(mna_matrix, node_map)
            component.stamp_rhs(rhs, node_map)

        # Handle voltage sources (B and C matrices)
        self._stamp_voltage_sources(mna_matrix, rhs, node_map, num_nodes, voltages=None)

        # Solve the linear system
        try:
            solution = _solve_mna(mna_matrix, rhs)
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

    def _apply_gmin_stepping(self, mna_matrix, node_map: Dict[str, int], gmin: float) -> None:
        """Add minimum conductance from each node to ground for convergence aid."""
        for node, idx in node_map.items():
            mna_matrix[idx, idx] += gmin

    def _solve_newton(self) -> Dict[str, float]:
        """
        Solve non-linear circuit using Newton-Raphson iteration.

        Features:
        - Source stepping homotopy for improved convergence
        - GMIN stepping (opt-in) for bistable circuits (SRAM latches)
        - Adaptive damping with supply-relative thresholds
        - Oscillation detection with averaged-solution acceptance
        - Hard .ic mode (force_ic) via temporary voltage source constraints

        Returns:
            Dictionary mapping node names to voltage values

        Raises:
            RuntimeError: If Newton-Raphson fails to converge
            np.linalg.LinAlgError: If the circuit matrix is singular
        """
        # Reset damping to default for each solve
        self.damping_factor = 1.0

        # Get circuit topology
        nodes = self.circuit.get_nodes()
        node_map = self.circuit.get_node_map()
        num_nodes = len(nodes)

        # --- Force IC: add temporary voltage source constraints ---
        _ic_temp_sources: List[VoltageSource] = []
        if self.force_ic and self.initial_guess:
            # Find nodes already constrained by existing voltage sources
            vs_constrained = set()
            for comp in self.circuit.components:
                if isinstance(comp, VoltageSource):
                    if comp.nodes[1] in ("0", "GND"):
                        vs_constrained.add(comp.nodes[0])
                    elif comp.nodes[0] in ("0", "GND"):
                        vs_constrained.add(comp.nodes[1])
            # Add temp VS for IC nodes not already constrained
            for node_name, voltage in self.initial_guess.items():
                if (node_name not in ("0", "GND")
                        and node_name not in vs_constrained
                        and node_name in node_map):
                    vs = VoltageSource(f"_V_ic_{node_name}", [node_name, "0"], voltage)
                    self.circuit.components.append(vs)
                    _ic_temp_sources.append(vs)

        num_voltage_sources = self.circuit.count_voltage_sources()
        matrix_size = num_nodes + num_voltage_sources

        # Store original voltage source values for source stepping
        original_voltages = []
        for component in self.circuit.components:
            if isinstance(component, VoltageSource):
                original_voltages.append(component.voltage)

        # Estimate supply voltage for supply-relative damping threshold
        max_vs_voltage = max((abs(v) for v in original_voltages), default=1.0) or 1.0

        # --- GMIN stepping schedule ---
        # V5 Phase A retry-design (2026-05-07): reduced from the 4-level
        # schedule [1e-6, 1e-8, 1e-10, self.gmin] to a 2-level schedule
        # [1e-8, self.gmin] when retry is invoked. Source stepping is
        # tried once per level, so 4 levels x 5 steps = 20 NR sweeps per
        # GMIN-on solve, which dominated the verify wall-time when GMIN
        # was default-on. 2 levels keeps the homotopy useful for the
        # cells that need it (TSMC5 BSIMAR-M VTC trip-point overflow)
        # while halving the slow-path cost.
        if self.use_gmin_stepping:
            gmin_schedule = [1e-8, self.gmin]
        else:
            gmin_schedule = [self.gmin]

        voltages: Dict[str, float] = {}
        final_converged = False

        for gmin_level in gmin_schedule:
            # Source stepping: gradually increase voltage source values
            num_steps = self.source_stepping_steps if (self._has_non_linear_components() and self.use_source_stepping) else 1
            for step in range(num_steps):
                # Scale voltage sources
                scale = (step + 1) / num_steps
                vs_idx = 0
                for component in self.circuit.components:
                    if isinstance(component, VoltageSource):
                        component.voltage = original_voltages[vs_idx] * scale
                        vs_idx += 1

                # Initial guess
                if step == 0 and gmin_level == gmin_schedule[0]:
                    if self.initial_guess is not None:
                        voltages = {node: 0.0 for node in nodes}
                        for node, voltage in self.initial_guess.items():
                            if node in voltages:
                                voltages[node] = voltage
                    else:
                        voltages = {node: 0.0 for node in nodes}
                voltages["0"] = 0.0
                voltages["GND"] = 0.0

                # --- Adaptive damping state ---
                damping = 1.0
                prev_max_delta = float('inf')
                stuck_counter = 0
                voltage_history: List[Dict[str, float]] = []

                # Newton-Raphson iteration for this source step
                nr_converged = False
                max_change = 0.0
                for iteration in range(self.max_iterations // num_steps):
                    # Initialize MNA matrix and RHS vector
                    mna_matrix = _create_mna_matrix(matrix_size)
                    rhs = np.zeros(matrix_size)

                    # Stamp linear components (resistors, current sources)
                    for component in self.circuit.components:
                        if not _is_mosfet(component):
                            component.stamp_conductance(mna_matrix, node_map)
                            component.stamp_rhs(rhs, node_map)

                    # Stamp MOSFET conductances and currents
                    for component in self.circuit.components:
                        if _is_mosfet(component):
                            self._stamp_mosfet(component, mna_matrix, rhs, node_map, voltages)

                    # Handle voltage sources (B and C matrices)
                    self._stamp_voltage_sources(mna_matrix, rhs, node_map, num_nodes, voltages=voltages)

                    # Apply node-level GMIN (conductance from each node to ground)
                    if gmin_level > self.gmin:
                        self._apply_gmin_stepping(mna_matrix, node_map, gmin_level)

                    # Solve the MNA system
                    try:
                        solution = _solve_mna(mna_matrix, rhs)
                    except (np.linalg.LinAlgError, RuntimeError) as e:
                        raise np.linalg.LinAlgError(
                            f"Circuit matrix is singular at source step {step + 1}, iteration {iteration + 1}. "
                            f"Check circuit topology or initial guess."
                        ) from e

                    # V5' trust-region: cap NN per-iteration |ΔV| at one supply rail to kill NR runaway.
                    if _has_nn_device(self.circuit):
                        for idx, node in enumerate(nodes):
                            d = solution[idx] - voltages[node]
                            if d > max_vs_voltage:
                                solution[idx] = voltages[node] + max_vs_voltage
                            elif d < -max_vs_voltage:
                                solution[idx] = voltages[node] - max_vs_voltage

                    # Calculate deltas
                    max_delta = 0.0
                    deltas: Dict[str, float] = {}
                    for idx, node in enumerate(nodes):
                        delta_v = solution[idx] - voltages[node]
                        max_delta = max(max_delta, abs(delta_v))

                    # Track voltage history for oscillation detection
                    voltage_snapshot = {node: solution[idx] for idx, node in enumerate(nodes)}
                    voltage_history.append(voltage_snapshot)
                    if len(voltage_history) > 5:
                        voltage_history.pop(0)

                    # --- Adaptive damping ---
                    improvement_ratio = max_delta / (prev_max_delta + 1e-15)

                    if improvement_ratio > 0.9 and iteration > 15:
                        stuck_counter += 1
                        if stuck_counter >= 2:
                            damping = max(0.1, damping * 0.8)
                            stuck_counter = 0
                    elif improvement_ratio < 0.5:
                        damping = min(1.0, damping * 1.1)
                        stuck_counter = 0
                    else:
                        stuck_counter = 0

                    # Supply-relative large-delta damping
                    if max_delta > 0.5 * max_vs_voltage:
                        damping = min(damping, 0.5)
                    elif max_delta < 0.05 * max_vs_voltage:
                        damping = min(1.0, damping * 1.2)

                    prev_max_delta = max_delta

                    # Identify voltage-source-constrained nodes
                    vs_constrained_nodes = set()
                    for component in self.circuit.components:
                        if isinstance(component, VoltageSource):
                            pos_node = component.nodes[0]
                            neg_node = component.nodes[1]
                            if neg_node == "0":
                                vs_constrained_nodes.add(pos_node)
                            elif pos_node == "0":
                                vs_constrained_nodes.add(neg_node)

                    # Update voltages with damping
                    max_change = 0.0
                    for idx, node in enumerate(nodes):
                        old_voltage = voltages[node]
                        new_voltage_solution = solution[idx]

                        if node in vs_constrained_nodes:
                            new_voltage = new_voltage_solution
                        else:
                            new_voltage = damping * new_voltage_solution + (1.0 - damping) * old_voltage

                        deltas[node] = abs(new_voltage - old_voltage)
                        voltages[node] = new_voltage
                        max_change = max(max_change, abs(new_voltage - old_voltage))

                    # Log iteration if logger is available
                    if self.logger:
                        currents = {}
                        conductances = {}
                        for comp in self.circuit.components:
                            try:
                                current = comp.calculate_current(voltages)
                                currents[comp.name] = current
                                if _is_mosfet(comp):
                                    g_ds, g_m, g_mb = comp.get_conductance(voltages)
                                    conductances[comp.name] = {"gm": g_m, "gds": g_ds, "gmb": g_mb}
                            except (NotImplementedError, AttributeError):
                                pass

                        iter_info = IterationInfo(
                            iteration=iteration,
                            voltages=voltages.copy(),
                            deltas=deltas,
                            currents=currents,
                            conductances=conductances
                        )
                        self.logger.log_iteration(point_num=0, iter_info=iter_info)

                    # Check convergence: SPICE-standard RELTOL + VNTOL
                    all_converged = True
                    for node in nodes:
                        dv = deltas.get(node, 0.0)
                        v_new = voltages[node]
                        threshold = self.vntol + self.reltol * max(abs(v_new), abs(v_new - dv))
                        if dv >= threshold:
                            all_converged = False
                            break

                    if all_converged:
                        if self.logger:
                            self.logger.log_convergence(
                                point_num=0,
                                converged=True,
                                iterations=iteration + 1,
                                tolerance=max_change
                            )
                        nr_converged = True
                        break

                # --- Oscillation detection (NR exhausted without converging) ---
                if not nr_converged and len(voltage_history) >= 3:
                    max_rel_variance = 0.0
                    avg_voltages: Dict[str, float] = {}
                    for node in nodes:
                        values = [s.get(node, 0.0) for s in voltage_history[-3:]]
                        avg_voltages[node] = sum(values) / 3.0
                        variance = max(values) - min(values)
                        v_abs = max(abs(v) for v in values) if values else 0.0
                        threshold = self.vntol + self.reltol * v_abs
                        max_rel_variance = max(max_rel_variance, variance / (threshold + 1e-30))

                    if max_rel_variance < 10.0:
                        # Oscillating within tolerance — accept averaged solution
                        for node in nodes:
                            voltages[node] = avg_voltages[node]
                        voltages["0"] = 0.0
                        voltages["GND"] = 0.0
                        nr_converged = True

                if nr_converged:
                    final_converged = True

            # GMIN continuation: use converged solution as initial guess for next level
            if final_converged:
                self.initial_guess = voltages.copy()

        # Restore original voltage source values
        vs_idx = 0
        for component in self.circuit.components:
            if isinstance(component, VoltageSource):
                component.voltage = original_voltages[vs_idx]
                vs_idx += 1

        # --- Force IC cleanup ---
        if _ic_temp_sources:
            for vs in _ic_temp_sources:
                self.circuit.components.remove(vs)
            # Re-solve without IC constraints using constrained result as guess
            saved_force_ic = self.force_ic
            self.force_ic = False
            self.initial_guess = voltages.copy()
            num_voltage_sources = self.circuit.count_voltage_sources()
            matrix_size = num_nodes + num_voltage_sources
            try:
                voltages = self._solve_newton()
            finally:
                self.force_ic = saved_force_ic
            return voltages

        # Extract and store voltage source currents from final operating point
        mna_matrix_final = _create_mna_matrix(matrix_size)
        rhs_final = np.zeros(matrix_size)

        for component in self.circuit.components:
            if not _is_mosfet(component):
                component.stamp_conductance(mna_matrix_final, node_map)
                component.stamp_rhs(rhs_final, node_map)

        for component in self.circuit.components:
            if _is_mosfet(component):
                self._stamp_mosfet(component, mna_matrix_final, rhs_final, node_map, voltages)

        self._stamp_voltage_sources(mna_matrix_final, rhs_final, node_map, num_nodes, voltages=voltages)

        try:
            solution_final = _solve_mna(mna_matrix_final, rhs_final)
            self._store_source_currents(solution_final, nodes)
        except (np.linalg.LinAlgError, RuntimeError):
            pass

        # V5 Phase A retry-design: surface convergence + finite-output
        # status so the simulation orchestrator can decide whether to
        # retry with GMIN homotopy on. Bad outputs (NaN / Inf / >1e10)
        # also count as failures because the DC solver does not raise
        # on NR exhaustion — it just returns the last (possibly garbage)
        # voltage vector.
        finite_voltages = all(
            (np.isfinite(v) and abs(v) < 1.0e10)
            for v in voltages.values()
        )
        self._last_solve_converged = bool(final_converged and finite_voltages)

        return voltages

    def _stamp_mosfet(
        self,
        mosfet,
        mna_matrix: np.ndarray,
        rhs: np.ndarray,
        node_map: Dict[str, int],
        voltages: Dict[str, float],
    ) -> None:
        """Stamp MOSFET conductance and NR current source to MNA matrix (DC)."""
        _stamp_mosfet_dc(mosfet, mna_matrix, rhs, node_map, voltages, self.gmin)

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
                 pseudo_transient_cap: float = 1e-12,
                 nr_tolerance: float = 1e-7,
                 reltol: float = 1e-4, vntol: float = 1e-7, gmin: float = 1e-12,
                 max_substeps: int = 1, lte_safety_factor: float = 0.5):
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
            nr_tolerance: Newton-Raphson convergence tolerance (default: 1e-7 V)
            reltol: Relative convergence tolerance (default: 1e-4, tighter than SPICE 1e-3)
            vntol: Absolute voltage tolerance (default: 1e-7 V, tighter than SPICE 1e-6)
            gmin: Minimum MOSFET channel conductance (SPICE GMIN, default: 1e-12 S)
            max_substeps: Max LTE-adaptive sub-steps per output interval (1=disabled, default: 1)
            lte_safety_factor: LTE acceptance threshold (default: 0.5)

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

        # Newton-Raphson convergence tolerance
        self.nr_tolerance = nr_tolerance

        # SPICE-standard convergence parameters
        self.reltol = reltol
        self.vntol = vntol
        self.gmin = gmin

        # LTE adaptive sub-stepping parameters
        self.max_substeps = max_substeps
        self.lte_safety_factor = lte_safety_factor

        # Active internal timestep (may differ from self.dt during sub-stepping)
        self._current_dt = dt

        # Integration method: 'be', 'trap', or 'bdf2'
        self._integration_method = 'be'

        # Store pseudo-capacitor references for cleanup
        self._pseudo_capacitors: List = []

        # V5 Phase A — A3: dt-halve fallback event log. Each entry is a
        # dict with {step, sub_idx, sim_time, halve_num, dt_before, dt_after,
        # error_msg}. Read by verification scripts after a transient run
        # to flag cells that needed >1 halving.
        self._dt_halve_events: List[Dict] = []

    def _has_non_linear_components(self) -> bool:
        """Check if circuit contains non-linear components (MOSFETs)."""
        return _has_non_linear(self.circuit)

    def _has_nn_devices(self) -> bool:
        """Return True if the circuit contains any NN compact-model device
        (LEVEL >= 73). Used to gate the V5 Phase A dt-halve fallback so
        BSIM-CMG (LEVEL=72) transients keep their existing behaviour.
        """
        return _has_nn_device(self.circuit)

    def _add_pseudo_capacitors(self) -> None:
        """Add pseudo-capacitors scaled to circuit capacitance for initialization."""
        from pycircuitsim.models.passive import Capacitor

        # Auto-detect max circuit capacitance
        max_circuit_cap = 0.0
        for component in self.circuit.components:
            if isinstance(component, Capacitor) and not component.name.startswith("_pseudo_"):
                max_circuit_cap = max(max_circuit_cap, component.capacitance)

        # Scale pseudo-cap: 5x the largest circuit cap, or use user-specified value
        if max_circuit_cap > 0 and self.pseudo_transient_cap > 10 * max_circuit_cap:
            effective_cap = 5.0 * max_circuit_cap
            if self.debug:
                print(f"  Auto-scaling pseudo-cap: {self.pseudo_transient_cap:.2e} -> "
                      f"{effective_cap:.2e} (5x max circuit cap {max_circuit_cap:.2e})")
        else:
            effective_cap = self.pseudo_transient_cap

        nodes = self.circuit.get_nodes()
        pseudo_cap_idx = 0
        for node in nodes:
            cap = Capacitor(f"_pseudo_{pseudo_cap_idx}", [node, "0"], effective_cap)
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
        step_index: int = 0,
        use_gmin: bool = True
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
        # Matrix size: num_nodes + num_voltage_sources
        matrix_size = num_nodes + num_voltage_sources

        # Use previous timestep's voltages as initial guess
        voltages = initial_voltages.copy()

        # Newton-Raphson parameters (aligned with DC solver)
        tolerance = self.nr_tolerance
        max_iterations = 200  # Increased from 100 for difficult convergence

        # Calculate Gmin value for this timestep (if enabled)
        gmin = self.gmin_final
        if use_gmin and self.use_gmin_stepping and step_index < self.gmin_steps:
            # Exponential decay from gmin_initial to gmin_final
            alpha = step_index / (self.gmin_steps - 1) if self.gmin_steps > 1 else 1.0
            gmin = self.gmin_initial * (1 - alpha) + self.gmin_final * alpha
            if self.debug:
                print(f"  Gmin stepping: step {step_index}, gmin = {gmin:.2e}")

        # Start with full damping (no damping); reduce if needed during iteration
        damping = 1.0

        # Track previous max_delta for adaptive damping
        prev_max_delta = float('inf')
        stuck_counter = 0  # Count iterations with minimal improvement

        # Track recent voltages for oscillation detection
        voltage_history = []

        # Debug: Track convergence behavior (if enabled)
        debug_log = [] if self.debug else None

        for iteration in range(max_iterations):
            # Build MNA matrix and RHS
            mna_matrix = _create_mna_matrix(matrix_size)
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
                solution = _solve_mna(mna_matrix, rhs)
            except (np.linalg.LinAlgError, RuntimeError):
                raise RuntimeError(
                    f"Circuit matrix is singular at t={time:.6e}s during Newton-Raphson iteration {iteration+1}"
                )

            # V5' trust-region: cap NN per-iteration |ΔV| at one supply rail to kill NR runaway.
            if _has_nn_device(self.circuit):
                vdd_cap = max(
                    (abs(c.voltage) for c in self.circuit.components if isinstance(c, VoltageSource)),
                    default=1.0,
                ) or 1.0
                for idx, node in enumerate(nodes):
                    d = solution[idx] - voltages[node]
                    if d > vdd_cap:
                        solution[idx] = voltages[node] + vdd_cap
                    elif d < -vdd_cap:
                        solution[idx] = voltages[node] - vdd_cap

            # Extract voltages from solution (matches DC solver approach)
            # Solution contains NEW voltages, not deltas (due to MNA formulation)
            max_delta = 0.0
            deltas = {}

            # Identify voltage-source-constrained nodes (exempt from damping)
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

            # Check convergence: SPICE-standard RELTOL + VNTOL
            all_converged = True
            for idx, node in enumerate(nodes):
                dv = abs(solution[idx] - voltages[node])
                v_abs = max(abs(solution[idx]), abs(voltages[node]))
                threshold = self.vntol + self.reltol * v_abs
                if dv >= threshold:
                    all_converged = False
                    break

            if all_converged:
                # Converged! Use new voltages directly
                for idx, node in enumerate(nodes):
                    voltages[node] = solution[idx]
                self._last_nr_iterations = iteration + 1
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

                # Check oscillation: variance relative to SPICE tolerance
                max_rel_variance = 0.0
                for node in nodes:
                    values = [s.get(node, 0.0) for s in voltage_history[-3:]]
                    variance = max(values) - min(values)
                    v_abs = max(abs(v) for v in values) if values else 0.0
                    threshold = self.vntol + self.reltol * v_abs
                    max_rel_variance = max(max_rel_variance, variance / (threshold + 1e-30))

                # Accept if oscillation is within 10x convergence tolerance
                if max_rel_variance < 10.0:
                    if self.debug:
                        print(f"  WARNING: Newton-Raphson oscillating at t={time:.6e}s")
                        print(f"  Max variance = {max_rel_variance:.2e} (accepting averaged solution)")
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
        """Stamp MOSFET conductance/current (DC part) + charge-based capacitance for transient."""
        # DC conductance + NR current source stamping (shared with DCSolver)
        _stamp_mosfet_dc(mosfet, mna_matrix, rhs, node_map, voltages, self.gmin)

        # --- Charge-based intrinsic capacitance stamping ---
        # Supports BE, Trapezoidal, and BDF-2 integration methods.
        # Theory: I_t(n+1) = coeff * Q_t(n+1) - history_terms
        drain, gate, source, bulk = mosfet.nodes
        if hasattr(mosfet, '_q_prev') and mosfet._q_prev is not None:
            charges = mosfet.get_charges(voltages)
            caps = mosfet.get_capacitances(voltages)
            dt = self._current_dt

            # Select integration method coefficients
            method = getattr(self, '_integration_method', 'trap')
            if method == 'bdf2' and hasattr(mosfet, '_q_prev2') and mosfet._q_prev2 is not None:
                coeff = 1.5 / dt
                h_g = (2.0 / dt) * mosfet._q_prev["qg"] - (0.5 / dt) * mosfet._q_prev2["qg"]
                h_d = (2.0 / dt) * mosfet._q_prev["qd"] - (0.5 / dt) * mosfet._q_prev2["qd"]
            elif method == 'trap' or (method == 'bdf2' and mosfet._q_prev2 is None):
                # Trapezoidal (or fallback when BDF-2 history not yet available)
                coeff = 2.0 / dt
                h_g = coeff * mosfet._q_prev["qg"] + getattr(mosfet, '_i_prev_gate', 0.0)
                h_d = coeff * mosfet._q_prev["qd"] + getattr(mosfet, '_i_prev_drain', 0.0)
            else:
                # Backward Euler: no i_prev history term
                coeff = 1.0 / dt
                h_g = coeff * mosfet._q_prev["qg"]
                h_d = coeff * mosfet._q_prev["qd"]

            # Terminal voltages at current NR iterate
            v_g = voltages.get(gate, 0.0)
            v_d = voltages.get(drain, 0.0)
            v_s = voltages.get(source, 0.0)

            # Capacitive currents at NR iterate V0
            i_g_cap = coeff * charges["qg"] - h_g
            i_d_cap = coeff * charges["qd"] - h_d
            # Source by charge conservation: i_s = -(i_g + i_d)

            # Jacobian entries: dI_t/dV_j = coeff * C_tj
            cgg = caps.get("cgg", 0.0)
            cgd = caps.get("cgd", 0.0)
            cgs = caps.get("cgs", 0.0)
            cdg = caps.get("cdg", 0.0)
            cdd = caps.get("cdd", 0.0)
            # Derived from charge conservation on each terminal
            cds = -(cdg + cdd)
            csg = -(cgg + cdg)
            csd = -(cgd + cdd)
            css = -(cgs + cds)

            scale = coeff

            # Node indices (None if ground)
            g_idx = node_map.get(gate) if gate != "0" else None
            d_idx = node_map.get(drain) if drain != "0" else None
            s_idx = node_map.get(source) if source != "0" else None

            # --- Stamp Jacobian (conductance matrix) ---
            # Gate row: dI_g/dV_g, dI_g/dV_d, dI_g/dV_s
            if g_idx is not None:
                mna_matrix[g_idx, g_idx] += scale * cgg
                if d_idx is not None:
                    mna_matrix[g_idx, d_idx] += scale * cgd
                if s_idx is not None:
                    mna_matrix[g_idx, s_idx] += scale * cgs

            # Drain row: dI_d/dV_g, dI_d/dV_d, dI_d/dV_s
            if d_idx is not None:
                if g_idx is not None:
                    mna_matrix[d_idx, g_idx] += scale * cdg
                mna_matrix[d_idx, d_idx] += scale * cdd
                if s_idx is not None:
                    mna_matrix[d_idx, s_idx] += scale * cds

            # Source row: dI_s/dV_g, dI_s/dV_d, dI_s/dV_s
            if s_idx is not None:
                if g_idx is not None:
                    mna_matrix[s_idx, g_idx] += scale * csg
                if d_idx is not None:
                    mna_matrix[s_idx, d_idx] += scale * csd
                mna_matrix[s_idx, s_idx] += scale * css

            # --- Stamp RHS (NR constant) ---
            # e_t = I_t(V0) - Σ_j(scale * C_tj * V0_j)
            e_g = i_g_cap - scale * (cgg * v_g + cgd * v_d + cgs * v_s)
            e_d = i_d_cap - scale * (cdg * v_g + cdd * v_d + cds * v_s)
            e_s = -(e_g + e_d)  # Charge conservation

            if g_idx is not None:
                rhs[g_idx] -= e_g
            if d_idx is not None:
                rhs[d_idx] -= e_d
            if s_idx is not None:
                rhs[s_idx] -= e_s


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

        # V5 Phase A — A3.2: track the highest committed step so the
        # verify_nn_dc_tran inverter-tran runner can recover a partial
        # waveform when NR exhausts mid-transient (turns ERROR row into
        # numeric FAIL row).
        self._last_committed_step = 0
        self._partial_time = time
        self._partial_voltages = voltages_over_time

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

        # Initialize MOSFET charge state for intrinsic capacitance tracking
        for component in self.circuit.components:
            if _is_mosfet(component) and hasattr(component, 'init_charge_state'):
                component.init_charge_state(initial_voltages)

        # Store initial voltages
        time[0] = 0.0
        for node in nodes:
            voltages_over_time[node][0] = initial_voltages.get(node, 0.0)

        # Debug: Print initial voltages
        if self.debug:
            print(f"Initial transient voltages:")
            for node in sorted(nodes)[:5]:
                print(f"  V{node} = {initial_voltages.get(node, 0.0):.4f} V")

        # Step 2: Add pseudo-capacitors if enabled AND no DC OP provided
        # If a valid DC operating point was provided as initial_guess, skip
        # pseudo-transient and Gmin stepping — they create startup artifacts.
        has_dc_op = self.initial_guess is not None and len(self.initial_guess) > 0
        if has_dc_op:
            # DC OP provides correct initial conditions; convergence aids not needed
            effective_use_pseudo = False
            effective_use_gmin = False
            if self.debug:
                print(f"DC operating point provided — skipping pseudo-transient and Gmin stepping")
        else:
            effective_use_pseudo = self.use_pseudo_transient
            effective_use_gmin = self.use_gmin_stepping

        if effective_use_pseudo and self._has_non_linear_components():
            if self.debug:
                print(f"Adding pseudo-capacitors for better DC convergence (first {self.pseudo_transient_steps} steps)")
            self._add_pseudo_capacitors()

        # Step 3: Adaptive time-stepping with LTE-based sub-stepping
        # V5 Phase A — A3: NN circuits use a 4-halve cap (16x sub-resolution)
        # with explicit event logging. BSIM-CMG keeps the original 5-halve
        # behaviour to preserve byte-identical verify_bsimcmg_* output.
        has_non_linear = self._has_non_linear_components()
        is_nn_circuit = self._has_nn_devices()
        max_dt_reductions = 4 if is_nn_circuit else 5

        # LTE-adaptive sub-stepping: uses constructor parameters
        adaptive_substeps = 1
        max_substeps = self.max_substeps
        lte_safety_factor = self.lte_safety_factor

        # Stiffness tracking for BDF-2 auto-switching
        _stiff_switched = False  # Once True, stays on BDF-2

        for step in range(1, num_steps):
            # Current output time
            current_time = step * self.dt
            time[step] = min(current_time, self.t_stop)

            # Remove pseudo-capacitors after specified steps
            if effective_use_pseudo and step == self.pseudo_transient_steps + 1:
                if self.debug:
                    print(f"Removing pseudo-capacitors at step {step}")
                self._remove_pseudo_capacitors()

            # Integration method selection: BE (step 1) → Trap (step 2+) → BDF-2 (on stiffness)
            if step == 1:
                self._integration_method = 'be'
            elif _stiff_switched:
                self._integration_method = 'bdf2'
            else:
                self._integration_method = 'trap'

            # Set capacitor integration flags
            use_trap = self._integration_method == 'trap'
            self._use_trap_for_charges = use_trap
            for component in self.circuit.components:
                if isinstance(component, Capacitor):
                    component._use_trapezoidal = use_trap
                    component._method = self._integration_method

            # Starting voltages for this output interval
            current_voltages = {}
            for node in nodes:
                current_voltages[node] = voltages_over_time[node][step - 1]
            current_voltages["0"] = 0.0
            current_voltages["GND"] = 0.0

            # Sub-step within this output interval
            n_subs = adaptive_substeps
            sub_dt = self.dt / n_subs

            for sub_idx in range(n_subs):
                sub_time = time[step - 1] + (sub_idx + 1) * sub_dt
                self._current_dt = sub_dt

                # Retry loop for NR convergence failures
                dt_reduction_count = 0
                current_sub_dt = sub_dt

                while dt_reduction_count <= max_dt_reductions:
                    try:
                        self._current_dt = current_sub_dt

                        # Update capacitor companion models
                        for component in self.circuit.components:
                            if isinstance(component, Capacitor):
                                component.get_companion_model(current_sub_dt, component.v_prev)

                        # Solve for node voltages
                        if has_non_linear:
                            timestep_voltages = self._solve_timestep_newton(
                                nodes=nodes,
                                node_map=node_map,
                                num_nodes=num_nodes,
                                num_voltage_sources=num_voltage_sources,
                                initial_voltages=current_voltages,
                                time=sub_time,
                                step_index=step - 1,
                                use_gmin=effective_use_gmin
                            )
                        else:
                            matrix_size = num_nodes + num_voltage_sources
                            mna_matrix = _create_mna_matrix(matrix_size)
                            rhs = np.zeros(matrix_size)

                            for component in self.circuit.components:
                                component.stamp_conductance(mna_matrix, node_map)
                                component.stamp_rhs(rhs, node_map)

                            self._stamp_voltage_sources(mna_matrix, rhs, node_map, num_nodes, sub_time)

                            try:
                                solution = _solve_mna(mna_matrix, rhs)
                            except (np.linalg.LinAlgError, RuntimeError) as e:
                                raise np.linalg.LinAlgError(
                                    f"Circuit matrix is singular at t={sub_time:.6f}s. "
                                    f"Check for floating nodes or short circuits."
                                ) from e

                            timestep_voltages = {}
                            for idx, node in enumerate(nodes):
                                timestep_voltages[node] = float(solution[idx])
                            timestep_voltages["0"] = 0.0
                            timestep_voltages["GND"] = 0.0

                        # Sub-step succeeded — commit state
                        for component in self.circuit.components:
                            if isinstance(component, Capacitor):
                                component.update_voltage(timestep_voltages)

                        for component in self.circuit.components:
                            if _is_mosfet(component) and hasattr(component, 'update_charge_state'):
                                terminal_currents = {}
                                if hasattr(component, '_q_prev') and component._q_prev is not None:
                                    charges_new = component.get_charges(timestep_voltages)
                                    dt_eff = current_sub_dt
                                    method = self._integration_method
                                    if method == 'bdf2' and hasattr(component, '_q_prev2') and component._q_prev2 is not None:
                                        coeff = 1.5 / dt_eff
                                        h_g = (2.0 / dt_eff) * component._q_prev["qg"] - (0.5 / dt_eff) * component._q_prev2["qg"]
                                        h_d = (2.0 / dt_eff) * component._q_prev["qd"] - (0.5 / dt_eff) * component._q_prev2["qd"]
                                    elif method == 'trap':
                                        coeff = 2.0 / dt_eff
                                        h_g = coeff * component._q_prev["qg"] + getattr(component, '_i_prev_gate', 0.0)
                                        h_d = coeff * component._q_prev["qd"] + getattr(component, '_i_prev_drain', 0.0)
                                    else:  # 'be'
                                        coeff = 1.0 / dt_eff
                                        h_g = coeff * component._q_prev["qg"]
                                        h_d = coeff * component._q_prev["qd"]
                                    terminal_currents["i_gate"] = coeff * charges_new["qg"] - h_g
                                    terminal_currents["i_drain"] = coeff * charges_new["qd"] - h_d
                                component.update_charge_state(timestep_voltages, terminal_currents)

                        current_voltages = timestep_voltages
                        break

                    except RuntimeError as e:
                        dt_reduction_count += 1
                        if dt_reduction_count > max_dt_reductions:
                            raise RuntimeError(
                                f"Failed to converge at t={sub_time:.2e}s even with minimum dt. "
                                f"Original error: {e}"
                            ) from e
                        dt_before = current_sub_dt
                        current_sub_dt = sub_dt / (2 ** dt_reduction_count)
                        # V5 Phase A — A3: log every dt-halve event so
                        # verification scripts can flag cells that needed
                        # >1 halving (escalates as a model-fit issue).
                        self._dt_halve_events.append({
                            "step": step,
                            "sub_idx": sub_idx,
                            "sim_time": float(sub_time),
                            "halve_num": dt_reduction_count,
                            "dt_before": float(dt_before),
                            "dt_after": float(current_sub_dt),
                            "is_nn_circuit": bool(is_nn_circuit),
                            "error_msg": str(e),
                        })
                        if self.debug:
                            print(f"  WARNING: Convergence failed at t={sub_time:.2e}s, reducing dt to {current_sub_dt:.2e}")

            # Store at output point
            for node in nodes:
                voltages_over_time[node][step] = current_voltages[node]
            # V5 Phase A — A3.2: track committed step for partial-recovery.
            self._last_committed_step = step

            # Stiffness detection: if NR took > 20 iterations, switch to BDF-2
            if (not _stiff_switched and has_non_linear and step > 2
                    and getattr(self, '_last_nr_iterations', 0) > 20):
                _stiff_switched = True
                if self.debug:
                    print(f"  Stiffness detected at step {step} (NR iters={self._last_nr_iterations}) -> switching to BDF-2")

            # LTE estimation for adaptive sub-stepping (need >= 3 output points)
            if step >= 2:
                max_lte_ratio = 0.0
                for node in nodes:
                    v_np1 = voltages_over_time[node][step]
                    v_n = voltages_over_time[node][step - 1]
                    v_nm1 = voltages_over_time[node][step - 2]
                    d2v = abs(v_np1 - 2.0 * v_n + v_nm1)
                    lte = d2v / 12.0  # Trapezoidal LTE coefficient
                    threshold = self.vntol + self.reltol * max(abs(v_np1), abs(v_n))
                    if threshold > 0:
                        max_lte_ratio = max(max_lte_ratio, lte / threshold)

                # Account for current sub-stepping: effective error ~ raw / n^2
                # (Trapezoidal order 2: global error is O(h^2), h = dt/n)
                effective_lte = max_lte_ratio / (adaptive_substeps ** 2)

                # Compute optimal sub-steps: n = ceil(sqrt(raw_lte / threshold))
                if effective_lte > lte_safety_factor:
                    optimal_n = int(np.ceil(np.sqrt(max_lte_ratio / lte_safety_factor)))
                    adaptive_substeps = min(max(optimal_n, adaptive_substeps), max_substeps)
                    if self.debug:
                        print(f"  LTE={max_lte_ratio:.1f} eff={effective_lte:.2f} at t={current_time:.2e}s -> substeps={adaptive_substeps}")
                elif effective_lte < lte_safety_factor / 8 and adaptive_substeps > 1:
                    adaptive_substeps = max(adaptive_substeps // 2, 1)
                    if self.debug:
                        print(f"  LTE={max_lte_ratio:.1f} eff={effective_lte:.2f} at t={current_time:.2e}s -> substeps={adaptive_substeps}")

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

        # Add SPICE GMIN minimum conductance for numerical stability
        g_ds = max(g_ds, 1e-12)

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

        # NOTE: AC capacitance stamping (Cgs, Cgd, etc.) not yet implemented.
        # Only the resistive small-signal model is used for AC analysis.

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
