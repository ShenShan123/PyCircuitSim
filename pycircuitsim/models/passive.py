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
    Ideal DC voltage source with optional AC specification.

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
        ac_magnitude: AC magnitude for AC analysis (volts, default 0)
        ac_phase: AC phase for AC analysis (degrees, default 0)
    """

    def __init__(self, name: str, nodes: List[str], value: float,
                 ac_magnitude: float = 0.0, ac_phase: float = 0.0):
        """
        Initialize a voltage source.

        Args:
            name: Component identifier (e.g., 'V1', 'V_dd')
            nodes: List of exactly two node names [positive, negative]
            value: Voltage value in volts (DC value)
            ac_magnitude: AC magnitude in volts (default 0)
            ac_phase: AC phase in degrees (default 0)

        Raises:
            ValueError: If nodes count is not 2
        """
        # Validate number of nodes first
        if len(nodes) != 2:
            raise ValueError(f"VoltageSource must have exactly 2 nodes, got {len(nodes)}")

        # Initialize with value (stored in self.value by parent class)
        super().__init__(name, nodes, value)

        # Store current (will be set by solver)
        self._current = 0.0

        # AC analysis parameters
        self.ac_magnitude = float(ac_magnitude)
        self.ac_phase = float(ac_phase)

    @property
    def voltage(self) -> float:
        """Get voltage value (references self.value for consistency)."""
        return self.value if self.value is not None else 0.0

    @voltage.setter
    def voltage(self, value: float):
        """Set voltage value (updates self.value for consistency)."""
        self.value = float(value)

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
        Calculate current through voltage source.

        The current is stored by the solver after MNA solving.
        Current flows from positive terminal to negative terminal.

        Args:
            voltages: Dictionary mapping node names to voltage values (not used)

        Returns:
            Current through voltage source in amperes
        """
        return getattr(self, '_current', 0.0)

    def set_current(self, current: float) -> None:
        """
        Set the current through this voltage source (called by solver).

        Args:
            current: Current value in amperes
        """
        self._current = float(current)

    def __repr__(self) -> str:
        """String representation of the voltage source."""
        return f"VoltageSource({self.name}, nodes={self.nodes}, V={self.voltage}V)"


class PulseVoltageSource(VoltageSource):
    """
    PULSE voltage source for transient analysis.

    A PULSE source generates a periodic pulse waveform defined by:
    - V1: Initial value
    - V2: Pulsed value
    - TD: Delay time
    - TR: Rise time
    - TF: Fall time
    - PW: Pulse width
    - PER: Period

    The waveform repeats with period PER. The pulse goes from V1 to V2,
    stays at V2 for PW time, then returns to V1.

    Attributes:
        name: Component identifier (e.g., 'V1', 'V_in')
        nodes: List of two node names [positive, negative]
        v1: Initial voltage value (volts)
        v2: Pulsed voltage value (volts)
        td: Delay time before first pulse (seconds)
        tr: Rise time (seconds)
        tf: Fall time (seconds)
        pw: Pulse width at V2 (seconds)
        per: Period of the waveform (seconds)
    """

    def __init__(
        self,
        name: str,
        nodes: List[str],
        v1: float,
        v2: float,
        td: float,
        tr: float,
        tf: float,
        pw: float,
        per: float
    ):
        """
        Initialize a PULSE voltage source.

        Args:
            name: Component identifier (e.g., 'V1', 'V_in')
            nodes: List of exactly two node names [positive, negative]
            v1: Initial voltage value in volts
            v2: Pulsed voltage value in volts
            td: Delay time in seconds
            tr: Rise time in seconds
            tf: Fall time in seconds
            pw: Pulse width in seconds
            per: Period in seconds

        Raises:
            ValueError: If nodes count is not 2, or if timing parameters are invalid
        """
        # Validate number of nodes
        if len(nodes) != 2:
            raise ValueError(f"PulseVoltageSource must have exactly 2 nodes, got {len(nodes)}")

        # Initialize with v1 as initial value
        super().__init__(name, nodes, v1)

        # Store pulse parameters
        self.v1 = float(v1)
        self.v2 = float(v2)
        self.td = float(td)
        self.tr = float(tr)
        self.tf = float(tf)
        self.pw = float(pw)
        self.per = float(per)

        # Validate timing parameters
        if self.td < 0:
            raise ValueError(f"Delay time TD must be non-negative, got {td}")
        if self.tr <= 0:
            raise ValueError(f"Rise time TR must be positive, got {tr}")
        if self.tf <= 0:
            raise ValueError(f"Fall time TF must be positive, got {tf}")
        if self.pw <= 0:
            raise ValueError(f"Pulse width PW must be positive, got {pw}")
        if self.per <= 0:
            raise ValueError(f"Period PER must be positive, got {per}")
        if self.per < (self.pw + self.tr + self.tf):
            raise ValueError(f"Period ({per}) must be >= PW+TR+TF ({self.pw + self.tr + self.tf})")

    def get_voltage_at_time(self, time: float) -> float:
        """
        Get the voltage value at a specific time.

        The PULSE waveform follows this pattern:
        - 0 ≤ t < TD: V1 (initial delay)
        - TD ≤ t < TD+TR: Ramp from V1 to V2 (rising edge)
        - TD+TR ≤ t < TD+TR+PW: V2 (pulse high)
        - TD+TR+PW ≤ t < TD+TR+PW+TF: Ramp from V2 to V1 (falling edge)
        - TD+TR+PW+TF ≤ t < PER: V1 (off time)
        - Pattern repeats every PER

        Args:
            time: Time in seconds

        Returns:
            Voltage value at the given time
        """
        # Handle negative time (use initial value)
        if time < 0:
            return self.v1

        # Calculate position within current period
        t_in_period = time % self.per

        # Check if we're still in initial delay
        if time < self.td:
            return self.v1

        # Time since delay ended
        t_since_delay = time - self.td
        t_in_pulse_period = t_since_delay % self.per

        # Rising edge
        if t_in_pulse_period < self.tr:
            # Linear ramp from V1 to V2
            fraction = t_in_pulse_period / self.tr
            return self.v1 + fraction * (self.v2 - self.v1)

        # Pulse high
        elif t_in_pulse_period < (self.tr + self.pw):
            return self.v2

        # Falling edge
        elif t_in_pulse_period < (self.tr + self.pw + self.tf):
            # Linear ramp from V2 to V1
            t_fall = t_in_pulse_period - (self.tr + self.pw)
            fraction = t_fall / self.tf
            return self.v2 + fraction * (self.v1 - self.v2)

        # Off time (back to V1)
        else:
            return self.v1

    @property
    def voltage(self) -> float:
        """
        Get current voltage value.

        For time-varying sources, this returns the initial value (v1).
        Use get_voltage_at_time(t) for time-dependent values.

        Returns:
            Initial voltage value (v1)
        """
        return self.v1

    @voltage.setter
    def voltage(self, value: float) -> None:
        """
        Set voltage value for DC analysis compatibility.

        For pulse sources, setting the voltage scales both v1 and v2 proportionally.
        This allows DC source stepping to work with pulse sources.

        Args:
            value: New voltage value (scales the pulse waveform)
        """
        # Calculate scaling factor based on current v1
        if self.v1 != 0:
            scale = value / self.v1
            # Scale both v1 and v2 to maintain pulse amplitude
            self.v1 = float(value)
            self.v2 = self.v2 * scale
        else:
            # If v1 is 0, just set v1 and leave v2 unchanged
            self.v1 = float(value)

    def __repr__(self) -> str:
        """String representation of the PULSE voltage source."""
        return (f"PulseVoltageSource({self.name}, nodes={self.nodes}, "
                f"V1={self.v1}V, V2={self.v2}V, TD={self.td}s, "
                f"TR={self.tr}s, TF={self.tf}s, PW={self.pw}s, PER={self.per}s)")


class CurrentSource(Component):
    """
    Ideal DC current source.

    A current source maintains a fixed current flow from its positive terminal
    to its negative terminal. In MNA formulation, current sources contribute
    directly to the RHS vector:

    - Adds +I to the source node (node[0], where current flows from)
    - Adds -I to the sink node (node[1], where current flows to)

    Current sources do not contribute to the conductance matrix since they
    are independent sources (not dependent on voltage).

    Attributes:
        name: Component identifier (e.g., 'I1', 'I_bias')
        nodes: List of two node names [source, sink]
        current: Current value in amperes (flows from node[0] to node[1])
    """

    def __init__(self, name: str, nodes: List[str], value: float):
        """
        Initialize a current source.

        Args:
            name: Component identifier (e.g., 'I1', 'I_bias')
            nodes: List of exactly two node names [source, sink]
            value: Current value in amperes

        Raises:
            ValueError: If nodes count is not 2
        """
        super().__init__(name, nodes, value)

        # Validate number of nodes
        if len(nodes) != 2:
            raise ValueError(f"CurrentSource must have exactly 2 nodes, got {len(nodes)}")

        # Store current value
        self.current = float(value)

    def get_nodes(self) -> List[str]:
        """
        Return list of node names this current source connects to.

        Returns:
            List of two node names [source, sink]
        """
        return self.nodes

    def stamp_conductance(self, matrix: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Interface for conductance stamping (pass-through for current sources).

        Current sources are independent sources and do not contribute to
        the conductance matrix in MNA formulation. They only affect the
        RHS vector through stamp_rhs().

        Args:
            matrix: The MNA matrix (not modified by current sources)
            node_map: Mapping from node names to matrix indices
        """
        # Current sources don't stamp to conductance matrix
        pass

    def stamp_rhs(self, rhs: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Add current source terms to the RHS vector.

        For a current source from node_i to node_j:
        - Add +I to node_i (current flows out of source node)
        - Add -I to node_j (current flows into sink node)

        Ground node (node "0") is not in the node_map and is skipped.

        Args:
            rhs: The RHS vector to modify (in-place)
            node_map: Mapping from node names to matrix indices
        """
        node_i, node_j = self.nodes[0], self.nodes[1]

        # Add +I to source node (current flows out)
        if node_i != "0" and node_i in node_map:
            idx_i = node_map[node_i]
            rhs[idx_i] += self.current

        # Add -I to sink node (current flows in)
        if node_j != "0" and node_j in node_map:
            idx_j = node_map[node_j]
            rhs[idx_j] -= self.current

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """
        Calculate the current through the current source.

        For an ideal current source, the current is fixed regardless of
        the voltage across its terminals. This method returns the
        specified current value.

        Args:
            voltages: Dictionary mapping node names to voltage values
                      (not used for ideal current sources)

        Returns:
            Current value in amperes (constant)
        """
        # Ideal current source always returns its specified current
        return self.current

    def __repr__(self) -> str:
        """String representation of the current source."""
        return f"CurrentSource({self.name}, nodes={self.nodes}, I={self.current}A)"


class Capacitor(Component):
    """
    Linear capacitor using Backward Euler companion model for transient analysis.

    For DC analysis, a capacitor is an open circuit (I = 0).
    For transient analysis, the capacitor is discretized using Backward Euler:

    The companion model represents the capacitor as:
    - Equivalent conductance: G_eq = C / dt
    - Equivalent current source: I_eq = G_eq * V_prev

    where V_prev is the voltage across the capacitor at the previous timestep.

    This allows the capacitor to be modeled as a resistor in parallel with
    a current source during each timestep of transient analysis.

    Attributes:
        name: Component identifier (e.g., 'C1')
        nodes: List of two node names
        capacitance: Capacitance value in farads
        v_prev: Voltage across capacitor at previous timestep (starts at 0)
        _g_eq: Equivalent conductance from companion model (C/dt)
        _i_eq: Equivalent current from companion model (G_eq * V_prev)
    """

    def __init__(self, name: str, nodes: List[str], value: float):
        """
        Initialize a capacitor.

        Args:
            name: Component identifier (e.g., 'C1', 'C_load')
            nodes: List of exactly two node names (e.g., ['n1', 'n2'])
            value: Capacitance value in farads (must be positive)

        Raises:
            ValueError: If capacitance is not positive or nodes count is not 2
        """
        super().__init__(name, nodes, value)

        # Validate number of nodes
        if len(nodes) != 2:
            raise ValueError(f"Capacitor must have exactly 2 nodes, got {len(nodes)}")

        # Validate capacitance value
        if value is None or value <= 0:
            raise ValueError(f"Capacitance must be positive, got {value}")

        self.capacitance = float(value)
        self.v_prev = 0.0  # Initial voltage across capacitor

        # Companion model parameters (set during transient analysis)
        self._g_eq = 0.0  # Equivalent conductance
        self._i_eq = 0.0  # Equivalent current source

    def get_nodes(self) -> List[str]:
        """
        Return list of node names this capacitor connects to.

        Returns:
            List of two node names
        """
        return self.nodes

    def get_companion_model(self, dt: float, v_prev: float) -> tuple[float, float]:
        """
        Calculate Backward Euler companion model parameters.

        The companion model represents the discrete-time capacitor as:
        - G_eq = C / dt (equivalent conductance)
        - I_eq = G_eq * V_prev (equivalent current source)

        Args:
            dt: Timestep size in seconds
            v_prev: Voltage across capacitor at previous timestep

        Returns:
            Tuple of (G_eq, I_eq) where:
                G_eq: Equivalent conductance in siemens
                I_eq: Equivalent current in amperes
        """
        g_eq = self.capacitance / dt
        i_eq = g_eq * v_prev

        # Store for stamping
        self._g_eq = g_eq
        self._i_eq = i_eq

        return g_eq, i_eq

    def stamp_conductance(self, matrix: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Add equivalent conductance (G_eq) to the MNA matrix.

        After the companion model is set, this stamps G_eq the same way
        a resistor stamps its conductance. For a capacitor between nodes i and j:
        - Add G_eq to diagonal entries G[i,i] and G[j,j]
        - Subtract G_eq from off-diagonal entries G[i,j] and G[j,i]

        Ground node (node "0") is not in the node_map and is skipped.

        Args:
            matrix: The MNA matrix to modify (in-place)
            node_map: Mapping from node names to matrix indices
        """
        node_i, node_j = self.nodes[0], self.nodes[1]
        g = self._g_eq

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
        Add equivalent current source (I_eq) to the RHS vector.

        After the companion model is set, this stamps I_eq the same way
        a current source stamps its current. For a capacitor between nodes i and j:
        - Add +I_eq to node_i
        - Add -I_eq to node_j

        Ground node (node "0") is not in the node_map and is skipped.

        Args:
            rhs: The RHS vector to modify (in-place)
            node_map: Mapping from node names to matrix indices
        """
        node_i, node_j = self.nodes[0], self.nodes[1]

        # Add +I_eq to node_i
        if node_i != "0" and node_i in node_map:
            idx_i = node_map[node_i]
            rhs[idx_i] += self._i_eq

        # Add -I_eq to node_j
        if node_j != "0" and node_j in node_map:
            idx_j = node_map[node_j]
            rhs[idx_j] -= self._i_eq

    def update_voltage(self, voltages: Dict[str, float]) -> None:
        """
        Update the previous voltage after a timestep completes.

        This should be called after each timestep in transient analysis
        to store the current voltage for the next timestep's companion model.

        Args:
            voltages: Dictionary mapping node names to voltage values
        """
        node_i, node_j = self.nodes[0], self.nodes[1]

        # Get voltages (default to 0 if node not found)
        v_i = voltages.get(node_i, 0.0)
        v_j = voltages.get(node_j, 0.0)

        # Update v_prev for next timestep
        self.v_prev = v_i - v_j

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """
        Calculate current flowing through the capacitor.

        For DC analysis, the capacitor is an open circuit (I = 0).
        For transient analysis, the actual current is calculated by the solver
        using the companion model.

        Args:
            voltages: Dictionary mapping node names to voltage values

        Returns:
            Current flowing from first node to second node (0 for DC analysis)
        """
        # In DC analysis, capacitor is open circuit
        # In transient analysis, current is calculated by solver using companion model
        return 0.0

    def __repr__(self) -> str:
        """String representation of the capacitor."""
        return f"Capacitor({self.name}, nodes={self.nodes}, C={self.capacitance}F)"
