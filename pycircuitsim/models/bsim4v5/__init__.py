"""
BSIM4V5 MOSFET Model Component

This module provides the Python Component class for BSIM4V5 MOSFETs,
integrating the C bridge library with the PyCircuitSim solver.

The BSIM4V5 model is an industry-standard compact model for advanced
CMOS technologies (below 100nm), providing accurate simulation of:
- Short-channel effects
- Velocity saturation
- Mobility degradation
- DIBL (Drain-Induced Barrier Lowering)
- And many other physical effects
"""

from typing import List, Dict, Any
import numpy as np

from pycircuitsim.models.base import Component
from .bsim4_wrapper import BSIM4Model, BSIM4Device, BSIM4Output as BSIM4_Output


class BSIM4V5_NMOS(Component):
    """
    N-Channel MOSFET using BSIM4V5 model.

    This component wraps the BSIM4V5 C library to provide accurate
    MOSFET simulation for deep-submicron technologies.

    Terminal order: [drain, gate, source, bulk]

    Attributes:
        name: Component identifier (e.g., 'M1')
        nodes: List of four node names [drain, gate, source, bulk]
        L: Channel length (m)
        W: Channel width (m)
        params: Dictionary of BSIM4V5 model parameters
    """

    def __init__(
        self,
        name: str,
        nodes: List[str],
        L: float,
        W: float,
        params: Dict[str, float] = None
    ):
        """
        Initialize a BSIM4V5 NMOS transistor.

        Args:
            name: Component identifier
            nodes: List of exactly four node names [drain, gate, source, bulk]
            L: Channel length in meters
            W: Channel width in meters
            params: Optional dictionary of model parameters (uses defaults if None)

        Raises:
            ValueError: If node count is not 4, or L/W are not positive
        """
        super().__init__(name, nodes, None)

        if len(nodes) != 4:
            raise ValueError(f"BSIM4V5_NMOS must have exactly 4 nodes, got {len(nodes)}")

        if L <= 0 or W <= 0:
            raise ValueError(f"Channel dimensions must be positive, got L={L}, W={W}")

        self.L = float(L)
        self.W = float(W)
        self.mos_type = "nmos"

        # Create BSIM4 model and device
        self._model = BSIM4Model("nmos")
        self._device = BSIM4Device(self._model, L, W)

        # Apply user-provided parameters
        if params:
            for param_name, param_value in params.items():
                self._model.set_param(param_name.upper(), param_value)

        # Cached conductances and currents from last evaluation
        self._gm = 0.0
        self._gds = 0.0
        self._gmbs = 0.0
        self._id = 0.0
        self._last_voltages = {}

    def get_nodes(self) -> List[str]:
        """Return list of node names this component connects to."""
        return self.nodes

    def _get_term_voltages(self, voltages: Dict[str, float]) -> tuple:
        """
        Extract terminal voltages from voltage dictionary.

        Args:
            voltages: Dictionary of node name -> voltage

        Returns:
            Tuple of (Vds, Vgs, Vbs)
        """
        v_d = voltages.get(self.nodes[0], 0.0)
        v_g = voltages.get(self.nodes[1], 0.0)
        v_s = voltages.get(self.nodes[2], 0.0)
        v_b = voltages.get(self.nodes[3], 0.0)

        vds = v_d - v_s
        vgs = v_g - v_s
        vbs = v_b - v_s

        return vds, vgs, vbs

    def stamp_conductance(self, matrix: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Stamp conductances to the MNA matrix.

        The C bridge returns POSITIVE conductances for both NMOS and PMOS.

        The stamping pattern follows the standard 4-terminal MOSFET model.
        Since Vgs = Vg - Vs and Vbs = Vb - Vs, the contributions are:

        At drain node:  gds*(Vd-Vs) + gm*(Vg-Vs) + gmbs*(Vb-Vs)
        At source node: -gds*(Vd-Vs) - gm*(Vg-Vs) - gmbs*(Vb-Vs)

        This gives the stamping pattern:
               d      g         s               b
          d  +gds   +gm    -(gds+gm+gmbs)     +gmbs
          s  -gds   -gm     (gds+gm+gmbs)     -gmbs
        """
        d = self.nodes[0]
        g = self.nodes[1]
        s = self.nodes[2]
        b = self.nodes[3]

        # Get matrix indices
        idx_d = node_map.get(d)
        idx_g = node_map.get(g)
        idx_s = node_map.get(s)
        idx_b = node_map.get(b)

        # Use conductances directly (positive for both NMOS and PMOS)
        gds = self._gds
        gm = self._gm
        gmbs = self._gmbs

        # Add minimum conductance to prevent numerical instability
        # (same as Level 1 MOSFET solver - see solver.py line 556)
        g_min = 1e-6  # 1 microSiemens minimum conductance (~1 MΩ)
        gds = max(gds, g_min)

        # Total conductance affecting source node
        g_total = gds + gm + gmbs

        # Stamp drain row
        if idx_d is not None:
            matrix[idx_d, idx_d] += gds
            if idx_g is not None:
                matrix[idx_d, idx_g] += gm
            if idx_s is not None:
                matrix[idx_d, idx_s] -= g_total
            if idx_b is not None:
                matrix[idx_d, idx_b] += gmbs

        # Stamp source row
        if idx_s is not None:
            matrix[idx_s, idx_s] += g_total
            if idx_d is not None:
                matrix[idx_s, idx_d] -= gds
            if idx_g is not None:
                matrix[idx_s, idx_g] -= gm
            if idx_b is not None:
                matrix[idx_s, idx_b] -= gmbs

    def stamp_rhs(self, rhs: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Stamp equivalent current source to RHS vector.

        For the companion model in MNA (G*V = I), the MOSFET linearization is:
            Id = gm*Vgs + gds*Vds + gmbs*Vbs + i_eq

        The MNA conductance stamp adds gm, gds, gmbs to the G matrix.
        From KCL at drain: (sum of resistor currents) = Id
        In MNA form: G*V = -i_eq  (i_eq goes to RHS with NEGATIVE sign)

        This is because KCL gives:
            G_resistor*V1 - gm*Vg - (G_resistor + gds)*Vd = i_eq
        Multiplying by -1 to match MNA sign convention (resistor stamps
        have negative off-diagonal entries):
            -G_resistor*V1 + gm*Vg + (G_resistor + gds)*Vd = -i_eq
        """
        if not self._last_voltages:
            return  # No previous evaluation

        vds, vgs, vbs = self._get_term_voltages(self._last_voltages)

        # Norton equivalent current source
        i_eq = self._id - (self._gm * vgs + self._gds * vds + self._gmbs * vbs)

        d = self.nodes[0]
        s = self.nodes[2]

        idx_d = node_map.get(d)
        idx_s = node_map.get(s)

        # RHS gets NEGATIVE i_eq at drain (current flows OUT of node into MOSFET)
        # and POSITIVE i_eq at source (current flows INTO node from MOSFET)
        if idx_d is not None:
            rhs[idx_d] -= i_eq
        if idx_s is not None:
            rhs[idx_s] += i_eq

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """
        Calculate drain current given terminal voltages.

        This also updates the cached conductances for stamping.

        Args:
            voltages: Dictionary mapping node names to voltages

        Returns:
            Drain current (positive for current flowing into drain)
        """
        # Get terminal voltages
        vds, vgs, vbs = self._get_term_voltages(voltages)

        # Evaluate BSIM4V5 model
        output = self._device.evaluate(vds, vgs, vbs)

        # Cache results for stamping
        self._id = output.Id
        self._gm = output.Gm
        self._gds = output.Gds
        self._gmbs = output.Gmbs
        self._last_voltages = voltages.copy()

        return output.Id

    def get_conductance(self, voltages: Dict[str, float]) -> tuple:
        """
        Get conductances at given bias point.

        Args:
            voltages: Dictionary of node voltages

        Returns:
            Tuple of (gds, gm, gmbs) in Siemens
        """
        # Update if voltages changed
        if voltages != self._last_voltages:
            self.calculate_current(voltages)

        return (self._gds, self._gm, self._gmbs)


class BSIM4V5_PMOS(Component):
    """
    P-Channel MOSFET using BSIM4V5 model.

    This component wraps the BSIM4V5 C library for PMOS simulation.

    Terminal order: [drain, gate, source, bulk]

    Attributes:
        name: Component identifier (e.g., 'M1')
        nodes: List of four node names [drain, gate, source, bulk]
        L: Channel length (m)
        W: Channel width (m)
        params: Dictionary of BSIM4V5 model parameters
    """

    def __init__(
        self,
        name: str,
        nodes: List[str],
        L: float,
        W: float,
        params: Dict[str, float] = None
    ):
        """Initialize a BSIM4V5 PMOS transistor."""
        super().__init__(name, nodes, None)

        if len(nodes) != 4:
            raise ValueError(f"BSIM4V5_PMOS must have exactly 4 nodes, got {len(nodes)}")

        if L <= 0 or W <= 0:
            raise ValueError(f"Channel dimensions must be positive, got L={L}, W={W}")

        self.L = float(L)
        self.W = float(W)
        self.mos_type = "pmos"

        # Create BSIM4 model and device
        self._model = BSIM4Model("pmos")
        self._device = BSIM4Device(self._model, L, W)

        # Apply user-provided parameters
        if params:
            for param_name, param_value in params.items():
                self._model.set_param(param_name.upper(), param_value)

        # Cached conductances and currents
        self._gm = 0.0
        self._gds = 0.0
        self._gmbs = 0.0
        self._id = 0.0
        self._last_voltages = {}

    def get_nodes(self) -> List[str]:
        """Return list of node names this component connects to."""
        return self.nodes

    def _get_term_voltages(self, voltages: Dict[str, float]) -> tuple:
        """Extract terminal voltages."""
        v_d = voltages.get(self.nodes[0], 0.0)
        v_g = voltages.get(self.nodes[1], 0.0)
        v_s = voltages.get(self.nodes[2], 0.0)
        v_b = voltages.get(self.nodes[3], 0.0)

        vds = v_d - v_s
        vgs = v_g - v_s
        vbs = v_b - v_s

        return vds, vgs, vbs

    def stamp_conductance(self, matrix: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Stamp conductances to the MNA matrix.

        The C bridge returns POSITIVE conductances for PMOS.

        The stamping pattern follows the standard 4-terminal MOSFET model.
        Since Vgs = Vg - Vs and Vbs = Vb - Vs, the contributions are:

        At drain node:  gds*(Vd-Vs) + gm*(Vg-Vs) + gmbs*(Vb-Vs)
        At source node: -gds*(Vd-Vs) - gm*(Vg-Vs) - gmbs*(Vb-Vs)

        This gives the stamping pattern:
               d      g         s               b
          d  +gds   +gm    -(gds+gm+gmbs)     +gmbs
          s  -gds   -gm     (gds+gm+gmbs)     -gmbs
        """
        d = self.nodes[0]
        g = self.nodes[1]
        s = self.nodes[2]
        b = self.nodes[3]

        idx_d = node_map.get(d)
        idx_g = node_map.get(g)
        idx_s = node_map.get(s)
        idx_b = node_map.get(b)

        # Use conductances directly (positive for PMOS)
        gds = self._gds
        gm = self._gm
        gmbs = self._gmbs

        # Add minimum conductance to prevent numerical instability
        g_min = 1e-6  # 1 microSiemens minimum conductance (~1 MΩ)
        gds = max(gds, g_min)

        # Total conductance affecting source node
        g_total = gds + gm + gmbs

        # Stamp drain row
        if idx_d is not None:
            matrix[idx_d, idx_d] += gds
            if idx_g is not None:
                matrix[idx_d, idx_g] += gm
            if idx_s is not None:
                matrix[idx_d, idx_s] -= g_total
            if idx_b is not None:
                matrix[idx_d, idx_b] += gmbs

        # Stamp source row
        if idx_s is not None:
            matrix[idx_s, idx_s] += g_total
            if idx_d is not None:
                matrix[idx_s, idx_d] -= gds
            if idx_g is not None:
                matrix[idx_s, idx_g] -= gm
            if idx_b is not None:
                matrix[idx_s, idx_b] -= gmbs

    def stamp_rhs(self, rhs: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Stamp equivalent current source to RHS vector.

        For Newton-Raphson with companion model, the MOSFET drain current
        Id flows FROM the external circuit INTO the drain terminal.
        From the circuit node's perspective, this current LEAVES the node.

        The linearized current is: Id = gds*Vds + gm*Vgs + gmbs*Vbs + i_eq
        where i_eq = Id_actual - (gds*Vds + gm*Vgs + gmbs*Vbs)

        PMOS note: Even though PMOS Id is negative when ON, the same stamping
        pattern applies because i_eq accounts for the sign automatically.
        """
        if not self._last_voltages:
            return

        vds, vgs, vbs = self._get_term_voltages(self._last_voltages)

        # Calculate Norton equivalent current source
        i_eq = self._id - (self._gm * vgs + self._gds * vds + self._gmbs * vbs)

        d = self.nodes[0]
        s = self.nodes[2]

        idx_d = node_map.get(d)
        idx_s = node_map.get(s)

        # RHS gets NEGATIVE i_eq at drain (current flows OUT of node into MOSFET)
        # and POSITIVE i_eq at source (current flows INTO node from MOSFET)
        if idx_d is not None:
            rhs[idx_d] -= i_eq
        if idx_s is not None:
            rhs[idx_s] += i_eq

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """Calculate drain current given terminal voltages."""
        vds, vgs, vbs = self._get_term_voltages(voltages)

        output = self._device.evaluate(vds, vgs, vbs)

        self._id = output.Id
        self._gm = output.Gm
        self._gds = output.Gds
        self._gmbs = output.Gmbs
        self._last_voltages = voltages.copy()

        return output.Id

    def get_conductance(self, voltages: Dict[str, float]) -> tuple:
        """Get conductances at given bias point."""
        if voltages != self._last_voltages:
            self.calculate_current(voltages)

        return (self._gds, self._gm, self._gmbs)


# Convenience aliases
BSIM4V5 = BSIM4V5_NMOS  # Default to NMOS
