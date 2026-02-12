"""
MOSFET Level 1 (Shichman-Hodges) model.

This module implements NMOS and PMOS transistors using the Level 1 compact model,
also known as the Shichman-Hodges model. This is a simple yet physically-based
model suitable for digital circuit simulation and basic analog design.

The model implements three operating regions:
1. Cutoff: V_gs < V_th (for NMOS), I_ds = 0
2. Linear: V_ds < V_ov, I_ds = K * [(V_gs - V_th) * V_ds - 0.5 * V_ds^2]
3. Saturation: V_ds >= V_ov, I_ds = 0.5 * K * (V_gs - V_th)^2

Where:
- K = KP * (W/L) is the transconductance parameter
- V_ov = V_gs - V_th is the overdrive voltage
- V_th is the threshold voltage
- KP is the process transconductance parameter

Note: This implementation ignores body effect and channel-length modulation
for simplicity, as specified in Level 1 model requirements.
"""
from typing import List, Dict, Tuple
import numpy as np

from pycircuitsim.models.base import Component


class NMOS(Component):
    """
    N-Channel MOSFET Level 1 (Shichman-Hodges) model.

    The NMOS transistor is a three-terminal (or four-terminal with body) device
    where current flows from drain to source. The gate-source voltage controls
    the channel conductivity.

    For an NMOS:
    - V_th > 0 (positive threshold voltage)
    - KP > 0 (positive transconductance parameter)
    - Current flows from drain to source when V_gs > V_th

    Terminal order: [drain, gate, source, bulk]

    Attributes:
        name: Component identifier (e.g., 'M1', 'Mn1')
        nodes: List of four node names [drain, gate, source, bulk]
        L: Channel length in meters
        W: Channel width in meters
        VTO: Threshold voltage in volts (default: 0.7V for typical NMOS)
        KP: Process transconductance parameter in A/V^2 (default: 20u A/V^2)
        K: Transconductance parameter K = KP * (W/L)
    """

    # Voltage clamping limits to prevent numerical overflow
    _MAX_V_OVERDRIVE = 5.0  # Maximum V_gs - V_th before clamping (volts)
    _MAX_V_DS = 10.0  # Maximum V_ds before clamping (volts)

    def __init__(
        self,
        name: str,
        nodes: List[str],
        L: float,
        W: float,
        VTO: float = 0.7,
        KP: float = 20e-6
    ):
        """
        Initialize an NMOS transistor.

        Args:
            name: Component identifier (e.g., 'M1', 'Mn1')
            nodes: List of exactly four node names [drain, gate, source, bulk]
            L: Channel length in meters (must be positive)
            W: Channel width in meters (must be positive)
            VTO: Threshold voltage in volts (default: 0.7V)
            KP: Process transconductance in A/V^2 (default: 20e-6 A/V^2)

        Raises:
            ValueError: If node count is not 4, or L/W are not positive
        """
        super().__init__(name, nodes, None)

        # Validate number of nodes
        if len(nodes) != 4:
            raise ValueError(f"NMOS must have exactly 4 nodes, got {len(nodes)}")

        # Validate channel dimensions
        if L <= 0:
            raise ValueError(f"Channel length L must be positive, got {L}")
        if W <= 0:
            raise ValueError(f"Channel width W must be positive, got {W}")

        # Store parameters
        self.L = float(L)
        self.W = float(W)
        self.VTO = float(VTO)
        self.KP = float(KP)

        # Calculate transconductance parameter K = KP * (W/L)
        self.K = self.KP * (self.W / self.L)

    def get_nodes(self) -> List[str]:
        """
        Return list of node names this NMOS connects to.

        Returns:
            List of four node names [drain, gate, source, bulk]
        """
        return self.nodes

    def stamp_conductance(self, matrix: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Add conductance terms to the MNA matrix.

        For non-linear devices like MOSFETs, the conductance depends on the
        operating point. This method stamps the small-signal conductance
        calculated at the current bias point.

        The Newton-Raphson solver will call this method iteratively with
        updated voltages until convergence.

        Args:
            matrix: The MNA matrix to modify (in-place)
            node_map: Mapping from node names to matrix indices

        Note:
            This is a placeholder for the Newton-Raphson solver integration.
            The actual stamping uses the conductance values from get_conductance().
        """
        # For non-linear devices, the solver will:
        # 1. Get terminal voltages from current solution
        # 2. Call get_conductance() to get (g_ds, g_m)
        # 3. Stamp these values to the matrix
        # For now, this method exists to satisfy the Component interface
        pass

    def stamp_rhs(self, rhs: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Add current/source terms to the RHS vector.

        MOSFETs are non-linear devices and don't contribute to the RHS
        directly in the MNA formulation. Their behavior is captured through
        the conductance matrix stamping.

        Args:
            rhs: The RHS vector to modify (in-place)
            node_map: Mapping from node names to matrix indices
        """
        # MOSFETs don't contribute to RHS directly
        # Their non-linear behavior is captured through conductance stamping
        pass

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """
        Calculate drain current I_ds given terminal voltages.

        Implements the Shichman-Hodges equations for the three operating regions:
        - Cutoff: V_gs < V_th, I_ds = 0
        - Linear: V_ds < V_ov, I_ds = K * [(V_gs - V_th) * V_ds - 0.5 * V_ds^2]
        - Saturation: V_ds >= V_ov, I_ds = 0.5 * K * (V_gs - V_th)^2

        Args:
            voltages: Dictionary mapping node names to voltage values

        Returns:
            Drain current I_ds flowing from drain to source (in amperes)
        """
        # Extract terminal voltages
        v_d = voltages.get(self.nodes[0], 0.0)  # Drain
        v_g = voltages.get(self.nodes[1], 0.0)  # Gate
        v_s = voltages.get(self.nodes[2], 0.0)  # Source
        # v_b = voltages.get(self.nodes[3], 0.0)  # Bulk (not used in Level 1)

        # Calculate terminal voltages relative to source
        v_gs = v_g - v_s  # Gate-source voltage
        v_ds = v_d - v_s  # Drain-source voltage

        # Check if in cutoff region (V_gs < V_th)
        if v_gs < self.VTO:
            return 0.0

        # Calculate overdrive voltage
        v_ov = v_gs - self.VTO

        # Apply voltage clamping to prevent numerical overflow
        # This ensures numerical stability during Newton-Raphson iterations
        v_ov = np.clip(v_ov, -self._MAX_V_OVERDRIVE, self._MAX_V_OVERDRIVE)
        v_ds = np.clip(v_ds, -self._MAX_V_DS, self._MAX_V_DS)

        # Check operating region (linear vs saturation)
        if v_ds < v_ov:
            # Linear region (triode region)
            # I_ds = K * [(V_gs - V_th) * V_ds - 0.5 * V_ds^2]
            i_ds = self.K * (v_ov * v_ds - 0.5 * v_ds**2)
        else:
            # Saturation region
            # I_ds = 0.5 * K * (V_gs - V_th)^2
            i_ds = 0.5 * self.K * v_ov**2

        return i_ds

    def get_conductance(self, voltages: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate small-signal conductance parameters for Newton-Raphson.

        Returns the partial derivatives of I_ds with respect to terminal voltages:
        - g_ds = dI_ds/dV_ds (output conductance)
        - g_m = dI_ds/dV_gs (transconductance)

        These values are used by the Newton-Raphson solver to linearize the
        non-linear MOSFET equations at each iteration.

        Args:
            voltages: Dictionary mapping node names to voltage values

        Returns:
            Tuple of (g_ds, g_m) in siemens
        """
        # Extract terminal voltages
        v_d = voltages.get(self.nodes[0], 0.0)  # Drain
        v_g = voltages.get(self.nodes[1], 0.0)  # Gate
        v_s = voltages.get(self.nodes[2], 0.0)  # Source
        # v_b = voltages.get(self.nodes[3], 0.0)  # Bulk (not used in Level 1)

        # Calculate terminal voltages relative to source
        v_gs = v_g - v_s  # Gate-source voltage
        v_ds = v_d - v_s  # Drain-source voltage

        # Check if in cutoff region (V_gs < V_th)
        if v_gs < self.VTO:
            # In cutoff, no current, so conductances are zero
            return 0.0, 0.0

        # Calculate overdrive voltage
        v_ov = v_gs - self.VTO

        # Apply voltage clamping to ensure consistent behavior with calculate_current
        # This ensures numerical stability during Newton-Raphson iterations
        v_ov = np.clip(v_ov, -self._MAX_V_OVERDRIVE, self._MAX_V_OVERDRIVE)
        v_ds = np.clip(v_ds, -self._MAX_V_DS, self._MAX_V_DS)

        # Check operating region (linear vs saturation)
        if v_ds < v_ov:
            # Linear region (triode region)
            # g_m = dI_ds/dV_gs = K * V_ds
            # g_ds = dI_ds/dV_ds = K * (V_ov - V_ds)
            g_m = self.K * v_ds
            g_ds = self.K * (v_ov - v_ds)
        else:
            # Saturation region
            # g_m = dI_ds/dV_gs = K * V_ov
            # g_ds = dI_ds/dV_ds = 0 (ideal saturation, no channel-length modulation)
            g_m = self.K * v_ov
            g_ds = 0.0

        return g_ds, g_m

    def get_capacitances(self, voltages: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate MOSFET internal capacitances for transient analysis.

        Uses Meyer capacitance model (simplified):
        - C_gs: Gate-source overlap + channel capacitance
        - C_gd: Gate-drain overlap + channel capacitance
        - C_db: Drain-bulk junction capacitance
        - C_sb: Source-bulk junction capacitance

        Args:
            voltages: Dictionary mapping node names to voltage values

        Returns:
            Dictionary with capacitance values in Farads
        """
        # Extract voltages
        v_d = voltages.get(self.nodes[0], 0.0)  # Drain
        v_g = voltages.get(self.nodes[1], 0.0)  # Gate
        v_s = voltages.get(self.nodes[2], 0.0)  # Source
        v_b = voltages.get(self.nodes[3], 0.0)  # Bulk

        v_gs = v_g - v_s
        v_gd = v_g - v_d
        v_ds = v_d - v_s

        # Oxide capacitance (C_ox = epsilon_ox * W * L / t_ox)
        # Assuming t_ox = 10nm for typical 180nm process
        epsilon_ox = 3.45e-11  # F/m (SiO2)
        t_ox = 10e-9  # meters
        C_ox = epsilon_ox * self.W * self.L / t_ox

        # Overlap capacitances (fixed)
        C_ov = 0.2 * C_ox  # 20% of oxide as overlap

        # Junction capacitances (simplified)
        C_j0 = 1e-15  # 1fF zero-bias junction capacitance

        # Voltage-dependent junction capacitances
        phi_bi = 0.7  # Built-in potential (V)
        # Clamp junction voltages to avoid division issues
        v_db = max(v_d - v_b, -phi_bi + 0.01)
        v_sb = max(v_s - v_b, -phi_bi + 0.01)

        C_db = C_j0 / (1 + v_db / phi_bi)**0.5
        C_sb = C_j0 / (1 + v_sb / phi_bi)**0.5

        # Check operating region for channel capacitance division
        v_ov = v_gs - self.VTO

        if v_gs < self.VTO:  # Cutoff
            C_gs = C_ov
            C_gd = C_ov
        elif v_ds < v_ov:  # Linear
            C_gs = C_ov + 0.5 * C_ox
            C_gd = C_ov + 0.5 * C_ox
        else:  # Saturation
            C_gs = C_ov + 0.66 * C_ox  # 2/3 of C_ox
            C_gd = C_ov  # Only overlap

        return {
            "cgs": C_gs,
            "cgd": C_gd,
            "cdb": C_db,
            "csb": C_sb,
        }

    def __repr__(self) -> str:
        """String representation of the NMOS transistor."""
        return (f"NMOS({self.name}, nodes={self.nodes}, "
                f"L={self.L*1e6:.1f}μm, W={self.W*1e6:.1f}μm, "
                f"VTO={self.VTO}V, KP={self.KP*1e6:.1f}μA/V²)")


class PMOS(Component):
    """
    P-Channel MOSFET Level 1 (Shichman-Hodges) model.

    The PMOS transistor is complementary to NMOS, with opposite polarities:
    - V_th < 0 (negative threshold voltage)
    - KP < 0 (negative transconductance parameter)
    - Current flows from source to drain when |V_gs| > |V_th|

    Terminal order: [drain, gate, source, bulk]

    Note: PMOS uses the same equations as NMOS but with negative parameters.
    The current direction is from source to drain (opposite of NMOS).

    Attributes:
        name: Component identifier (e.g., 'M1', 'Mp1')
        nodes: List of four node names [drain, gate, source, bulk]
        L: Channel length in meters
        W: Channel width in meters
        VTO: Threshold voltage in volts (default: -0.7V for typical PMOS)
        KP: Process transconductance parameter in A/V^2 (default: -20u A/V^2)
        K: Transconductance parameter K = KP * (W/L)
    """

    # Voltage clamping limits to prevent numerical overflow
    _MAX_V_OVERDRIVE = 5.0  # Maximum |V_gs - V_th| before clamping (volts)
    _MAX_V_DS = 10.0  # Maximum |V_ds| before clamping (volts)

    def __init__(
        self,
        name: str,
        nodes: List[str],
        L: float,
        W: float,
        VTO: float = -0.7,
        KP: float = -20e-6
    ):
        """
        Initialize a PMOS transistor.

        Args:
            name: Component identifier (e.g., 'M1', 'Mp1')
            nodes: List of exactly four node names [drain, gate, source, bulk]
            L: Channel length in meters (must be positive)
            W: Channel width in meters (must be positive)
            VTO: Threshold voltage in volts (default: -0.7V, negative for PMOS)
            KP: Process transconductance in A/V^2 (default: -20e-6 A/V^2, negative)

        Raises:
            ValueError: If node count is not 4, or L/W are not positive
        """
        super().__init__(name, nodes, None)

        # Validate number of nodes
        if len(nodes) != 4:
            raise ValueError(f"PMOS must have exactly 4 nodes, got {len(nodes)}")

        # Validate channel dimensions
        if L <= 0:
            raise ValueError(f"Channel length L must be positive, got {L}")
        if W <= 0:
            raise ValueError(f"Channel width W must be positive, got {W}")

        # Store parameters
        self.L = float(L)
        self.W = float(W)
        self.VTO = float(VTO)
        self.KP = float(KP)

        # Calculate transconductance parameter K = KP * (W/L)
        self.K = self.KP * (self.W / self.L)

    def get_nodes(self) -> List[str]:
        """
        Return list of node names this PMOS connects to.

        Returns:
            List of four node names [drain, gate, source, bulk]
        """
        return self.nodes

    def stamp_conductance(self, matrix: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Add conductance terms to the MNA matrix.

        For non-linear devices like MOSFETs, the conductance depends on the
        operating point. This method stamps the small-signal conductance
        calculated at the current bias point.

        The Newton-Raphson solver will call this method iteratively with
        updated voltages until convergence.

        Args:
            matrix: The MNA matrix to modify (in-place)
            node_map: Mapping from node names to matrix indices

        Note:
            This is a placeholder for the Newton-Raphson solver integration.
            The actual stamping uses the conductance values from get_conductance().
        """
        # For non-linear devices, the solver will:
        # 1. Get terminal voltages from current solution
        # 2. Call get_conductance() to get (g_ds, g_m)
        # 3. Stamp these values to the matrix
        # For now, this method exists to satisfy the Component interface
        pass

    def stamp_rhs(self, rhs: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Add current/source terms to the RHS vector.

        MOSFETs are non-linear devices and don't contribute to the RHS
        directly in the MNA formulation. Their behavior is captured through
        the conductance matrix stamping.

        Args:
            rhs: The RHS vector to modify (in-place)
            node_map: Mapping from node names to matrix indices
        """
        # MOSFETs don't contribute to RHS directly
        # Their non-linear behavior is captured through conductance stamping
        pass

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """
        Calculate drain current I_ds given terminal voltages.

        PMOS uses the same Shichman-Hodges equations as NMOS but with negative
        parameters. The current flows from source to drain when conducting.

        Operating regions:
        - Cutoff: V_gs > V_th (remember V_th is negative for PMOS), I_ds = 0
        - Linear: |V_ds| < |V_ov|
        - Saturation: |V_ds| >= |V_ov|

        Args:
            voltages: Dictionary mapping node names to voltage values

        Returns:
            Drain current I_ds (positive current flows from drain to source)
        """
        # Extract terminal voltages
        v_d = voltages.get(self.nodes[0], 0.0)  # Drain
        v_g = voltages.get(self.nodes[1], 0.0)  # Gate
        v_s = voltages.get(self.nodes[2], 0.0)  # Source
        # v_b = voltages.get(self.nodes[3], 0.0)  # Bulk (not used in Level 1)

        # Calculate terminal voltages relative to source
        v_gs = v_g - v_s  # Gate-source voltage
        v_ds = v_d - v_s  # Drain-source voltage

        # Check if in cutoff region (V_gs > V_th for PMOS, since V_th is negative)
        if v_gs > self.VTO:
            return 0.0

        # Calculate overdrive voltage (will be negative for PMOS)
        v_ov = v_gs - self.VTO

        # Apply voltage clamping to prevent numerical overflow
        # This ensures numerical stability during Newton-Raphson iterations
        v_ov = np.clip(v_ov, -self._MAX_V_OVERDRIVE, self._MAX_V_OVERDRIVE)
        v_ds = np.clip(v_ds, -self._MAX_V_DS, self._MAX_V_DS)

        # Check operating region (linear vs saturation)
        # For PMOS, both v_ds and v_ov are negative in normal operation
        # Linear region: |V_ds| < |V_ov| equivalent to v_ds > v_ov (since both negative)
        if v_ds > v_ov:
            # Linear region (triode region)
            # Same equation as NMOS, but KP is negative
            i_ds = self.K * (v_ov * v_ds - 0.5 * v_ds**2)
        else:
            # Saturation region
            i_ds = 0.5 * self.K * v_ov**2

        return i_ds

    def get_conductance(self, voltages: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate small-signal conductance parameters for Newton-Raphson.

        Returns the partial derivatives of I_ds with respect to terminal voltages:
        - g_ds = dI_ds/dV_ds (output conductance)
        - g_m = dI_ds/dV_gs (transconductance)

        These values are used by the Newton-Raphson solver to linearize the
        non-linear MOSFET equations at each iteration.

        Args:
            voltages: Dictionary mapping node names to voltage values

        Returns:
            Tuple of (g_ds, g_m) in siemens
        """
        # Extract terminal voltages
        v_d = voltages.get(self.nodes[0], 0.0)  # Drain
        v_g = voltages.get(self.nodes[1], 0.0)  # Gate
        v_s = voltages.get(self.nodes[2], 0.0)  # Source
        # v_b = voltages.get(self.nodes[3], 0.0)  # Bulk (not used in Level 1)

        # Calculate terminal voltages relative to source
        v_gs = v_g - v_s  # Gate-source voltage
        v_ds = v_d - v_s  # Drain-source voltage

        # Check if in cutoff region (V_gs > V_th for PMOS)
        if v_gs > self.VTO:
            # In cutoff, no current, so conductances are zero
            return 0.0, 0.0

        # Calculate overdrive voltage (will be negative for PMOS)
        v_ov = v_gs - self.VTO

        # Apply voltage clamping to ensure consistent behavior with calculate_current
        # This ensures numerical stability during Newton-Raphson iterations
        v_ov = np.clip(v_ov, -self._MAX_V_OVERDRIVE, self._MAX_V_OVERDRIVE)
        v_ds = np.clip(v_ds, -self._MAX_V_DS, self._MAX_V_DS)

        # Check operating region (linear vs saturation)
        # For PMOS, both v_ds and v_ov are negative in normal operation
        # Linear region: |V_ds| < |V_ov| equivalent to v_ds > v_ov
        if v_ds > v_ov:
            # Linear region (triode region)
            # Use absolute value of K (conductance magnitude) and absolute voltage values
            g_m = abs(self.K) * abs(v_ds)
            g_ds = abs(self.K) * abs(v_ov - v_ds)
        else:
            # Saturation region
            g_m = abs(self.K) * abs(v_ov)
            g_ds = 0.0

        return g_ds, g_m

    def get_capacitances(self, voltages: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate MOSFET internal capacitances for transient analysis.

        Uses Meyer capacitance model (simplified):
        - C_gs: Gate-source overlap + channel capacitance
        - C_gd: Gate-drain overlap + channel capacitance
        - C_db: Drain-bulk junction capacitance
        - C_sb: Source-bulk junction capacitance

        Args:
            voltages: Dictionary mapping node names to voltage values

        Returns:
            Dictionary with capacitance values in Farads
        """
        # Extract voltages
        v_d = voltages.get(self.nodes[0], 0.0)  # Drain
        v_g = voltages.get(self.nodes[1], 0.0)  # Gate
        v_s = voltages.get(self.nodes[2], 0.0)  # Source
        v_b = voltages.get(self.nodes[3], 0.0)  # Bulk

        v_gs = v_g - v_s
        v_gd = v_g - v_d
        v_ds = v_d - v_s

        # Oxide capacitance (C_ox = epsilon_ox * W * L / t_ox)
        # Assuming t_ox = 10nm for typical 180nm process
        epsilon_ox = 3.45e-11  # F/m (SiO2)
        t_ox = 10e-9  # meters
        C_ox = epsilon_ox * self.W * self.L / t_ox

        # Overlap capacitances (fixed)
        C_ov = 0.2 * C_ox  # 20% of oxide as overlap

        # Junction capacitances (simplified)
        C_j0 = 1e-15  # 1fF zero-bias junction capacitance

        # Voltage-dependent junction capacitances
        phi_bi = 0.7  # Built-in potential (V)
        # Clamp junction voltages to avoid division issues
        v_db = max(v_d - v_b, -phi_bi + 0.01)
        v_sb = max(v_s - v_b, -phi_bi + 0.01)

        C_db = C_j0 / (1 + v_db / phi_bi)**0.5
        C_sb = C_j0 / (1 + v_sb / phi_bi)**0.5

        # Check operating region for channel capacitance division
        v_ov = v_gs - self.VTO

        if v_gs > self.VTO:  # Cutoff (PMOS: V_gs > V_th)
            C_gs = C_ov
            C_gd = C_ov
        elif v_ds > v_ov:  # Linear (PMOS: v_ds > v_ov when both negative)
            C_gs = C_ov + 0.5 * C_ox
            C_gd = C_ov + 0.5 * C_ox
        else:  # Saturation
            C_gs = C_ov + 0.66 * C_ox  # 2/3 of C_ox
            C_gd = C_ov  # Only overlap

        return {
            "cgs": C_gs,
            "cgd": C_gd,
            "cdb": C_db,
            "csb": C_sb,
        }

    def __repr__(self) -> str:
        """String representation of the PMOS transistor."""
        return (f"PMOS({self.name}, nodes={self.nodes}, "
                f"L={self.L*1e6:.1f}μm, W={self.W*1e6:.1f}μm, "
                f"VTO={self.VTO}V, KP={self.KP*1e6:.1f}μA/V²)")
