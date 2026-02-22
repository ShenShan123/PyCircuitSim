"""
BSIM-CMG compact model integration.

This module provides NMOS_CMG and PMOS_CMG classes that wrap the PyCMG
BSIM-CMG compact model (LEVEL=72), enabling production-grade FinFET simulation
using OSDI-compiled Verilog-A models.

The BSIM-CMG model supports:
- FinFET geometry effects (NFIN, TFIN, HFIN, FPITCH)
- Advanced short-channel effects
- Temperature dependence
- Bulk coupling (4-terminal device)
- Capacitance extraction for AC analysis

Terminal order: [drain, gate, source, bulk]
"""

from typing import List, Dict, Tuple, Optional
import sys
from pathlib import Path

# Add PyCMG to Python path if not already present
PYCMG_PATH = Path(__file__).parent.parent.parent / "models" / "PyCMG"
if str(PYCMG_PATH) not in sys.path:
    sys.path.insert(0, str(PYCMG_PATH))

try:
    from pycmg import Model, Instance
except ImportError as e:
    raise ImportError(
        f"Failed to import PyCMG: {e}. "
        "Ensure PyCMG is built and accessible in the project directory."
    )

from pycircuitsim.models.base import Component
from pycircuitsim.config import BSIMCMG_OSDI_PATH, DEFAULT_TEMPERATURE


class NMOS_CMG(Component):
    """
    N-Channel FinFET using BSIM-CMG compact model (LEVEL=72).

    This class wraps the PyCMG BSIM-CMG compact model to provide a pycircuitsim-
    compatible interface. It supports production-grade FinFET simulation with
    advanced geometric and physical effects.

    Terminal order: [drain, gate, source, bulk]

    Attributes:
        name: Component identifier (e.g., 'Mn1')
        nodes: List of four node names [drain, gate, source, bulk]
        L: Channel length in meters
        NFIN: Number of fins (integer or float)
        TFIN: Fin thickness in meters (optional, uses modelcard default)
        HFIN: Fin height in meters (optional, uses modelcard default)
        FPITCH: Fin pitch in meters (optional, uses modelcard default)
        temperature: Device temperature in Kelvin
    """

    def __init__(
        self,
        name: str,
        nodes: List[str],
        osdi_path: str,
        modelcard_path: str,
        model_name: str,
        L: float,
        NFIN: float,
        TFIN: Optional[float] = None,
        HFIN: Optional[float] = None,
        FPITCH: Optional[float] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        model_card_name: Optional[str] = None,
    ):
        """
        Initialize an NMOS FinFET using BSIM-CMG.

        Args:
            name: Component identifier
            nodes: List of exactly four node names [drain, gate, source, bulk]
            osdi_path: Path to BSIM-CMG OSDI binary (.osdi file)
            modelcard_path: Path to modelcard file (.pm, .lib, or .model)
            model_name: Model name from netlist (e.g., "nmos1")
            L: Channel length in meters
            NFIN: Number of fins
            TFIN: Fin thickness in meters (optional)
            HFIN: Fin height in meters (optional)
            FPITCH: Fin pitch in meters (optional)
            temperature: Device temperature in Kelvin (default: 300.15K = 27°C)
            model_card_name: Name of the model in the modelcard file (e.g., "nmos_rvt").
                If None, falls back to model_name.

        Raises:
            ValueError: If node count is not 4, or L/NFIN are invalid
            FileNotFoundError: If OSDI binary or modelcard not found
        """
        super().__init__(name, nodes, None)

        # Validate number of nodes
        if len(nodes) != 4:
            raise ValueError(f"NMOS_CMG must have exactly 4 nodes, got {len(nodes)}")

        # Validate channel length and fin count
        if L <= 0:
            raise ValueError(f"Channel length L must be positive, got {L}")
        if NFIN <= 0:
            raise ValueError(f"Number of fins NFIN must be positive, got {NFIN}")

        # Validate file paths
        if not Path(osdi_path).exists():
            raise FileNotFoundError(f"OSDI binary not found: {osdi_path}")
        if not Path(modelcard_path).exists():
            raise FileNotFoundError(f"Modelcard not found: {modelcard_path}")

        # Store parameters
        self.L = float(L)
        self.NFIN = float(NFIN)
        self.TFIN = float(TFIN) if TFIN is not None else None
        self.HFIN = float(HFIN) if HFIN is not None else None
        self.FPITCH = float(FPITCH) if FPITCH is not None else None
        self.temperature = float(temperature)

        # Create PyCMG model (loads modelcard parameters)
        # model_card_name overrides model_name for modelcard lookup
        # This allows netlist model names (e.g., "nmos1") to differ from
        # modelcard model names (e.g., "nmos_rvt" in ASAP7)
        self._pycmg_model = Model(
            osdi_path=osdi_path,
            modelcard_path=modelcard_path,
            model_name=model_card_name or model_name,
            model_card_name=model_card_name,
        )

        # Build instance parameters dictionary
        inst_params = {"L": self.L, "NFIN": self.NFIN}
        if self.TFIN is not None:
            inst_params["TFIN"] = self.TFIN
        if self.HFIN is not None:
            inst_params["HFIN"] = self.HFIN
        if self.FPITCH is not None:
            inst_params["FPITCH"] = self.FPITCH

        # Create PyCMG instance
        self._pycmg_instance = Instance(
            model=self._pycmg_model,
            params=inst_params,
            temperature=self.temperature
        )

        # Cache for eval results (cleared each Newton-Raphson iteration)
        self._eval_cache: Optional[Dict[str, float]] = None
        self._cache_voltages: Optional[Tuple[float, float, float, float]] = None

        # Charge state for transient analysis
        self._q_prev: Optional[Dict[str, float]] = None
        self._v_prev_tran: Optional[Dict[str, float]] = None

    def get_nodes(self) -> List[str]:
        """
        Return list of node names this NMOS_CMG connects to.

        Returns:
            List of four node names [drain, gate, source, bulk]
        """
        return self.nodes

    def stamp_conductance(self, matrix, node_map: Dict[str, int]) -> None:
        """
        Placeholder for Component interface.

        For MOSFETs, the solver handles conductance stamping by calling
        get_conductance() and calculate_current() directly.
        """
        pass

    def stamp_rhs(self, rhs, node_map: Dict[str, int]) -> None:
        """
        Placeholder for Component interface.

        MOSFETs don't contribute to RHS directly in the MNA formulation.
        """
        pass

    def _eval_dc(self, voltages: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate DC operating point (with caching).

        Args:
            voltages: Dictionary mapping node names to voltage values

        Returns:
            Dictionary with keys: id, ig, is, ie, ids, qg, qd, qs, qb, gm, gds, gmb, etc.
        """
        # Extract terminal voltages
        v_d = voltages.get(self.nodes[0], 0.0)  # Drain
        v_g = voltages.get(self.nodes[1], 0.0)  # Gate
        v_s = voltages.get(self.nodes[2], 0.0)  # Source
        v_b = voltages.get(self.nodes[3], 0.0)  # Bulk

        # Check cache
        v_tuple = (v_d, v_g, v_s, v_b)
        if self._cache_voltages == v_tuple and self._eval_cache is not None:
            return self._eval_cache

        # Call PyCMG eval_dc
        result = self._pycmg_instance.eval_dc({
            "d": v_d,
            "g": v_g,
            "s": v_s,
            "e": v_b  # Bulk terminal is 'e' in BSIM-CMG
        })

        # Update cache
        self._eval_cache = result
        self._cache_voltages = v_tuple

        return result

    def clear_cache(self) -> None:
        """
        Clear evaluation cache.

        Should be called at the start of each Newton-Raphson iteration.
        """
        self._eval_cache = None
        self._cache_voltages = None

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """
        Calculate drain terminal current magnitude.

        Uses the BSIM-CMG compact model via PyCMG. Returns the drain terminal
        current 'id' (not the internal channel current 'ids').

        For NMOS ON: PyCMG id < 0 (SPICE: current OUT of drain), so -id > 0.
        The solver expects positive values; NMOS/PMOS sign difference is
        handled by the RHS stamping in solver.py.

        Args:
            voltages: Dictionary mapping node names to voltage values

        Returns:
            Drain terminal current (positive for NMOS ON)
        """
        result = self._eval_dc(voltages)
        # NMOS: negate SPICE id (negative when ON) to get positive value
        return -result["id"]

    def get_conductance(self, voltages: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Calculate small-signal conductance parameters for Newton-Raphson.

        Returns the conductances extracted from the BSIM-CMG Jacobian:
        - g_ds = dI_ds/dV_ds (output conductance)
        - g_m = dI_ds/dV_gs (transconductance)
        - g_mb = dI_ds/dV_bs (bulk transconductance)

        Args:
            voltages: Dictionary mapping node names to voltage values

        Returns:
            Tuple of (g_ds, g_m, g_mb) in siemens
        """
        result = self._eval_dc(voltages)

        g_ds = result.get("gds", 0.0)
        g_m = result.get("gm", 0.0)
        g_mb = result.get("gmb", 0.0)

        # IMPORTANT: gds should always be positive (output conductance)
        # Negative gds (negative resistance) causes divergence
        # gm and gmb are signed transconductances, preserve their signs
        g_ds = abs(g_ds)

        return (g_ds, g_m, g_mb)

    def get_capacitances(self, voltages: Dict[str, float]) -> Dict[str, float]:
        """
        Get terminal capacitances for AC analysis.

        Returns:
            Dictionary with keys: cgg, cgd, cgs, cdg, cdd, cds, csg, csd, css
        """
        result = self._eval_dc(voltages)

        return {
            "cgg": result.get("cgg", 0.0),
            "cgd": result.get("cgd", 0.0),
            "cgs": result.get("cgs", 0.0),
            "cdg": result.get("cdg", 0.0),
            "cdd": result.get("cdd", 0.0),
        }

    def get_charges(self, voltages: Dict[str, float]) -> Dict[str, float]:
        """Get terminal charges from BSIM-CMG eval_dc().

        Returns:
            Dictionary with keys: qg, qd, qs, qb (Coulombs)
        """
        result = self._eval_dc(voltages)
        return {
            "qg": result.get("qg", 0.0),
            "qd": result.get("qd", 0.0),
            "qs": result.get("qs", 0.0),
            "qb": result.get("qb", 0.0),
        }

    def init_charge_state(self, voltages: Dict[str, float]) -> None:
        """Initialize charge state from DC operating point.
        Must be called before transient analysis starts.
        """
        charges = self.get_charges(voltages)
        self._q_prev = charges.copy()
        self._v_prev_tran = {
            "d": voltages.get(self.nodes[0], 0.0),
            "g": voltages.get(self.nodes[1], 0.0),
            "s": voltages.get(self.nodes[2], 0.0),
            "b": voltages.get(self.nodes[3], 0.0),
        }
        # Trapezoidal integration state for intrinsic caps
        self._i_prev_cgd = 0.0
        self._i_prev_cgs = 0.0
        self._i_prev_cdd = 0.0

    def update_charge_state(self, voltages: Dict[str, float],
                            cap_currents: Optional[Dict[str, float]] = None) -> None:
        """Update charge state after a converged timestep."""
        charges = self.get_charges(voltages)
        self._q_prev = charges.copy()
        self._v_prev_tran = {
            "d": voltages.get(self.nodes[0], 0.0),
            "g": voltages.get(self.nodes[1], 0.0),
            "s": voltages.get(self.nodes[2], 0.0),
            "b": voltages.get(self.nodes[3], 0.0),
        }
        if cap_currents is not None:
            self._i_prev_cgd = cap_currents.get("i_cgd", 0.0)
            self._i_prev_cgs = cap_currents.get("i_cgs", 0.0)
            self._i_prev_cdd = cap_currents.get("i_cdd", 0.0)



class PMOS_CMG(Component):
    """
    P-Channel FinFET using BSIM-CMG compact model (LEVEL=72).

    Similar to NMOS_CMG but for PMOS devices. All voltages and currents
    follow standard PMOS conventions.

    Terminal order: [drain, gate, source, bulk]

    NOTE: PMOS current sign convention may differ from NMOS due to
    hole current flow direction.
    """

    def __init__(
        self,
        name: str,
        nodes: List[str],
        osdi_path: str,
        modelcard_path: str,
        model_name: str,
        L: float,
        NFIN: float,
        TFIN: Optional[float] = None,
        HFIN: Optional[float] = None,
        FPITCH: Optional[float] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        model_card_name: Optional[str] = None,
    ):
        """
        Initialize a PMOS FinFET using BSIM-CMG.

        Args:
            Same as NMOS_CMG

        Raises:
            ValueError: If node count is not 4, or L/NFIN are invalid
            FileNotFoundError: If OSDI binary or modelcard not found
        """
        super().__init__(name, nodes, None)

        # Validate number of nodes
        if len(nodes) != 4:
            raise ValueError(f"PMOS_CMG must have exactly 4 nodes, got {len(nodes)}")

        # Validate channel length and fin count
        if L <= 0:
            raise ValueError(f"Channel length L must be positive, got {L}")
        if NFIN <= 0:
            raise ValueError(f"Number of fins NFIN must be positive, got {NFIN}")

        # Validate file paths
        if not Path(osdi_path).exists():
            raise FileNotFoundError(f"OSDI binary not found: {osdi_path}")
        if not Path(modelcard_path).exists():
            raise FileNotFoundError(f"Modelcard not found: {modelcard_path}")

        # Store parameters
        self.L = float(L)
        self.NFIN = float(NFIN)
        self.TFIN = float(TFIN) if TFIN is not None else None
        self.HFIN = float(HFIN) if HFIN is not None else None
        self.FPITCH = float(FPITCH) if FPITCH is not None else None
        self.temperature = float(temperature)

        # Create PyCMG model (loads modelcard parameters)
        self._pycmg_model = Model(
            osdi_path=osdi_path,
            modelcard_path=modelcard_path,
            model_name=model_card_name or model_name,
            model_card_name=model_card_name,
        )

        # Build instance parameters dictionary
        inst_params = {"L": self.L, "NFIN": self.NFIN}
        if self.TFIN is not None:
            inst_params["TFIN"] = self.TFIN
        if self.HFIN is not None:
            inst_params["HFIN"] = self.HFIN
        if self.FPITCH is not None:
            inst_params["FPITCH"] = self.FPITCH

        # Create PyCMG instance
        self._pycmg_instance = Instance(
            model=self._pycmg_model,
            params=inst_params,
            temperature=self.temperature
        )

        # Cache for eval results
        self._eval_cache: Optional[Dict[str, float]] = None
        self._cache_voltages: Optional[Tuple[float, float, float, float]] = None

        # Charge state for transient analysis
        self._q_prev: Optional[Dict[str, float]] = None
        self._v_prev_tran: Optional[Dict[str, float]] = None

    def get_nodes(self) -> List[str]:
        """Return list of node names."""
        return self.nodes

    def stamp_conductance(self, matrix, node_map: Dict[str, int]) -> None:
        """Placeholder for Component interface."""
        pass

    def stamp_rhs(self, rhs, node_map: Dict[str, int]) -> None:
        """Placeholder for Component interface."""
        pass

    def _eval_dc(self, voltages: Dict[str, float]) -> Dict[str, float]:
        """Evaluate DC operating point (with caching)."""
        # Extract terminal voltages
        v_d = voltages.get(self.nodes[0], 0.0)  # Drain
        v_g = voltages.get(self.nodes[1], 0.0)  # Gate
        v_s = voltages.get(self.nodes[2], 0.0)  # Source
        v_b = voltages.get(self.nodes[3], 0.0)  # Bulk

        # Check cache
        v_tuple = (v_d, v_g, v_s, v_b)
        if self._cache_voltages == v_tuple and self._eval_cache is not None:
            return self._eval_cache

        # Call PyCMG eval_dc
        result = self._pycmg_instance.eval_dc({
            "d": v_d,
            "g": v_g,
            "s": v_s,
            "e": v_b  # Bulk terminal is 'e' in BSIM-CMG
        })

        # Update cache
        self._eval_cache = result
        self._cache_voltages = v_tuple

        return result

    def clear_cache(self) -> None:
        """Clear evaluation cache."""
        self._eval_cache = None
        self._cache_voltages = None

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """
        Calculate drain terminal current magnitude.

        For PMOS ON: PyCMG id > 0 (SPICE: current INTO drain), so id > 0.
        The solver expects positive values; PMOS RHS stamping in solver.py
        handles the opposite current direction vs NMOS.

        Returns:
            Drain terminal current (positive for PMOS ON)
        """
        result = self._eval_dc(voltages)
        # PMOS: id is already positive when ON (SPICE: current INTO drain)
        return result["id"]

    def get_conductance(self, voltages: Dict[str, float]) -> Tuple[float, float, float]:
        """Calculate small-signal conductance parameters."""
        result = self._eval_dc(voltages)

        g_ds = result.get("gds", 0.0)
        g_m = result.get("gm", 0.0)
        g_mb = result.get("gmb", 0.0)

        # IMPORTANT: gds should always be positive (output conductance)
        # gm and gmb are signed transconductances, preserve their signs
        g_ds = abs(g_ds)

        return (g_ds, g_m, g_mb)

    def get_capacitances(self, voltages: Dict[str, float]) -> Dict[str, float]:
        """Get terminal capacitances for AC analysis."""
        result = self._eval_dc(voltages)

        return {
            "cgg": result.get("cgg", 0.0),
            "cgd": result.get("cgd", 0.0),
            "cgs": result.get("cgs", 0.0),
            "cdg": result.get("cdg", 0.0),
            "cdd": result.get("cdd", 0.0),
        }

    def get_charges(self, voltages: Dict[str, float]) -> Dict[str, float]:
        """Get terminal charges from BSIM-CMG eval_dc().

        Returns:
            Dictionary with keys: qg, qd, qs, qb (Coulombs)
        """
        result = self._eval_dc(voltages)
        return {
            "qg": result.get("qg", 0.0),
            "qd": result.get("qd", 0.0),
            "qs": result.get("qs", 0.0),
            "qb": result.get("qb", 0.0),
        }

    def init_charge_state(self, voltages: Dict[str, float]) -> None:
        """Initialize charge state from DC operating point.
        Must be called before transient analysis starts.
        """
        charges = self.get_charges(voltages)
        self._q_prev = charges.copy()
        self._v_prev_tran = {
            "d": voltages.get(self.nodes[0], 0.0),
            "g": voltages.get(self.nodes[1], 0.0),
            "s": voltages.get(self.nodes[2], 0.0),
            "b": voltages.get(self.nodes[3], 0.0),
        }
        # Trapezoidal integration state for intrinsic caps
        self._i_prev_cgd = 0.0
        self._i_prev_cgs = 0.0
        self._i_prev_cdd = 0.0

    def update_charge_state(self, voltages: Dict[str, float],
                            cap_currents: Optional[Dict[str, float]] = None) -> None:
        """Update charge state after a converged timestep."""
        charges = self.get_charges(voltages)
        self._q_prev = charges.copy()
        self._v_prev_tran = {
            "d": voltages.get(self.nodes[0], 0.0),
            "g": voltages.get(self.nodes[1], 0.0),
            "s": voltages.get(self.nodes[2], 0.0),
            "b": voltages.get(self.nodes[3], 0.0),
        }
        if cap_currents is not None:
            self._i_prev_cgd = cap_currents.get("i_cgd", 0.0)
            self._i_prev_cgs = cap_currents.get("i_cgs", 0.0)
            self._i_prev_cdd = cap_currents.get("i_cdd", 0.0)
