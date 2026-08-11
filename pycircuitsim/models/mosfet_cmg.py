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
PYCMG_PATH = Path(__file__).parent.parent.parent / "external_compact_models" / "PyCMG"
if str(PYCMG_PATH) not in sys.path:
    sys.path.insert(0, str(PYCMG_PATH))

try:
    from pycmg import Instance, get_shared_model
except ImportError as e:
    raise ImportError(
        f"Failed to import PyCMG: {e}. "
        "Ensure PyCMG is built and accessible in the project directory."
    )

from pycircuitsim.models.base import Component
from pycircuitsim.config import BSIMCMG_OSDI_PATH, DEFAULT_TEMPERATURE


class MOSFET_CMG(Component):
    """Base class for BSIM-CMG FinFET compact models (LEVEL=72).

    Handles PyCMG model/instance creation, evaluation caching, conductance
    extraction, capacitance/charge queries, and transient charge state.

    Subclasses (NMOS_CMG, PMOS_CMG) override only calculate_current()
    to apply the correct sign convention.

    Terminal order: [drain, gate, source, bulk]
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
        device_label: str = "MOSFET_CMG",
        multiplier: float = 1.0,
    ):
        """Initialize a BSIM-CMG FinFET device.

        Args:
            name: Component identifier (e.g., 'Mn1')
            nodes: List of exactly four node names [drain, gate, source, bulk]
            osdi_path: Path to BSIM-CMG OSDI binary (.osdi file)
            modelcard_path: Path to modelcard file (.pm, .lib, or .model)
            model_name: Model name from netlist (e.g., "nmos1")
            L: Channel length in meters
            NFIN: Number of fins
            TFIN: Fin thickness in meters (optional)
            HFIN: Fin height in meters (optional)
            FPITCH: Fin pitch in meters (optional)
            temperature: Device temperature in Kelvin (default: 300.15K = 27C)
            model_card_name: Name of the model in the modelcard file (e.g., "nmos_rvt").
                If None, falls back to model_name.
            device_label: Label for error messages (e.g., "NMOS_CMG", "PMOS_CMG")
            multiplier: Instance multiplier `m=` — the number of IDENTICAL
                devices in parallel (default 1.0 == a single device).
                Measured on ngspice-45.2 (nsvt_l109_f2, m=1 vs m=4): i(vd)
                scales exactly x4 while @nm[gm] / @nm[cgg] are IDENTICAL, i.e.
                m multiplies the residual and the resistive/reactive Jacobian
                and does NOT change device physics. It is therefore applied in
                the accessors below, never inside `_eval_dc` (whose cache must
                stay raw), and never via NFIN or parallel copies.

        Raises:
            ValueError: If node count is not 4, or L/NFIN/multiplier are invalid
            FileNotFoundError: If OSDI binary or modelcard not found
        """
        super().__init__(name, nodes, None)

        # Validate number of nodes
        if len(nodes) != 4:
            raise ValueError(f"{device_label} must have exactly 4 nodes, got {len(nodes)}")

        # Validate channel length and fin count
        if L <= 0:
            raise ValueError(f"Channel length L must be positive, got {L}")
        if NFIN <= 0:
            raise ValueError(f"Number of fins NFIN must be positive, got {NFIN}")
        if multiplier <= 0:
            raise ValueError(
                f"Instance multiplier m must be positive, got {multiplier}")

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
        self.m = float(multiplier)

        # Build instance parameters dictionary
        inst_params = {"L": self.L, "NFIN": self.NFIN}
        if self.TFIN is not None:
            inst_params["TFIN"] = self.TFIN
        if self.HFIN is not None:
            inst_params["HFIN"] = self.HFIN
        if self.FPITCH is not None:
            inst_params["FPITCH"] = self.FPITCH

        # Get the PyCMG model (loads modelcard parameters).
        # model_card_name overrides model_name for modelcard lookup: this
        # allows netlist model names (e.g., "nmos1") to differ from modelcard
        # model names (e.g., "nmos_rvt" in ASAP7).
        #
        # Memoised on (osdi, card, resolved name, geometry) — one 444 KB card
        # parse per distinct card instead of one per DEVICE. The geometry is
        # part of the key because L/NFIN/TFIN are MODEL-kind OSDI params living
        # in the shared model buffer (see pycmg.model.get_shared_model).
        self._pycmg_model = get_shared_model(
            osdi_path=osdi_path,
            modelcard_path=modelcard_path,
            model_name=model_name,
            model_card_name=model_card_name,
            geometry=inst_params,
        )

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
        self._q_prev2: Optional[Dict[str, float]] = None  # Two-step-ago charges (BDF-2)
        self._v_prev_tran: Optional[Dict[str, float]] = None

    def get_nodes(self) -> List[str]:
        """Return list of node names [drain, gate, source, bulk]."""
        return self.nodes

    def stamp_conductance(self, matrix, node_map: Dict[str, int]) -> None:
        """Placeholder — solver handles MOSFET conductance stamping directly."""
        pass

    def stamp_rhs(self, rhs, node_map: Dict[str, int]) -> None:
        """Placeholder — MOSFETs don't contribute to RHS directly."""
        pass

    def _eval_dc(self, voltages: Dict[str, float]) -> Dict[str, float]:
        """Evaluate DC operating point (with caching).

        Args:
            voltages: Dictionary mapping node names to voltage values

        Returns:
            Dictionary with keys: id, ig, is, ie, ids, qg, qd, qs, qb,
            gm, gds, gmb, cgg, cgd, cgs, cdg, cdd, etc.
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

    def set_temperature(self, temperature_kelvin: float) -> None:
        """Rebind this device at a new temperature, in place.

        Temperature is otherwise a CONSTRUCTION parameter, so a temperature
        sweep would have to rebuild every device per point (measured 818.7 ms
        per device). Rebinding the PyCMG instance is measured at 0.255 ms per
        device and bit-identical to a fresh instance at that temperature.

        The ``clear_cache()`` below is MANDATORY, not hygiene: ``_eval_cache``
        is keyed on (v_d, v_g, v_s, v_b) only, so a temperature change at
        unchanged voltages would otherwise return the STALE result — a silent
        wrong answer. The transient charge history is reset for the same
        reason (it belongs to the old temperature).

        Args:
            temperature_kelvin: New device temperature in KELVIN

        Raises:
            ValueError: If the value looks like Celsius (<= 200 K)
        """
        if temperature_kelvin <= 200.0:
            raise ValueError(
                f"Temperature must be in Kelvin (> 200 K), got "
                f"{temperature_kelvin}. Use temp_K = temp_C + 273.15.")

        self.temperature = float(temperature_kelvin)
        self._pycmg_instance.set_temperature(self.temperature)

        # Voltage-keyed caches and charge history are temperature-stale now.
        self.clear_cache()
        self._q_prev = None
        self._q_prev2 = None
        self._v_prev_tran = None

    def clear_cache(self) -> None:
        """Clear evaluation cache.

        Should be called at the start of each Newton-Raphson iteration.
        """
        self._eval_cache = None
        self._cache_voltages = None

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """Calculate drain terminal current. Must be overridden by subclass."""
        raise NotImplementedError("Subclasses must implement calculate_current()")

    def get_conductance(self, voltages: Dict[str, float]) -> Tuple[float, float, float]:
        """Calculate small-signal conductance parameters for Newton-Raphson.

        Returns the conductances extracted from the BSIM-CMG Jacobian:
        - g_ds = dI_ds/dV_ds (output conductance)
        - g_m = dI_ds/dV_gs (transconductance)
        - g_mb = dI_ds/dV_bs (bulk transconductance)

        All three are scaled by the instance multiplier ``m`` — they are the
        derivative of the (also m-scaled) stamped residual, so current,
        conductance, capacitance and charge MUST scale together or the NR
        Jacobian stops being the derivative of what is stamped.

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

        return (g_ds * self.m, g_m * self.m, g_mb * self.m)

    def get_capacitances(self, voltages: Dict[str, float]) -> Dict[str, float]:
        """Get terminal capacitances for AC analysis (x instance multiplier)."""
        result = self._eval_dc(voltages)
        m = self.m

        return {
            "cgg": result.get("cgg", 0.0) * m,
            "cgd": result.get("cgd", 0.0) * m,
            "cgs": result.get("cgs", 0.0) * m,
            "cdg": result.get("cdg", 0.0) * m,
            "cdd": result.get("cdd", 0.0) * m,
        }

    def get_charges(self, voltages: Dict[str, float]) -> Dict[str, float]:
        """Get terminal charges from BSIM-CMG eval_dc() (x instance multiplier).

        This is the ONLY route into ``init_charge_state`` /
        ``update_charge_state``, so the transient companion model inherits the
        multiplier from here.

        Returns:
            Dictionary with keys: qg, qd, qs, qb (Coulombs)
        """
        result = self._eval_dc(voltages)
        m = self.m
        return {
            "qg": result.get("qg", 0.0) * m,
            "qd": result.get("qd", 0.0) * m,
            "qs": result.get("qs", 0.0) * m,
            "qb": result.get("qb", 0.0) * m,
        }

    def init_charge_state(self, voltages: Dict[str, float]) -> None:
        """Initialize charge state from DC operating point.
        Must be called before transient analysis starts.
        """
        charges = self.get_charges(voltages)
        self._q_prev = charges.copy()
        self._q_prev2 = charges.copy()  # BDF-2: same as q_prev at DC
        self._v_prev_tran = {
            "d": voltages.get(self.nodes[0], 0.0),
            "g": voltages.get(self.nodes[1], 0.0),
            "s": voltages.get(self.nodes[2], 0.0),
            "b": voltages.get(self.nodes[3], 0.0),
        }
        # Terminal capacitive currents for charge-based trapezoidal integration
        # At DC operating point, dQ/dt = 0, so all initial currents are 0
        self._i_prev_gate = 0.0
        self._i_prev_drain = 0.0

    def update_charge_state(self, voltages: Dict[str, float],
                            cap_currents: Optional[Dict[str, float]] = None) -> None:
        """Update charge state after a converged timestep."""
        charges = self.get_charges(voltages)
        self._q_prev2 = self._q_prev.copy() if self._q_prev is not None else charges.copy()
        self._q_prev = charges.copy()
        self._v_prev_tran = {
            "d": voltages.get(self.nodes[0], 0.0),
            "g": voltages.get(self.nodes[1], 0.0),
            "s": voltages.get(self.nodes[2], 0.0),
            "b": voltages.get(self.nodes[3], 0.0),
        }
        if cap_currents is not None:
            self._i_prev_gate = cap_currents.get("i_gate", 0.0)
            self._i_prev_drain = cap_currents.get("i_drain", 0.0)


class NMOS_CMG(MOSFET_CMG):
    """N-Channel FinFET using BSIM-CMG compact model (LEVEL=72).

    Terminal order: [drain, gate, source, bulk]
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
        multiplier: float = 1.0,
    ):
        super().__init__(
            name, nodes, osdi_path, modelcard_path, model_name,
            L, NFIN, TFIN, HFIN, FPITCH, temperature, model_card_name,
            device_label="NMOS_CMG", multiplier=multiplier,
        )

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """Calculate drain terminal current (x instance multiplier).

        For NMOS ON: PyCMG id < 0 (SPICE: current OUT of drain), so -id > 0.
        The solver expects positive values; NMOS/PMOS sign difference is
        handled by the RHS stamping in solver.py.

        Returns:
            Drain terminal current (positive for NMOS ON)
        """
        result = self._eval_dc(voltages)
        # NMOS: negate SPICE id (negative when ON) to get positive value
        return -result["id"] * self.m


class PMOS_CMG(MOSFET_CMG):
    """P-Channel FinFET using BSIM-CMG compact model (LEVEL=72).

    Terminal order: [drain, gate, source, bulk]
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
        multiplier: float = 1.0,
    ):
        super().__init__(
            name, nodes, osdi_path, modelcard_path, model_name,
            L, NFIN, TFIN, HFIN, FPITCH, temperature, model_card_name,
            device_label="PMOS_CMG", multiplier=multiplier,
        )

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """Calculate drain terminal current (x instance multiplier).

        For PMOS ON: PyCMG id > 0 (SPICE: current INTO drain), so id > 0.
        The solver expects positive values; PMOS RHS stamping in solver.py
        handles the opposite current direction vs NMOS.

        Returns:
            Drain terminal current (positive for PMOS ON)
        """
        result = self._eval_dc(voltages)
        # PMOS: id is already positive when ON (SPICE: current INTO drain)
        return result["id"] * self.m
