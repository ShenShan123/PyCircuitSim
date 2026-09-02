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
import math
import sys
from pathlib import Path

# Add PyCMG to Python path if not already present
PYCMG_PATH = Path(__file__).parent.parent.parent / "external_compact_models" / "bsim_cmg"
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

    evaluator_boundary: str = "native"

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

        # --- SPICE-style NR voltage-limiting state (V7.5.0) ---
        # (vgs, vds, vbs) this device was last EVALUATED at, in the
        # NMOS-normalized frame (PMOS pairs are negated). None until the
        # first limited stamp of a solve; solvers reset it at solve entry.
        self._v_lim_prev: Optional[Tuple[float, float, float]] = None
        # True while the last nr_limit_voltages() call actually clamped —
        # the solvers refuse to declare convergence on such an iteration.
        self._nr_limited: bool = False
        # The terminal-voltage mapping of the last limited evaluation
        # (the ORIGINAL dict object when limiting was a no-op, so the
        # inactive path stays bit-identical).
        self._nr_v_eval: Optional[Dict[str, float]] = None
        # Retreat state for nr_retreat_voltages(): the previous anchor
        # pairs and the source reference of the current eval frame.
        self._v_lim_anchor: Optional[Tuple[float, float, float]] = None
        self._v_lim_source: float = 0.0

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

        # Call PyCMG eval_dc in the SOURCE-REFERENCED frame (V7.5.0).
        # BSIM-CMG physics depends on terminal differences only, but the
        # OSDI internal-node solve is NOT robust at large absolute
        # offsets: measured on nch_svt_mac l68n/nfin4, the identical
        # (vgs,vds,vbs)=(2.5,2.5,2.5) bias evaluates at s=-8.5 V and
        # diverges at s=-16.1 V (warm or cold). Newton runaway in a
        # floating subcircuit can park absolute levels anywhere, so the
        # eval frame is pinned to the source terminal. This mirrors NN
        # Rule 2 (source-relative frame; shift invariance makes it exact).
        try:
            result = self._pycmg_instance.eval_dc({
                "d": v_d - v_s,
                "g": v_g - v_s,
                "s": 0.0,
                "e": v_b - v_s  # Bulk terminal is 'e' in BSIM-CMG
            })
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc} [device {self.name}, L={self.L:g}, "
                f"NFIN={self.NFIN:g}, m={self.m:g}]") from exc

        # V7.5.0 — condense the just-loaded resistive Jacobian for the
        # full-terminal Newton stamp. The channel-only gm/gds/gmb opvars
        # miss the body-junction conductances entirely, which at high
        # temperature carry the drain current (measured at 125 C:
        # id = +1.8 mA against gds = 4.3e-13 S) — a Jacobian blind to
        # them locks circuit NR into a limit cycle.
        result["jac4"] = self._pycmg_instance.condense_last_jacobian()
        # ... and its reactive twin: the TRUE dQ/dV (verified == finite
        # differences of (qd,qg,qs,qb), bulk row/column included). The
        # SPICE 5-cap view reconstructed a transcap matrix with flipped
        # signs for floating-bulk devices, which at small dt turns the
        # transient Newton iteration into an amplifier.
        result["cjac4"] = self._pycmg_instance.condense_last_react()

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

    # ------------------------------------------------------------------
    # SPICE-style Newton-Raphson voltage limiting (V7.5.0)
    #
    # The solver's dv_limit trust region caps the per-iteration NODE step
    # but does nothing about where the device is EVALUATED: capped
    # iterations can still walk a gate to -3 V on a 0.65 V rail, at which
    # point the OSDI internal-node solve diverges and the whole timestep
    # dies (AnalogGym charge pump, amplifier slew edges). The classic
    # SPICE answer is damped limiting: evaluate the device at a limited
    # bias derived from the last evaluation point, and let the stamp
    # linearize about THAT point. The linear extrapolation keeps the
    # true derivative (a hard clamp would zero it and stall NR), and the
    # limit functions are the identity for small steps, so the converged
    # fixed point is bit-identical for every circuit that never strays.
    # ------------------------------------------------------------------

    #: NMOS-normalized polarity sign; PMOS_CMG overrides with -1.0.
    _nr_sign: float = 1.0
    #: Absolute |Vgs|/|Vds|/|Vbs| evaluation window (V). Legit FinFET
    #: operating points (incl. 2xVDD charge-pump boosts on 0.65-0.8 V
    #: rails) stay well inside; the OSDI internal solve stays sane inside.
    _NR_LIM_WINDOW: float = 2.5
    #: Threshold estimate for the fetlim shape (normalized frame). Only
    #: shapes step sizes — the converged answer never depends on it.
    _NR_LIM_VTO: float = 0.25
    #: Critical body-junction voltage for pnjlim. Only shapes step sizes;
    #: identity for converged (small-step) iterations.
    _NR_LIM_VCRIT: float = 0.6

    @staticmethod
    def _fetlim(vnew: float, vold: float, vto: float) -> float:
        """SPICE3 DEVfetlim: damped limiting for a FET gate-source step."""
        vtsthi = abs(2.0 * (vold - vto)) + 2.0
        vtstlo = 0.5 * vtsthi + 2.0
        vtox = vto + 3.5
        delv = vnew - vold
        if vold >= vto:
            if vold >= vtox:
                if delv <= 0.0:
                    # Going off
                    if vnew >= vtox:
                        if -delv > vtstlo:
                            vnew = vold - vtstlo
                    else:
                        vnew = max(vnew, vto + 2.0)
                else:
                    # Staying on
                    if delv >= vtsthi:
                        vnew = vold + vtsthi
            else:
                # Middle region
                if delv <= 0.0:
                    vnew = max(vnew, vto - 0.5)
                else:
                    vnew = min(vnew, vto + 4.0)
        else:
            # Off
            if delv <= 0.0:
                if -delv > vtsthi:
                    vnew = vold - vtsthi
            else:
                vtemp = vto + 0.5
                if vnew <= vtemp:
                    if delv > vtstlo:
                        vnew = vold + vtstlo
                else:
                    vnew = vtemp
        return vnew

    @staticmethod
    def _pnjlim(vnew: float, vold: float, vt: float, vcrit: float) -> float:
        """SPICE3 DEVpnjlim: logarithmic limiting of a forward junction step.

        Beyond vcrit a junction current is exponential in the bias, so a
        volt-scale Newton step means amp-scale currents (measured: a PMOS
        drain-body junction at +2.5 V / 125 C evaluates to 542 A). Steps
        past vcrit are compressed to thermal-voltage log increments.
        """
        if vnew > vcrit and abs(vnew - vold) > vt + vt:
            if vold > 0.0:
                arg = 1.0 + (vnew - vold) / vt
                vnew = vold + vt * math.log(arg) if arg > 0.0 else vcrit
            else:
                vnew = vt * math.log(vnew / vt)
        return vnew

    @staticmethod
    def _limvds(vnew: float, vold: float) -> float:
        """SPICE3 DEVlimvds, sign-symmetric.

        Classic SPICE swaps drain/source so vds >= 0 always; this
        simulator does not, and BSIM-CMG legitimately sees negative vds.
        Applying the classic function in the sign frame of ``vold``
        preserves its behaviour on the positive side and mirrors it on
        the negative side; a zero-crossing lands at most 0.5 V past zero
        per iteration (the classic -0.5 floor).
        """
        s = 1.0 if vold >= 0.0 else -1.0
        an, ao = s * vnew, s * vold
        if ao >= 3.5:
            if an > ao:
                an = min(an, 3.0 * ao + 2.0)
            elif an < 3.5:
                an = max(an, 2.0)
        else:
            if an > ao:
                an = min(an, 4.0)
            else:
                an = max(an, -0.5)
        return s * an

    def reset_nr_limits(self) -> None:
        """Forget the limiting anchor; solvers call this at solve entry."""
        self._v_lim_prev = None
        self._nr_limited = False
        self._nr_v_eval = None
        self._v_lim_anchor = None
        self._v_lim_source = 0.0

    def nr_retreat_voltages(self) -> Optional[Dict[str, float]]:
        """Halve the last limited step back toward the previous anchor.

        Called by the stamp when the OSDI evaluation at the limited bias
        still failed (internal-node NR divergence at extreme bias). The
        anchor is the previous iteration's evaluation point, which did
        evaluate; bisecting toward it always reaches evaluable territory.
        Returns the new terminal mapping, or None when there is no anchor
        (first evaluation of a solve) or no meaningful room left.
        """
        if self._v_lim_prev is None:
            return None
        anchor = getattr(self, "_v_lim_anchor", None)
        if anchor is None:
            # First evaluation of a solve (anchors were just reset):
            # retreat toward zero bias, which always evaluates.
            anchor = (0.0, 0.0, 0.0)
        cur = self._v_lim_prev
        span = max(abs(c - a) for c, a in zip(cur, anchor))
        if span < 1e-6:
            return None
        mid = tuple(0.5 * (c + a) for c, a in zip(cur, anchor))
        self._v_lim_prev = mid
        self._nr_limited = True
        sgn = self._nr_sign
        drain, gate, source, bulk = self.nodes
        v_s = self._v_lim_source
        v_eval = {
            source: v_s,
            drain: v_s + sgn * mid[1],
            gate: v_s + sgn * mid[0],
            bulk: v_s + sgn * mid[2],
        }
        v_eval[source] = v_s
        self._nr_v_eval = v_eval
        return v_eval

    def nr_limit_voltages(self, voltages: Dict[str, float]) -> Dict[str, float]:
        """Return the terminal-voltage mapping to EVALUATE this device at.

        Applies fetlim to vgs, sign-symmetric limvds to vds, a plain step
        cap to vbs — all in the NMOS-normalized frame relative to the
        last evaluation point — then clamps every pair into the absolute
        window. Terminals sharing one node (diode-connected devices)
        are reconciled to a single value (the most-limited proposal).

        Side effects: records the accepted pairs as the next anchor,
        sets ``_nr_limited``, and caches the returned mapping in
        ``_nr_v_eval``. Returns the ORIGINAL dict object when nothing
        was clamped, keeping the inactive path bit-identical.
        """
        drain, gate, source, bulk = self.nodes
        v_d = voltages.get(drain, 0.0)
        v_g = voltages.get(gate, 0.0)
        v_s = voltages.get(source, 0.0)
        v_b = voltages.get(bulk, 0.0)

        sgn = self._nr_sign
        raw = {
            "gs": sgn * (v_g - v_s),
            "ds": sgn * (v_d - v_s),
            "bs": sgn * (v_b - v_s),
        }

        win = self._NR_LIM_WINDOW
        lim: Dict[str, float] = {}
        old = self._v_lim_prev
        # Retreat anchor for nr_retreat_voltages(): the pairs of the last
        # evaluation, plus the source reference the eval frame hangs off.
        self._v_lim_anchor = old
        self._v_lim_source = v_s
        if old is None:
            for key in ("gs", "ds", "bs"):
                lim[key] = min(max(raw[key], -win), win)
        else:
            old_map = {"gs": old[0], "ds": old[1], "bs": old[2]}
            lim["gs"] = self._fetlim(raw["gs"], old_map["gs"], self._NR_LIM_VTO)
            lim["ds"] = self._limvds(raw["ds"], old_map["ds"])
            dbs = raw["bs"] - old_map["bs"]
            lim["bs"] = (old_map["bs"] + (1.0 if dbs > 0.0 else -1.0)
                         if abs(dbs) > 1.0 else raw["bs"])
            # Junction limiting (V7.5.0): the body diodes sit on the
            # (b,s) and (b,d) pairs, both forward at POSITIVE normalized
            # bias for either polarity. The b-d limit is honoured by
            # adjusting ds while keeping bs, mirroring SPICE's
            # vds = vbs - vbd reconstruction.
            vt_th = 8.617333262e-5 * self.temperature
            vcrit = self._NR_LIM_VCRIT
            lim["bs"] = self._pnjlim(lim["bs"], old_map["bs"], vt_th, vcrit)
            xbd_new = lim["bs"] - lim["ds"]
            xbd_old = old_map["bs"] - old_map["ds"]
            xbd_lim = self._pnjlim(xbd_new, xbd_old, vt_th, vcrit)
            if xbd_lim != xbd_new:
                lim["ds"] = lim["bs"] - xbd_lim
            for key in ("gs", "ds", "bs"):
                lim[key] = min(max(lim[key], -win), win)
            # Aliased terminals (e.g. diode-connected gate==drain) must
            # evaluate at ONE voltage: keep the most-limited proposal.
            groups: Dict[str, list] = {}
            for key, node in (("gs", gate), ("ds", drain), ("bs", bulk)):
                groups.setdefault(node, []).append(key)
            for keys in groups.values():
                if len(keys) > 1:
                    chosen = min(keys, key=lambda k: abs(lim[k] - old_map[k]))
                    for k in keys:
                        lim[k] = lim[chosen]

        self._v_lim_prev = (lim["gs"], lim["ds"], lim["bs"])
        if lim == raw:
            self._nr_limited = False
            self._nr_v_eval = voltages
            return voltages

        self._nr_limited = True
        v_eval = {
            source: v_s,
            drain: v_s + sgn * lim["ds"],
            gate: v_s + sgn * lim["gs"],
            bulk: v_s + sgn * lim["bs"],
        }
        # Source overwrites any alias above it in construction order;
        # rebuild with source last so a drain/gate/bulk tied to the
        # source node keeps the source reference exact.
        v_eval[source] = v_s
        self._nr_v_eval = v_eval
        return v_eval

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

        # Critical Design Rule 4: never abs(gds) — flipping a legitimately
        # negative gds to large-positive biases the Jacobian away from the
        # stamped current's true derivative, and NR settles into a limit
        # cycle around a residual it can never cancel (measured on the
        # 125 C amplifier start: a permanent 1.4 mA KCL violation with
        # vout bouncing +-30 mV forever). The gmin FLOOR lives at the
        # stamp (`max(g_ds, gmin)`), which handles the negative case.
        return (g_ds * self.m, g_m * self.m, g_mb * self.m)

    def get_terminal_stamp(self, voltages: Dict[str, float]):
        """Full 4-terminal Newton companion: currents + Jacobian (x m).

        Returns ``(i_out, g4)`` where ``i_out[t]`` is the current LEAVING
        node t into the device (A) and ``g4[t, j] = d i_out[t] / d V_j``
        (S), terminal order [d, g, s, b]. PyCMG reports currents in the
        opposite orientation and ``jac4`` is their derivative, so both
        are negated here; the [d,:] row of ``g4`` reproduces the classic
        (gds, gm, gmb) opvars exactly when junctions are off, and keeps
        the junction/gate-leakage terms the opvars never carried.
        """
        result = self._eval_dc(voltages)
        m = self.m
        i_out = [-result["id"] * m, -result["ig"] * m,
                 -result["is"] * m, -result["ie"] * m]
        g4 = result["jac4"] * (-m)
        return i_out, g4

    def get_charge_stamp(self, voltages: Dict[str, float]):
        """Full 4-terminal charge companion: charges + dQ/dV (x m).

        Returns ``(q4, c4)`` in terminal order [d, g, s, b]: ``q4[t]`` is
        the SPICE terminal charge (Coulombs) and ``c4[t, j] = dq4[t]/dV_j``
        (Farads), the condensed reactive OSDI Jacobian — bulk row and
        column included, signs exact (verified against finite differences
        of the reported charges). Charges keep PyCMG's orientation: the
        gate-row companion the old 3x3 block built from qg/cgg passes the
        NGSPICE transient gates, so +q with +dq/dV is the node-consistent
        pairing.
        """
        result = self._eval_dc(voltages)
        m = self.m
        q4 = [result["qd"] * m, result["qg"] * m,
              result["qs"] * m, result["qb"] * m]
        c4 = result["cjac4"] * m
        return q4, c4

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
        # V7.5.1 full charge companion tracks all four terminals.
        self._i_prev_source = 0.0
        self._i_prev_bulk = 0.0

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
            self._i_prev_source = cap_currents.get("i_source", 0.0)
            self._i_prev_bulk = cap_currents.get("i_bulk", 0.0)


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

    #: NR limiting works in the NMOS-normalized frame; PMOS pairs negate.
    _nr_sign: float = -1.0

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
