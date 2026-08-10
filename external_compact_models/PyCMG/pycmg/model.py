"""PyCMG ctypes-based OSDI interface -- public API.

All temperatures in this module are in **Kelvin**.
Convert from Celsius: ``temp_K = temp_C + 273.15``.
"""

from __future__ import annotations

import ctypes
import os
import warnings
from typing import Dict, List, Optional

import numpy as np

from .osdi_types import (
    ACCESS_FLAG_INSTANCE,
    ACCESS_FLAG_SET,
    ANALYSIS_DC,
    ANALYSIS_IC,
    ANALYSIS_STATIC,
    ANALYSIS_TRAN,
    CALC_OP,
    CALC_REACT_JACOBIAN,
    CALC_REACT_LIM_RHS,
    CALC_REACT_RESIDUAL,
    CALC_RESIST_JACOBIAN,
    CALC_RESIST_LIM_RHS,
    CALC_RESIST_RESIDUAL,
    ENABLE_LIM,
    EVAL_RET_FLAG_FATAL,
    INIT_LIM,
    PARA_KIND_MASK,
    PARA_KIND_OPVAR,
    PARA_TY_INT,
    PARA_TY_MASK,
    PARA_TY_REAL,
    OsdiDescriptor,
)
from .core import (
    OsdiInstance,
    OsdiLibrary,
    OsdiModel,
    OsdiSimulation,
    apply_param,
)
from .parser import ParsedModel, parse_modelcard, parse_number_with_suffix


class Model:
    """
    BSIM-CMG model wrapper for OSDI binary interface.

    The Model class loads an OSDI compiled model and associated modelcard parameters.
    It provides the foundation for creating device instances with specific geometry
    and operating conditions.

    Temperature handling:
        - The Model class itself does not store temperature
        - Temperature is specified when creating Instance objects
        - Temperature must be in KELVIN (see module docstring for conversion)

    Example:
        >>> model = Model(
        ...     "bsimcmg.osdi",
        ...     "asap7.pm",
        ...     "nmos_rvt"
        ... )
    """

    def __init__(self, osdi_path: str, modelcard_path: str, model_name: str,
                 model_card_name: Optional[str] = None) -> None:
        self._lib = OsdiLibrary(osdi_path)
        desc = self._lib.descriptor_by_name(model_name) if model_name else None
        if desc is None:
            desc = self._lib.descriptor(0)
        if desc is None:
            raise RuntimeError("OSDI descriptor not found")
        self._desc = desc
        self._model = OsdiModel(desc)
        self._modelcard_params: Dict[str, float] = {}
        if modelcard_path:
            # Use model_card_name if explicitly provided, otherwise fall back to
            # model_name so parse_modelcard targets the correct .model block.
            # Without this, parse_modelcard(target=None) matches the FIRST model
            # in the file -- which is NMOS in multi-model files like ASAP7,
            # causing PMOS models to get DEVTYPE=1 (NMOS) instead of DEVTYPE=0.
            target = model_card_name if model_card_name else model_name
            parsed = parse_modelcard(modelcard_path, target)
            self._modelcard_params = dict(parsed.params)

    @property
    def descriptor(self) -> OsdiDescriptor:
        return self._desc

    @property
    def model(self) -> OsdiModel:
        return self._model

    @property
    def modelcard_params(self) -> Dict[str, float]:
        return dict(self._modelcard_params)


class Instance:
    """
    BSIM-CMG device instance for DC and transient evaluation.

    An Instance represents a specific device with geometry parameters and
    operating conditions (temperature, voltages). It provides methods for
    DC operating point analysis and transient simulation.

    Temperature parameter:
        - The temperature parameter MUST be in KELVIN
        - Default is 300.15 K (27C, typical room temperature)
        - To convert from Celsius: temp_K = temp_C + 273.15

    Args:
        model: Model object containing OSDI descriptor and modelcard
        params: Instance-specific parameters (L, TFIN, NFIN, etc.)
        temperature: Operating temperature in KELVIN (default: 300.15 K = 27C)
        model_overrides: Optional model-level parameter overrides for process variation.

            WARNING: model_overrides writes directly to the shared OsdiModel buffer.
            Creating multiple Instances from the same Model with different
            model_overrides will cause earlier Instances to silently use the
            latest override values. For per-instance process variation, create
            a separate Model() per override set.

    Example:
        >>> # Create instance at room temperature (27C)
        >>> inst = Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2},
        ...                 temperature=300.15)  # 27C in Kelvin

        >>> # Create instance at elevated temperature (85C)
        >>> inst = Instance(model, params={"L": 16e-9},
        ...                 temperature=358.15)  # 85C = 85 + 273.15

        >>> # Create instance at cold temperature (-40C)
        >>> inst = Instance(model, params={"L": 16e-9},
        ...                 temperature=233.15)  # -40C = -40 + 273.15
    """

    def __init__(self, model: Model, params: Optional[Dict[str, float]] = None,
                 temperature: float = 300.15,
                 model_overrides: Optional[Dict[str, float]] = None) -> None:
        self._model = model
        self._inst = OsdiInstance(model.descriptor)
        self._temperature = temperature
        if temperature < 200.0:
            warnings.warn(
                f"Temperature {temperature} K is very low (< 200 K). "
                f"Did you pass Celsius instead of Kelvin? "
                f"Use temp_K = temp_C + 273.15 to convert.",
                stacklevel=2,
            )
        self._sim = OsdiSimulation()
        self._connected_terminals = int(model.descriptor.num_terminals)
        for key, val in model.modelcard_params.items():
            apply_param(model.descriptor, self._inst, model.model, key, val, True)
        # 2. Apply process variation overrides (model-level, overrides modelcard)
        # WARNING: model_overrides writes to shared OsdiModel buffer. Do not reuse
        # an Instance after creating another Instance from the same Model with
        # different model_overrides — the shared buffer will have the latest values.
        if model_overrides:
            for key, val in model_overrides.items():
                apply_param(model.descriptor, self._inst, model.model, key, val, True)
        if params:
            for key, val in params.items():
                apply_param(model.descriptor, self._inst, model.model, key, val, False)
        self._model.model.process_params()
        self._inst.bind_simulation(self._sim, model.model, self._connected_terminals, temperature)
        self._cache_terminal_positions()
        self._has_prev_solve = False
        self._has_prev_q = False
        self._prev_qg = 0.0
        self._prev_qd = 0.0
        self._prev_qs = 0.0
        self._prev_qb = 0.0

    def _cache_terminal_positions(self) -> None:
        """Cache g/d/s positions in terminal_indices to avoid per-call linear scans."""
        def _find(name: str) -> int:
            for i, idx in enumerate(self._sim.terminal_indices):
                if self._sim.node_names[idx] == name:
                    return i
            return -1

        self._term_g: int = _find("g")
        self._term_d: int = _find("d")
        self._term_s: int = _find("s")

    def set_params(self, params: Dict[str, float], allow_rebind: bool = False) -> None:
        for key, val in params.items():
            apply_param(self._model.descriptor, self._inst, self._model.model, key, val, False)
        self._model.model.process_params()
        internal = self._inst.process_params(self._model.model, self._connected_terminals, self._temperature)
        if len(internal) != len(self._sim.internal_indices):
            if not allow_rebind:
                raise RuntimeError("topology changed; rebind required")
            self._sim = OsdiSimulation()
            self._inst.bind_simulation(self._sim, self._model.model, self._connected_terminals, self._temperature)
            self._cache_terminal_positions()
            self._has_prev_solve = False
            self._has_prev_q = False
            self._prev_qg = 0.0
            self._prev_qd = 0.0
            self._prev_qs = 0.0
            self._prev_qb = 0.0

    def internal_node_count(self) -> int:
        return len(self._sim.internal_indices)

    def state_count(self) -> int:
        return int(self._model.descriptor.num_states)

    def _set_node_voltages(self, nodes: Dict[str, float], seed_internal: bool) -> None:
        for name in ("d", "g", "s", "e"):
            value = float(nodes.get(name, 0.0))
            self._sim.set_voltage(name, value)
        if seed_internal:
            if "di" in self._sim.node_index and "di" not in nodes:
                self._sim.set_voltage("di", self._sim.solve[self._sim.node_index["d"]])
            if "si" in self._sim.node_index and "si" not in nodes:
                self._sim.set_voltage("si", self._sim.solve[self._sim.node_index["s"]])

    def _read_current(self, name: str) -> float:
        idx = self._sim.node_index.get(name)
        if idx is None:
            return 0.0
        return -self._sim.residual_resist[idx]

    def _read_current_from(self, residuals: List[float], name: str) -> float:
        idx = self._sim.node_index.get(name)
        if idx is None or idx >= len(residuals):
            return 0.0
        return -residuals[idx]

    def _read_terminal_current(self, term: str, internal: str) -> float:
        """Read terminal current from internal node residual.

        OSDI residual_resist stores KCL residuals (current INTO node).
        Terminal current convention: I = -F (current OUT of device terminal).
        Must negate, same as _read_current() / _read_current_from().
        """
        idx_internal = self._sim.node_index.get(internal)
        if idx_internal is not None and idx_internal < len(self._sim.residual_resist):
            return -float(self._sim.residual_resist[idx_internal])
        return self._read_current(term)

    def _read_opvar(self, name: str, alias: str) -> Optional[float]:
        desc = self._model.descriptor
        name_lower = name.lower()
        alias_lower = alias.lower()
        total = int(desc.num_params + desc.num_opvars)
        for i in range(total):
            param = desc.param_opvar[i]
            if (param.flags & PARA_KIND_MASK) != PARA_KIND_OPVAR:
                continue
            matched = False
            if param.num_alias == 0:
                if param.name and param.name[0]:
                    if param.name[0].decode("utf-8", errors="replace").lower() == name_lower:
                        matched = True
            else:
                for a in range(param.num_alias):
                    alias_name = param.name[a]
                    if not alias_name:
                        continue
                    alias_str = alias_name.decode("utf-8", errors="replace").lower()
                    if alias_str in (name_lower, alias_lower):
                        matched = True
                        break
            if not matched:
                continue
            ptr = desc.access(self._inst.data(), self._model.model.data(), i,
                              ACCESS_FLAG_SET | ACCESS_FLAG_INSTANCE)
            if not ptr:
                return None
            ty = param.flags & PARA_TY_MASK
            if ty == PARA_TY_INT:
                return float(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int32))[0])
            if ty == PARA_TY_REAL:
                return float(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))[0])
            return None
        return None

    @staticmethod
    def _build_full_jacobian(sim: OsdiSimulation, values: "ctypes.Array") -> np.ndarray:
        n = len(sim.node_names)
        out = np.zeros((n, n), dtype=float)
        for k, (row, col) in enumerate(sim.jacobian_info):
            if row < n and col < n and k < len(values):
                out[row, col] = values[k]
        return out

    @staticmethod
    def _schur_condense(full: np.ndarray,
                        external: List[int],
                        internal: List[int]) -> Optional[np.ndarray]:
        """Schur complement condensation: reduce NxN matrix to external-only.

        Computes: M_ee - M_ei @ M_ii^{-1} @ M_ie

        Works for both real and complex matrices (capacitance uses complex
        Y = G + jωC; resistive Jacobian uses real G).

        Args:
            full: NxN matrix (real or complex)
            external: indices of external (terminal) nodes
            internal: indices of internal nodes

        Returns:
            ne×ne condensed matrix (same dtype as input), or None if
            the internal node matrix is singular (LinAlgError).
        """
        ne = len(external)
        ni = len(internal)
        dtype = full.dtype

        m_ee = np.zeros((ne, ne), dtype=dtype)
        for r in range(ne):
            for c in range(ne):
                m_ee[r, c] = full[external[r], external[c]]

        if ni == 0:
            return m_ee

        m_ei = np.zeros((ne, ni), dtype=dtype)
        m_ie = np.zeros((ni, ne), dtype=dtype)
        m_ii = np.zeros((ni, ni), dtype=dtype)
        for r in range(ne):
            for c in range(ni):
                m_ei[r, c] = full[external[r], internal[c]]
        for r in range(ni):
            for c in range(ne):
                m_ie[r, c] = full[internal[r], external[c]]
            for c in range(ni):
                m_ii[r, c] = full[internal[r], internal[c]]

        try:
            m_ie_sol = np.linalg.solve(m_ii, m_ie)
        except np.linalg.LinAlgError:
            return None

        return m_ee - m_ei @ m_ie_sol

    @staticmethod
    def _condense_capacitance(g_full: np.ndarray,
                              c_full: np.ndarray,
                              external: List[int],
                              internal: List[int]) -> np.ndarray:
        ne = len(external)
        c_condensed = np.zeros((ne, ne), dtype=float)
        if ne == 0:
            return c_condensed
        y_full = g_full.astype(complex) + 1j * c_full.astype(complex)
        y_condensed = Instance._schur_condense(y_full, external, internal)
        if y_condensed is None:
            return c_condensed  # zeros on singular internal matrix (matches old behavior)
        return np.imag(y_condensed).astype(float)

    def _condense_caps(self) -> Dict[str, float]:
        """Extract condensed capacitance matrix matching SPICE convention.

        The OSDI reactive Jacobian stores dQ/dV in KCL convention where
        off-diagonal entries are negative (Y-matrix convention). SPICE
        capacitance parameters (cgg, cgd, etc.) use the opposite sign for
        off-diagonal elements: C_ij = -Y_ij/jw for i != j.

        This method negates off-diagonal entries so that output matches
        NGSPICE's @n1[cXX] operating-point variables.
        """
        g_full = self._build_full_jacobian(self._sim, self._sim.jacobian_resist)
        c_full = self._build_full_jacobian(self._sim, self._sim.jacobian_react)
        c_condensed = self._condense_capacitance(g_full, c_full,
                                                 self._sim.terminal_indices,
                                                 self._sim.internal_indices)
        caps = {"cgg": 0.0, "cgd": 0.0, "cgs": 0.0, "cdg": 0.0, "cdd": 0.0}
        if c_condensed.size == 0:
            return caps
        g, d, s = self._term_g, self._term_d, self._term_s
        if g >= 0:
            # Diagonal: direct value (self-capacitance, positive)
            caps["cgg"] = float(c_condensed[g, g])
            # Off-diagonal: negate to convert from Y-matrix to SPICE cap convention
            if d >= 0:
                caps["cgd"] = -float(c_condensed[g, d])
            if s >= 0:
                caps["cgs"] = -float(c_condensed[g, s])
        if d >= 0 and g >= 0:
            caps["cdg"] = -float(c_condensed[d, g])
            # Diagonal: direct value
            caps["cdd"] = float(c_condensed[d, d])
        return caps

    def get_jacobian_matrix(self, nodes: Dict[str, float]) -> np.ndarray:
        """Extract the condensed 4x4 resistive Jacobian matrix.

        BSIM-CMG has internal nodes (di, si, etc.) making the raw Jacobian
        7x7 or larger. This method condenses it to the 4 external terminals
        using Schur complement elimination:

            G_ext = G_ee - G_ei * G_ii^-1 * G_ie

        This is the matrix a circuit simulator's Newton-Raphson solver sees.

        Returns a 4x4 numpy array where:
        - Rows/cols correspond to terminal order in sim.terminal_indices
          (typically d, g, s, e)
        - G[i,j] = dI_i / dV_j (conductance, Siemens)

        Args:
            nodes: Dict with keys 'd', 'g', 's', 'e' and voltage values

        Returns:
            4x4 condensed Jacobian matrix as numpy array
        """
        # Run DC evaluation to populate OSDI Jacobian buffers
        self.eval_dc(nodes)

        # Build full NxN resistive Jacobian from OSDI
        g_full = self._build_full_jacobian(self._sim, self._sim.jacobian_resist)

        # Condense to external-only using Schur complement
        ext = self._sim.terminal_indices
        intn = self._sim.internal_indices
        g_condensed = self._schur_condense(g_full, ext, intn)

        if g_condensed is None:
            # Fallback: return external-only block negated (matches old behavior)
            ne = len(ext)
            g_ee = np.zeros((ne, ne))
            for r in range(ne):
                for c in range(ne):
                    g_ee[r, c] = g_full[ext[r], ext[c]]
            return -g_ee

        # Negate: OSDI jacobian_resist stores dF/dV where F is KCL residual
        # (current into node). Terminal currents use I = -F, so dI/dV = -dF/dV.
        return -g_condensed

    def eval_dc(self, nodes: Dict[str, float]) -> Dict[str, float]:
        """
        Perform DC operating point analysis.

        Evaluates the device at specified terminal voltages and returns
        terminal currents, charges, derivatives, and capacitances.

        Temperature:
            Uses the temperature specified during Instance initialization (in KELVIN).
            To change temperature, create a new Instance with the desired temperature.

        Args:
            nodes: Dictionary mapping terminal names to voltages
                   Required keys: "d" (drain), "g" (gate), "s" (source), "e" (bulk)
                   Example: {"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0}

        Returns:
            Dictionary with 18 output values:
            - Currents (A): id, ig, is, ie, ids
            - Charges (C): qg, qd, qs, qb
            - Derivatives (S): gm, gds, gmb
            - Capacitances (F): cgg, cgd, cgs, cdg, cdd

        Example:
            >>> inst = Instance(model, params={"L": 16e-9}, temperature=300.15)  # 27C
            >>> result = inst.eval_dc({"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0})
            >>> print(f"Drain current: {result['id']:.6e} A")
            >>> print(f"Transconductance: {result['gm']:.6e} S")
        """
        self._set_node_voltages(nodes, True)
        # V6.4.7 S9b generator floor fix: the internal-node NR tolerance is
        # env-overridable via ``NN_DC_SOLVE_TOL``. The legacy default (1e-9)
        # lets a warm-started solve satisfy the residual test WITHOUT moving
        # the internal nodes for any true |id| < ~1e-9 A, so deep-subthreshold
        # / OFF-state rows came back as EXACT 0.0 (the 6-8 % zero-row
        # artifact). NN data generation exports ``NN_DC_SOLVE_TOL=1e-12`` so
        # sub-nA currents resolve; ``Instance.eval_dc`` default is unchanged.
        _dc_tol = float(os.environ.get("NN_DC_SOLVE_TOL", "1e-9"))
        converged = self._inst.solve_internal_nodes(self._model.model, self._sim, 200, _dc_tol)
        if not converged:
            raise RuntimeError(
                "Internal node NR failed to converge at "
                + ", ".join(f"{k}={nodes.get(k, 0.0):.4f}" for k in ("d", "g", "s", "e"))
            )
        flags = (ANALYSIS_DC | ANALYSIS_STATIC | CALC_RESIST_JACOBIAN |
                 CALC_RESIST_RESIDUAL | CALC_RESIST_LIM_RHS |
                 CALC_REACT_JACOBIAN | CALC_REACT_RESIDUAL |
                 CALC_REACT_LIM_RHS | CALC_OP | ENABLE_LIM | INIT_LIM)
        ret = self._inst.eval(self._model.model, self._sim, flags)
        if ret & EVAL_RET_FLAG_FATAL:
            raise RuntimeError(f"OSDI eval fatal error: flags=0x{ret:08x}")
        self._sim.clear()
        self._inst.load_residuals(self._model.model, self._sim)
        self._inst.load_jacobian(self._model.model, self._sim)

        out: Dict[str, float] = {
            "id": self._read_current("d"),
            "ig": self._read_current("g"),
            "is": self._read_current("s"),
            "ie": self._read_current("e"),
        }
        # Drain-source current (Ids = Id - Is for common-source configuration)
        out["ids"] = out["id"] - out["is"]

        qg = self._read_opvar("qg", "qgate") or 0.0
        qd = self._read_opvar("qd", "qdrain") or 0.0
        qs = self._read_opvar("qs", "qsource") or 0.0
        qb = self._read_opvar("qb", "qbulk")
        if qb is None:
            qb = self._read_opvar("qe", "qe") or 0.0
        out.update({"qg": qg, "qd": qd, "qs": qs, "qb": qb})

        gm = self._read_opvar("gm", "gm") or 0.0
        gds = self._read_opvar("gds", "gds") or 0.0
        gmb = self._read_opvar("gmbs", "gmbs")
        if gmb is None:
            gmb = self._read_opvar("gmb", "gmb") or 0.0
        out.update({"gm": gm, "gds": gds, "gmb": gmb})

        out.update(self._condense_caps())
        return out

    def eval_tran(self, nodes: Dict[str, float], time: float, delta_t: float,
                  prev_state: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Perform transient analysis at a specific time point.

        Evaluates the device with time-dependent effects including charge storage
        and capacitive currents. Suitable for transient simulation and AC analysis.

        Temperature:
            Uses the temperature specified during Instance initialization (in KELVIN).
            Temperature effects on capacitances and charges are evaluated at
            the initialization temperature.

        Args:
            nodes: Dictionary mapping terminal names to voltages
                   Required keys: "d" (drain), "g" (gate), "s" (source), "e" (bulk)
                   Example: {"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0}
            time: Current simulation time in seconds
            delta_t: Time step in seconds (must be positive)
            prev_state: Optional previous state vector for multi-step simulations

        Returns:
            Dictionary with 9 output values:
            - Currents (A): id, ig, is, ie, ids (includes displacement currents)
            - Charges (C): qg, qd, qs, qb

        Example:
            >>> inst = Instance(model, params={"L": 16e-9}, temperature=358.15)  # 85C
            >>> result = inst.eval_tran(
            ...     nodes={"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0},
            ...     time=1e-9,
            ...     delta_t=1e-12
            ... )
            >>> print(f"Drain current (with dQ/dt): {result['id']:.6e} A")
        """
        if delta_t <= 0.0:
            raise RuntimeError("delta_t must be positive")
        if prev_state is not None:
            if len(prev_state) != len(self._sim.state_prev):
                raise RuntimeError("prev_state size mismatch")
            for i, val in enumerate(prev_state):
                self._sim.state_prev[i] = val
        self._set_node_voltages(nodes, True)
        self._sim.copy_solve_to_prev()
        self._sim.has_prev_solve = True
        if not self._has_prev_solve:
            ic_flags = (ANALYSIS_IC | CALC_RESIST_RESIDUAL |
                        CALC_RESIST_LIM_RHS | CALC_REACT_RESIDUAL |
                        CALC_REACT_LIM_RHS | CALC_OP | ENABLE_LIM | INIT_LIM)
            ret = self._inst.eval_with_time(self._model.model, self._sim, ic_flags, time)
            if ret & EVAL_RET_FLAG_FATAL:
                raise RuntimeError(f"OSDI eval fatal error: flags=0x{ret:08x}")
            if len(self._sim.state_prev) == len(self._sim.state_next):
                self._sim.state_prev, self._sim.state_next = self._sim.state_next, self._sim.state_prev
        num_states = self.state_count()
        alpha = 1.0 / delta_t
        if num_states > 0:
            for key, val in [
                ("dt", delta_t), ("delta_t", delta_t), ("delta", delta_t),
                ("h", delta_t), ("step", delta_t), ("alpha", alpha),
                ("t", time), ("time", time), ("abstime", time),
            ]:
                self._sim.set_sim_param(key, val)
        if num_states == 0:
            # Relaxed tolerance vs eval_dc's 1e-9: after the first-call IC
            # eval the internal-node residual saturates at ~|Id| because the
            # internal-only Jacobian cannot null the external coupling term.
            # The circuit-level NR provides the outer convergence loop.
            converged = self._inst.solve_internal_nodes(self._model.model, self._sim, 200, 1e-3)
            if not converged:
                warnings.warn(
                    "Internal node NR did not converge (tran) at "
                    + ", ".join(f"{k}={nodes.get(k, 0.0):.4f}" for k in ("d", "g", "s", "e")),
                    stacklevel=2,
                )
            flags = (ANALYSIS_DC | ANALYSIS_STATIC | CALC_RESIST_JACOBIAN |
                     CALC_RESIST_RESIDUAL | CALC_RESIST_LIM_RHS |
                     CALC_REACT_JACOBIAN | CALC_REACT_RESIDUAL |
                     CALC_REACT_LIM_RHS | CALC_OP | ENABLE_LIM | INIT_LIM)
            ret = self._inst.eval(self._model.model, self._sim, flags)
            if ret & EVAL_RET_FLAG_FATAL:
                raise RuntimeError(f"OSDI eval fatal error: flags=0x{ret:08x}")
            self._sim.clear()
            self._inst.load_residuals(self._model.model, self._sim)
        else:
            converged = self._inst.solve_internal_nodes_tran(self._model.model, self._sim, time, alpha, 200, 1e-9)
            if not converged:
                warnings.warn(
                    "Internal node NR did not converge (tran) at "
                    + ", ".join(f"{k}={nodes.get(k, 0.0):.4f}" for k in ("d", "g", "s", "e")),
                    stacklevel=2,
                )
            flags = (ANALYSIS_TRAN | CALC_RESIST_JACOBIAN | CALC_RESIST_RESIDUAL |
                     CALC_RESIST_LIM_RHS | CALC_REACT_JACOBIAN |
                     CALC_REACT_RESIDUAL | CALC_REACT_LIM_RHS |
                     CALC_OP | ENABLE_LIM | INIT_LIM)
            ret = self._inst.eval_with_time(self._model.model, self._sim, flags, time)
            if ret & EVAL_RET_FLAG_FATAL:
                raise RuntimeError(f"OSDI eval fatal error: flags=0x{ret:08x}")
            self._sim.clear()
            self._inst.load_residuals(self._model.model, self._sim)
            self._inst.load_jacobian_tran(self._model.model, self._sim, alpha)
            self._inst.load_spice_rhs_tran(self._model.model, self._sim, alpha)

        total_residual = [
            self._sim.residual_resist[i] + alpha * self._sim.residual_react[i] - self._sim.rhs_tran[i]
            for i in range(len(self._sim.residual_resist))
        ]

        qg = self._read_opvar("qg", "qgate") or 0.0
        qd = self._read_opvar("qd", "qdrain") or 0.0
        qs = self._read_opvar("qs", "qsource") or 0.0
        qb = self._read_opvar("qb", "qbulk")
        if qb is None:
            qb = self._read_opvar("qe", "qe") or 0.0

        out: Dict[str, float] = {"qg": qg, "qd": qd, "qs": qs, "qb": qb}
        if num_states == 0:
            dqg_dt = dqd_dt = dqs_dt = dqb_dt = 0.0
            if self._has_prev_q:
                dqg_dt = (qg - self._prev_qg) * alpha
                dqd_dt = (qd - self._prev_qd) * alpha
                dqs_dt = (qs - self._prev_qs) * alpha
                dqb_dt = (qb - self._prev_qb) * alpha
            out["id"] = self._read_terminal_current("d", "di") + dqd_dt
            out["ig"] = self._read_current("g") + dqg_dt
            out["is"] = self._read_terminal_current("s", "si") + dqs_dt
            out["ie"] = self._read_current("e") + dqb_dt
            self._prev_qg = qg
            self._prev_qd = qd
            self._prev_qs = qs
            self._prev_qb = qb
            self._has_prev_q = True
        else:
            out["id"] = self._read_current_from(total_residual, "d")
            out["ig"] = self._read_current_from(total_residual, "g")
            out["is"] = self._read_current_from(total_residual, "s")
            out["ie"] = self._read_current_from(total_residual, "e")

        # Drain-source current (Ids = Id - Is for common-source configuration)
        out["ids"] = out["id"] - out["is"]

        self._sim.copy_solve_to_prev()
        self._has_prev_solve = True
        self._sim.has_prev_solve = True
        if len(self._sim.state_prev) == len(self._sim.state_next):
            self._sim.state_prev, self._sim.state_next = self._sim.state_next, self._sim.state_prev
        return out
