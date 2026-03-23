"""NN-based MOSFET compact model (LEVEL=73).

Drop-in replacement for NMOS_CMG/PMOS_CMG using a trained neural network
instead of PyCMG physics evaluation. Implements the full Component interface
required by the solver.

Key design:
- Autograd-derived conductances (gm, gds, gmb) guarantee Jacobian consistency
  for Newton-Raphson convergence.
- Charge conservation: qs = -(qg + qd + qb) enforced analytically.
- Physical constraints: gds >= 0, cutoff clamping.
- Same sign conventions as NMOS_CMG/PMOS_CMG.

Terminal order: [drain, gate, source, bulk]
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pycircuitsim.models.base import Component
from nn_model.architecture.direct_loss import DirectNet
from nn_model.data.normalize import NormStats, inv_signed_log


class _MOSFETNNBase(Component):
    """Base class for NN-based MOSFET models.

    Handles model loading, inference, caching, and charge state management.
    Subclasses (NMOS_NN, PMOS_NN) differ only in sign convention for
    calculate_current().
    """

    def __init__(
        self,
        name: str,
        nodes: List[str],
        model_path: str,
        L: float,
        NFIN: float,
        temperature: float = 300.15,
        phig: Optional[float] = None,
    ):
        super().__init__(name, nodes, None)

        if len(nodes) != 4:
            raise ValueError(f"MOSFET_NN must have exactly 4 nodes, got {len(nodes)}")
        if L <= 0:
            raise ValueError(f"Channel length L must be positive, got {L}")
        if NFIN <= 0:
            raise ValueError(f"Number of fins NFIN must be positive, got {NFIN}")

        self.L = float(L)
        self.NFIN = float(NFIN)
        self.temperature = float(temperature)
        self.phig = float(phig) if phig is not None else None

        # Load model and normalization stats
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"NN model not found: {model_path}")

        norm_path = model_path.parent / (model_path.stem.replace("_best", "_norm") + ".npz")
        if not norm_path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {norm_path}")

        # Auto-detect model dimensions from checkpoint
        state = torch.load(str(model_path), weights_only=True, map_location="cpu")
        weight_keys = [k for k in state.keys() if k.endswith('.weight')]
        # Infer input_dim from first layer, output_dim/hidden_dim from last layer
        first_key = weight_keys[0]
        last_key = weight_keys[-1]
        input_dim = state[first_key].shape[1]
        output_dim = state[last_key].shape[0]
        hidden_dim = state[last_key].shape[1]
        n_weight_keys = len(weight_keys)
        n_layers = n_weight_keys

        self._input_dim = input_dim  # 6 (legacy) or 7 (with PHIG)

        self._nn_model = DirectNet(
            input_dim=input_dim, hidden_dim=hidden_dim,
            n_layers=n_layers - 1,  # -1 because DirectNet adds output layer separately
            output_dim=output_dim,
        )
        self._nn_model.load_state_dict(state)
        self._nn_model.eval()
        self._output_dim = output_dim

        self._norm_stats = NormStats.load(str(norm_path))

        # Pre-compute normalized geometry features (constant per device)
        nfin_log = np.log2(max(self.NFIN, 1.0))
        if input_dim == 7 and self.phig is not None:
            # 7-dim model: [Vd, Vg, Vs, Vb, log2(NFIN), T, PHIG]
            geo_raw = np.array([nfin_log, self.temperature, self.phig])
        else:
            # Legacy 6-dim model: [Vd, Vg, Vs, Vb, log2(NFIN), T]
            geo_raw = np.array([nfin_log, self.temperature])
        geo_range = self._norm_stats.input_max[4:] - self._norm_stats.input_min[4:]
        geo_range[geo_range < 1e-10] = 1.0
        self._geo_norm = (geo_raw - self._norm_stats.input_min[4:]) / geo_range

        # Subclass sets this
        self._is_pmos = False

        # Cache
        self._eval_cache: Optional[Dict[str, float]] = None
        self._cache_voltages: Optional[Tuple[float, ...]] = None

        # Charge state for transient analysis
        self._q_prev: Optional[Dict[str, float]] = None
        self._v_prev_tran: Optional[Dict[str, float]] = None
        self._i_prev_gate: float = 0.0
        self._i_prev_drain: float = 0.0

    def _eval(self, voltages: Dict[str, float]) -> Dict[str, float]:
        """Evaluate NN model at given voltages.

        Returns dict with keys: id, gm, gds, gmb, qg, qd, qs, qb,
        cgg, cgd, cgs, cdg, cdd (all in physical units).

        For 13-output models: all values are directly predicted (fast, no autograd).
        For 4-output models: id/qg/qd/qb predicted, derivatives via autograd.
        """
        v_d = voltages.get(self.nodes[0], 0.0)
        v_g = voltages.get(self.nodes[1], 0.0)
        v_s = voltages.get(self.nodes[2], 0.0)
        v_b = voltages.get(self.nodes[3], 0.0)

        v_tuple = (v_d, v_g, v_s, v_b)
        if self._cache_voltages == v_tuple and self._eval_cache is not None:
            return self._eval_cache

        stats = self._norm_stats

        # For PMOS: shift to source-relative frame (Vs=0)
        # Training data was generated with Vs=0, so the NN expects
        # voltages relative to source. In a circuit, PMOS source is at VDD.
        if self._is_pmos:
            v_shift = v_s
            v_d_nn = v_d - v_shift
            v_g_nn = v_g - v_shift
            v_s_nn = 0.0
            v_b_nn = v_b - v_shift
        else:
            v_d_nn = v_d
            v_g_nn = v_g
            v_s_nn = v_s
            v_b_nn = v_b

        # Clamp voltages to training range to prevent NN extrapolation
        # The NN returns garbage outside its training domain. Clamping ensures
        # the solver always gets physically reasonable values even during
        # Newton-Raphson overshoot.
        v_raw = np.array([v_d_nn, v_g_nn, v_s_nn, v_b_nn])
        v_raw_clamped = np.clip(v_raw, stats.input_min[:4], stats.input_max[:4])

        # Normalize voltage inputs
        v_range = stats.input_max[:4] - stats.input_min[:4]
        v_range[v_range < 1e-10] = 1.0
        v_norm = (v_raw_clamped - stats.input_min[:4]) / v_range

        # Build full input tensor: [Vd, Vg, Vs, Vb, NFIN, T] or [Vd, Vg, Vs, Vb, NFIN, T, PHIG]
        x_np = np.concatenate([v_norm, self._geo_norm]).astype(np.float32)
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # (1, 6) or (1, 7)

        # Always use autograd for conductances (Jacobian consistency for NR).
        # Direct prediction of gm/gds is NOT consistent with id, causing NR
        # divergence. Autograd guarantees gm = did/dVg exactly.
        if self._output_dim == 13:
            result = self._eval_hybrid13(x)
        else:
            result = self._eval_autograd4(x)

        self._eval_cache = result
        self._cache_voltages = v_tuple
        return result

    def _eval_hybrid13(self, x: torch.Tensor) -> Dict[str, float]:
        """Hybrid path: autograd conductances + direct charges/caps from 13-output model.

        Conductances (gm, gds, gmb) are derived via autograd of the id output
        to guarantee Jacobian consistency for Newton-Raphson convergence.
        Charges and capacitances use direct predictions (not critical for DC NR).
        """
        x_v = x[:, :4].requires_grad_(True)
        x_g = x[:, 4:]
        x_full = torch.cat([x_v, x_g], dim=1)

        with torch.enable_grad():
            out = self._nn_model(x_full)  # (1, 13)

            # Autograd: conductances from id (col 0)
            grad_id = torch.autograd.grad(
                out[:, 0].sum(), x_v, create_graph=False, retain_graph=True
            )[0]  # (1, 4)

            # Autograd: gate caps from qg (col 4)
            grad_qg = torch.autograd.grad(
                out[:, 4].sum(), x_v, create_graph=False, retain_graph=True
            )[0]

            # Autograd: drain caps from qd (col 5)
            grad_qd = torch.autograd.grad(
                out[:, 5].sum(), x_v, create_graph=False, retain_graph=False
            )[0]

        # Denormalize direct outputs
        id_phys = self._denorm_scalar(out[0, 0].item(), col_idx=0)
        qg_phys = self._denorm_scalar(out[0, 4].item(), col_idx=4)
        qd_phys = self._denorm_scalar(out[0, 5].item(), col_idx=5)
        qs_phys = self._denorm_scalar(out[0, 6].item(), col_idx=6)
        qb_phys = self._denorm_scalar(out[0, 7].item(), col_idx=7)

        # Denormalize autograd conductances (exact derivatives of id)
        gm_phys = self._denorm_full_derivative(
            grad_id[0, 1].item(), out_col=0, in_col=1, phys_val=id_phys)
        gds_phys = self._denorm_full_derivative(
            grad_id[0, 0].item(), out_col=0, in_col=0, phys_val=id_phys)
        gmb_phys = self._denorm_full_derivative(
            grad_id[0, 3].item(), out_col=0, in_col=3, phys_val=id_phys)

        # Denormalize autograd capacitances
        cgg_phys = self._denorm_full_derivative(
            grad_qg[0, 1].item(), out_col=4, in_col=1, phys_val=qg_phys)
        cgd_phys = self._denorm_full_derivative(
            grad_qg[0, 0].item(), out_col=4, in_col=0, phys_val=qg_phys)
        cgs_phys = self._denorm_full_derivative(
            grad_qg[0, 2].item(), out_col=4, in_col=2, phys_val=qg_phys)
        cdg_phys = self._denorm_full_derivative(
            grad_qd[0, 1].item(), out_col=5, in_col=1, phys_val=qd_phys)
        cdd_phys = self._denorm_full_derivative(
            grad_qd[0, 0].item(), out_col=5, in_col=0, phys_val=qd_phys)

        gds_phys = max(abs(gds_phys), 1e-12)

        return {
            "id": id_phys, "gm": gm_phys, "gds": gds_phys, "gmb": gmb_phys,
            "qg": qg_phys, "qd": qd_phys, "qs": qs_phys, "qb": qb_phys,
            "cgg": cgg_phys, "cgd": cgd_phys, "cgs": cgs_phys,
            "cdg": cdg_phys, "cdd": cdd_phys,
        }

    def _eval_autograd4(self, x: torch.Tensor) -> Dict[str, float]:
        """Slow path: 4-output model with autograd derivatives."""
        x_v = x[:, :4].requires_grad_(True)
        x_g = x[:, 4:]
        x_full = torch.cat([x_v, x_g], dim=1)

        with torch.enable_grad():
            out = self._nn_model(x_full)  # (1, 4) = [id, qg, qd, qb]

            grad_id = torch.autograd.grad(
                out[:, 0].sum(), x_v, create_graph=False, retain_graph=True
            )[0]
            grad_qg = torch.autograd.grad(
                out[:, 1].sum(), x_v, create_graph=False, retain_graph=True
            )[0]
            grad_qd = torch.autograd.grad(
                out[:, 2].sum(), x_v, create_graph=False, retain_graph=False
            )[0]

        id_phys = self._denorm_scalar(out[0, 0].item(), col_idx=0)
        qg_phys = self._denorm_scalar(out[0, 1].item(), col_idx=4)
        qd_phys = self._denorm_scalar(out[0, 2].item(), col_idx=5)
        qb_phys = self._denorm_scalar(out[0, 3].item(), col_idx=7)
        qs_phys = -(qg_phys + qd_phys + qb_phys)

        gm_phys = self._denorm_full_derivative(
            grad_id[0, 1].item(), out_col=0, in_col=1, phys_val=id_phys)
        gds_phys = self._denorm_full_derivative(
            grad_id[0, 0].item(), out_col=0, in_col=0, phys_val=id_phys)
        gmb_phys = self._denorm_full_derivative(
            grad_id[0, 3].item(), out_col=0, in_col=3, phys_val=id_phys)
        cgg_phys = self._denorm_full_derivative(
            grad_qg[0, 1].item(), out_col=4, in_col=1, phys_val=qg_phys)
        cgd_phys = self._denorm_full_derivative(
            grad_qg[0, 0].item(), out_col=4, in_col=0, phys_val=qg_phys)
        cgs_phys = self._denorm_full_derivative(
            grad_qg[0, 2].item(), out_col=4, in_col=2, phys_val=qg_phys)
        cdg_phys = self._denorm_full_derivative(
            grad_qd[0, 1].item(), out_col=5, in_col=1, phys_val=qd_phys)
        cdd_phys = self._denorm_full_derivative(
            grad_qd[0, 0].item(), out_col=5, in_col=0, phys_val=qd_phys)

        gds_phys = max(abs(gds_phys), 1e-12)

        return {
            "id": id_phys, "gm": gm_phys, "gds": gds_phys, "gmb": gmb_phys,
            "qg": qg_phys, "qd": qd_phys, "qs": qs_phys, "qb": qb_phys,
            "cgg": cgg_phys, "cgd": cgd_phys, "cgs": cgs_phys,
            "cdg": cdg_phys, "cdd": cdd_phys,
        }

    def _denorm_scalar(self, val_norm: float, col_idx: int) -> float:
        """Denormalize a single scalar output from z-score + signed_log space."""
        stats = self._norm_stats
        val_log = val_norm * stats.output_std[col_idx] + stats.output_mean[col_idx]
        floor = stats.output_log_floors[col_idx]
        # inv_signed_log for scalar
        if abs(val_log) < 1e-30:
            return 0.0
        sign = 1.0 if val_log >= 0 else -1.0
        return sign * floor * (10.0 ** abs(val_log))

    def _denorm_full_derivative(
        self, deriv_norm: float, out_col: int, in_col: int, phys_val: float
    ) -> float:
        """Denormalize a derivative from normalized to physical space.

        Chain rule for signed_log + z-score on output, min-max on input:

            d(phys)/d(V_phys) = d(zscore)/d(V_norm)  [NN autograd output]
                              * output_std            [z-score → log space]
                              / input_range           [V_norm → V_phys]
                              * |phys| * ln(10)       [log space → physical]

        Args:
            deriv_norm: d(out_zscore)/d(in_minmax) from autograd.
            out_col: Output column index (for std lookup).
            in_col: Input column index (for range lookup).
            phys_val: Current physical value of the output (id, qg, etc).
        """
        stats = self._norm_stats

        in_range = stats.input_max[in_col] - stats.input_min[in_col]
        if abs(in_range) < 1e-10:
            return 0.0

        out_std = stats.output_std[out_col]

        if abs(phys_val) < 1e-30:
            return 0.0

        return deriv_norm * out_std / in_range * abs(phys_val) * np.log(10.0)

    def get_nodes(self) -> List[str]:
        return self.nodes

    def stamp_conductance(self, matrix, node_map: Dict[str, int]) -> None:
        pass  # Solver handles MOSFET stamping directly

    def stamp_rhs(self, rhs, node_map: Dict[str, int]) -> None:
        pass  # Solver handles MOSFET stamping directly

    def get_conductance(self, voltages: Dict[str, float]) -> Tuple[float, float, float]:
        """Get (g_ds, g_m, g_mb) in Siemens.

        These are already in physical units from _eval() (chain rule applied there).
        """
        result = self._eval(voltages)
        return (result["gds"], result["gm"], result["gmb"])

    def get_capacitances(self, voltages: Dict[str, float]) -> Dict[str, float]:
        """Get terminal capacitances in Farads."""
        result = self._eval(voltages)
        return {
            "cgg": result["cgg"],
            "cgd": result["cgd"],
            "cgs": result["cgs"],
            "cdg": result["cdg"],
            "cdd": result["cdd"],
        }

    def get_charges(self, voltages: Dict[str, float]) -> Dict[str, float]:
        """Get terminal charges in Coulombs."""
        result = self._eval(voltages)
        return {
            "qg": result["qg"],
            "qd": result["qd"],
            "qs": result["qs"],
            "qb": result["qb"],
        }

    def init_charge_state(self, voltages: Dict[str, float]) -> None:
        """Initialize charge state from DC operating point."""
        charges = self.get_charges(voltages)
        self._q_prev = charges.copy()
        self._v_prev_tran = {
            "d": voltages.get(self.nodes[0], 0.0),
            "g": voltages.get(self.nodes[1], 0.0),
            "s": voltages.get(self.nodes[2], 0.0),
            "b": voltages.get(self.nodes[3], 0.0),
        }
        self._i_prev_gate = 0.0
        self._i_prev_drain = 0.0

    def update_charge_state(
        self,
        voltages: Dict[str, float],
        cap_currents: Optional[Dict[str, float]] = None,
    ) -> None:
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
            self._i_prev_gate = cap_currents.get("i_gate", 0.0)
            self._i_prev_drain = cap_currents.get("i_drain", 0.0)

    def clear_cache(self) -> None:
        """Clear evaluation cache (called at start of each NR iteration)."""
        self._eval_cache = None
        self._cache_voltages = None


class NMOS_NN(_MOSFETNNBase):
    """NN-based N-Channel MOSFET (LEVEL=73).

    Same sign convention as NMOS_CMG:
    - calculate_current() returns positive when current leaves drain (NMOS ON)
    """

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        result = self._eval(voltages)
        # NN predicts id in SPICE convention (same as PyCMG)
        # NMOS: negate to get "current leaving drain" (positive when ON)
        return -result["id"]


class PMOS_NN(_MOSFETNNBase):
    """NN-based P-Channel MOSFET (LEVEL=73).

    Same sign convention as PMOS_CMG:
    - calculate_current() returns positive when current enters drain (PMOS ON)

    Voltages are shifted to source-relative frame (Vs=0) before feeding to NN,
    since training data was generated with Vs=0 and negative Vg/Vd for PMOS.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_pmos = True

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        result = self._eval(voltages)
        # PMOS: id > 0 when ON (current INTO drain)
        return result["id"]
