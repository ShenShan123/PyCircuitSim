"""DirectNet MLP compact model (LEVEL=73).

Drop-in replacement for NMOS_CMG/PMOS_CMG using a trained DirectNet
MLP instead of PyCMG physics evaluation. Implements the full
Component interface required by the solver.

Uses DirectNet with tech-code embedding: 7-dim continuous input
[Vd, Vg, Vs, Vb, NFIN_log, L, T] + discrete tech-variant code via
``nn.Embedding``. Uses asinh + zscore normalisation.

Key design:
- Autograd-derived conductances (gm, gds, gmb) guarantee Jacobian
  consistency for Newton-Raphson convergence.
- Charge conservation: qs = -(qg + qd + qb) enforced analytically.
- Physical constraints: gds >= 0, cutoff clamping.
- Same sign conventions as NMOS_CMG/PMOS_CMG.
- Normalisation: ``BSIMARNormStats(mode='asinh')``. Input clamping uses
  the ``input_min``/``input_max`` metadata stored in the normaliser
  stats.

Terminal order: [drain, gate, source, bulk]
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys

import math

import numpy as np
import torch

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Make the new `bsimar` package importable regardless of cwd.
# Layout: external_compact_models/bsimar/ (package)
_BSIMAR_PARENT = PROJECT_ROOT / "external_compact_models"
if str(_BSIMAR_PARENT) not in sys.path:
    sys.path.insert(0, str(_BSIMAR_PARENT))

from pycircuitsim.models.base import Component
from bsimar.models.direct_net import DirectNet
from bsimar.config import UNKNOWN_CODE_ID
from bsimar.data.normalize import BSIMARNormStats


import logging

_logger = logging.getLogger(__name__)

_NN_DEVICE: Optional[torch.device] = None


def _get_nn_device() -> torch.device:
    """Return the best available device for NN inference (singleton)."""
    global _NN_DEVICE
    if _NN_DEVICE is None:
        if torch.cuda.is_available():
            _NN_DEVICE = torch.device("cuda")
        else:
            _NN_DEVICE = torch.device("cpu")
    return _NN_DEVICE


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
        tech_code: Optional[int] = None,
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

        # Load model and normalization stats
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"NN model not found: {model_path}")

        norm_path = model_path.parent / (model_path.stem.replace("_best", "_norm") + ".npz")
        if not norm_path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {norm_path}")

        # Load checkpoint and infer architecture from weight shapes
        state = torch.load(str(model_path), weights_only=True, map_location="cpu")

        net_weight_keys = [k for k in state.keys()
                           if k.startswith("net.") and k.endswith(".weight")]
        first_key = net_weight_keys[0]
        last_key = net_weight_keys[-1]
        output_dim = state[last_key].shape[0]
        hidden_dim = state[last_key].shape[1]
        n_layers = len(net_weight_keys) - 1  # -1 for output layer

        num_tech_codes = state["tech_embedding.weight"].shape[0]
        tech_embed_dim = state["tech_embedding.weight"].shape[1]
        # First layer input = continuous_dim + tech_embed_dim
        input_dim = state[first_key].shape[1] - tech_embed_dim
        self._input_dim = input_dim

        self._nn_model = DirectNet(
            input_dim=input_dim, hidden_dim=hidden_dim,
            n_layers=n_layers, output_dim=output_dim,
            num_tech_codes=num_tech_codes, tech_embed_dim=tech_embed_dim,
        )
        self._nn_model.load_state_dict(state)
        self._nn_model.eval()

        # Store tech code
        self._tech_code = tech_code if tech_code is not None else UNKNOWN_CODE_ID
        self._tech_code_tensor = torch.tensor(
            [self._tech_code], dtype=torch.long)

        self._output_dim = output_dim

        self._norm_stats = BSIMARNormStats.load(str(norm_path))

        # Pre-compute normalized geometry features (constant per device).
        # 3 geometry features [NFIN_log, L, T] at indices [4:7]
        nfin_log = np.log2(max(self.NFIN, 1.0))
        geo_raw = np.array([nfin_log, self.L, self.temperature])
        geo_std = self._norm_stats.input_std[4:7].copy()
        geo_std[geo_std < 1e-12] = 1.0
        self._geo_norm = (geo_raw - self._norm_stats.input_mean[4:7]) / geo_std

        # Subclass sets this
        self._is_pmos = False

        # Cache
        self._eval_cache: Optional[Dict[str, float]] = None
        self._cache_voltages: Optional[Tuple[float, ...]] = None

        # Charge state for transient analysis
        self._q_prev: Optional[Dict[str, float]] = None
        self._q_prev2: Optional[Dict[str, float]] = None  # Two-step-ago charges (BDF-2)
        self._v_prev_tran: Optional[Dict[str, float]] = None
        self._i_prev_gate: float = 0.0
        self._i_prev_drain: float = 0.0

        # Move model + constants to GPU
        self._setup_gpu()

    def _setup_gpu(self) -> None:
        """Move model and pre-computed constants to the best available device."""
        self._device = _get_nn_device()
        self._nn_model.to(self._device)
        self._tech_code_tensor = self._tech_code_tensor.to(self._device)

        stats = self._norm_stats
        self._geo_norm_t = torch.tensor(
            self._geo_norm.astype(np.float32), dtype=torch.float32, device=self._device
        )
        v_std = stats.input_std[:4].copy()
        v_std[v_std < 1e-12] = 1.0
        self._v_mean = torch.tensor(stats.input_mean[:4], dtype=torch.float32, device=self._device)
        self._v_std_t = torch.tensor(v_std, dtype=torch.float32, device=self._device)
        self._v_min = torch.tensor(stats.input_min[:4], dtype=torch.float32, device=self._device)
        self._v_max = torch.tensor(stats.input_max[:4], dtype=torch.float32, device=self._device)
        # Smooth clamp sharpness: beta = 1/margin where margin = 5% of per-dim range
        v_range = self._v_max - self._v_min
        v_range = torch.clamp(v_range, min=0.01)  # guard against zero-range dims
        self._clamp_beta = (1.0 / (0.05 * v_range)).to(self._device)  # (4,)

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

        # For PMOS: shift to source-relative frame (Vs=0)
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

        # Smooth-clamp & normalise voltages directly on device (GPU if available).
        # Per-element softplus clamp: C1-continuous, avoids zero-gradient cliff
        # at training domain boundaries that stalls NR convergence.
        v_raw = torch.tensor(
            [v_d_nn, v_g_nn, v_s_nn, v_b_nn],
            dtype=torch.float32, device=self._device,
        )
        # Manual softplus: (1/beta) * log(1 + exp(beta * x))  (per-element beta)
        bx_lo = self._clamp_beta * (v_raw - self._v_min)
        v_clamped = self._v_min + torch.where(
            bx_lo > 20.0, v_raw - self._v_min,  # linear for large x (avoid overflow)
            torch.log1p(torch.exp(bx_lo)) / self._clamp_beta)
        bx_hi = self._clamp_beta * (self._v_max - v_clamped)
        v_clamped = self._v_max - torch.where(
            bx_hi > 20.0, self._v_max - v_clamped,
            torch.log1p(torch.exp(bx_hi)) / self._clamp_beta)
        v_norm = (v_clamped - self._v_mean) / self._v_std_t
        x = torch.cat([v_norm, self._geo_norm_t]).unsqueeze(0)  # (1, input_dim)

        # Always use autograd for conductances (Jacobian consistency for NR).
        # Direct prediction of gm/gds is NOT consistent with id, causing NR
        # divergence. Autograd guarantees gm = did/dVg exactly.
        if self._output_dim == 13:
            result = self._eval_hybrid13(x)
        else:
            result = self._eval_autograd4(x)

        # Enforce Id(Vds=0) = 0 via analytical Vds correction
        vds = v_d_nn - v_s_nn
        result = self._apply_vds_correction(result, vds)

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
            out = self._nn_model(x_full, tech_codes=self._tech_code_tensor)

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
        qb_phys = self._denorm_scalar(out[0, 7].item(), col_idx=7)
        # Enforce charge conservation: qs = -(qg + qd + qb)
        qs_phys = -(qg_phys + qd_phys + qb_phys)

        # Denormalize autograd conductances (exact derivatives of id).
        # The NN predicts id in PyCMG terminal-current convention (negative
        # for NMOS ON), so d(id)/d(Vg) is negative.  The solver needs
        # d(i_leaving)/d(Vgs) = d(-id)/d(Vg) = -d(id)/d(Vg), which is
        # positive and matches PyCMG's always-positive gm.  Negate here.
        gm_phys = -self._denorm_full_derivative(
            grad_id[0, 1].item(), out_col=0, in_col=1, phys_val=id_phys)
        gds_phys = self._denorm_full_derivative(
            grad_id[0, 0].item(), out_col=0, in_col=0, phys_val=id_phys)
        gmb_phys = -self._denorm_full_derivative(
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

        # Physics-based gds floor: gds >= |id| * lambda_min.
        # NN learns near-flat Id-Vds in saturation, so autograd gds ≈ 0.
        # Without a reasonable floor, inverter gain → ∞ and NR diverges.
        # At FinFET 16nm, BSIM-CMG lambda = 0.3-1.2 V⁻¹ (strong DIBL+CLM).
        # The floor only affects the NR Jacobian, not the converged current,
        # so a generous value is safe for accuracy.
        gds_floor = max(abs(id_phys) * 0.5, 1e-12)
        if gds_phys < gds_floor:
            _logger.debug("gds=%.3e below floor=%.3e (|id|=%.3e) — clamped",
                          gds_phys, gds_floor, abs(id_phys))
        gds_phys = max(gds_phys, gds_floor)

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
            out = self._nn_model(x_full, tech_codes=self._tech_code_tensor)

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

        # Negate gm/gmb: d(id)/d(V) → d(-id)/d(V) = d(i_leaving)/d(V)
        gm_phys = -self._denorm_full_derivative(
            grad_id[0, 1].item(), out_col=0, in_col=1, phys_val=id_phys)
        gds_phys = self._denorm_full_derivative(
            grad_id[0, 0].item(), out_col=0, in_col=0, phys_val=id_phys)
        gmb_phys = -self._denorm_full_derivative(
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

        # Physics-based gds floor (same as _eval_hybrid13 path)
        gds_floor = max(abs(id_phys) * 0.02, 1e-12)
        if gds_phys < gds_floor:
            _logger.debug("gds=%.3e below floor=%.3e (|id|=%.3e, autograd4) — clamped",
                          gds_phys, gds_floor, abs(id_phys))
        gds_phys = max(gds_phys, gds_floor)

        return {
            "id": id_phys, "gm": gm_phys, "gds": gds_phys, "gmb": gmb_phys,
            "qg": qg_phys, "qd": qd_phys, "qs": qs_phys, "qb": qb_phys,
            "cgg": cgg_phys, "cgd": cgd_phys, "cgs": cgs_phys,
            "cdg": cdg_phys, "cdd": cdd_phys,
        }

    def _denorm_scalar(self, val_norm: float, col_idx: int) -> float:
        """Denormalize a single scalar output from normalized space.

        Dispatches by normaliser mode:
        - zscore: ``y_phys = y_norm * std + mean``
        - asinh:  ``y_phys = scale * sinh(y_norm * std + mean)``
        """
        stats = self._norm_stats
        u = float(val_norm * stats.output_std[col_idx] + stats.output_mean[col_idx])
        if stats.mode == "asinh":
            return float(stats.asinh_scale[col_idx]) * float(np.sinh(u))
        return u  # zscore: u is already the physical value

    def _denorm_full_derivative(
        self, deriv_norm: float, out_col: int, in_col: int, phys_val: float
    ) -> float:
        """Denormalize a derivative from normalized to physical space.

        Dispatches by normaliser mode:

        **zscore** (linear):
            d(y_phys)/d(v_phys) = d(y_norm)/d(v_norm) * out_std / in_std

        **asinh** (nonlinear):
            d(y_phys)/d(v_phys) = d(y_norm)/d(v_norm)
                * out_std * sqrt(scale² + y_phys²) / in_std

        Args:
            deriv_norm: d(out_norm)/d(in_norm) from autograd.
            out_col: Output column index.
            in_col: Input column index (voltage column 0..3).
            phys_val: Physical-space value of the output (needed for asinh chain rule).
        """
        stats = self._norm_stats
        in_std = float(stats.input_std[in_col])
        if in_std < 1e-12:
            return 0.0
        out_std = float(stats.output_std[out_col])
        if stats.mode == "asinh":
            asinh_scale = float(stats.asinh_scale[out_col])
            dy_phys_dy_zscore = out_std * np.sqrt(
                asinh_scale * asinh_scale + phys_val * phys_val)
            return float(deriv_norm) * dy_phys_dy_zscore / in_std
        return float(deriv_norm) * out_std / in_std

    def _apply_vds_correction(
        self, result: Dict[str, float], vds: float,
    ) -> Dict[str, float]:
        """Enforce Id=0 at Vds=0 AND for reverse-direction Vds.

        Two-part analytical correction applied to the NN-predicted
        drain current:

        1. **Normal direction** (NMOS Vds>0, PMOS Vds<0):
           ``Id_corr = Id_nn * (1 - exp(-|Vds|/Vt))``
           Goes to 0 at Vds=0 and ≈Id_nn for |Vds| >> Vt.

        2. **Reverse direction** (NMOS Vds<0, PMOS Vds>0):
           ``Id_corr = 0``
           The NN has no training data for reverse-bias operation;
           any prediction is extrapolation noise.  Forcing Id=0
           prevents NR overshoot from driving the output beyond
           the supply rails.

        The gds correction uses the **symmetric** ``1-exp(-|Vds|/Vt)``
        in both directions so that the NR Jacobian always has a
        reasonable linear-region conductance entry (≈ |Id_nn|/Vt at
        Vds=0), even when Id itself is forced to zero.  This
        prevents floating-node singularities.
        """
        # Use 2×kT/q as the transition width.  Physical motivation:
        # the subthreshold slope factor n ≈ 1.3 for FinFETs, but the
        # NN's Vds=0 residual requires extra suppression to prevent
        # wrong-sign leakage from destabilising inverter rail states.
        # n=2 gives a comfortable margin: at Vds = 0.24 V (minimum
        # in the NMOS pulse test), f ≈ 0.99 — negligible accuracy
        # impact — while at Vds = 0.05 V, f ≈ 0.62 (vs 0.85 for
        # Vt = 0.026), providing stronger suppression of the NN's
        # spurious leakage near Vds ≈ 0.
        VT = 0.052  # 2 × kT/q at 300 K
        abs_vds = abs(vds)

        # Is Vds in the physically normal direction for this device?
        # NMOS: current flows when Vds > 0 (drain above source)
        # PMOS: current flows when Vds < 0 (drain below source, NN frame)
        if self._is_pmos:
            normal_dir = vds < 0.0
        else:
            normal_dir = vds > 0.0

        # Fast path: normal direction with large |Vds| → correction ≈ 1
        if normal_dir and abs_vds > 20.0 * VT:
            return result

        # Symmetric correction factor (always used for gds)
        exp_sym = math.exp(-abs_vds / VT) if abs_vds <= 20.0 * VT else 0.0
        f_sym = 1.0 - exp_sym

        # One-sided correction factor for Id/gm/gmb:
        # 0 for reverse direction, same as f_sym for normal
        f_id = f_sym if normal_dir else 0.0

        id_raw = result["id"]

        # Scale current and transconductances (one-sided)
        result["id"] = id_raw * f_id
        result["gm"] = result["gm"] * f_id
        result["gmb"] = result["gmb"] * f_id

        # gds: symmetric scaling + linear-region conductance term.
        # The term |id_raw| · exp(-|Vds|/Vt) / Vt comes from the
        # product rule and provides the physically correct linear-
        # region gds (≈ |id_raw|/Vt at Vds=0).  Using the symmetric
        # form ensures gds is large at the rail states (Vds≈0) even
        # when Id is forced to zero by the one-sided correction.
        gds_linear = abs(id_raw) * exp_sym / VT
        result["gds"] = result["gds"] * f_sym + gds_linear

        # Re-apply gds floor on corrected values
        gds_floor = max(abs(result["id"]) * 0.5, 1e-12)
        result["gds"] = max(result["gds"], gds_floor)

        # Sign enforcement: NMOS id must be ≤ 0 (current into drain),
        # PMOS id must be ≥ 0 (current into drain in PMOS convention).
        # Wrong-sign predictions are NN extrapolation artefacts that
        # destabilise feedback circuits (inverter rail states).
        wrong_sign = (
            (self._is_pmos and result["id"] < 0.0) or
            (not self._is_pmos and result["id"] > 0.0)
        )
        if wrong_sign:
            result["id"] = 0.0
            result["gm"] = 0.0
            result["gmb"] = 0.0

        return result

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
        self._q_prev2 = charges.copy()  # BDF-2: same as q_prev at DC
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
