"""Shared base class for NN-based MOSFET compact models.

Used by ``mosfet_directnet`` (LEVEL=73, MLP) and ``mosfet_bsimar``
(LEVEL=74, autoregressive Transformer). Both models share:

* terminal voltage prep (PMOS source-shift + softplus clamp + z-score)
* a 1-sample autograd pass that gives id/qg/qd plus their Jacobians
* normalised → physical chain rule (delegated to the normalizer)
* analytical Vds correction (rule 19) including rail-restoring extrapolation
* charge state + caching used by the transient solver

Subclasses provide:

1. ``model_factory(state)`` returning the un-loaded ``nn.Module``.
2. ``output_layout`` selecting how columns are read from the model
   output: ``"standard"`` reads ``OUTPUT_COLUMN_ORDER`` directly
   (DirectNet); ``"bsimar"`` permutes from ``BSIMAR_COLUMN_ORDER``
   back to ``OUTPUT_COLUMN_ORDER`` (Transformer).
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Make `bsimar` importable regardless of cwd.
PROJECT_ROOT = Path(__file__).parent.parent.parent
_BSIMAR_PARENT = PROJECT_ROOT / "external_compact_models"
if str(_BSIMAR_PARENT) not in sys.path:
    sys.path.insert(0, str(_BSIMAR_PARENT))

from pycircuitsim.models.base import Component
from bsimar.config import UNKNOWN_CODE_ID
from bsimar.data.normalize import (
    NormStats, normalizer_from_stats,
    OUTPUT_COLUMN_ORDER, BSIMAR_COLUMN_ORDER,
)


_logger = logging.getLogger(__name__)
_NN_DEVICE: Optional[torch.device] = None


def _get_nn_device() -> torch.device:
    """Return the best available device (singleton)."""
    global _NN_DEVICE
    if _NN_DEVICE is None:
        _NN_DEVICE = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu"))
    return _NN_DEVICE


# Column indices into OUTPUT_COLUMN_ORDER (the canonical order the
# normalizer's stats are stored in).
_OC = {n: i for i, n in enumerate(OUTPUT_COLUMN_ORDER)}
_OC_ID = _OC["id"]
_OC_QG = _OC["qg"]
_OC_QD = _OC["qd"]
_OC_QB = _OC["qb"]


class _MOSFETNNBase(Component):
    """Shared NN-MOSFET implementation used by LEVEL=73 and LEVEL=74."""

    # Subclasses (DirectNet / BSIMAR) set these.
    _is_pmos: bool = False
    _output_layout: str = "standard"   # "standard" or "bsimar"

    def __init__(
        self,
        name: str,
        nodes: List[str],
        model_path: str,
        L: float,
        NFIN: float,
        temperature: float = 300.15,
        tech_code: Optional[int] = None,
        *,
        model_factory: Optional[
            Callable[[Dict[str, torch.Tensor]], torch.nn.Module]
        ] = None,
        output_layout: str = "standard",
    ) -> None:
        super().__init__(name, nodes, None)
        if len(nodes) != 4:
            raise ValueError(
                f"NN MOSFET must have 4 nodes, got {len(nodes)}")
        if L <= 0:
            raise ValueError(f"L must be positive, got {L}")
        if NFIN <= 0:
            raise ValueError(f"NFIN must be positive, got {NFIN}")

        self.L = float(L)
        self.NFIN = float(NFIN)
        self.temperature = float(temperature)
        self._output_layout = output_layout

        # ── Resolve the checkpoint, norm.npz, and arch config ─────────
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"NN model not found: {model_path_obj}")

        base_stem = model_path_obj.stem
        for sfx in (".phys", ".ar"):
            if base_stem.endswith(sfx):
                base_stem = base_stem[: -len(sfx)]
                break
        norm_path = model_path_obj.parent / (
            base_stem.replace("_best", "_norm") + ".npz")
        if not norm_path.exists():
            raise FileNotFoundError(
                f"Norm stats not found: {norm_path}")

        state = torch.load(
            str(model_path_obj), weights_only=True, map_location="cpu")

        # ── Build the model (subclass-supplied) and load weights ──────
        assert model_factory is not None, (
            "_MOSFETNNBase requires model_factory")
        self._nn_model = model_factory(state)
        self._nn_model.load_state_dict(state)
        self._nn_model.eval()

        # ── Norm stats + normalizer ───────────────────────────────────
        self._norm_stats: NormStats = NormStats.load(str(norm_path))
        self._normalizer = normalizer_from_stats(self._norm_stats)

        # ── Tech code ────────────────────────────────────────────────
        self._tech_code = (
            tech_code if tech_code is not None else UNKNOWN_CODE_ID)
        self._tech_code_tensor = torch.tensor(
            [self._tech_code], dtype=torch.long)

        # ── Pre-compute normalised geometry (constant per device) ────
        nfin_log = float(np.log2(max(self.NFIN, 1.0)))
        geo_raw = np.array(
            [nfin_log, self.L, self.temperature], dtype=np.float64)
        geo_std = self._norm_stats.input_std[4:7].copy()
        geo_std[geo_std < 1e-12] = 1.0
        self._geo_norm = (
            (geo_raw - self._norm_stats.input_mean[4:7]) / geo_std)

        # Derive the model-output → column-name lookup. Three cases:
        #
        # (1) E2 4-output head: norm.npz declares ``output_columns``
        #     (e.g. ["id", "qg", "qd", "qb"]). Map names to that
        #     subset's indices; missing names are unavailable.
        # (2) Transformer (BSIMAR layout): outputs in BSIMAR_COLUMN_ORDER.
        # (3) Standard 13-output DirectNet: OUTPUT_COLUMN_ORDER.
        if self._norm_stats.output_columns is not None:
            cols = self._norm_stats.output_columns
            self._out_col = {n: cols.index(n) for n in cols}
        elif self._output_layout == "bsimar":
            self._out_col = {
                n: BSIMAR_COLUMN_ORDER.index(n) for n in OUTPUT_COLUMN_ORDER
            }
        else:
            self._out_col = {
                n: OUTPUT_COLUMN_ORDER.index(n) for n in OUTPUT_COLUMN_ORDER
            }

        # ── VDD estimate from the training-domain box ────────────────
        vd_range = max(
            abs(float(self._norm_stats.input_max[0])),
            abs(float(self._norm_stats.input_min[0])))
        self._vdd_estimate = vd_range / 2.0

        # ── Cache + transient state ──────────────────────────────────
        self._eval_cache: Optional[Dict[str, float]] = None
        self._cache_voltages: Optional[Tuple[float, ...]] = None
        self._q_prev: Optional[Dict[str, float]] = None
        self._q_prev2: Optional[Dict[str, float]] = None
        self._v_prev_tran: Optional[Dict[str, float]] = None
        self._i_prev_gate: float = 0.0
        self._i_prev_drain: float = 0.0

        self._setup_gpu()

    # ── GPU setup ─────────────────────────────────────────────────────

    def _setup_gpu(self) -> None:
        self._device = _get_nn_device()
        self._nn_model.to(self._device)
        self._tech_code_tensor = self._tech_code_tensor.to(self._device)

        s = self._norm_stats
        self._geo_norm_t = torch.tensor(
            self._geo_norm, dtype=torch.float32, device=self._device)
        v_std = s.input_std[:4].copy()
        v_std[v_std < 1e-12] = 1.0
        self._v_mean = torch.tensor(
            s.input_mean[:4], dtype=torch.float32, device=self._device)
        self._v_std_t = torch.tensor(
            v_std, dtype=torch.float32, device=self._device)
        self._v_min = torch.tensor(
            s.input_min[:4], dtype=torch.float32, device=self._device)
        self._v_max = torch.tensor(
            s.input_max[:4], dtype=torch.float32, device=self._device)
        v_range = torch.clamp(self._v_max - self._v_min, min=0.01)
        # Smooth-clamp sharpness; margin = 5% of per-dim training range
        self._clamp_beta = (1.0 / (0.05 * v_range)).to(self._device)

    # ── Voltage prep: PMOS shift + smooth clamp + z-score ────────────

    def _prep_voltages(
        self, voltages: Dict[str, float],
    ) -> Tuple[torch.Tensor, float, float]:
        """Returns (x_full normalised, v_d_nn, v_s_nn)."""
        v_d = voltages.get(self.nodes[0], 0.0)
        v_g = voltages.get(self.nodes[1], 0.0)
        v_s = voltages.get(self.nodes[2], 0.0)
        v_b = voltages.get(self.nodes[3], 0.0)

        if self._is_pmos:
            v_d_nn = v_d - v_s
            v_g_nn = v_g - v_s
            v_s_nn = 0.0
            v_b_nn = v_b - v_s
        else:
            v_d_nn, v_g_nn, v_s_nn, v_b_nn = v_d, v_g, v_s, v_b

        v_raw = torch.tensor(
            [v_d_nn, v_g_nn, v_s_nn, v_b_nn],
            dtype=torch.float32, device=self._device)

        # Per-element softplus clamp to [v_min, v_max]:
        # softplus(beta*x)/beta with linear branch for large arg.
        beta = self._clamp_beta
        bx_lo = beta * (v_raw - self._v_min)
        v_clamped = self._v_min + torch.where(
            bx_lo > 20.0, v_raw - self._v_min,
            torch.log1p(torch.exp(bx_lo)) / beta)
        bx_hi = beta * (self._v_max - v_clamped)
        v_clamped = self._v_max - torch.where(
            bx_hi > 20.0, self._v_max - v_clamped,
            torch.log1p(torch.exp(bx_hi)) / beta)

        v_norm = (v_clamped - self._v_mean) / self._v_std_t
        x = torch.cat([v_norm, self._geo_norm_t]).unsqueeze(0)
        return x, v_d_nn, v_s_nn

    # ── Core eval: forward + autograd + denorm ───────────────────────

    def _eval(self, voltages: Dict[str, float]) -> Dict[str, float]:
        v_tuple = (
            voltages.get(self.nodes[0], 0.0),
            voltages.get(self.nodes[1], 0.0),
            voltages.get(self.nodes[2], 0.0),
            voltages.get(self.nodes[3], 0.0),
        )
        if self._cache_voltages == v_tuple and self._eval_cache is not None:
            return self._eval_cache

        x, v_d_nn, v_s_nn = self._prep_voltages(voltages)
        x_v = x[:, :4].requires_grad_(True)
        x_g = x[:, 4:]
        x_full = torch.cat([x_v, x_g], dim=1)

        with torch.enable_grad():
            out = self._forward_model(x_full)
            grad_id = torch.autograd.grad(
                out[:, self._mcol("id")].sum(), x_v,
                create_graph=False, retain_graph=True)[0]
            grad_qg = torch.autograd.grad(
                out[:, self._mcol("qg")].sum(), x_v,
                create_graph=False, retain_graph=True)[0]
            grad_qd = torch.autograd.grad(
                out[:, self._mcol("qd")].sum(), x_v,
                create_graph=False, retain_graph=False)[0]

        # Scalar predictions → physical units. The normalizer's stats
        # are stored in OUTPUT_COLUMN_ORDER, so look up by name.
        id_phys = self._denorm("id", out[0, self._mcol("id")].item())
        qg_phys = self._denorm("qg", out[0, self._mcol("qg")].item())
        qd_phys = self._denorm("qd", out[0, self._mcol("qd")].item())
        qb_phys = self._denorm("qb", out[0, self._mcol("qb")].item())
        qs_phys = -(qg_phys + qd_phys + qb_phys)  # charge conservation

        # Conductances from autograd. The NN predicts id in PyCMG sign
        # convention (negative for NMOS ON), so d(id)/dV is negative;
        # negate gm/gmb so the solver's "current leaving drain" frame
        # gets always-positive transconductance.
        gm_phys = -self._denorm_deriv(
            "id", in_col=1, deriv_norm=grad_id[0, 1].item(),
            phys_val=id_phys)
        gds_phys = self._denorm_deriv(
            "id", in_col=0, deriv_norm=grad_id[0, 0].item(),
            phys_val=id_phys)
        gmb_phys = -self._denorm_deriv(
            "id", in_col=3, deriv_norm=grad_id[0, 3].item(),
            phys_val=id_phys)

        cgg_phys = self._denorm_deriv(
            "qg", in_col=1, deriv_norm=grad_qg[0, 1].item(),
            phys_val=qg_phys)
        cgd_phys = self._denorm_deriv(
            "qg", in_col=0, deriv_norm=grad_qg[0, 0].item(),
            phys_val=qg_phys)
        cgs_phys = self._denorm_deriv(
            "qg", in_col=2, deriv_norm=grad_qg[0, 2].item(),
            phys_val=qg_phys)
        cdg_phys = self._denorm_deriv(
            "qd", in_col=1, deriv_norm=grad_qd[0, 1].item(),
            phys_val=qd_phys)
        cdd_phys = self._denorm_deriv(
            "qd", in_col=0, deriv_norm=grad_qd[0, 0].item(),
            phys_val=qd_phys)

        gds_phys = self._floor_gds(id_phys, gds_phys)

        result = {
            "id": id_phys, "gm": gm_phys, "gds": gds_phys, "gmb": gmb_phys,
            "qg": qg_phys, "qd": qd_phys, "qs": qs_phys, "qb": qb_phys,
            "cgg": cgg_phys, "cgd": cgd_phys, "cgs": cgs_phys,
            "cdg": cdg_phys, "cdd": cdd_phys,
        }
        result = self._apply_vds_correction(result, vds=v_d_nn - v_s_nn)

        self._eval_cache = result
        self._cache_voltages = v_tuple
        return result

    # — small helpers —

    def _forward_model(self, x_full: torch.Tensor) -> torch.Tensor:
        """Override in BSIMAR subclass to call the AR-inference forward."""
        return self._nn_model(x_full, tech_codes=self._tech_code_tensor)

    def _mcol(self, name: str) -> int:
        """Model-output column index for ``name``."""
        return self._out_col[name]

    def _stats_col(self, name: str) -> int:
        """Index of column ``name`` in the normalizer's stats arrays."""
        cols = self._norm_stats.output_columns or OUTPUT_COLUMN_ORDER
        return cols.index(name)

    def _denorm(self, name: str, val_norm: float) -> float:
        """Physical value of a single scalar output column."""
        i = self._stats_col(name)
        s = self._norm_stats
        u = float(val_norm) * float(s.output_std[i]) + float(s.output_mean[i])
        if s.mode == "asinh":
            return float(s.asinh_scale[i]) * float(np.sinh(u))
        return u

    def _denorm_deriv(
        self, out_name: str, in_col: int, deriv_norm: float, phys_val: float,
    ) -> float:
        """Chain-rule denormalise a derivative via the normalizer."""
        i = self._stats_col(out_name)
        return self._normalizer.denormalize_derivative(
            deriv_norm=deriv_norm,
            out_idx=i, in_idx=in_col, y_phys=phys_val)

    @staticmethod
    def _floor_gds(id_phys: float, gds_phys: float) -> float:
        """Physics-based gds floor (rule 5): max(|id|·0.5, 1e-12)."""
        return max(gds_phys, max(abs(id_phys) * 0.5, 1e-12))

    # ── Vds correction (rule 19) ─────────────────────────────────────

    def _apply_vds_correction(
        self, result: Dict[str, float], vds: float,
    ) -> Dict[str, float]:
        """Enforce Id(Vds=0)=0 and Id=0 for reverse Vds, plus rail-
        restoring extrapolation past the training Vds range.

        Four-part correction (order matters):

        (a) Quadratic-then-linear ramp when |Vds| > VDD_train. Replicates
            PyCMG's restoring leakage so NR converges to the true rail
            instead of locking on the NN's flat-zero plateau.
        (b) One-sided 1−exp(−|Vds|/VT) factor on Id/gm/gmb in the normal
            direction; zero in the reverse direction.
        (c) Symmetric Vds factor on gds plus a linear-region term so the
            Jacobian has finite slope even when Id is forced to zero.
        (d) Sign enforcement (NMOS id≤0, PMOS id≥0).
        """
        VDD_train = self._vdd_estimate
        VT = max(0.06 * VDD_train, 0.026)
        abs_vds = abs(vds)
        normal_dir = (vds < 0.0) if self._is_pmos else (vds > 0.0)

        # (a) Rail-restoring extrapolation
        if abs_vds > VDD_train:
            overshoot = abs_vds - VDD_train
            g_max = 1.0e-3       # 1 mS scale
            x_ref = 0.5 * VDD_train
            x_cap = 5.0 * x_ref  # transition to linear past 5·x_ref
            if overshoot <= x_cap:
                id_extra = 0.5 * g_max * overshoot * overshoot / x_ref
                g_extra = g_max * overshoot / x_ref
            else:
                id_at_cap = 0.5 * g_max * x_cap * x_cap / x_ref
                g_at_cap = g_max * x_cap / x_ref
                id_extra = id_at_cap + g_at_cap * (overshoot - x_cap)
                g_extra = g_at_cap
            # Sign convention for restoring leakage (Rule 20 fix, V6.2):
            # In PyCMG sign convention, NMOS conducting id < 0 (current
            # leaving drain in CMG's frame) and PMOS conducting id > 0. At
            # rail-overshoot, the physical restoring leakage drives id in
            # the *same* direction as conducting (more |id|), pulling the
            # drain node back toward the source rail via the device. The
            # original V4-re ship used the opposite sign here; the
            # wrong-sign clamp at (d) then wiped the contribution inside
            # the band VDD_train < |Vds| < 20·VT, creating a current-free
            # dead-band where V(out) could settle at ~±100 mV outside the
            # rails (the V6.1 TSMC7 transient bottleneck — see Rule 20).
            if normal_dir:
                if self._is_pmos:
                    result["id"] += id_extra      # PMOS: id more positive
                else:
                    result["id"] -= id_extra      # NMOS: id more negative
            result["gds"] = max(result["gds"], g_extra)

        # Fast path: well into the normal-direction regime.
        if normal_dir and abs_vds > 20.0 * VT:
            return result

        exp_sym = math.exp(-abs_vds / VT) if abs_vds <= 20.0 * VT else 0.0
        f_sym = 1.0 - exp_sym
        f_id = f_sym if normal_dir else 0.0

        id_raw = result["id"]
        result["id"] = id_raw * f_id
        result["gm"] *= f_id
        result["gmb"] *= f_id
        result["gds"] = result["gds"] * f_sym + abs(id_raw) * exp_sym / VT
        result["gds"] = self._floor_gds(result["id"], result["gds"])

        # (d) wrong-sign clamp
        wrong = (
            (self._is_pmos and result["id"] < 0.0)
            or (not self._is_pmos and result["id"] > 0.0))
        if wrong:
            result["id"] = 0.0
            result["gm"] = 0.0
            result["gmb"] = 0.0

        return result

    # ── Solver-side interface ────────────────────────────────────────

    def get_nodes(self) -> List[str]:
        return self.nodes

    def stamp_conductance(self, matrix, node_map):  # noqa: D401
        pass  # solver stamps MOSFETs directly

    def stamp_rhs(self, rhs, node_map):
        pass

    def get_conductance(
        self, voltages: Dict[str, float],
    ) -> Tuple[float, float, float]:
        r = self._eval(voltages)
        return r["gds"], r["gm"], r["gmb"]

    def get_capacitances(
        self, voltages: Dict[str, float],
    ) -> Dict[str, float]:
        r = self._eval(voltages)
        return {k: r[k] for k in ("cgg", "cgd", "cgs", "cdg", "cdd")}

    def get_charges(
        self, voltages: Dict[str, float],
    ) -> Dict[str, float]:
        r = self._eval(voltages)
        return {k: r[k] for k in ("qg", "qd", "qs", "qb")}

    # ── Transient charge state ───────────────────────────────────────

    def init_charge_state(self, voltages: Dict[str, float]) -> None:
        q = self.get_charges(voltages)
        self._q_prev = q.copy()
        self._q_prev2 = q.copy()
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
        q = self.get_charges(voltages)
        self._q_prev2 = (
            self._q_prev.copy() if self._q_prev is not None else q.copy())
        self._q_prev = q.copy()
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
        self._eval_cache = None
        self._cache_voltages = None
