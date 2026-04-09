"""BSIM-AR: Autoregressive Transformer compact model (LEVEL=74).

Drop-in replacement for NMOS_CMG/PMOS_CMG / NMOS_NN/PMOS_NN using the
BSIMAR v3 Transformer. Implements the full Component interface required
by the solver, reusing ``_MOSFETNNBase`` for cache/charge-state/terminal
bookkeeping and overriding model loading + ``_eval`` for two reasons:

1. The Transformer outputs sit in ``BSIMAR_COLUMN_ORDER`` (paper's Q → I
   → C order), not ``OUTPUT_COLUMN_ORDER``. The autograd column
   indices are therefore different from DirectNet's.

2. BSIMAR v3 uses the **asinh + z-score** normaliser. DirectNet's
   ``_denorm_scalar`` and ``_denorm_full_derivative`` expect the
   legacy signed-log math, which is wrong for asinh. We replace them
   here with the asinh chain rule:

       y_phys = asinh_scale * sinh(y_zscore * out_std + out_mean)
       d(y_phys)/d(v_phys) = out_std * sqrt(asinh_scale² + y_phys²) *
                             (autograd_deriv) / in_std

Terminal order: [drain, gate, source, bulk]
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys

import numpy as np
import torch

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_BSIMAR_PARENT = PROJECT_ROOT / "external_compact_models"
if str(_BSIMAR_PARENT) not in sys.path:
    sys.path.insert(0, str(_BSIMAR_PARENT))

from pycircuitsim.models.mosfet_directnet import _MOSFETNNBase
from bsimar.models.transformer import TransformerEncoderModel
from bsimar.config import PROCESS_PARAM_NAMES
from bsimar.data.normalize import BSIMARNormStats, BSIMAR_COLUMN_ORDER, OUTPUT_COLUMN_ORDER


# Column indices for BSIMAR's AR output order
# (qg, qb, qd, qs, id, gm, gds, gmb, cgg, cgd, cgs, cdg, cdd).
_BSIMAR_IDX = {name: i for i, name in enumerate(BSIMAR_COLUMN_ORDER)}
_BSIMAR_IDX_ID = _BSIMAR_IDX["id"]
_BSIMAR_IDX_QG = _BSIMAR_IDX["qg"]
_BSIMAR_IDX_QD = _BSIMAR_IDX["qd"]
_BSIMAR_IDX_QS = _BSIMAR_IDX["qs"]
_BSIMAR_IDX_QB = _BSIMAR_IDX["qb"]

# Column indices for OUTPUT_COLUMN_ORDER
# (id, gm, gds, gmb, qg, qd, qs, qb, cgg, cgd, cgs, cdg, cdd).
# The normaliser stats are stored in this order.
_OUT_IDX = {name: i for i, name in enumerate(OUTPUT_COLUMN_ORDER)}


class _MOSFETBSIMARBase(_MOSFETNNBase):
    """Base class for BSIM-AR Transformer MOSFET models (LEVEL=74).

    Overrides model loading and ``_eval`` from ``_MOSFETNNBase``; all
    other behaviour (cache, charge state, stamping, terminal naming,
    PMOS source-shift) is inherited unchanged.
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
        process_params: Optional[Dict[str, float]] = None,
    ):
        # Skip _MOSFETNNBase.__init__ — we replicate it with Transformer
        # model loading + asinh normaliser loading.
        from pycircuitsim.models.base import Component
        Component.__init__(self, name, nodes, None)

        if len(nodes) != 4:
            raise ValueError(f"MOSFET_BSIMAR must have exactly 4 nodes, got {len(nodes)}")
        if L <= 0:
            raise ValueError(f"Channel length L must be positive, got {L}")
        if NFIN <= 0:
            raise ValueError(f"Number of fins NFIN must be positive, got {NFIN}")

        self.L = float(L)
        self.NFIN = float(NFIN)
        self.temperature = float(temperature)
        self.phig = float(phig) if phig is not None else None
        self.process_params = process_params

        # Load model + normalisation + architecture config
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"BSIM-AR model not found: {model_path}")

        norm_path = model_path.parent / (model_path.stem.replace("_best", "_norm") + ".npz")
        if not norm_path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {norm_path}")

        config_path = model_path.parent / (model_path.stem.replace("_best", "_config") + ".npz")
        if not config_path.exists():
            raise FileNotFoundError(f"Architecture config not found: {config_path}")

        cfg = np.load(str(config_path))
        input_dim = int(cfg["input_dim"])
        target_dim = int(cfg["target_dim"])
        d_model = int(cfg["d_model"])
        nhead = int(cfg["nhead"])
        num_layers = int(cfg["num_layers"])
        dim_feedforward = int(cfg["dim_feedforward"])
        dropout = float(cfg["dropout"])

        self._input_dim = input_dim
        self._output_dim = target_dim

        self._nn_model = TransformerEncoderModel(
            input_dim=input_dim,
            target_dim=target_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        state = torch.load(str(model_path), weights_only=True, map_location="cpu")
        self._nn_model.load_state_dict(state)
        self._nn_model.eval()

        # Asinh normaliser (the BSIMAR v3 default)
        self._norm_stats = BSIMARNormStats.load(str(norm_path))
        assert self._norm_stats.mode == "asinh", (
            f"BSIMAR LEVEL=74 expects an asinh-mode normaliser, "
            f"got mode={self._norm_stats.mode}"
        )

        # Pre-compute normalised geometry features (constant per device).
        n_proc = input_dim - 6
        nfin_log = np.log2(max(self.NFIN, 1.0))
        if self.process_params is not None and n_proc > 1:
            pp = self.process_params
            proc_names = PROCESS_PARAM_NAMES[:n_proc]
            proc_vals = [pp.get(p.lower(), 0.0) for p in proc_names]
            geo_raw = np.array([nfin_log, self.temperature] + proc_vals)
        elif n_proc == 1 and (self.phig is not None or
                               (self.process_params and "phig" in self.process_params)):
            phig_val = self.phig or self.process_params["phig"]
            geo_raw = np.array([nfin_log, self.temperature, phig_val])
        else:
            geo_raw = np.array([nfin_log, self.temperature])
        geo_std = self._norm_stats.input_std[4:].copy()
        geo_std[geo_std < 1e-12] = 1.0
        self._geo_norm = (geo_raw - self._norm_stats.input_mean[4:]) / geo_std

        self._is_pmos = False

        # Cache and charge state (inherited API expectations)
        self._eval_cache: Optional[Dict[str, float]] = None
        self._cache_voltages: Optional[Tuple[float, ...]] = None
        self._q_prev: Optional[Dict[str, float]] = None
        self._q_prev2: Optional[Dict[str, float]] = None
        self._v_prev_tran: Optional[Dict[str, float]] = None
        self._i_prev_gate: float = 0.0
        self._i_prev_drain: float = 0.0

    # ── asinh denormalisation ────────────────────────────────────────────

    def _denorm_scalar(self, val_norm: float, out_name: str) -> float:
        """Denormalise a single scalar output from asinh + z-score space.

        ``out_name`` is the column name (``'id'``, ``'qg'``, ...). We
        look up the normaliser statistics in ``OUTPUT_COLUMN_ORDER``
        indices (the normaliser was fit on non-reordered data).
        """
        stats = self._norm_stats
        col = _OUT_IDX[out_name]
        u = val_norm * float(stats.output_std[col]) + float(stats.output_mean[col])
        return float(stats.asinh_scale[col]) * float(np.sinh(u))

    def _denorm_derivative(
        self,
        deriv_norm: float,
        out_name: str,
        in_col: int,
        phys_val: float,
    ) -> float:
        """Denormalise a derivative ``d(y_norm)/d(v_zscore)`` under asinh.

        Chain rule:

            d(y_phys)/d(v_phys)
                = d(y_phys)/d(y_zscore) * d(y_zscore)/d(v_zscore) / in_std
                = out_std * sqrt(asinh_scale² + y_phys²)  *  deriv_norm  /  in_std

        ``out_name`` indexes the normaliser stats in OUTPUT_COLUMN_ORDER;
        ``in_col`` is the voltage column index (0..3).
        """
        stats = self._norm_stats
        col = _OUT_IDX[out_name]
        in_std = float(stats.input_std[in_col])
        if in_std < 1e-12:
            return 0.0
        asinh_scale = float(stats.asinh_scale[col])
        out_std = float(stats.output_std[col])
        dy_phys_dy_zscore = out_std * np.sqrt(asinh_scale * asinh_scale + phys_val * phys_val)
        return float(deriv_norm) * float(dy_phys_dy_zscore) / in_std

    # ── Evaluation ───────────────────────────────────────────────────────

    def _eval(self, voltages: Dict[str, float]) -> Dict[str, float]:
        """Evaluate BSIMAR Transformer at given voltages.

        Returns a dict with physical-space values for id/gm/gds/gmb,
        qg/qd/qs/qb, cgg/cgd/cgs/cdg/cdd.

        Mirrors DirectNet's hybrid13 path: charges are denormalised
        directly, conductances come from autograd of id, and cap
        derivatives come from autograd of qg/qd. All denormalisation
        uses the asinh chain rule. Column indices refer to BSIMAR's
        AR output order (qg=0, qb=1, qd=2, qs=3, id=4, gm=..., caps=8..12).
        """
        v_d = voltages.get(self.nodes[0], 0.0)
        v_g = voltages.get(self.nodes[1], 0.0)
        v_s = voltages.get(self.nodes[2], 0.0)
        v_b = voltages.get(self.nodes[3], 0.0)

        v_tuple = (v_d, v_g, v_s, v_b)
        if self._cache_voltages == v_tuple and self._eval_cache is not None:
            return self._eval_cache

        stats = self._norm_stats

        # PMOS source-shift (training data has Vs = 0).
        if self._is_pmos:
            v_shift = v_s
            v_d_nn = v_d - v_shift
            v_g_nn = v_g - v_shift
            v_s_nn = 0.0
            v_b_nn = v_b - v_shift
        else:
            v_d_nn, v_g_nn, v_s_nn, v_b_nn = v_d, v_g, v_s, v_b

        # Clamp voltages to training range (prevents NR-overshoot garbage).
        v_raw = np.array([v_d_nn, v_g_nn, v_s_nn, v_b_nn])
        v_raw_clamped = np.clip(
            v_raw, stats.input_min[:4], stats.input_max[:4])

        # Z-score normalise voltage inputs.
        v_std = stats.input_std[:4].copy()
        v_std[v_std < 1e-12] = 1.0
        v_norm = (v_raw_clamped - stats.input_mean[:4]) / v_std

        # Assemble the full input tensor: [4V, NFIN_log, L, T, 12_proc].
        x_np = np.concatenate([v_norm, self._geo_norm]).astype(np.float32)
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # (1, 19)

        # Forward with first-order autograd on the voltage slice.
        # Note: SDPA flash/efficient kernels do NOT support double
        # backward, but we only need first-order derivatives here
        # (create_graph=False), which they handle fine.
        x_v = x[:, :4].requires_grad_(True)
        x_g = x[:, 4:]
        x_full = torch.cat([x_v, x_g], dim=1)

        with torch.enable_grad():
            out = self._nn_model(x_full)  # (1, 13) in BSIMAR order

            grad_id = torch.autograd.grad(
                out[:, _BSIMAR_IDX_ID].sum(), x_v,
                create_graph=False, retain_graph=True,
            )[0]
            grad_qg = torch.autograd.grad(
                out[:, _BSIMAR_IDX_QG].sum(), x_v,
                create_graph=False, retain_graph=True,
            )[0]
            grad_qd = torch.autograd.grad(
                out[:, _BSIMAR_IDX_QD].sum(), x_v,
                create_graph=False, retain_graph=False,
            )[0]

        # Denormalise scalar predictions under asinh.
        id_phys = self._denorm_scalar(out[0, _BSIMAR_IDX_ID].item(), "id")
        qg_phys = self._denorm_scalar(out[0, _BSIMAR_IDX_QG].item(), "qg")
        qd_phys = self._denorm_scalar(out[0, _BSIMAR_IDX_QD].item(), "qd")
        qs_phys = self._denorm_scalar(out[0, _BSIMAR_IDX_QS].item(), "qs")
        qb_phys = self._denorm_scalar(out[0, _BSIMAR_IDX_QB].item(), "qb")

        # Denormalise autograd conductances (x_v columns: Vd=0, Vg=1, Vs=2, Vbs=3).
        gm_phys = self._denorm_derivative(
            grad_id[0, 1].item(), "id", in_col=1, phys_val=id_phys)
        gds_phys = self._denorm_derivative(
            grad_id[0, 0].item(), "id", in_col=0, phys_val=id_phys)
        gmb_phys = self._denorm_derivative(
            grad_id[0, 3].item(), "id", in_col=3, phys_val=id_phys)

        # Denormalise autograd capacitances.
        cgg_phys = self._denorm_derivative(
            grad_qg[0, 1].item(), "qg", in_col=1, phys_val=qg_phys)
        cgd_phys = self._denorm_derivative(
            grad_qg[0, 0].item(), "qg", in_col=0, phys_val=qg_phys)
        cgs_phys = self._denorm_derivative(
            grad_qg[0, 2].item(), "qg", in_col=2, phys_val=qg_phys)
        cdg_phys = self._denorm_derivative(
            grad_qd[0, 1].item(), "qd", in_col=1, phys_val=qd_phys)
        cdd_phys = self._denorm_derivative(
            grad_qd[0, 0].item(), "qd", in_col=0, phys_val=qd_phys)

        gds_phys = max(abs(gds_phys), 1e-12)

        result = {
            "id": id_phys, "gm": gm_phys, "gds": gds_phys, "gmb": gmb_phys,
            "qg": qg_phys, "qd": qd_phys, "qs": qs_phys, "qb": qb_phys,
            "cgg": cgg_phys, "cgd": cgd_phys, "cgs": cgs_phys,
            "cdg": cdg_phys, "cdd": cdd_phys,
        }
        self._eval_cache = result
        self._cache_voltages = v_tuple
        return result


class NMOS_BSIMAR(_MOSFETBSIMARBase):
    """BSIM-AR N-Channel MOSFET (LEVEL=74).

    Same sign convention as NMOS_CMG / NMOS_NN:
    - calculate_current() returns positive when current leaves drain (NMOS ON).
    """

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        result = self._eval(voltages)
        return -result["id"]


class PMOS_BSIMAR(_MOSFETBSIMARBase):
    """BSIM-AR P-Channel MOSFET (LEVEL=74).

    Same sign convention as PMOS_CMG / PMOS_NN:
    - calculate_current() returns positive when current enters drain (PMOS ON).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_pmos = True

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        result = self._eval(voltages)
        return result["id"]
