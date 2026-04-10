"""BSIM-AR: Autoregressive Transformer compact model (LEVEL=74).

Supports two model versions:

- **v3** (``use_tech_codes=False`` in config): 19-dim input with 12
  continuous process parameters.
- **v4** (``use_tech_codes=True`` in config): 7-dim input + discrete
  tech-variant code. No process params needed at inference time.

Both versions use the asinh + z-score normaliser and the same autograd
Jacobian extraction.

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
from bsimar.config import PROCESS_PARAM_NAMES, UNKNOWN_CODE_ID
from bsimar.data.normalize import BSIMARNormStats, BSIMAR_COLUMN_ORDER, OUTPUT_COLUMN_ORDER


# Column indices for BSIMAR's AR output order
_BSIMAR_IDX = {name: i for i, name in enumerate(BSIMAR_COLUMN_ORDER)}
_BSIMAR_IDX_ID = _BSIMAR_IDX["id"]
_BSIMAR_IDX_QG = _BSIMAR_IDX["qg"]
_BSIMAR_IDX_QD = _BSIMAR_IDX["qd"]
_BSIMAR_IDX_QS = _BSIMAR_IDX["qs"]
_BSIMAR_IDX_QB = _BSIMAR_IDX["qb"]

# Column indices for OUTPUT_COLUMN_ORDER (normaliser stats order)
_OUT_IDX = {name: i for i, name in enumerate(OUTPUT_COLUMN_ORDER)}


class _MOSFETBSIMARBase(_MOSFETNNBase):
    """Base class for BSIM-AR Transformer MOSFET models (LEVEL=74).

    Supports both v3 (process params) and v4 (tech codes).
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
        tech_code: Optional[int] = None,
    ):
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

        # Detect v4 mode from config
        self._use_tech_codes = bool(cfg.get("use_tech_codes", False))
        num_tech_codes = int(cfg["num_tech_codes"]) if "num_tech_codes" in cfg else 22

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
            use_tech_codes=self._use_tech_codes,
            num_tech_codes=num_tech_codes,
        )
        state = torch.load(str(model_path), weights_only=True, map_location="cpu")
        self._nn_model.load_state_dict(state)
        self._nn_model.eval()

        # Asinh normaliser
        self._norm_stats = BSIMARNormStats.load(str(norm_path))
        assert self._norm_stats.mode == "asinh", (
            f"BSIMAR LEVEL=74 expects an asinh-mode normaliser, "
            f"got mode={self._norm_stats.mode}")

        # Pre-compute normalised geometry features (constant per device).
        nfin_log = np.log2(max(self.NFIN, 1.0))

        if self._use_tech_codes:
            # v4: 3 geometry features [NFIN_log, L, T], no process params.
            # Tech code is handled as a separate integer.
            self._tech_code = tech_code if tech_code is not None else UNKNOWN_CODE_ID
            self._tech_code_tensor = torch.tensor(
                [self._tech_code], dtype=torch.long)
            geo_raw = np.array([nfin_log, self.L, self.temperature])
            geo_std = self._norm_stats.input_std[4:7].copy()
            geo_std[geo_std < 1e-12] = 1.0
            self._geo_norm = (geo_raw - self._norm_stats.input_mean[4:7]) / geo_std
        else:
            # v3: geometry + process params
            self._tech_code = None
            self._tech_code_tensor = None
            n_proc = input_dim - 6
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

        # Cache and charge state
        self._eval_cache: Optional[Dict[str, float]] = None
        self._cache_voltages: Optional[Tuple[float, ...]] = None
        self._q_prev: Optional[Dict[str, float]] = None
        self._q_prev2: Optional[Dict[str, float]] = None
        self._v_prev_tran: Optional[Dict[str, float]] = None
        self._i_prev_gate: float = 0.0
        self._i_prev_drain: float = 0.0

    # ── asinh denormalisation ────────────────────────────────────────────

    def _denorm_scalar(self, val_norm: float, out_name: str) -> float:
        stats = self._norm_stats
        col = _OUT_IDX[out_name]
        u = val_norm * float(stats.output_std[col]) + float(stats.output_mean[col])
        return float(stats.asinh_scale[col]) * float(np.sinh(u))

    def _denorm_derivative(
        self, deriv_norm: float, out_name: str, in_col: int, phys_val: float,
    ) -> float:
        stats = self._norm_stats
        col = _OUT_IDX[out_name]
        in_std = float(stats.input_std[in_col])
        if in_std < 1e-12:
            return 0.0
        asinh_scale = float(stats.asinh_scale[col])
        out_std = float(stats.output_std[col])
        dy_phys_dy_zscore = out_std * np.sqrt(
            asinh_scale * asinh_scale + phys_val * phys_val)
        return float(deriv_norm) * float(dy_phys_dy_zscore) / in_std

    # ── Evaluation ───────────────────────────────────────────────────────

    def _eval(self, voltages: Dict[str, float]) -> Dict[str, float]:
        v_d = voltages.get(self.nodes[0], 0.0)
        v_g = voltages.get(self.nodes[1], 0.0)
        v_s = voltages.get(self.nodes[2], 0.0)
        v_b = voltages.get(self.nodes[3], 0.0)

        v_tuple = (v_d, v_g, v_s, v_b)
        if self._cache_voltages == v_tuple and self._eval_cache is not None:
            return self._eval_cache

        stats = self._norm_stats

        # PMOS source-shift
        if self._is_pmos:
            v_shift = v_s
            v_d_nn = v_d - v_shift
            v_g_nn = v_g - v_shift
            v_s_nn = 0.0
            v_b_nn = v_b - v_shift
        else:
            v_d_nn, v_g_nn, v_s_nn, v_b_nn = v_d, v_g, v_s, v_b

        # Clamp voltages to training range
        v_raw = np.array([v_d_nn, v_g_nn, v_s_nn, v_b_nn])
        v_raw_clamped = np.clip(
            v_raw, stats.input_min[:4], stats.input_max[:4])

        # Z-score normalise voltage inputs
        v_std = stats.input_std[:4].copy()
        v_std[v_std < 1e-12] = 1.0
        v_norm = (v_raw_clamped - stats.input_mean[:4]) / v_std

        # Assemble input tensor (v3: 19-dim, v4: 7-dim)
        x_np = np.concatenate([v_norm, self._geo_norm]).astype(np.float32)
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)

        # Forward with first-order autograd on voltage slice
        x_v = x[:, :4].requires_grad_(True)
        x_g = x[:, 4:]
        x_full = torch.cat([x_v, x_g], dim=1)

        with torch.enable_grad():
            out = self._nn_model(
                x_full, tech_codes=self._tech_code_tensor)

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

        # Denormalise scalar predictions
        id_phys = self._denorm_scalar(out[0, _BSIMAR_IDX_ID].item(), "id")
        qg_phys = self._denorm_scalar(out[0, _BSIMAR_IDX_QG].item(), "qg")
        qd_phys = self._denorm_scalar(out[0, _BSIMAR_IDX_QD].item(), "qd")
        qs_phys = self._denorm_scalar(out[0, _BSIMAR_IDX_QS].item(), "qs")
        qb_phys = self._denorm_scalar(out[0, _BSIMAR_IDX_QB].item(), "qb")

        # Denormalise autograd conductances
        gm_phys = self._denorm_derivative(
            grad_id[0, 1].item(), "id", in_col=1, phys_val=id_phys)
        gds_phys = self._denorm_derivative(
            grad_id[0, 0].item(), "id", in_col=0, phys_val=id_phys)
        gmb_phys = self._denorm_derivative(
            grad_id[0, 3].item(), "id", in_col=3, phys_val=id_phys)

        # Denormalise autograd capacitances
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
    """BSIM-AR N-Channel MOSFET (LEVEL=74)."""

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        result = self._eval(voltages)
        return -result["id"]


class PMOS_BSIMAR(_MOSFETBSIMARBase):
    """BSIM-AR P-Channel MOSFET (LEVEL=74)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_pmos = True

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        result = self._eval(voltages)
        return result["id"]
