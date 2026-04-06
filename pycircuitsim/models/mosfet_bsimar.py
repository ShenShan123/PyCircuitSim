"""BSIM-AR: Autoregressive Transformer compact model (LEVEL=74).

Drop-in replacement for NMOS_NN/PMOS_NN using a Transformer encoder instead
of a DirectNet MLP. Implements the full Component interface required by the
solver, sharing the same normalization pipeline and evaluation patterns.

Key differences from LEVEL=73 (DirectNet):
- Architecture: Transformer encoder with autoregressive inference (13 sequential steps)
- Slower per-evaluation (13x forward pass vs 1x), but potentially higher accuracy
- Same normalization (signed_log + z-score), same 18-in/13-out format

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

# Make the new `bsimar` package importable regardless of cwd.
_BSIMAR_PARENT = PROJECT_ROOT / "external_compact_models" / "BSIMAR"
if str(_BSIMAR_PARENT) not in sys.path:
    sys.path.insert(0, str(_BSIMAR_PARENT))

from pycircuitsim.models.mosfet_nn import _MOSFETNNBase
from bsimar.models.transformer import TransformerEncoderModel
from bsimar.config import PROCESS_PARAM_NAMES
from bsimar.data.normalize import NormStats


class _MOSFETBSIMARBase(_MOSFETNNBase):
    """Base class for BSIM-AR Transformer MOSFET models.

    Overrides model loading from _MOSFETNNBase to use TransformerEncoderModel
    instead of DirectNet. All evaluation, denormalization, charge state, and
    caching logic is inherited unchanged.
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
        # Skip _MOSFETNNBase.__init__ — we replicate it with Transformer model loading.
        # Call Component.__init__ directly.
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

        # Load model checkpoint and architecture config
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"BSIM-AR model not found: {model_path}")

        norm_path = model_path.parent / (model_path.stem.replace("_best", "_norm") + ".npz")
        if not norm_path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {norm_path}")

        config_path = model_path.parent / (model_path.stem.replace("_best", "_config") + ".npz")
        if not config_path.exists():
            raise FileNotFoundError(f"Architecture config not found: {config_path}")

        # Load architecture config
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

        # Construct and load Transformer model
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

        # Load normalization stats
        self._norm_stats = NormStats.load(str(norm_path))

        # Pre-compute normalized geometry features (constant per device)
        # Number of process params the model expects = input_dim - 6 (4V + NFIN + T)
        n_proc = input_dim - 6
        nfin_log = np.log2(max(self.NFIN, 1.0))
        if self.process_params is not None and n_proc > 1:
            # Universal model: use exactly the number of process params the model expects
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
        self._q_prev2: Optional[Dict[str, float]] = None
        self._v_prev_tran: Optional[Dict[str, float]] = None
        self._i_prev_gate: float = 0.0
        self._i_prev_drain: float = 0.0


class NMOS_BSIMAR(_MOSFETBSIMARBase):
    """BSIM-AR N-Channel MOSFET (LEVEL=74).

    Same sign convention as NMOS_CMG / NMOS_NN:
    - calculate_current() returns positive when current leaves drain (NMOS ON)
    """

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        result = self._eval(voltages)
        return -result["id"]


class PMOS_BSIMAR(_MOSFETBSIMARBase):
    """BSIM-AR P-Channel MOSFET (LEVEL=74).

    Same sign convention as PMOS_CMG / PMOS_NN:
    - calculate_current() returns positive when current enters drain (PMOS ON)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_pmos = True

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        result = self._eval(voltages)
        return result["id"]
