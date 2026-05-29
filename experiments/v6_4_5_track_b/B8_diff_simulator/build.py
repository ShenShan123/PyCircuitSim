"""Build torch DirectNet devices for TSMC7 exactly as the inference path does.

Loads a DirectNet checkpoint + its norm.npz, reconstructs the architecture
from the state-dict shapes (matching ``_DirectNetMixin._build_from_state``),
and wraps it in a :class:`TorchDirectNetDevice` with float64 weights so the
differentiable RO sim is numerically clean.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[3]
if str(PROJECT_ROOT / "external_compact_models") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from bsimar.data.normalize import NormStats  # noqa: E402
from bsimar.models.direct_net import DirectNet  # noqa: E402

from ro_torch import TorchDirectNetDevice  # noqa: E402

CKPT_DIR = PROJECT_ROOT / "external_compact_models" / "bsimar" / "checkpoints"

# TSMC7 BenchTech constants (from tests.common.complex.BENCH['TSMC7']).
TSMC7_VDD = 0.75
TSMC7_NFIN = 2
TSMC7_L_NMOS = 16e-9
TSMC7_L_PMOS = 20e-9
TSMC7_TECH_CODE = 2  # local_variant_code('tsmc7','tsmc7','ulvt')


def build_directnet_from_state(state: dict) -> DirectNet:
    """Reconstruct DirectNet arch from state-dict shapes (inference path)."""
    net_keys = [k for k in state.keys()
                if k.startswith("net.") and k.endswith(".weight")]
    net_keys_sorted = sorted(
        net_keys, key=lambda k: int(k.split(".")[1]))
    output_dim = state[net_keys_sorted[-1]].shape[0]
    hidden_dim = state[net_keys_sorted[-1]].shape[1]
    n_layers = len(net_keys_sorted) - 1
    num_tech_codes = state["tech_embedding.weight"].shape[0]
    tech_embed_dim = state["tech_embedding.weight"].shape[1]
    input_dim = state[net_keys_sorted[0]].shape[1] - tech_embed_dim
    monotonic = any(k.startswith("mono.") for k in state)
    monotone_sign = 1.0
    monotone_hidden = 64
    if monotonic:
        monotone_sign = float(state["mono.sign"].item())
        monotone_hidden = state["mono.w_vg_raw"].shape[0]
    model = DirectNet(
        input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers,
        output_dim=output_dim, num_tech_codes=num_tech_codes,
        tech_embed_dim=tech_embed_dim, monotonic=monotonic,
        monotone_sign=monotone_sign, monotone_hidden=monotone_hidden,
    )
    model.load_state_dict(state)
    return model


def load_device(
    stem: str, *, is_pmos: bool, device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> TorchDirectNetDevice:
    pt = CKPT_DIR / f"{stem}_best.pt"
    nz = CKPT_DIR / f"{stem}_norm.npz"
    state = torch.load(str(pt), weights_only=True, map_location="cpu")
    model = build_directnet_from_state(state)
    model = model.to(dtype=dtype, device=device)
    model.eval()  # disable embedding dropout — inference parity
    stats = NormStats.load(str(nz))
    L = TSMC7_L_PMOS if is_pmos else TSMC7_L_NMOS
    return TorchDirectNetDevice(
        model, stats, is_pmos=is_pmos, L=L, NFIN=TSMC7_NFIN,
        tech_code=TSMC7_TECH_CODE, device=device, dtype=dtype,
    )


def load_pair(
    nmos_stem: str, pmos_stem: str, *, device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> Tuple[TorchDirectNetDevice, TorchDirectNetDevice]:
    nmos = load_device(nmos_stem, is_pmos=False, device=device, dtype=dtype)
    pmos = load_device(pmos_stem, is_pmos=True, device=device, dtype=dtype)
    return nmos, pmos
