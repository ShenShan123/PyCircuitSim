"""DirectNet MLP compact model (LEVEL=73).

Drop-in MOSFET model that evaluates a trained DirectNet MLP per
Newton-Raphson iteration. Inherits all the heavy lifting (input
clamping, autograd Jacobian, normalised → physical chain rule,
Vds correction, charge state) from ``_MOSFETNNBase`` in
``mosfet_nn.py``; this module only wires up the model class.

Terminal order: [drain, gate, source, bulk]
"""

from typing import Dict, List, Optional

import torch

from pycircuitsim.models.mosfet_nn import _MOSFETNNBase

# Re-export _MOSFETNNBase so existing solver code that imports it from
# ``mosfet_directnet`` (e.g. solver.py's `_has_nn_device` check) keeps
# working.
__all__ = ["_MOSFETNNBase", "NMOS_NN", "PMOS_NN", "_get_nn_device"]


def _get_nn_device():  # back-compat re-export
    from pycircuitsim.models.mosfet_nn import _get_nn_device as _impl
    return _impl()


class _DirectNetMixin(_MOSFETNNBase):
    """LEVEL=73 base: loads a DirectNet checkpoint via _MOSFETNNBase."""

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
        from neural_network.models.direct_net import DirectNet

        def _build_from_state(state: Dict[str, torch.Tensor]) -> torch.nn.Module:
            net_keys = [
                k for k in state.keys()
                if k.startswith("net.") and k.endswith(".weight")
            ]
            output_dim = state[net_keys[-1]].shape[0]
            hidden_dim = state[net_keys[-1]].shape[1]
            n_layers = len(net_keys) - 1
            num_tech_codes = state["tech_embedding.weight"].shape[0]
            tech_embed_dim = state["tech_embedding.weight"].shape[1]
            input_dim = state[net_keys[0]].shape[1] - tech_embed_dim
            # Phase 7a — a checkpoint trained with `--monotonic` carries
            # `mono.*` keys. Auto-detect them so the monotone residual is
            # rebuilt at the right size; a stock checkpoint has none.
            monotonic = any(k.startswith("mono.") for k in state)
            monotone_sign = 1.0
            monotone_hidden = 64
            if monotonic:
                monotone_sign = float(state["mono.sign"].item())
                monotone_hidden = state["mono.w_vg_raw"].shape[0]
            # V6.4.8 S3 — a checkpoint trained with `--ekv-core` carries
            # `core.*` keys (incl. norm/sign buffers that round-trip). Rebuild
            # the EKV backbone at the right hidden size; a stock checkpoint has
            # none. `load_state_dict` then restores the real buffers, so no
            # norm-stat plumbing is needed here.
            ekv_core = any(k.startswith("core.") for k in state)
            ekv_hidden = 64
            if ekv_core:
                ekv_hidden = state["core.param_head.0.weight"].shape[0]
            return DirectNet(
                input_dim=input_dim, hidden_dim=hidden_dim,
                n_layers=n_layers, output_dim=output_dim,
                num_tech_codes=num_tech_codes,
                tech_embed_dim=tech_embed_dim,
                monotonic=monotonic,
                monotone_sign=monotone_sign,
                monotone_hidden=monotone_hidden,
                ekv_core=ekv_core,
                ekv_hidden=ekv_hidden,
            )

        super().__init__(
            name=name, nodes=nodes, model_path=model_path,
            L=L, NFIN=NFIN, temperature=temperature, tech_code=tech_code,
            model_factory=_build_from_state,
            output_layout="standard",
        )


class NMOS_NN(_DirectNetMixin):
    """N-channel DirectNet MOSFET (LEVEL=73)."""

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        return -self._eval(voltages)["id"]


class PMOS_NN(_DirectNetMixin):
    """P-channel DirectNet MOSFET (LEVEL=73). Source-relative frame."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_pmos = True

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        return self._eval(voltages)["id"]
