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
        from bsimar.models.direct_net import DirectNet

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
            return DirectNet(
                input_dim=input_dim, hidden_dim=hidden_dim,
                n_layers=n_layers, output_dim=output_dim,
                num_tech_codes=num_tech_codes,
                tech_embed_dim=tech_embed_dim,
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
