"""DirectNet: baseline MLP for MOSFET compact modeling.

Predicts all 13 outputs `[id, gm, gds, gmb, qg, qd, qs, qb, cgg, cgd, cgs, cdg, cdd]`
in a single forward pass. Fast to train (~2s/epoch on a modern GPU) and the
reference model that the Transformer-based BSIM-AR is compared against.

Architecture: plain feed-forward MLP with SiLU activations. Conductance and
capacitance targets come from PyCMG as direct supervision — they are NOT
derived via autograd during training. Jacobian consistency at inference time
is provided by autograd inside `pycircuitsim/models/mosfet_nn.py`
(single-sample, fast).
"""

import torch
import torch.nn as nn


class DirectNet(nn.Module):
    """MLP predicting MOSFET outputs directly.

    Supports two output modes:
    - ``output_dim=4``  — predict [id, qg, qd, qb] only (legacy Phase 1)
    - ``output_dim=13`` — predict all 13 outputs (default, current)

    Input: normalized feature vector. The expected input_dim varies by
    dataset format (6 legacy, 7 Phase 13 PHIG, 13 universal-7-param,
    18 universal-12-param, 19 universal-12-param+L).
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        n_layers: int = 4,
        output_dim: int = 4,
    ):
        super().__init__()
        self.output_dim = output_dim

        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
