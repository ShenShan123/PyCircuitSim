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


class DirectNetV4(nn.Module):
    """DirectNet v4: MLP with discrete tech-code embedding.

    Same 7-dim continuous input as BSIMAR v4 Transformer
    [Vd, Vg, Vs, Vb, NFIN_log, L, T] plus an integer tech-variant code
    mapped through ``nn.Embedding``. The embedding vector is concatenated
    with the continuous features before the MLP trunk.

    This mirrors the BSIMAR v4 Transformer's input scheme, making the
    two models directly comparable (same information budget).
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 384,
        n_layers: int = 6,
        output_dim: int = 13,
        num_tech_codes: int = 18,
        tech_embed_dim: int = 32,
        tech_embed_dropout: float = 0.1,
        unknown_code_id: int = 17,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_tech_codes = num_tech_codes
        self.tech_embed_dim = tech_embed_dim
        self._tech_embed_dropout = tech_embed_dropout
        self._unknown_code_id = unknown_code_id

        self.tech_embedding = nn.Embedding(num_tech_codes, tech_embed_dim)

        layers: list[nn.Module] = []
        in_dim = input_dim + tech_embed_dim
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

    def forward(
        self,
        x: torch.Tensor,
        tech_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert tech_codes is not None, "DirectNetV4 requires tech_codes"

        # Embedding dropout: randomly replace codes with UNKNOWN during training.
        if self.training and self._tech_embed_dropout > 0.0:
            mask = (torch.rand(tech_codes.size(0), device=tech_codes.device)
                    < self._tech_embed_dropout)
            tech_codes = tech_codes.clone()
            tech_codes[mask] = self._unknown_code_id

        emb = self.tech_embedding(tech_codes)  # (B, tech_embed_dim)
        combined = torch.cat([x, emb], dim=-1)  # (B, input_dim + tech_embed_dim)
        return self.net(combined)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
