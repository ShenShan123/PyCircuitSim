"""DirectNet MLP for six-surface full-terminal compact modeling."""

from __future__ import annotations

import torch
from torch import nn


class DirectNet(nn.Module):
    """MLP with a discrete technology embedding.

    The six outputs are ``i_d, i_g, i_b, qd, qg, qb`` in the canonical
    full-terminal order. Source current and charge are reconstructed by the
    runtime so every terminal stamp closes exactly.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 384,
        n_layers: int = 6,
        output_dim: int = 6,
        num_tech_codes: int = 18,
        tech_embed_dim: int = 32,
        tech_embed_dropout: float = 0.1,
        unknown_code_id: int = 17,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.num_tech_codes = num_tech_codes
        self.tech_embed_dim = tech_embed_dim
        self._tech_embed_dropout = tech_embed_dropout
        self._unknown_code_id = unknown_code_id

        self.tech_embedding = nn.Embedding(num_tech_codes, tech_embed_dim)
        layers: list[nn.Module] = []
        layer_input = input_dim + tech_embed_dim
        for _ in range(n_layers):
            layers.extend((nn.Linear(layer_input, hidden_dim), nn.SiLU()))
            layer_input = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        tech_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tech_codes is None:
            raise ValueError("DirectNet requires tech_codes")
        if self.training and self._tech_embed_dropout > 0.0:
            mask = (
                torch.rand(tech_codes.size(0), device=tech_codes.device)
                < self._tech_embed_dropout
            )
            tech_codes = tech_codes.clone()
            tech_codes[mask] = self._unknown_code_id
        embedding = self.tech_embedding(tech_codes)
        return self.net(torch.cat((x, embedding), dim=-1))

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


__all__ = ["DirectNet"]
