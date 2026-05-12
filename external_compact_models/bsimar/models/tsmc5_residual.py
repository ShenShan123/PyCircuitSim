"""TSMC5-only residual head for the V6 small-probe DirectNet backbone.

Tier M2 of `docs/superpowers/plans/2026-05-11-tsmc5-inverter-tiered-fix.md`.

Adds a small (~2-3 K param) MLP residual on top of a *frozen* DirectNet
backbone, conditioned on the discrete tech-code (only TSMC5's four
variants — codes 0..3 — see ``bsimar.config.TECH_VARIANT_CODES``).
The residual is added in **normalised** output space (i.e. before
denormalisation), so the simulator's existing asinh-zscore chain-rule
path stays bit-identical and the autograd Jacobian flows through both
the backbone and the residual.

Inference rule (rule 19 still applies on top):
  y_norm_pred = backbone(x, tech_codes)
              + residual_head(x, tech_codes)  * gate(tech_codes)

``gate(tech_codes)`` is a {0,1} mask that is **1 only on tech_codes <
``num_tsmc5_codes``** (default 4) so non-TSMC5 cells must be
bit-identical to the bare backbone.

The final linear layer is initialised to **zero** so at training start
the residual contributes exactly zero — additive identity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TSMC5ResidualHead(nn.Module):
    """Small residual MLP added to backbone in normalised output space.

    Architecture (default):

        tech_code (int in [0, num_tech_codes))     — only codes <
                                                     num_tsmc5_codes
                                                     produce non-zero
                                                     residual
            └─► nn.Embedding(num_tech_codes, tech_embed_dim=8)
                    │
                    ├── concat with the 6 continuous input features
                    │     (the 7-dim DirectNet input minus the
                    │      tech-code column, which is consumed via
                    │      the embedding instead).
                    ▼
        Linear(6 + 8, hidden=32) → SiLU
            ▼
        Linear(hidden, hidden)   → SiLU
            ▼
        Linear(hidden, out_dim)  ── zero-initialised weights+bias
            ▼
        gate(tech_code) * residual

    Parameter count at defaults (num_tech_codes=18, tech_embed_dim=8,
    hidden=32, out_dim=4): 18×8 + (14×32+32) + (32×32+32) + (32×4+4)
    = 144 + 480 + 1056 + 132 = 1812 params. Well within the 2 K-3 K
    target.
    """

    def __init__(
        self,
        input_dim: int = 7,
        out_dim: int = 4,
        num_tech_codes: int = 18,
        tech_embed_dim: int = 8,
        hidden: int = 32,
        num_tsmc5_codes: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_tech_codes = num_tech_codes
        self.tech_embed_dim = tech_embed_dim
        self.num_tsmc5_codes = num_tsmc5_codes

        self.tech_embedding = nn.Embedding(num_tech_codes, tech_embed_dim)

        # The continuous part of the input is everything except the
        # discrete tech-code column. The 7-dim DirectNet input is
        # actually 6 continuous (Vd, Vg, Vs, Vb, log2(NFIN), L, T — 7
        # dims) plus a separate tech_codes long-tensor argument; we
        # mirror the backbone's structure exactly.
        in_dim = input_dim + tech_embed_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, out_dim)
        self.act = nn.SiLU()

        # Zero-init the FINAL linear layer so the residual starts at
        # exactly zero (epoch-0 forward pass is bit-identical to the
        # bare backbone).
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

        # Modest init on the other linears so gradients flow.
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.tech_embedding.weight, std=0.1)

    def forward(
        self,
        x: torch.Tensor,
        tech_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the gated residual (zero for non-TSMC5 codes)."""
        emb = self.tech_embedding(tech_codes)
        h = torch.cat([x, emb], dim=-1)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        r = self.fc_out(h)

        # Gate: hard mask on tech_code < num_tsmc5_codes. Boolean
        # tensor broadcast against (B, out_dim). Non-TSMC5 rows get
        # exactly zero residual (Jacobian preserved for those rows).
        gate = (tech_codes < self.num_tsmc5_codes).to(r.dtype).unsqueeze(-1)
        return r * gate

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
