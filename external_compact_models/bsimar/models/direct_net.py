"""DirectNet: baseline MLP with tech-code embedding for MOSFET compact modeling.

Predicts all 13 outputs `[id, gm, gds, gmb, qg, qd, qs, qb, cgg, cgd, cgs, cdg, cdd]`
in a single forward pass. Fast to train (~2s/epoch on a modern GPU) and the
reference model that the Transformer-based BSIM-AR is compared against.

Architecture: 7-dim continuous input [Vd, Vg, Vs, Vb, NFIN_log, L, T] plus
a discrete tech-variant code mapped through ``nn.Embedding``. The embedding
vector is concatenated with the continuous features before the MLP trunk.
SiLU activations throughout.

Conductance and capacitance targets come from PyCMG as direct supervision —
they are NOT derived via autograd during training. Jacobian consistency at
inference time is provided by autograd inside
``pycircuitsim/models/mosfet_directnet.py`` (single-sample, fast).

Phase 7 (V6.4.2) — optional soft physics constraints, default OFF, each
behind a train-CLI flag:

* **7a — monotonicity** (``--monotonic``). A residual-construction monotone
  sub-network is *added* to the ``id`` output column. The sub-network is
  monotone in normalised ``Vg`` by construction (non-negative first-layer
  weights w.r.t. ``Vg`` + monotone SiLU + non-negative output projection),
  so the autograd Jacobian ``gm = ∂id/∂Vg`` that NR consumes (Rule 1) gains
  a guaranteed-sign component. The base MLP is untouched, so the monotone
  term only *biases* the surface — it does not hard-clip it. Shapes of the
  base ``net`` / ``tech_embedding`` keys are unchanged, so a non-monotone
  checkpoint still loads. The extra ``mono.*`` keys are auto-detected by the
  simulator's ``_build_from_state``.

This module does NOT add any loss term (Rule 10): all Phase-7 work shapes
the network, never the loss.
"""

import torch
import torch.nn as nn


class _MonotoneVgResidual(nn.Module):
    """Residual sub-network monotone in normalised ``Vg`` (input column 1).

    Construction (Sill 1997 / "Scalable Monotonic NN", ICLR 2024):

    * the first linear's column acting on ``Vg`` is constrained to a fixed
      sign via ``softplus`` of a free parameter — so ∂(hidden)/∂Vg keeps a
      single sign;
    * SiLU is *not* globally monotone, so the hidden activation is a plain
      ``Softplus`` (monotone, smooth, non-zero gradient — Rule 4 friendly);
    * the output projection is likewise sign-constrained.

    The composition of a sign-fixed affine map, a monotone activation and a
    sign-fixed projection is monotone in ``Vg``. Inputs other than ``Vg``
    (Vd, Vs, Vb, NFIN, L, T, tech-embedding) flow through unconstrained
    weights, so the term still depends on operating point — it is only the
    ``Vg`` *partial derivative* whose sign is pinned.

    ``sign`` is +1 (id increases with Vg) or -1 (id decreases with Vg, the
    PyCMG NMOS convention where ON current is negative). It is stored as a
    buffer so it round-trips through the checkpoint.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        sign: float = -1.0,
        vg_col: int = 1,
    ) -> None:
        super().__init__()
        self.vg_col = vg_col
        self.register_buffer("sign", torch.tensor(float(sign)))

        # First layer is split so the Vg dependence is isolated: `w_rest`
        # carries the (free, unconstrained) weights for every input column
        # EXCEPT Vg; `w_vg_raw` is the sign-constrained Vg weight. Keeping
        # Vg out of `w_rest` entirely is what makes ∂(hidden)/∂Vg
        # single-signed — there is no unconstrained Vg path.
        self.w_rest = nn.Parameter(torch.empty(hidden_dim, in_dim - 1))
        self.w_vg_raw = nn.Parameter(torch.empty(hidden_dim, 1))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.act = nn.Softplus()
        # Output projection onto the scalar id-correction; sign-constrained.
        self.w_out_raw = nn.Parameter(torch.empty(1, hidden_dim))
        self.b_out = nn.Parameter(torch.zeros(1))

        nn.init.xavier_uniform_(self.w_rest)
        nn.init.normal_(self.w_vg_raw, mean=-2.0, std=0.1)   # softplus≈0.13
        nn.init.normal_(self.w_out_raw, mean=-2.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar (B, 1) correction monotone in ``x[:, vg_col]``."""
        # Non-negative Vg weight (magnitude only; sign carried by `self.sign`).
        w_vg = torch.nn.functional.softplus(self.w_vg_raw)          # (H, 1)
        vg = x[:, self.vg_col : self.vg_col + 1]                    # (B, 1)
        rest = torch.cat(
            [x[:, : self.vg_col], x[:, self.vg_col + 1 :]], dim=-1)  # (B, in-1)
        h = rest @ self.w_rest.t() + vg @ w_vg.t() + self.b1
        h = self.act(h)                                            # monotone↑
        w_out = torch.nn.functional.softplus(self.w_out_raw)        # (1, H)≥0
        out = h @ w_out.t() + self.b_out                           # (B, 1)
        return self.sign * out


class DirectNet(nn.Module):
    """MLP with discrete tech-code embedding predicting MOSFET outputs.

    Input: 7-dim continuous features + integer tech-variant code.
    Output: 13-dim [id, gm, gds, gmb, qg, qd, qs, qb, cgg, cgd, cgs, cdg, cdd].

    ``monotonic`` (Phase 7a, default False): add a residual sub-network
    monotone in ``Vg`` to the ``id`` output column. ``monotone_sign`` picks
    the pinned sign of ∂id/∂Vg (-1 for the PyCMG NMOS/PMOS convention where
    ON current is negative w.r.t. the column ordering). When True the model
    carries extra ``mono.*`` state-dict keys.
    """

    # Index of the `id` column in OUTPUT_COLUMN_ORDER.
    _ID_COL: int = 0

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
        monotonic: bool = False,
        monotone_sign: float = -1.0,
        monotone_hidden: int = 64,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_tech_codes = num_tech_codes
        self.tech_embed_dim = tech_embed_dim
        self._tech_embed_dropout = tech_embed_dropout
        self._unknown_code_id = unknown_code_id
        self._monotonic = monotonic

        self.tech_embedding = nn.Embedding(num_tech_codes, tech_embed_dim)

        layers: list[nn.Module] = []
        in_dim = input_dim + tech_embed_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Phase 7a — monotone residual on the `id` column. Acts on the same
        # [continuous || tech-embedding] vector the trunk sees; Vg is the
        # input-column index 1.
        self.mono: _MonotoneVgResidual | None = None
        if monotonic:
            self.mono = _MonotoneVgResidual(
                in_dim=input_dim + tech_embed_dim,
                hidden_dim=monotone_hidden,
                sign=monotone_sign,
                vg_col=1,
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        tech_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert tech_codes is not None, "DirectNet requires tech_codes"

        # Embedding dropout: randomly replace codes with UNKNOWN during training.
        if self.training and self._tech_embed_dropout > 0.0:
            mask = (torch.rand(tech_codes.size(0), device=tech_codes.device)
                    < self._tech_embed_dropout)
            tech_codes = tech_codes.clone()
            tech_codes[mask] = self._unknown_code_id

        emb = self.tech_embedding(tech_codes)  # (B, tech_embed_dim)
        combined = torch.cat([x, emb], dim=-1)  # (B, input_dim + tech_embed_dim)
        out = self.net(combined)

        if self.mono is not None:
            # Add the monotone-in-Vg residual to the `id` column only. The
            # remaining 12 columns are untouched. Indexing then re-stacking
            # keeps the output a single differentiable tensor so the
            # inference-time autograd Jacobian (Rule 1) still flows.
            corr = self.mono(combined).squeeze(-1)  # (B,)
            id_col = out[:, self._ID_COL] + corr
            out = torch.cat(
                [id_col.unsqueeze(-1), out[:, self._ID_COL + 1 :]], dim=-1)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
