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

* **B9 — hard-monotone id head** (``mono_full_id=True``). The trunk's ``id``
  column is *replaced* (not augmented) by a dedicated ``_MonotoneIdHead`` that
  is globally monotone-decreasing in normalised ``Vg`` by construction. The
  head uses a 3-layer network where every weight on the Vg-monotone path is
  softplus-constrained non-negative, Softplus activations between layers
  (globally monotone, C¹ — Rule 4), and a sign flip at the output.
  The 12 other output columns remain from the shared trunk unchanged.
  Extra keys ``mono_id.*`` are auto-detected by ``_build_from_state``.

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


class _MonotoneIdHead(nn.Module):
    """Hard-monotone id head for B9 (V6.4.5 Track-B).

    Globally monotone-decreasing in normalised ``Vg`` (input column 1).
    The entire id column is computed by this network; the shared trunk's
    id output is discarded when this head is active.

    Construction (3-layer, Daniels & Velikova 2010 / "Certified Monotonic NN"):
    - Layer 1: split weight on Vg (non-negative via softplus) + unconstrained
      weights on all other inputs → h1 (monotone-increasing in Vg)
    - Softplus activation (C¹, globally monotone — Rule 4)
    - Layer 2: all weights on the h1 path are non-negative (softplus) +
      a separate unconstrained "skip" path for non-Vg inputs → h2
      (composition of non-negative map on monotone-increasing input preserves
       monotone-increasing in Vg)
    - Softplus activation
    - Layer 3: non-negative output weights (softplus) → scalar (monotone-increasing)
    - Final: ``sign * scalar`` where sign = -1 makes output monotone-DECREASING

    This gives a function where ∂id/∂Vg < 0 (id becomes more negative / larger
    magnitude as Vg increases), matching the NMOS PyCMG data (id < 0 when ON,
    |id| increases with Vg).

    The sign is stored as a buffer for checkpoint round-trips. For both NMOS
    and PMOS (source-relative frame), the training data shows id_norm decreasing
    as Vg_norm increases at high Vd, so sign = -1 for both.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        sign: float = -1.0,
        vg_col: int = 1,
    ) -> None:
        super().__init__()
        self.vg_col = vg_col
        self.register_buffer("sign", torch.tensor(float(sign)))
        self._hidden_dim = hidden_dim

        # ---- Layer 1 ----
        # Vg path: non-negative weight (sign constrained via softplus of free param)
        # Other inputs: unconstrained
        self.l1_w_vg_raw = nn.Parameter(torch.empty(hidden_dim, 1))
        self.l1_w_rest = nn.Parameter(torch.empty(hidden_dim, in_dim - 1))
        self.l1_b = nn.Parameter(torch.zeros(hidden_dim))

        # ---- Layer 2 ----
        # All weights on h1 (which is monotone-increasing in Vg) are non-negative
        # to preserve monotonicity. Additional skip from non-Vg inputs (unconstrained)
        # modulates the output without breaking the monotone guarantee on the Vg path.
        self.l2_w_h1_raw = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.l2_w_skip = nn.Parameter(torch.empty(hidden_dim, in_dim - 1))
        self.l2_b = nn.Parameter(torch.zeros(hidden_dim))

        # ---- Layer 3 (output) ----
        # Non-negative weights on h2 → scalar monotone-increasing in Vg
        self.l3_w_raw = nn.Parameter(torch.empty(1, hidden_dim))
        self.l3_b = nn.Parameter(torch.zeros(1))

        self.act = nn.Softplus()

        # Initialise: small positive values via small-mean normal so softplus ≈ 0.13
        nn.init.normal_(self.l1_w_vg_raw, mean=-2.0, std=0.3)
        nn.init.xavier_uniform_(self.l1_w_rest)
        nn.init.normal_(self.l2_w_h1_raw, mean=-2.0, std=0.3)
        nn.init.xavier_uniform_(self.l2_w_skip)
        nn.init.normal_(self.l3_w_raw, mean=-2.0, std=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar (B,) monotone-decreasing in x[:, vg_col]."""
        F = torch.nn.functional
        vg = x[:, self.vg_col : self.vg_col + 1]                    # (B, 1)
        rest = torch.cat(
            [x[:, : self.vg_col], x[:, self.vg_col + 1 :]], dim=-1)  # (B, in-1)

        # Layer 1: h1 is monotone-increasing in Vg
        w1_vg = F.softplus(self.l1_w_vg_raw)                         # (H, 1) >= 0
        h1 = rest @ self.l1_w_rest.t() + vg @ w1_vg.t() + self.l1_b # (B, H)
        h1 = self.act(h1)  # Softplus: monotone, C1, positive                 # (B, H)

        # Layer 2: non-negative weights on h1 preserve monotone-increasing in Vg
        w2_h1 = F.softplus(self.l2_w_h1_raw)                         # (H, H) >= 0
        h2 = h1 @ w2_h1.t() + rest @ self.l2_w_skip.t() + self.l2_b # (B, H)
        h2 = self.act(h2)                                             # (B, H)

        # Layer 3: non-negative output → positive scalar, monotone-increasing in Vg
        w3 = F.softplus(self.l3_w_raw)                                # (1, H) >= 0
        out = h2 @ w3.t() + self.l3_b                                 # (B, 1)

        # Sign flip → monotone-DECREASING in Vg
        return (self.sign * out).squeeze(-1)                          # (B,)


class DirectNet(nn.Module):
    """MLP with discrete tech-code embedding predicting MOSFET outputs.

    Input: 7-dim continuous features + integer tech-variant code.
    Output: 13-dim [id, gm, gds, gmb, qg, qd, qs, qb, cgg, cgd, cgs, cdg, cdd].

    ``monotonic`` (Phase 7a, default False): add a residual sub-network
    monotone in ``Vg`` to the ``id`` output column. ``monotone_sign`` picks
    the pinned sign of ∂id/∂Vg (-1 for the PyCMG NMOS/PMOS convention where
    ON current is negative w.r.t. the column ordering). When True the model
    carries extra ``mono.*`` state-dict keys.

    ``mono_full_id`` (B9, default False): replace the trunk's ``id`` column
    with a ``_MonotoneIdHead`` that is globally monotone-decreasing in ``Vg``.
    Carries extra ``mono_id.*`` state-dict keys; auto-detected by
    ``_build_from_state`` in ``mosfet_directnet.py``.
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
        mono_full_id: bool = False,
        mono_id_hidden: int = 128,
        mono_id_sign: float = -1.0,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_tech_codes = num_tech_codes
        self.tech_embed_dim = tech_embed_dim
        self._tech_embed_dropout = tech_embed_dropout
        self._unknown_code_id = unknown_code_id
        self._monotonic = monotonic
        self._mono_full_id = mono_full_id

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

        # B9 — hard-monotone id head (replaces trunk's id column entirely).
        self.mono_id: _MonotoneIdHead | None = None
        if mono_full_id:
            self.mono_id = _MonotoneIdHead(
                in_dim=input_dim + tech_embed_dim,
                hidden_dim=mono_id_hidden,
                sign=mono_id_sign,
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

        if self.mono_id is not None:
            # B9: Replace the trunk's id column entirely with the hard-monotone
            # head. The head is globally monotone-decreasing in Vg by
            # construction; the trunk's id output is discarded for the id column.
            # Re-stacking keeps a single differentiable tensor for autograd.
            id_col = self.mono_id(combined)  # (B,) monotone-decreasing in Vg
            out = torch.cat(
                [id_col.unsqueeze(-1), out[:, self._ID_COL + 1 :]], dim=-1)

        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
