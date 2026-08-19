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
  so the autograd Jacobian ``gm = ∂id/∂Vg`` that NR consumes gains
  a guaranteed-sign component. The base MLP is untouched, so the monotone
  term only *biases* the surface — it does not hard-clip it. Shapes of the
  base ``net`` / ``tech_embedding`` keys are unchanged, so a non-monotone
  checkpoint still loads. The extra ``mono.*`` keys are auto-detected by the
  simulator's ``_build_from_state``.

This module does NOT add any loss term: all Phase-7 work shapes
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
      ``Softplus`` (monotone, smooth, non-zero gradient);
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


class _EKVCore(nn.Module):
    """Differentiable EKV/charge-sheet analytic backbone for the ``id`` column.

    V6.4.8 Stage S3 / V6.5.8 high-r_o variant. Composes the ``id`` output as a
    *physically structured* closed-form drain current plus a **floor-scaled**
    bounded NN residual:

        Id_phys(V) = Id_core + sqrt(Id_core² + (κ·id_s)²) · α·tanh(trunk_id)

    (see ``forward``) — a bounded *fraction* of the local current in conduction,
    with a small absolute floor in cutoff. ``Id_core`` is the canonical EKV
    forward-minus-reverse current

        Id_core = pol · I0 · ( sp(vov)² − sp(vov − vds/(n·φt))² ) · (1 + λ·sp(vd))

    with ``sp`` = softplus and the per-device coefficients ``{I0, Vth, n, λ,
    γ}`` produced by a tiny MLP head from the (geometry ‖ tech-embedding)
    features (constant per device). This wires the **V-shape** of the surface
    to physics — monotone in Vgs, self-limiting as Vds→0 (the switchcap triode
    fix), saturating in strong Vds (the opamp locus) — while leaving the
    coefficients learnable per geometry/tech. autograd through the core yields a
    structurally-correct gm/gds/gmb; **zero loss terms**.

    Operates in the same space the trunk does:

    * inputs are the *normalised* 7-dim features; the core de-standardises the 4
      voltage columns to physical Volts using stored ``v_mean/v_std`` buffers;
    * the result is converted back to the standardised-asinh ``id`` column via
      the stored ``id_s/id_mean/id_std`` buffers (the exact inverse of the
      inference-time ``_denorm``), so the loss, the 12 other heads, and the
      autograd→physical chain rule are all untouched.

    All norm/sign constants are **buffers** so they round-trip through the
    checkpoint: at inference ``_build_from_state`` rebuilds this module with
    neutral defaults and ``load_state_dict`` restores the real values — no
    norm-stat plumbing needed on the simulator side.
    """

    # EKV coefficient-head last-layer bias inits (physical-sensible global
    # modelcard): [Vth, n_raw, beta_raw, gamma_raw, lam_raw, vdsat_raw,
    # theta_raw]. The last layer's WEIGHT is zero-init, so at init the core is
    # exactly this geometry-independent global EKV; training learns the
    # geometry/tech dependence on top.
    _PARAM_BIAS_INIT = (0.30, 0.50, -6.0, -3.0, 0.0, -1.0, -3.0)

    def __init__(
        self,
        geo_dim: int = 3,
        tech_embed_dim: int = 32,
        hidden_dim: int = 64,
        alpha: float = 0.5,
        resid_floor: float = 2.0,
        lam_lo: float = 0.05,
    ) -> None:
        super().__init__()
        # Coefficient head: (geo ‖ tech-embedding) → 7 raw EKV params. Keeping
        # the coefficients geometry-aware (NFIN/L/T enter via geo) is essential
        # — one checkpoint serves NFIN 2→20 and L 8→120nm, ~100× current range,
        # far beyond what a bounded residual could recover from per-code-only
        # constants.
        self.param_head = nn.Sequential(
            nn.Linear(geo_dim + tech_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 7),
        )
        nn.init.xavier_uniform_(self.param_head[0].weight)
        nn.init.zeros_(self.param_head[0].bias)
        nn.init.zeros_(self.param_head[2].weight)             # zero last layer
        with torch.no_grad():
            self.param_head[2].bias.copy_(
                torch.tensor(self._PARAM_BIAS_INIT, dtype=torch.float32))

        # — Norm / sign / physical constants (buffers; checkpoint round-trip) —
        self.register_buffer("v_mean", torch.zeros(4))
        self.register_buffer("v_std", torch.ones(4))
        self.register_buffer("id_s", torch.tensor(1.0))
        self.register_buffer("id_mean", torch.tensor(0.0))
        self.register_buffer("id_std", torch.tensor(1.0))
        # psign: voltage orientation = +1 (NMOS) / -1 (PMOS). The terminal
        # current is id_phys = -psign·|I| (NMOS ON negative, PMOS ON positive).
        self.register_buffer("psign", torch.tensor(1.0))
        self.register_buffer("phi0", torch.tensor(0.40))      # body-effect φ0
        self.register_buffer("ut", torch.tensor(0.025852))    # kT/q @ 300K
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        # V6.5.8 high-r_o variant: widen the CLM band's LOW end so the core can
        # represent near-flat saturation (high output resistance) — the
        # gain-163 tsmc7 opamp output stage needs r_o the old 0.30 floor clipped
        # from above. The coefficient head still picks the per-geometry value
        # from data; this only removes an artificial low-r_o floor. V6.5.8: the
        # floor also CAPS max r_o (= max opamp gain) — raise it to pull the
        # vout-weighted-KCL over-flattened gain back toward 163.
        self.register_buffer("lam_lo", torch.tensor(float(lam_lo)))  # CLM band lo
        self.register_buffer("lam_hi", torch.tensor(1.20))
        # V6.5.8 residual floor κ: the bounded id residual's absolute authority
        # in cutoff is κ·id_s·α (see forward()); κ·id_s ≈ the asinh current
        # resolution, so cutoff gets a modest, non-runaway correction budget.
        self.register_buffer("resid_floor", torch.tensor(float(resid_floor)))

    def set_norm(
        self,
        *,
        v_mean,
        v_std,
        id_s: float,
        id_mean: float,
        id_std: float,
        device_type: str,
    ) -> None:
        """Fill the norm/sign buffers from the fitted normalizer (train time)."""
        self.v_mean.copy_(torch.as_tensor(v_mean, dtype=torch.float32))
        self.v_std.copy_(torch.as_tensor(v_std, dtype=torch.float32))
        self.id_s.fill_(float(id_s))
        self.id_mean.fill_(float(id_mean))
        self.id_std.fill_(float(id_std))
        self.psign.fill_(1.0 if device_type == "nmos" else -1.0)

    def core_current(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Physical EKV drain current Id_core(V) in Amps. ``x`` is the
        normalised 7-dim input; ``emb`` the tech embedding (B, tech_dim).

        Charge-sheet EKV: Id = -psign · β·n·ut²·μ·(i_f − i_r)·clm, with
        i_{f,r} = softplus(·)² the forward/reverse interpolation. Self-limits
        as Vds→0 (i_f−i_r→0, the switchcap fix), saturates as Vds grows, and
        is monotone in the gate drive. All ops smooth."""
        sp = torch.nn.functional.softplus
        # De-standardise the 4 voltage columns back to physical Volts (Vs≡0).
        v_phys = x[:, :4] * self.v_std + self.v_mean          # (B,4): Vd,Vg,Vs,Vb
        vds = v_phys[:, 0] - v_phys[:, 2]
        vgs = v_phys[:, 1] - v_phys[:, 2]
        vbs = v_phys[:, 3] - v_phys[:, 2]

        # Device-oriented drives (>0 in forward conduction for both types).
        p = self.psign
        vg, vd, vb = p * vgs, p * vds, p * vbs

        # Per-device coefficients from the (geo ‖ tech) head.
        feat = torch.cat([x[:, 4:7], emb], dim=-1)
        raw = self.param_head(feat)                           # (B,7)
        vth = raw[:, 0]
        n = 1.0 + sp(raw[:, 1])
        beta = sp(raw[:, 2])
        gamma = sp(raw[:, 3])
        lam = self.lam_lo + (self.lam_hi - self.lam_lo) * torch.sigmoid(raw[:, 4])
        vdsat = sp(raw[:, 5]) + 1e-3
        theta = sp(raw[:, 6])

        ut = self.ut
        # Charge-sheet body effect: reverse body bias (vb<0) raises Vth.
        vth_b = vth + gamma * (torch.sqrt(self.phi0 + sp(-vb))
                               - torch.sqrt(self.phi0))
        vp = (vg - vth_b) / n
        i_f = sp(vp / (2.0 * ut)) ** 2
        i_r = sp((vp - vd) / (2.0 * ut)) ** 2
        mu_fac = 1.0 / (1.0 + theta * sp(vp))                 # mobility roll-off
        clm = 1.0 + lam * sp(vd - vdsat)                      # finite sat gds
        i_mag = beta * n * (ut * ut) * mu_fac * (i_f - i_r) * clm
        return -p * i_mag                                     # (B,) Amps

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor, trunk_id_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Return the composed ``id`` column in standardised-asinh space:
        the EKV physical backbone plus a **floor-scaled** bounded residual,
        added in *physical Amps* with its authority tied to the local current:

            id_phys = id_core + sqrt(id_core² + (κ·id_s)²) · α·tanh(trunk_id)

        In conduction sqrt(...) ≈ |id_core|, so the residual is a bounded
        fraction (≤ α·|id_core|, no sign flip) of the physical current; in
        cutoff (id_core→0) it floors at κ·id_s·α — a small absolute correction
        budget, NOT a runaway. This is the V6.5.8 fix for the V6.4.8 S3 KILL:
        the old additive-in-asinh-z form (``core_z + α·tanh``) carried a
        constant ±α·z authority that DOMINATED the offset-dominated µA band and
        wrecked the opamp output-stage locus. The sqrt magnitude is smooth (no
        kink at id_core=0), so the autograd gm/gds/gmb chain stays intact and
        the normalized-space Rule-1 FD gate passes."""
        id_core = self.core_current(x, emb)                  # (B,) Amps, signed
        floor = self.resid_floor * self.id_s
        mag = torch.sqrt(id_core * id_core + floor * floor)  # smooth |id|, ≥floor
        id_phys = id_core + mag * self.alpha * torch.tanh(trunk_id_norm)
        u = torch.asinh(id_phys / self.id_s)
        return (u - self.id_mean) / self.id_std              # standardised-asinh


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
        ekv_core: bool = False,
        ekv_alpha: float = 0.5,
        ekv_hidden: int = 64,
        ekv_lam_lo: float = 0.05,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_tech_codes = num_tech_codes
        self.tech_embed_dim = tech_embed_dim
        self._tech_embed_dropout = tech_embed_dropout
        self._unknown_code_id = unknown_code_id
        self._monotonic = monotonic
        self._ekv_core = ekv_core

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
            if ekv_core:
                raise ValueError(
                    "--monotonic and --ekv-core both target the id column; "
                    "enable at most one.")
            self.mono = _MonotoneVgResidual(
                in_dim=input_dim + tech_embed_dim,
                hidden_dim=monotone_hidden,
                sign=monotone_sign,
                vg_col=1,
            )

        # V6.4.8 S3 — EKV analytic backbone on the id column (default off).
        self.core: _EKVCore | None = None
        if ekv_core:
            self.core = _EKVCore(
                geo_dim=input_dim - 4,
                tech_embed_dim=tech_embed_dim,
                hidden_dim=ekv_hidden,
                alpha=ekv_alpha,
                lam_lo=ekv_lam_lo,
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

        if self.core is not None:
            # V6.4.8 S3 — replace the id column (col 0) with the EKV core +
            # bounded residual driven by the trunk's raw id output. The other
            # 12 columns are untouched; the composed column stays a single
            # differentiable tensor so the inference autograd Jacobian flows.
            id_col = self.core(x, emb, out[:, self._ID_COL])
            out = torch.cat(
                [id_col.unsqueeze(-1), out[:, self._ID_COL + 1 :]], dim=-1)

        if self.mono is not None:
            # Add the monotone-in-Vg residual to the `id` column only. The
            # remaining 12 columns are untouched. Indexing then re-stacking
            # keeps the output a single differentiable tensor so the
            # inference-time autograd Jacobian still flows.
            corr = self.mono(combined).squeeze(-1)  # (B,)
            id_col = out[:, self._ID_COL] + corr
            out = torch.cat(
                [id_col.unsqueeze(-1), out[:, self._ID_COL + 1 :]], dim=-1)
        return out

    def supports_fused_jacobian(self) -> bool:
        """True when ``forward_with_jacobian`` can serve this instance.

        The EKV core and the monotone residual both re-compose the ``id``
        column through their own sub-networks, which the closed-form
        propagation below does not model. Those variants keep the autograd
        path.
        """
        return self.core is None and self.mono is None

    def forward_with_jacobian(
        self,
        x: torch.Tensor,
        tech_codes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Value **and** ∂out/∂x[:, :4] in a single weight-streaming pass.

        V7.0.3. The simulator needs the Jacobian of the outputs w.r.t. the
        4 terminal voltages every Newton iteration, and gets it today from
        three ``autograd.grad`` calls — each of which re-streams the entire
        weight matrix. For a plain MLP the Jacobian can instead be
        propagated *forward* alongside the value in closed form: carry
        ``[h ; ∂h/∂v]`` as 5 rows and issue one GEMM per layer.

        One weight stream instead of four. Measured on DirectNet-large,
        batch 1, 1 thread: **1610 us -> 748 us**, and this returns the full
        13x4 Jacobian where the autograd path returns 3 rows of it.

        Returns ``(out, J)`` with ``out`` (B, output_dim) and ``J``
        (B, 4, output_dim), so ``J[:, :, k]`` matches the shape of
        ``autograd.grad(out[:, k].sum(), x_v)``.

        **Not bit-identical to autograd** — same mathematics, different
        summation order (values ~5e-7 abs, ∂id/∂V ~4e-6 relative). Gated
        behind ``PYCIRCUITSIM_NN_FUSED_JAC=1`` on the simulator side until
        a full complex-gate re-gate clears it.
        """
        assert tech_codes is not None, "DirectNet requires tech_codes"
        if not self.supports_fused_jacobian():
            raise NotImplementedError(
                "forward_with_jacobian does not model the EKV core / "
                "monotone residual; use the autograd path.")

        emb = self.tech_embedding(tech_codes)
        h = torch.cat([x, emb], dim=-1)                    # (B, D)
        bsz, d_in = h.shape

        # Row 0 carries the value; rows 1..4 carry ∂/∂v_k, seeded with the
        # identity on the 4 voltage columns.
        blk = h.new_zeros(bsz, 5, d_in)
        blk[:, 0] = h
        for k in range(4):
            blk[:, 1 + k, k] = 1.0

        n_lin = len(self.net)
        for i in range(0, n_lin - 1, 2):
            lin = self.net[i]                               # Linear
            blk = (blk.reshape(bsz * 5, -1) @ lin.weight.t()).reshape(
                bsz, 5, -1)
            pre = blk[:, 0] + lin.bias                      # bias: value only
            sig = torch.sigmoid(pre)
            # d/dz SiLU(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
            dact = sig * (1.0 + pre * (1.0 - sig))
            blk = torch.cat(
                [(pre * sig).unsqueeze(1), blk[:, 1:] * dact.unsqueeze(1)],
                dim=1)

        lin = self.net[n_lin - 1]                           # output Linear
        blk = (blk.reshape(bsz * 5, -1) @ lin.weight.t()).reshape(bsz, 5, -1)
        return blk[:, 0] + lin.bias, blk[:, 1:]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
