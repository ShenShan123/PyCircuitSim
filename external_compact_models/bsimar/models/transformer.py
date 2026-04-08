"""BSIM-AR: Autoregressive Transformer model for MOSFET compact modeling.

Architecture: causal Transformer encoder with teacher forcing during
training and autoregressive inference at test time.

Key paper-aligned design choices (`docs/main_V4.pdf` §4.2):

- **Scalar projection + learned token-type embedding.** Each input scalar
  (Vg, Vd, Vs, Vbs, geometry, process params, AR targets) is projected to
  d_model dimensions and added to a learned per-position token-type
  embedding. The token-type embedding is the paper's core trick — a
  pure scalar projection (which the previous version used) cannot
  distinguish a Vg=0.5 token from an id=0.5 token, breaking
  permutation invariance for self-attention.

- **Pre-LN encoder layers.** ``norm_first=True`` puts the LayerNorm
  before each sub-layer (GPT-2 / LLaMA / ViT style). This trains
  more stably at depth and is standard for ≥4-layer Transformers,
  which the paper's ModelL (6 layers) is.

- **GELU feed-forward activation.** PyTorch defaults to ReLU; the
  paper and modern Transformer practice use GELU.

- **Per-token output heads.** A separate ``nn.Linear(d_model, 1)`` per
  AR target makes the heterogeneous regression problem (id vs cgg
  differ by ~14 decades when un-normalized) easier to fit.

- **GPT-2 scaled residual init.** Output projections of attention and
  the FFN are re-initialized with gain ``1/sqrt(2 * num_layers)`` to
  control residual-stream variance compounding with depth.

Input:  (B, input_dim)  — normalized features (voltages + geometry + proc params)
Output: (B, target_dim) — outputs in the BSIMAR (paper) AR order.
"""

import math

import torch
import torch.nn as nn


class TransformerEncoderModel(nn.Module):
    """Autoregressive Transformer for MOSFET I-V / Q-V / C-V prediction.

    Args:
        input_dim:  Number of input features (default 18).
        target_dim: Number of output targets (default 13).
        d_model:    Transformer hidden dimension.
        nhead:      Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Feedforward network dimension.
        dropout:    Dropout rate.
        parallel_caps: P4 — if True, emit the 5 capacitance tokens in
            parallel from a single encoder hidden state instead of as 5
            sequential AR steps. The AR sequence shrinks from 13 to 8 +
            1 parallel cap step. The output shape and column order are
            unchanged. Defaults to False (baseline behavior).
        grouped_inputs: A2 — if True, collapse the 19 scalar input
            features into 3 semantic group tokens (voltages / geometry /
            process params) via small group-MLPs. The 4 voltage scalars
            (Vg, Vd, Vs, Vbs) become one voltage token, the 3 geometry
            scalars (NFIN_log, L, T) become one geometry token, and the
            12 process-parameter scalars become one process token. The
            encoder input sequence drops from ``input_dim + 1 + K`` to
            ``3 + 1 + K``, where ``K`` is the number of AR target tokens
            (8 with ``parallel_caps=True``, otherwise ``target_dim``).
            Assumes the canonical 19-column layout produced by
            ``bsimar.data.normalize._build_combined_input`` for the
            15-col geometry: ``[V(4), NFIN_log, L, T, 12_proc_params]``.
            Defaults to False (baseline behavior).
    """

    # P4 — parallel C-block constants. The cap block in BSIMAR_COLUMN_ORDER
    # starts at index ``CAP_START`` and contains ``N_CAPS`` tokens
    # (cgg, cgd, cgs, cdg, cdd). When ``parallel_caps=True`` we emit all
    # 5 cap tokens in a single parallel head conditioned on the encoder
    # state after the I-block (gmb), shrinking the AR sequence from 13 to
    # 8 (charges + currents/conds) + 1 parallel cap step.
    CAP_START: int = 8
    N_CAPS: int = 5

    # A2 — grouped input layout (matches the 19-column combined input:
    # [V(4), NFIN_log, L, T, 12_proc]).
    VOLTAGE_SLICE = slice(0, 4)
    GEOM_SLICE = slice(4, 7)
    PROC_SLICE = slice(7, 19)
    N_GROUPED_INPUT_TOKENS = 3

    def __init__(
        self,
        input_dim: int = 18,
        target_dim: int = 13,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        parallel_caps: bool = False,
        grouped_inputs: bool = False,
    ):
        super().__init__()
        self.raw_input_dim = input_dim
        self.target_dim = target_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.parallel_caps = parallel_caps
        self.grouped_inputs = grouped_inputs

        # A2 — when grouped_inputs is enabled, the encoder sees 3 context
        # tokens (voltage / geometry / process) instead of ``input_dim``
        # per-scalar tokens. ``self.input_dim`` is the number of context
        # positions before the start token and is used throughout the
        # rest of this module for AR bookkeeping.
        if grouped_inputs:
            assert input_dim == 19, (
                "grouped_inputs=True expects the canonical 19-column "
                "combined input layout ([V(4), NFIN_log, L, T, "
                "12_proc_params]), got input_dim={}".format(input_dim)
            )
            self.input_dim = self.N_GROUPED_INPUT_TOKENS
        else:
            self.input_dim = input_dim

        # P4: when parallel_caps is enabled, the AR sequence only carries
        # the first 8 targets (charges + currents/conds). The 5 cap tokens
        # are emitted in parallel from a single hidden state.
        if parallel_caps:
            assert target_dim == 13, (
                "parallel_caps=True assumes BSIMAR_COLUMN_ORDER (target_dim=13)"
            )
        self.ar_target_dim = (
            self.CAP_START if parallel_caps else target_dim
        )

        # Project each scalar feature to d_model. This remains in use
        # for the start token and the AR target tokens in both modes; in
        # the non-grouped mode it is also applied to every raw input
        # scalar, and in the grouped mode the raw input scalars go
        # through the group MLPs below instead.
        self.input_projection = nn.Linear(1, d_model)

        # A2 — grouped context tokenizers. Each group MLP collapses a
        # semantic chunk of the raw input into a single d_model token,
        # replacing the per-scalar ``input_projection`` pass for the
        # context. The AR target tokens still use ``input_projection``.
        if grouped_inputs:
            self.voltage_group = nn.Sequential(
                nn.Linear(4, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )
            self.geom_group = nn.Sequential(
                nn.Linear(3, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )
            self.proc_group = nn.Sequential(
                nn.Linear(12, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )

        # B1: Learned token-type embedding. One row per token position
        # in the full sequence: input features + start token + AR targets.
        # The embedding gives each scalar an *identity* (Vg vs id vs qg)
        # that the encoder can read; sinusoidal PE only encodes order.
        # Under A2 the number of context tokens is 3 instead of
        # ``input_dim``.
        self.n_tokens = self.input_dim + 1 + target_dim
        self.token_type_emb = nn.Embedding(self.n_tokens, d_model)

        # B2 + B5: Pre-LN encoder + GELU activation. norm_first=True
        # is the de facto standard for ≥4-layer Transformers and removes
        # the need for LR warmup.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        # Pre-LN networks need a final LayerNorm at the top of the stack.
        # enable_nested_tensor=False silences a benign PyTorch warning:
        # nested-tensor fast-path is mutually exclusive with norm_first=True.
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        # B3: Per-token output heads. One scalar projection per target,
        # vs the old single shared head. Trivial param overhead, much
        # better fit on heterogeneous targets (id, gm, qg, cgg, ...).
        self.output_heads = nn.ModuleList(
            [nn.Linear(d_model, 1) for _ in range(target_dim)]
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # B4: GPT-2-style scaled residual init. Down-weight the output
        # projections of attention and the FFN by 1/sqrt(2 * num_layers)
        # so that the residual stream's variance does not compound with
        # depth.
        scale = 1.0 / math.sqrt(2 * max(self.num_layers, 1))
        for layer in self.transformer_encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight, gain=scale)
            nn.init.xavier_uniform_(layer.linear2.weight, gain=scale)

    def _embed_context(self, x: torch.Tensor) -> torch.Tensor:
        """Embed the raw input context as ``(B, n_context, d_model)``.

        - Baseline: project each of the ``raw_input_dim`` scalars via
          ``input_projection`` to get ``(B, raw_input_dim, d_model)``.
        - A2 (grouped_inputs): run three small MLPs over the voltage,
          geometry, and process slices and stack to ``(B, 3, d_model)``.
        """
        if self.grouped_inputs:
            v_tok = self.voltage_group(x[:, self.VOLTAGE_SLICE])  # (B, d_model)
            g_tok = self.geom_group(x[:, self.GEOM_SLICE])        # (B, d_model)
            p_tok = self.proc_group(x[:, self.PROC_SLICE])        # (B, d_model)
            return torch.stack([v_tok, g_tok, p_tok], dim=1)      # (B, 3, d_model)
        return self.input_projection(x.unsqueeze(-1))              # (B, raw_input_dim, d_model)

    def _embed_ar_scalars(self, scalars: torch.Tensor) -> torch.Tensor:
        """Embed AR-side scalar tokens (start token + previous targets).

        ``scalars`` has shape ``(B, K)`` where K is the number of AR
        positions already materialized. Returns ``(B, K, d_model)``.
        The AR scalars are *always* projected via ``input_projection``
        regardless of ``grouped_inputs`` — the group MLPs only apply to
        the raw input context.
        """
        return self.input_projection(scalars.unsqueeze(-1))

    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Generate additive causal mask: mask[i,j] = -inf if j > i."""
        return torch.triu(
            torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1
        )

    def _add_token_type(self, embedded: torch.Tensor) -> torch.Tensor:
        """Add learned token-type embeddings to a (B, L, d_model) tensor.

        Position ``i`` in the full sequence always corresponds to the
        same semantic token (e.g. position 0 = Vg, position input_dim =
        start, position input_dim+1 = first AR target). During AR
        decoding ``L`` grows from ``input_dim+1`` up to ``input_dim+1+
        target_dim``; the token-type indices are simply ``arange(L)``.
        """
        L = embedded.size(1)
        token_ids = torch.arange(L, device=embedded.device)
        return embedded + self.token_type_emb(token_ids).unsqueeze(0)

    def _project_outputs(
        self, hidden: torch.Tensor, start_idx: int
    ) -> torch.Tensor:
        """Project hidden states with per-target heads.

        Args:
            hidden: (B, K, d_model) — the trailing K positions whose
                heads we want to apply.
            start_idx: index of the first AR target encoded in
                ``hidden[:, 0]`` (typically 0 for the full target slice
                or ``i`` for the i-th step of AR inference).
        """
        outs = []
        for k in range(hidden.size(1)):
            head = self.output_heads[start_idx + k]
            outs.append(head(hidden[:, k]).squeeze(-1))
        return torch.stack(outs, dim=1)

    def _parallel_cap_head(
        self, last_hidden: torch.Tensor
    ) -> torch.Tensor:
        """P4: emit all N_CAPS cap outputs in parallel from a single hidden state.

        Args:
            last_hidden: (B, d_model) — typically the encoder output for
                the final AR token (gmb), which conditions on the entire
                charge + I-V context.

        Returns:
            (B, N_CAPS) cap predictions in BSIMAR cap order
            (cgg, cgd, cgs, cdg, cdd).

        Each cap output gets its own learned token-type embedding so the
        shared head input can be distinguished per output. The token-type
        rows reused are the same ones the AR baseline would assign to
        positions ``input_dim + 1 + CAP_START + k`` for k in 0..N_CAPS-1.
        """
        device = last_hidden.device
        cap_token_ids = torch.arange(
            self.input_dim + 1 + self.CAP_START,
            self.input_dim + 1 + self.CAP_START + self.N_CAPS,
            device=device,
        )
        cap_te = self.token_type_emb(cap_token_ids)            # (N_CAPS, d_model)
        cap_h = last_hidden.unsqueeze(1) + cap_te.unsqueeze(0)  # (B, N_CAPS, d_model)
        return self._project_outputs(cap_h, start_idx=self.CAP_START)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional teacher forcing.

        Args:
            x: (B, input_dim) input features.
            y: (B, target_dim) ground-truth targets for teacher forcing.
               If None, uses autoregressive inference.

        Returns:
            (B, target_dim) predicted outputs in BSIMAR_COLUMN_ORDER.
        """
        batch_size = x.size(0)

        if y is not None:
            # Training: teacher forcing with start token
            context_emb = self._embed_context(x)  # (B, n_context, d_model)

            if self.parallel_caps:
                # Only feed the first ar_target_dim (= 8) AR tokens. The
                # cap block is emitted in parallel after the encoder run,
                # so it never appears as an input token. The shifted
                # sequence is [start, qg, qb, qd, qs, id, gm, gds] (length
                # ar_target_dim), and the corresponding 8 AR positions
                # produce hidden states for [qg, qb, qd, qs, id, gm, gds,
                # gmb].
                start_token = torch.zeros(
                    batch_size, 1, device=x.device, dtype=x.dtype)
                # y[:, :ar_target_dim - 1] = first 7 charges + currents
                y_shifted = torch.cat(
                    [start_token, y[:, :self.ar_target_dim - 1]], dim=1)

                ar_emb = self._embed_ar_scalars(y_shifted)
                embedded = torch.cat([context_emb, ar_emb], dim=1)
                embedded = self._add_token_type(embedded)

                L = embedded.size(1)
                causal_mask = self._generate_causal_mask(L).to(x.device)

                encoder_out = self.transformer_encoder(
                    embedded, mask=causal_mask)

                # Last ar_target_dim positions = q+I-V hidden states.
                qic_hidden = encoder_out[:, -self.ar_target_dim:]
                pred_qic = self._project_outputs(qic_hidden, start_idx=0)

                # Parallel cap head conditioned on the gmb hidden state
                # (last position of the encoder output).
                pred_caps = self._parallel_cap_head(encoder_out[:, -1, :])

                return torch.cat([pred_qic, pred_caps], dim=1)

            # Baseline TF path
            start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
            y_shifted = torch.cat([start_token, y[:, :-1]], dim=1)

            ar_emb = self._embed_ar_scalars(y_shifted)
            embedded = torch.cat([context_emb, ar_emb], dim=1)
            embedded = self._add_token_type(embedded)

            L = embedded.size(1)
            causal_mask = self._generate_causal_mask(L).to(x.device)

            encoder_out = self.transformer_encoder(embedded, mask=causal_mask)

            target_hidden = encoder_out[:, -self.target_dim:]
            return self._project_outputs(target_hidden, start_idx=0)

        # Inference: autoregressive generation
        context_emb = self._embed_context(x)  # (B, n_context, d_model)

        # Running buffer of AR-side scalars: starts with the start token
        # and grows by one slot per AR step as predictions are appended.
        start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        ar_scalars = start_token  # (B, K)
        predictions = []
        last_encoder_out: torch.Tensor | None = None

        # Under parallel_caps the AR loop only runs for ar_target_dim (=8)
        # steps; the 5 caps are emitted in one parallel head step after the
        # loop using the final encoder hidden state.
        ar_steps = self.ar_target_dim

        for i in range(ar_steps):
            ar_emb = self._embed_ar_scalars(ar_scalars)
            embedded = torch.cat([context_emb, ar_emb], dim=1)
            embedded = self._add_token_type(embedded)

            L = embedded.size(1)
            causal_mask = self._generate_causal_mask(L).to(x.device)

            out = self.transformer_encoder(embedded, mask=causal_mask)
            last_encoder_out = out

            head = self.output_heads[i]
            next_pred = head(out[:, -1, :]).squeeze(-1)
            predictions.append(next_pred)

            if i < ar_steps - 1:
                ar_scalars = torch.cat(
                    [ar_scalars, next_pred.unsqueeze(1)], dim=1
                )

        if self.parallel_caps:
            # last_encoder_out is the encoder run from the final AR step
            # (which fed gmb as the trailing input token? no — the loop
            # appends a token only AFTER each step, so the final encoder
            # run still corresponds to predicting gmb from the
            # ar_target_dim-token sequence). The hidden state at -1
            # encodes the gmb prediction context, which is what we want
            # to condition the cap head on.
            assert last_encoder_out is not None
            pred_caps = self._parallel_cap_head(last_encoder_out[:, -1, :])
            pred_qic = torch.stack(predictions, dim=1)
            return torch.cat([pred_qic, pred_caps], dim=1)

        return torch.stack(predictions, dim=1)

    def forward_scheduled(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        ss_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass with scheduled sampling.

        For each AR target position, with probability ``ss_ratio``,
        feed the model's own previous prediction (detached) instead
        of the ground-truth token. ``ss_ratio == 1.0`` is the pure-AR
        finetune mode used by N3 (no teacher forcing).

        Under ``parallel_caps``, the AR loop only walks the first 8
        targets (charges + currents/conds); the 5 cap outputs are
        emitted in parallel from the final encoder hidden state, the
        same way ``forward()`` does it. Caps therefore see no
        scheduled sampling, but they fully participate in the loss
        because they depend on the AR-perturbed encoder context.
        """
        if ss_ratio <= 0.0:
            return self.forward(x, y)

        batch_size = x.size(0)
        context_emb = self._embed_context(x)
        start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        ar_scalars = start_token
        predictions = []

        ar_steps = self.ar_target_dim
        last_encoder_out: torch.Tensor | None = None

        for t in range(ar_steps):
            ar_emb = self._embed_ar_scalars(ar_scalars)
            embedded = torch.cat([context_emb, ar_emb], dim=1)
            embedded = self._add_token_type(embedded)

            L = embedded.size(1)
            causal_mask = self._generate_causal_mask(L).to(x.device)

            out = self.transformer_encoder(embedded, mask=causal_mask)
            last_encoder_out = out
            head = self.output_heads[t]
            next_pred = head(out[:, -1, :]).squeeze(-1)
            predictions.append(next_pred)

            if t < ar_steps - 1:
                use_pred = torch.rand(batch_size, device=x.device) < ss_ratio
                next_token = torch.where(
                    use_pred, next_pred.detach(), y[:, t]
                )
                ar_scalars = torch.cat(
                    [ar_scalars, next_token.unsqueeze(1)], dim=1
                )

        if self.parallel_caps:
            assert last_encoder_out is not None
            pred_caps = self._parallel_cap_head(last_encoder_out[:, -1, :])
            pred_qic = torch.stack(predictions, dim=1)
            return torch.cat([pred_qic, pred_caps], dim=1)

        return torch.stack(predictions, dim=1)

    def forward_curriculum(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_targets: int = -1,
        ss_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass predicting only the first n_targets outputs."""
        if self.parallel_caps:
            raise NotImplementedError(
                "parallel_caps not supported under curriculum yet "
                "(only used by --loss direct). Use the standard forward()."
            )
        if n_targets <= 0 or n_targets >= self.target_dim:
            return self.forward_scheduled(x, y, ss_ratio=ss_ratio)

        batch_size = x.size(0)
        context_emb = self._embed_context(x)
        start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        ar_scalars = start_token
        predictions = []

        for t in range(self.target_dim):
            ar_emb = self._embed_ar_scalars(ar_scalars)
            embedded = torch.cat([context_emb, ar_emb], dim=1)
            embedded = self._add_token_type(embedded)

            L = embedded.size(1)
            causal_mask = self._generate_causal_mask(L).to(x.device)

            out = self.transformer_encoder(embedded, mask=causal_mask)
            head = self.output_heads[t]
            next_pred = head(out[:, -1, :]).squeeze(-1)

            if t < n_targets:
                predictions.append(next_pred)
            else:
                predictions.append(y[:, t])

            if t < self.target_dim - 1:
                if t < n_targets:
                    use_pred = torch.rand(batch_size, device=x.device) < ss_ratio
                    next_token = torch.where(use_pred, next_pred.detach(), y[:, t])
                else:
                    next_token = y[:, t]
                ar_scalars = torch.cat(
                    [ar_scalars, next_token.unsqueeze(1)], dim=1
                )

        return torch.stack(predictions, dim=1)
