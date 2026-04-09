"""BSIM-AR: Autoregressive Transformer model for MOSFET compact modeling.

Architecture: causal Transformer encoder with teacher forcing during
training and autoregressive inference at test time.

Design choices (paper + v3 sprint findings):

- **Grouped input tokens (A2)** — always on. The 19 raw scalars
  (4 voltages + NFIN_log + L + T + 12 process params) are collapsed
  into 3 semantic group tokens via small GELU MLPs. Drops the
  encoder sequence length from 28 to 12 and cuts wall-clock 30–50 %
  at medium/large tiers.

- **Parallel cap head (P4)** — always on. The 5 capacitance outputs
  emit in a single parallel projection from the gmb hidden state
  instead of as 5 sequential AR steps. Shrinks the AR sequence from
  13 to 8.

- **Scalar projection + learned token-type embedding (B1)** — each
  input scalar is projected to d_model dims and added to a per-
  position token-type embedding. The embedding gives each scalar
  an *identity* (Vg vs id vs qg) that the encoder can read;
  sinusoidal PE only encodes order.

- **Pre-LN encoder layers (B2)** — ``norm_first=True``. Standard for
  ≥4-layer Transformers and removes the need for LR warmup.

- **GELU feed-forward activation (B5)**.

- **Per-token output heads (B3)** — one ``nn.Linear(d_model, 1)``
  per target. Trivial param overhead, much better fit on
  heterogeneous targets.

- **GPT-2 scaled residual init (B4)** — attention and FFN output
  projections use gain ``1/sqrt(2 * num_layers)``.

Input:  (B, 19) — normalized features [V(4), NFIN_log, L, T, 12_proc_params]
Output: (B, 13) — outputs in BSIMAR (paper) AR order.
"""

import math

import torch
import torch.nn as nn


class TransformerEncoderModel(nn.Module):
    """Autoregressive Transformer for MOSFET I-V / Q-V / C-V prediction.

    The v3 architecture always runs with grouped inputs (A2) and the
    parallel cap head (P4). These were flags during the sprint; they
    are now structural. The constructor therefore expects
    ``input_dim=19`` and ``target_dim=13`` — no other shapes make
    sense under grouped_inputs + parallel_caps.

    Args:
        input_dim:  Must be 19 (the canonical 19-column combined input
            layout [V(4), NFIN_log, L, T, 12_proc_params]).
        target_dim: Must be 13 (BSIMAR_COLUMN_ORDER).
        d_model:    Transformer hidden dimension.
        nhead:      Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Feedforward network dimension.
        dropout:    Dropout rate.
    """

    # P4 — parallel C-block constants. The cap block in BSIMAR_COLUMN_ORDER
    # starts at index ``CAP_START`` and contains ``N_CAPS`` tokens
    # (cgg, cgd, cgs, cdg, cdd). All 5 cap tokens emit in a single
    # parallel head conditioned on the encoder state after the I-block
    # (gmb). AR sequence length: 8 (charges + currents/conds).
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
        input_dim: int = 19,
        target_dim: int = 13,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert input_dim == 19, (
            "BSIMAR v3 expects the canonical 19-column combined input "
            "layout ([V(4), NFIN_log, L, T, 12_proc_params]), got "
            f"input_dim={input_dim}"
        )
        assert target_dim == 13, (
            f"BSIMAR v3 assumes BSIMAR_COLUMN_ORDER, got target_dim={target_dim}"
        )

        self.raw_input_dim = input_dim
        self.target_dim = target_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # A2 — the encoder sees 3 context tokens (voltage / geometry /
        # process) instead of 19 per-scalar tokens. ``self.input_dim``
        # is the number of context positions before the start token
        # and is used throughout the rest of this module for AR
        # bookkeeping.
        self.input_dim = self.N_GROUPED_INPUT_TOKENS

        # P4 — the AR sequence only carries the first 8 targets
        # (charges + currents/conds). The 5 cap tokens emit in
        # parallel from the gmb hidden state.
        self.ar_target_dim = self.CAP_START

        # Project each scalar feature to d_model. Used for the start
        # token and the AR target tokens; the raw input context
        # scalars go through the group MLPs below instead.
        self.input_projection = nn.Linear(1, d_model)

        # A2 — grouped context tokenizers. Each group MLP collapses a
        # semantic chunk of the raw input into a single d_model token.
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
        """Embed the raw input context as ``(B, 3, d_model)``.

        Runs three small MLPs over the voltage, geometry, and process
        slices of the 19-column combined input and stacks them into
        three semantic tokens.
        """
        v_tok = self.voltage_group(x[:, self.VOLTAGE_SLICE])  # (B, d_model)
        g_tok = self.geom_group(x[:, self.GEOM_SLICE])        # (B, d_model)
        p_tok = self.proc_group(x[:, self.PROC_SLICE])        # (B, d_model)
        return torch.stack([v_tok, g_tok, p_tok], dim=1)      # (B, 3, d_model)

    def _embed_ar_scalars(self, scalars: torch.Tensor) -> torch.Tensor:
        """Embed AR-side scalar tokens (start token + previous targets).

        ``scalars`` has shape ``(B, K)`` where K is the number of AR
        positions already materialized. Returns ``(B, K, d_model)``.
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

            # Teacher-forced training path. Feed the first 7 ground-
            # truth AR tokens (charges + first 3 currents/conds); the
            # 8th AR position (gmb) is generated by the encoder and
            # conditions the parallel cap head. The 5 cap tokens are
            # never input tokens — they emit in parallel from gmb's
            # hidden state.
            start_token = torch.zeros(
                batch_size, 1, device=x.device, dtype=x.dtype)
            y_shifted = torch.cat(
                [start_token, y[:, :self.ar_target_dim - 1]], dim=1)

            ar_emb = self._embed_ar_scalars(y_shifted)
            embedded = torch.cat([context_emb, ar_emb], dim=1)
            embedded = self._add_token_type(embedded)

            L = embedded.size(1)
            causal_mask = self._generate_causal_mask(L).to(x.device)

            encoder_out = self.transformer_encoder(
                embedded, mask=causal_mask)

            # Last ar_target_dim (= 8) positions = q+I-V hidden states.
            qic_hidden = encoder_out[:, -self.ar_target_dim:]
            pred_qic = self._project_outputs(qic_hidden, start_idx=0)

            # Parallel cap head conditioned on the gmb hidden state.
            pred_caps = self._parallel_cap_head(encoder_out[:, -1, :])

            return torch.cat([pred_qic, pred_caps], dim=1)

        # Inference: autoregressive generation over the 8 q+I-V
        # targets, then parallel cap emission.
        context_emb = self._embed_context(x)  # (B, 3, d_model)
        start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        ar_scalars = start_token
        predictions = []
        last_encoder_out: torch.Tensor | None = None

        for i in range(self.ar_target_dim):
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

            if i < self.ar_target_dim - 1:
                ar_scalars = torch.cat(
                    [ar_scalars, next_pred.unsqueeze(1)], dim=1)

        assert last_encoder_out is not None
        pred_caps = self._parallel_cap_head(last_encoder_out[:, -1, :])
        pred_qic = torch.stack(predictions, dim=1)
        return torch.cat([pred_qic, pred_caps], dim=1)

    def forward_scheduled(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        ss_ratio: float = 1.0,
    ) -> torch.Tensor:
        """AR fine-tune forward with scheduled sampling.

        For each AR target position (0..ar_target_dim-1), with
        probability ``ss_ratio`` feed the model's own previous
        prediction (detached) instead of the ground-truth token.
        ``ss_ratio == 1.0`` is the pure-AR finetune mode used by the
        N3 finetune phase; at 0.0 it is identical to teacher forcing
        (and this helper short-circuits to ``forward(x, y)``).

        The 5 cap outputs emit in parallel from the gmb hidden state
        the same way ``forward()`` does. Caps therefore see no
        scheduled sampling but fully participate in the loss because
        they depend on the AR-perturbed encoder context.
        """
        if ss_ratio <= 0.0:
            return self.forward(x, y)

        batch_size = x.size(0)
        context_emb = self._embed_context(x)
        start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        ar_scalars = start_token
        predictions = []
        last_encoder_out: torch.Tensor | None = None

        for t in range(self.ar_target_dim):
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

            if t < self.ar_target_dim - 1:
                use_pred = torch.rand(batch_size, device=x.device) < ss_ratio
                next_token = torch.where(
                    use_pred, next_pred.detach(), y[:, t])
                ar_scalars = torch.cat(
                    [ar_scalars, next_token.unsqueeze(1)], dim=1)

        assert last_encoder_out is not None
        pred_caps = self._parallel_cap_head(last_encoder_out[:, -1, :])
        pred_qic = torch.stack(predictions, dim=1)
        return torch.cat([pred_qic, pred_caps], dim=1)
