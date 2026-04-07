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
    """

    def __init__(
        self,
        input_dim: int = 18,
        target_dim: int = 13,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Project each scalar feature to d_model
        self.input_projection = nn.Linear(1, d_model)

        # B1: Learned token-type embedding. One row per token position
        # in the full sequence: input features + start token + AR targets.
        # The embedding gives each scalar an *identity* (Vg vs id vs qg)
        # that the encoder can read; sinusoidal PE only encodes order.
        self.n_tokens = input_dim + 1 + target_dim
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
            (B, target_dim) predicted outputs.
        """
        batch_size = x.size(0)

        if y is not None:
            # Training: teacher forcing with start token
            start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
            y_shifted = torch.cat([start_token, y[:, :-1]], dim=1)

            full_input = torch.cat([x, y_shifted], dim=1)

            embedded = self.input_projection(full_input.unsqueeze(-1))
            embedded = self._add_token_type(embedded)

            L = embedded.size(1)
            causal_mask = self._generate_causal_mask(L).to(x.device)

            encoder_out = self.transformer_encoder(embedded, mask=causal_mask)

            target_hidden = encoder_out[:, -self.target_dim:]
            return self._project_outputs(target_hidden, start_idx=0)

        # Inference: autoregressive generation
        start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        current_seq = torch.cat([x, start_token], dim=1)
        predictions = []

        for i in range(self.target_dim):
            embedded = self.input_projection(current_seq.unsqueeze(-1))
            embedded = self._add_token_type(embedded)

            L = embedded.size(1)
            causal_mask = self._generate_causal_mask(L).to(x.device)

            out = self.transformer_encoder(embedded, mask=causal_mask)

            head = self.output_heads[i]
            next_pred = head(out[:, -1, :]).squeeze(-1)
            predictions.append(next_pred)

            if i < self.target_dim - 1:
                current_seq = torch.cat(
                    [current_seq, next_pred.unsqueeze(1)], dim=1
                )

        return torch.stack(predictions, dim=1)

    def forward_scheduled(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        ss_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass with scheduled sampling.

        For each target position, with probability ss_ratio, use the model's
        own previous prediction instead of the ground-truth token.
        """
        if ss_ratio <= 0.0:
            return self.forward(x, y)

        batch_size = x.size(0)
        start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        current_seq = torch.cat([x, start_token], dim=1)
        predictions = []

        for t in range(self.target_dim):
            embedded = self.input_projection(current_seq.unsqueeze(-1))
            embedded = self._add_token_type(embedded)

            L = embedded.size(1)
            causal_mask = self._generate_causal_mask(L).to(x.device)

            out = self.transformer_encoder(embedded, mask=causal_mask)
            head = self.output_heads[t]
            next_pred = head(out[:, -1, :]).squeeze(-1)
            predictions.append(next_pred)

            if t < self.target_dim - 1:
                use_pred = torch.rand(batch_size, device=x.device) < ss_ratio
                next_token = torch.where(
                    use_pred, next_pred.detach(), y[:, t]
                )
                current_seq = torch.cat(
                    [current_seq, next_token.unsqueeze(1)], dim=1
                )

        return torch.stack(predictions, dim=1)

    def forward_curriculum(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_targets: int = -1,
        ss_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass predicting only the first n_targets outputs."""
        if n_targets <= 0 or n_targets >= self.target_dim:
            return self.forward_scheduled(x, y, ss_ratio=ss_ratio)

        batch_size = x.size(0)
        start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        current_seq = torch.cat([x, start_token], dim=1)
        predictions = []

        for t in range(self.target_dim):
            embedded = self.input_projection(current_seq.unsqueeze(-1))
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
                current_seq = torch.cat(
                    [current_seq, next_token.unsqueeze(1)], dim=1
                )

        return torch.stack(predictions, dim=1)
