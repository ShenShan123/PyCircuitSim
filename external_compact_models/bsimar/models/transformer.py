"""BSIM-AR: Autoregressive Transformer model for MOSFET compact modeling.

Architecture: causal Transformer encoder with teacher forcing during
training and autoregressive inference at test time.

Input: 7-dim continuous [V(4), NFIN_log, L, T] plus a discrete integer
``tech_code`` per sample. The tech code is looked up in an ``nn.Embedding``
table, providing a learned representation of the technology variant.
This supports a reserved UNKNOWN code for zero-shot inference on unseen
technologies.

Three context tokens (voltage / geometry / tech-embedding) feed into the
Transformer, followed by AR target tokens with causal masking.

Design choices (paper + v3 sprint findings):

- **Grouped input tokens (A2)** — always on.
- **Parallel cap head (P4)** — always on.
- **Scalar projection + learned token-type embedding (B1)**.
- **Pre-LN encoder layers (B2)** — ``norm_first=True``.
- **GELU feed-forward activation (B5)**.
- **Per-token output heads (B3)**.
- **GPT-2 scaled residual init (B4)**.

Input:  (B, 7) continuous + (B,) integer tech codes
Output: (B, 13) — outputs in BSIMAR (paper) AR order.
"""

import math

import torch
import torch.nn as nn


class TransformerEncoderModel(nn.Module):
    """Autoregressive Transformer for MOSFET I-V / Q-V / C-V prediction.

    Args:
        input_dim:  7 — continuous features [V(4), NFIN_log, L, T].
        target_dim: Must be 13 (BSIMAR_COLUMN_ORDER).
        d_model:    Transformer hidden dimension.
        nhead:      Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Feedforward network dimension.
        dropout:    Dropout rate.
        num_tech_codes: Vocabulary size for the tech embedding.
        tech_embed_dropout: During training, probability of replacing the
            real tech code with UNKNOWN_CODE_ID. Trains the UNKNOWN
            embedding to serve as a generic-device representation
            for zero-shot inference.
    """

    # P4 — parallel C-block constants.
    CAP_START: int = 8
    N_CAPS: int = 5

    # A2 — grouped input layout.
    VOLTAGE_SLICE = slice(0, 4)
    GEOM_SLICE = slice(4, 7)
    N_GROUPED_INPUT_TOKENS = 3

    def __init__(
        self,
        input_dim: int = 7,
        target_dim: int = 13,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.2,
        *,
        num_tech_codes: int = 22,
        tech_embed_dropout: float = 0.0,
    ):
        super().__init__()

        assert input_dim == 7, (
            "BSIMAR expects 7-column continuous input "
            "[V(4), NFIN_log, L, T], got "
            f"input_dim={input_dim}"
        )
        assert target_dim == 13, (
            f"BSIMAR assumes BSIMAR_COLUMN_ORDER, got target_dim={target_dim}"
        )

        self.raw_input_dim = input_dim
        self.target_dim = target_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # A2 — 3 context tokens.
        self.input_dim = self.N_GROUPED_INPUT_TOKENS

        # P4 — AR sequence = first 8 targets (charges + currents/conds).
        self.ar_target_dim = self.CAP_START

        # Scalar projection for start token + AR target tokens.
        self.input_projection = nn.Linear(1, d_model)

        # Grouped context tokenizers.
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

        # Discrete tech-variant embedding.
        self.tech_embedding = nn.Embedding(num_tech_codes, d_model)
        self.num_tech_codes = num_tech_codes
        self._tech_embed_dropout = tech_embed_dropout
        self._unknown_code_id = 17  # matches bsimar.config.UNKNOWN_CODE_ID

        # B1: Learned token-type embedding.
        self.n_tokens = self.input_dim + 1 + target_dim
        self.token_type_emb = nn.Embedding(self.n_tokens, d_model)

        # B2 + B5: Pre-LN encoder + GELU activation.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        # B3: Per-token output heads.
        self.output_heads = nn.ModuleList(
            [nn.Linear(d_model, 1) for _ in range(target_dim)]
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # B4: GPT-2-style scaled residual init.
        scale = 1.0 / math.sqrt(2 * max(self.num_layers, 1))
        for layer in self.transformer_encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight, gain=scale)
            nn.init.xavier_uniform_(layer.linear2.weight, gain=scale)

    def _embed_context(
        self,
        x: torch.Tensor,
        tech_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Embed the raw input context as ``(B, 3, d_model)``.

        Two MLPs (voltage, geometry) + one embedding lookup (tech code).
        """
        v_tok = self.voltage_group(x[:, self.VOLTAGE_SLICE])   # (B, d)
        g_tok = self.geom_group(x[:, self.GEOM_SLICE])         # (B, d)

        # Embedding dropout: during training, randomly replace codes
        # with UNKNOWN_CODE_ID to train the generic representation.
        if self.training and self._tech_embed_dropout > 0.0:
            mask = (torch.rand(tech_codes.size(0), device=tech_codes.device)
                    < self._tech_embed_dropout)
            tech_codes = tech_codes.clone()
            tech_codes[mask] = self._unknown_code_id

        p_tok = self.tech_embedding(tech_codes)                # (B, d)

        return torch.stack([v_tok, g_tok, p_tok], dim=1)       # (B, 3, d)

    def _embed_ar_scalars(self, scalars: torch.Tensor) -> torch.Tensor:
        """Embed AR-side scalar tokens (start token + previous targets)."""
        return self.input_projection(scalars.unsqueeze(-1))

    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Generate additive causal mask: mask[i,j] = -inf if j > i."""
        return torch.triu(
            torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1
        )

    def _add_token_type(self, embedded: torch.Tensor) -> torch.Tensor:
        """Add learned token-type embeddings to a (B, L, d_model) tensor."""
        L = embedded.size(1)
        token_ids = torch.arange(L, device=embedded.device)
        return embedded + self.token_type_emb(token_ids).unsqueeze(0)

    def _project_outputs(
        self, hidden: torch.Tensor, start_idx: int
    ) -> torch.Tensor:
        """Project hidden states with per-target heads."""
        outs = []
        for k in range(hidden.size(1)):
            head = self.output_heads[start_idx + k]
            outs.append(head(hidden[:, k]).squeeze(-1))
        return torch.stack(outs, dim=1)

    def _parallel_cap_head(
        self, last_hidden: torch.Tensor
    ) -> torch.Tensor:
        """P4: emit all N_CAPS cap outputs in parallel from a single hidden state."""
        device = last_hidden.device
        cap_token_ids = torch.arange(
            self.input_dim + 1 + self.CAP_START,
            self.input_dim + 1 + self.CAP_START + self.N_CAPS,
            device=device,
        )
        cap_te = self.token_type_emb(cap_token_ids)
        cap_h = last_hidden.unsqueeze(1) + cap_te.unsqueeze(0)
        return self._project_outputs(cap_h, start_idx=self.CAP_START)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        *,
        tech_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional teacher forcing.

        Args:
            x: (B, 7) input features.
            y: (B, target_dim) ground-truth targets for teacher forcing.
               If None, uses autoregressive inference.
            tech_codes: (B,) integer tech-variant codes (required).

        Returns:
            (B, target_dim) predicted outputs in BSIMAR_COLUMN_ORDER.
        """
        assert tech_codes is not None, "tech_codes is required"
        batch_size = x.size(0)

        if y is not None:
            # Training: teacher forcing with start token
            context_emb = self._embed_context(x, tech_codes)

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

            qic_hidden = encoder_out[:, -self.ar_target_dim:]
            pred_qic = self._project_outputs(qic_hidden, start_idx=0)
            pred_caps = self._parallel_cap_head(encoder_out[:, -1, :])

            return torch.cat([pred_qic, pred_caps], dim=1)

        # Inference: autoregressive generation
        context_emb = self._embed_context(x, tech_codes)
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
