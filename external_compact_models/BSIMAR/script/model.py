"""BSIM-AR: Autoregressive Transformer model for MOSFET compact modeling.

Architecture: Transformer encoder with causal mask, teacher forcing during
training, autoregressive inference at test time.

Input:  (B, input_dim)  — 18 normalized features (voltages + geometry + 12 process params)
Output: (B, target_dim) — 13 outputs matching DirectNet OUTPUT_COLUMNS
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer sequence positions."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)
        return x + pe.to(x.device)


class TransformerEncoderModel(nn.Module):
    """Autoregressive Transformer for MOSFET I-V / Q-V prediction.

    Each scalar feature is projected to d_model dimensions, then processed
    by a causal Transformer encoder. During training, teacher forcing feeds
    ground-truth targets; during inference, outputs are generated one-by-one.

    Args:
        input_dim:  Number of input features (default 18 = 4V + 2geo + 12proc).
        target_dim: Number of output targets (default 13 = DirectNet OUTPUT_COLUMNS).
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

        # Project each scalar feature to d_model
        self.input_projection = nn.Linear(1, d_model)

        # Positional encoding (reserve slot for start token)
        max_len = input_dim + target_dim + 1  # +1 for start token
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # Standard nn.TransformerEncoder with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection: d_model -> 1 scalar per position
        self.output_layer = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Generate additive causal mask: mask[i,j] = -inf if j > i."""
        return torch.triu(
            torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1
        )

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
            # ── Training: teacher forcing with start token ──
            # Sequence: [x_1, ..., x_D, 0, y_1, ..., y_{T-1}]
            start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
            y_shifted = torch.cat([start_token, y[:, :-1]], dim=1)  # (B, target_dim)

            full_input = torch.cat([x, y_shifted], dim=1)  # (B, input_dim + target_dim)

            # Project and add positional encoding
            embedded = self.input_projection(full_input.unsqueeze(-1))  # (B, L, d_model)
            embedded = self.pos_encoder(embedded)

            # Causal mask (GPT-style)
            L = embedded.size(1)
            causal_mask = self._generate_causal_mask(L).to(x.device)

            encoder_out = self.transformer_encoder(embedded, mask=causal_mask)

            # Take last target_dim positions as predictions
            predictions = self.output_layer(encoder_out).squeeze(-1)  # (B, L)
            return predictions[:, -self.target_dim:]  # (B, target_dim)

        else:
            # ── Inference: autoregressive generation ──
            # Start with [x, 0] and generate target_dim outputs one by one
            start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
            current_seq = torch.cat([x, start_token], dim=1)  # (B, input_dim+1)
            predictions = []

            for i in range(self.target_dim):
                embedded = self.input_projection(current_seq.unsqueeze(-1))
                embedded = self.pos_encoder(embedded)

                L = embedded.size(1)
                causal_mask = self._generate_causal_mask(L).to(x.device)

                out = self.transformer_encoder(embedded, mask=causal_mask)

                # Last position predicts next output
                next_pred = self.output_layer(out[:, -1, :]).squeeze(-1)  # (B,)
                predictions.append(next_pred)

                # Append prediction to sequence for next step
                if i < self.target_dim - 1:
                    current_seq = torch.cat(
                        [current_seq, next_pred.unsqueeze(1)], dim=1
                    )

            return torch.stack(predictions, dim=1)  # (B, target_dim)

    def forward_scheduled(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        ss_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass with scheduled sampling.

        For each target position, with probability ss_ratio, use the model's
        own previous prediction instead of the ground-truth token.
        When ss_ratio=0, this is pure teacher forcing.
        When ss_ratio=1, this is fully autoregressive (but still supervised).

        Args:
            x: (B, input_dim) input features.
            y: (B, target_dim) ground-truth targets.
            ss_ratio: Probability of using model's own prediction at each step.

        Returns:
            (B, target_dim) predicted outputs.
        """
        if ss_ratio <= 0.0:
            return self.forward(x, y)

        batch_size = x.size(0)
        start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        current_seq = torch.cat([x, start_token], dim=1)  # (B, input_dim+1)
        predictions = []

        for t in range(self.target_dim):
            embedded = self.input_projection(current_seq.unsqueeze(-1))
            embedded = self.pos_encoder(embedded)

            L = embedded.size(1)
            causal_mask = self._generate_causal_mask(L).to(x.device)

            out = self.transformer_encoder(embedded, mask=causal_mask)
            next_pred = self.output_layer(out[:, -1, :]).squeeze(-1)  # (B,)
            predictions.append(next_pred)

            # Decide next token: model prediction or ground truth
            if t < self.target_dim - 1:
                use_pred = torch.rand(batch_size, device=x.device) < ss_ratio
                next_token = torch.where(
                    use_pred, next_pred.detach(), y[:, t]
                )
                current_seq = torch.cat(
                    [current_seq, next_token.unsqueeze(1)], dim=1
                )

        return torch.stack(predictions, dim=1)  # (B, target_dim)

    def forward_curriculum(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_targets: int = -1,
        ss_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass predicting only the first n_targets outputs.

        Uses scheduled sampling for the active targets. Inactive targets
        (positions >= n_targets) are filled with teacher-forced ground truth
        to maintain consistent sequence length for positional encoding.

        Args:
            x: (B, input_dim) input features.
            y: (B, target_dim) ground-truth targets.
            n_targets: Number of targets to actively predict (default: all).
            ss_ratio: Scheduled sampling ratio for active positions.

        Returns:
            (B, target_dim) predictions. Positions >= n_targets are copies of y.
        """
        if n_targets <= 0 or n_targets >= self.target_dim:
            return self.forward_scheduled(x, y, ss_ratio=ss_ratio)

        batch_size = x.size(0)
        start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        current_seq = torch.cat([x, start_token], dim=1)
        predictions = []

        for t in range(self.target_dim):
            embedded = self.input_projection(current_seq.unsqueeze(-1))
            embedded = self.pos_encoder(embedded)

            L = embedded.size(1)
            causal_mask = self._generate_causal_mask(L).to(x.device)

            out = self.transformer_encoder(embedded, mask=causal_mask)
            next_pred = self.output_layer(out[:, -1, :]).squeeze(-1)

            if t < n_targets:
                predictions.append(next_pred)
            else:
                # Beyond curriculum horizon: use ground truth as output
                predictions.append(y[:, t])

            if t < self.target_dim - 1:
                if t < n_targets:
                    # Scheduled sampling for active targets
                    use_pred = torch.rand(batch_size, device=x.device) < ss_ratio
                    next_token = torch.where(use_pred, next_pred.detach(), y[:, t])
                else:
                    # Teacher forcing for inactive targets
                    next_token = y[:, t]
                current_seq = torch.cat(
                    [current_seq, next_token.unsqueeze(1)], dim=1
                )

        return torch.stack(predictions, dim=1)
