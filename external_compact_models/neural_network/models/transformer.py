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
Output: (B, target_dim) — outputs in the configured autoregressive order.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# V7.0.4 — autoregressive prefix cache (plan lever I4). See
# ``_ar_generate_cached``.
#
# DEFAULT OFF, and it must stay off until a full 16-gate complex re-gate
# clears it — same bar as the V7.0.3 fused Jacobian. I4 was routed as
# "exact in exact arithmetic, needs verification"; the verification came
# back NEGATIVE. It is exact in real arithmetic but *not* bit-identical
# in float32, and the reason is a hard floor rather than a fixable
# implementation detail: on this CPU ``F.linear`` is not row-stable —
# a 1-row GEMV accumulates in a different order than the same row of an
# L-row GEMM (measured 0/96 shapes stable, ~8e-6 abs). Any formulation
# that computes fewer rows than the stock recompute therefore moves the
# last bits, no matter how attention is arranged. Measured end-to-end
# deviation on shipped checkpoints: outputs <= 5.3e-6, autograd Jacobian
# <= 1.6e-6 (see docs/CHANGELOG.md §V7.0.4).
_AR_CACHE = os.environ.get("PYCIRCUITSIM_NN_AR_CACHE", "0") == "1"


class TransformerEncoderModel(nn.Module):
    """Autoregressive Transformer for MOSFET I-V / Q-V / C-V prediction.

    Args:
        input_dim:  7 — continuous features [V(4), NFIN_log, L, T].
        target_dim: Number of predicted targets.
        ar_target_dim: Number of targets generated autoregressively. Remaining
            targets use the legacy parallel head. The default 8 preserves the
            13-target reduced BSIM-AR checkpoint contract.
        d_model:    Transformer hidden dimension.
        nhead:      Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Feedforward network dimension.
        dropout:    Dropout rate.
        num_tech_codes: Vocabulary size for the tech embedding.
        tech_embed_dropout: During training, probability of replacing the
            real tech code with ``unknown_code_id``. Trains the UNKNOWN
            embedding to serve as a generic-device representation
            for zero-shot inference.
        unknown_code_id: Index of the UNKNOWN slot inside the embedding
            vocabulary. Per-tech local vocabs (Rule 16) place it at
            ``num_tech_codes - 1``; ``None`` derives exactly that. The
            universal vocab's 17 falls out of the same rule (18 - 1).
    """

    # P4 — legacy parallel C-block defaults. Kept as class constants because
    # existing tests and checkpoints describe the reduced 8+5 contract with
    # these names; instances may now choose a different AR split.
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
        unknown_code_id: int | None = None,
        ar_target_dim: int = CAP_START,
    ) -> None:
        super().__init__()

        assert input_dim == 7, (
            "BSIMAR expects 7-column continuous input "
            "[V(4), NFIN_log, L, T], got "
            f"input_dim={input_dim}"
        )
        if target_dim <= 0:
            raise ValueError(f"target_dim must be positive, got {target_dim}")
        if not 1 <= ar_target_dim <= target_dim:
            raise ValueError(
                "ar_target_dim must be in [1, target_dim], got "
                f"{ar_target_dim} for target_dim={target_dim}")

        self.raw_input_dim = input_dim
        self.target_dim = target_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # A2 — 3 context tokens.
        self.input_dim = self.N_GROUPED_INPUT_TOKENS

        # P4 — legacy checkpoints use 8 AR targets followed by 5 parallel
        # capacitances. Full-terminal BSIM-AR records either a six-target AR
        # chain or a three-charge AR chain plus three parallel currents in its
        # architecture sidecar, without adding new state-dict keys.
        self.ar_target_dim = int(ar_target_dim)
        self.parallel_target_dim = target_dim - self.ar_target_dim

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
        # Rule 16: UNKNOWN sits at the tail of the (possibly local) vocab.
        # Hardcoding the universal 17 here would CUDA-assert on per-tech
        # vocabs (TSMC5 vocab=5 → unknown=4) the first time p_unknown fires.
        self._unknown_code_id = (
            unknown_code_id if unknown_code_id is not None
            else num_tech_codes - 1)

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

    def _parallel_tail_head(
        self, last_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Emit the configured non-AR tail from one hidden state."""
        if self.parallel_target_dim == 0:
            return last_hidden.new_empty((last_hidden.size(0), 0))
        device = last_hidden.device
        cap_token_ids = torch.arange(
            self.input_dim + 1 + self.ar_target_dim,
            self.input_dim + 1 + self.target_dim,
            device=device,
        )
        cap_te = self.token_type_emb(cap_token_ids)
        cap_h = last_hidden.unsqueeze(1) + cap_te.unsqueeze(0)
        return self._project_outputs(cap_h, start_idx=self.ar_target_dim)

    # ── V7.0.4: autoregressive prefix cache (plan lever I4) ──────────
    #
    # The stock inference loop below re-runs the ENTIRE encoder over the
    # whole growing prefix once per AR step: 8 passes over 4, 5, ..., 11
    # tokens = 60 token-passes to produce 11 distinct hidden states.
    # Under a causal mask a prefix hidden state cannot change once
    # computed, so 49 of those 60 are pure waste — and this recompute,
    # not model size, is why BSIM-AR inference is ~30-100x DirectNet.
    #
    # The cached path streams each token through the stack exactly once,
    # keeping per-layer K/V so the new token can still attend to the whole
    # prefix. Same algebra, one evaluation each — but see ``_AR_CACHE``:
    # the summation order is NOT the same, which is why it is opt-in.

    def _ar_cache_usable(self) -> bool:
        """Whether ``_ar_generate_cached`` models this module at all.

        The incremental path re-implements ``TransformerEncoderLayer``'s
        pre-LN block against ``self_attn``'s packed projection weights, so
        it is only valid for the exact configuration this class builds.
        Anything else (a hand-edited encoder, a training-mode call whose
        dropout is stochastic) falls back to the stock loop rather than
        silently computing something different. These are *correctness*
        conditions — the float deviation documented on ``_AR_CACHE`` is
        separate and applies whenever the cache runs at all.
        """
        if self.training:
            return False   # dropout is live; the stock loop owns that path
        norm = self.transformer_encoder.norm
        if norm is not None and not isinstance(norm, nn.LayerNorm):
            return False   # a non-per-position final norm would not split
        for layer in self.transformer_encoder.layers:
            sa = getattr(layer, "self_attn", None)
            if sa is None or not getattr(layer, "norm_first", False):
                return False
            if not (getattr(sa, "_qkv_same_embed_dim", False)
                    and getattr(sa, "batch_first", False)):
                return False
            if sa.bias_k is not None or sa.bias_v is not None:
                return False
            if getattr(sa, "add_zero_attn", False):
                return False
            if sa.num_heads != self.nhead or sa.embed_dim != self.d_model:
                return False
        return True

    def _encoder_append(
        self,
        x_new: torch.Tensor,
        cache: list[list[torch.Tensor | None]],
    ) -> torch.Tensor:
        """Push ``T`` new tokens through the stack, extending the K/V cache.

        Args:
            x_new: ``(B, T, d_model)`` layer-0 input for the new tokens
                (token-type embedding already added).
            cache: per-layer ``[k, v]`` of shape ``(B, nhead, S, head_dim)``
                for the ``S`` tokens already consumed; mutated in place.

        Returns:
            ``(B, T, d_model)`` encoder output for the new tokens only.
            Prefix outputs are not recomputed — under the causal mask they
            are unchanged, and the AR loop only ever reads the last one.
        """
        B, T, _ = x_new.shape
        n_head = self.nhead
        head_dim = self.d_model // n_head

        x = x_new
        for li, layer in enumerate(self.transformer_encoder.layers):
            sa = layer.self_attn

            # Pre-LN self-attention block, with K/V of the prefix reused.
            h = layer.norm1(x)
            qkv = F.linear(h, sa.in_proj_weight, sa.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, n_head, head_dim).transpose(1, 2)
            k = k.view(B, T, n_head, head_dim).transpose(1, 2)
            v = v.view(B, T, n_head, head_dim).transpose(1, 2)

            k_prev, v_prev = cache[li]
            if k_prev is not None:
                k = torch.cat([k_prev, k], dim=2)
                v = torch.cat([v_prev, v], dim=2)
            cache[li] = [k, v]

            # New token t sits at absolute position (S - T) + t, so it may
            # attend to every key up to there. Both shapes the AR loop
            # actually produces are mask-free: priming (T == S) is plain
            # causal attention, and an incremental token (T == 1) attends
            # to the entire cache.
            #
            # Priming passes ``is_causal=True`` rather than an equivalent
            # triangular mask on purpose. ``nn.TransformerEncoder`` runs
            # ``_detect_is_causal_mask`` over the mask it is handed and
            # forwards the hint, so the stock call reaches SDPA's fused
            # causal kernel with ``attn_mask=None``; rebuilding the mask
            # explicitly would select the math kernel and shift the primer
            # by ~7e-7 for no reason. Matching it costs nothing and keeps
            # the deviation confined to the incremental steps.
            S = k.size(2)
            attn_mask = None
            is_causal = False
            if T == S:
                is_causal = True
            elif T > 1:
                attn_mask = torch.triu(
                    torch.full((T, S), float("-inf"),
                               dtype=q.dtype, device=q.device),
                    diagonal=S - T + 1)
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal)
            attn = attn.transpose(1, 2).reshape(B, T, self.d_model)

            x = x + sa.out_proj(attn)
            x = x + layer.linear2(
                layer.activation(layer.linear1(layer.norm2(x))))

        norm = self.transformer_encoder.norm
        return x if norm is None else norm(x)

    def _ar_generate_cached(
        self,
        context_emb: torch.Tensor,
        start_token: torch.Tensor,
    ) -> torch.Tensor:
        """Cached equivalent of the stock autoregressive inference loop."""
        device = context_emb.device
        cache: list[list[torch.Tensor | None]] = [
            [None, None] for _ in self.transformer_encoder.layers]

        # Prime with the 3 context tokens + the start token — byte for byte
        # the sequence the stock loop's first pass sees.
        primer = torch.cat(
            [context_emb, self._embed_ar_scalars(start_token)], dim=1)
        primer = self._add_token_type(primer)
        last_hidden = self._encoder_append(primer, cache)[:, -1, :]

        next_pos = primer.size(1)
        predictions = []
        for i in range(self.ar_target_dim):
            next_pred = self.output_heads[i](last_hidden).squeeze(-1)
            predictions.append(next_pred)
            if i == self.ar_target_dim - 1:
                break
            token = self._embed_ar_scalars(next_pred.unsqueeze(1))
            token = token + self.token_type_emb(
                torch.tensor([next_pos], device=device)).unsqueeze(0)
            last_hidden = self._encoder_append(token, cache)[:, -1, :]
            next_pos += 1

        pred_tail = self._parallel_tail_head(last_hidden)
        pred_qic = torch.stack(predictions, dim=1)
        return torch.cat([pred_qic, pred_tail], dim=1)

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
            pred_tail = self._parallel_tail_head(encoder_out[:, -1, :])

            return torch.cat([pred_qic, pred_tail], dim=1)

        # Inference: autoregressive generation
        context_emb = self._embed_context(x, tech_codes)
        start_token = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)

        # V7.0.4 — same generation, each token encoded once (lever I4).
        if _AR_CACHE and self._ar_cache_usable():
            return self._ar_generate_cached(context_emb, start_token)

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
        pred_tail = self._parallel_tail_head(last_encoder_out[:, -1, :])
        pred_qic = torch.stack(predictions, dim=1)
        return torch.cat([pred_qic, pred_tail], dim=1)
