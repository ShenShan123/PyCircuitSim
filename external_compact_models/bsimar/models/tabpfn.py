"""TabPFN-style in-context compact model for MOSFET I-V / Q-V / C-V (LEVEL=75).

Scaled-down faithful port of the TabPFN v3 architecture
(``TabPFN-main/src/tabpfn/architectures/tabpfn_v3.py``) into the bsimar
stack. The three signature v3 stages are kept:

1. **Feature distribution embedder** — per-column induced self-attention
   (SetTransformer) over ROWS, so each query token sees where its feature
   value sits within the context distribution.
2. **Column aggregator** — per-row transformer over the FEATURE axis with
   learned CLS tokens read out by cross-attention (RoPE over the column
   sequence, v3 default).
3. **ICL transformer** — rows attend to context rows only (train-only
   keys/values); queries never attend to each other (row independence,
   required by the solver's ``batch_eval`` and the per-row MAE loss).

Two deliberate deviations from stock TabPFN v3:

- **Frozen learned context** instead of a user-supplied train set. During
  training a fresh K-row context is sampled from the training bank every
  step (episodic ICL); at save time a stratified K-row context is frozen
  into registered buffers (``ctx_x/ctx_y/ctx_tc``), so the checkpoint is
  self-contained. At inference the context-side computation (inducing
  hidden states + per-layer ICL K/V) is cached once per loaded module and
  every query costs only cross-attention reads.
- **Direct 13-output value head** instead of the 5000-bin bar
  distribution: Newton-Raphson needs cheap smooth first-order autograd
  Jacobians (gm/gds/gmb and the dQ/dV caps), and the trainer is
  MAE-on-asinh-normalized values.

Input:  (B, 7) continuous [V(4), NFIN_log, L, T] + (B,) integer tech codes
Output: (B, 13) in OUTPUT_COLUMN_ORDER (standard DirectNet layout)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TabPFNCompact"]


# ---------------------------------------------------------------------------
# Small shared pieces (ports of tabpfn_v3 helpers)
# ---------------------------------------------------------------------------


class _RMSNorm(nn.RMSNorm):
    """RMSNorm that casts its weight to the input dtype (v3 norm_factory)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is not None and self.weight.dtype != x.dtype:
            return F.rms_norm(
                x, self.normalized_shape, self.weight.to(x.dtype), self.eps)
        return super().forward(x)


def _apply_rope(t: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Rotary embedding along seq dim -2 (contiguous-halves layout, v3)."""
    dtype = t.dtype
    seq_len = t.shape[-2]
    positions = torch.arange(seq_len, device=t.device, dtype=inv_freq.dtype)
    freqs = positions[:, None] * inv_freq[None, :]           # (S, D/2)
    cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)      # (S, D)
    sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)
    half = t.shape[-1] // 2
    t_rot = torch.cat((-t[..., half:], t[..., :half]), dim=-1)
    return (t * cos + t_rot * sin).to(dtype)


class _RotaryEmbedding(nn.Module):
    """Rotary positional embedding over the feature-token axis."""

    def __init__(self, dim: int, theta: float = 100_000.0) -> None:
        super().__init__()
        assert dim % 2 == 0, f"RoPE head dim must be even, got {dim}"
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        # Non-learnable Parameter (not a buffer) to match upstream v3.
        self.freqs = nn.Parameter(inv_freq, requires_grad=False)

    def rotate(self, t_BSHD: torch.Tensor) -> torch.Tensor:
        # v3 rotates in (B, H, S, D) layout; keep the same transpose dance.
        return _apply_rope(
            t_BSHD.transpose(1, 2), self.freqs).transpose(1, 2)


class _MLP(nn.Sequential):
    """Two-layer GELU feed-forward with zero-initialized output (v3 MLP)."""

    def __init__(self, emsize: int, dim_feedforward: int) -> None:
        linear2 = nn.Linear(dim_feedforward, emsize, bias=False)
        nn.init.zeros_(linear2.weight)
        super().__init__(
            nn.Linear(emsize, dim_feedforward, bias=False),
            nn.GELU(),
            linear2,
        )


class _SoftmaxScalingMLP(nn.Module):
    """Query-aware attention scaling (v3 SoftmaxScalingMLP)."""

    def __init__(self, num_heads: int, head_dim: int,
                 n_hidden: int = 64) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.base_mlp = nn.Sequential(
            nn.Linear(1, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, num_heads * head_dim))
        self.query_mlp = nn.Sequential(
            nn.Linear(head_dim, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, head_dim))
        nn.init.zeros_(self.query_mlp[-1].weight)
        nn.init.zeros_(self.query_mlp[-1].bias)

    def forward(self, q_BSHD: torch.Tensor, n: int) -> torch.Tensor:
        logn = torch.tensor(
            [[math.log(max(float(n), 2.0))]],
            device=q_BSHD.device, dtype=q_BSHD.dtype)
        base = self.base_mlp(logn).view(1, 1, self.num_heads, self.head_dim)
        modulation = 1 + torch.tanh(self.query_mlp(q_BSHD))
        return q_BSHD * (base * modulation)


def _sdpa(q_BSHD: torch.Tensor, k_BVHD: torch.Tensor, v_BVHD: torch.Tensor,
          scaling: Optional[_SoftmaxScalingMLP] = None) -> torch.Tensor:
    """SDPA on (B, S, H, D) tensors with optional query scaling."""
    if scaling is not None:
        q_BSHD = scaling(q_BSHD, k_BVHD.shape[1])
    out = F.scaled_dot_product_attention(
        q_BSHD.transpose(1, 2), k_BVHD.transpose(1, 2),
        v_BVHD.transpose(1, 2))
    return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    """Multi-head self-attention, optionally with RoPE (v3 Attention)."""

    def __init__(self, emsize: int, nhead: int) -> None:
        super().__init__()
        assert emsize % nhead == 0
        self.nhead = nhead
        self.head_dim = emsize // nhead
        self.q_projection = nn.Linear(emsize, emsize, bias=False)
        self.k_projection = nn.Linear(emsize, emsize, bias=False)
        self.v_projection = nn.Linear(emsize, emsize, bias=False)
        self.out_projection = nn.Linear(emsize, emsize, bias=False)
        nn.init.xavier_uniform_(self.q_projection.weight)
        nn.init.xavier_uniform_(self.k_projection.weight)
        nn.init.xavier_uniform_(self.v_projection.weight)
        nn.init.zeros_(self.out_projection.weight)

    def forward(self, x_BSE: torch.Tensor,
                rope: Optional[_RotaryEmbedding] = None) -> torch.Tensor:
        B, S, _ = x_BSE.shape
        q = self.q_projection(x_BSE).view(B, S, self.nhead, self.head_dim)
        k = self.k_projection(x_BSE).view(B, S, self.nhead, self.head_dim)
        v = self.v_projection(x_BSE).view(B, S, self.nhead, self.head_dim)
        if rope is not None:
            q = rope.rotate(q)
            k = rope.rotate(k)
        out = _sdpa(q, k, v).reshape(B, S, -1)
        return self.out_projection(out)

    def forward_cross(self, q_BQE: torch.Tensor, kv_BVE: torch.Tensor,
                      rope: Optional[_RotaryEmbedding] = None
                      ) -> torch.Tensor:
        """Cross-attention readout reusing the same projections (v3
        TransformerBlock.forward_cross)."""
        B, Q, _ = q_BQE.shape
        V = kv_BVE.shape[1]
        q = self.q_projection(q_BQE).view(B, Q, self.nhead, self.head_dim)
        k = self.k_projection(kv_BVE).view(B, V, self.nhead, self.head_dim)
        v = self.v_projection(kv_BVE).view(B, V, self.nhead, self.head_dim)
        if rope is not None:
            q = rope.rotate(q)
            k = rope.rotate(k)
        out = _sdpa(q, k, v).reshape(B, Q, -1)
        return self.out_projection(out)


class _CrossAttention(nn.Module):
    """Multi-head cross-attention (v3 CrossAttention)."""

    def __init__(self, emsize: int, nhead: int,
                 scaling: Optional[_SoftmaxScalingMLP] = None) -> None:
        super().__init__()
        assert emsize % nhead == 0
        self.nhead = nhead
        self.head_dim = emsize // nhead
        self.scaling = scaling
        self.q_projection = nn.Linear(emsize, emsize, bias=False)
        self.k_projection = nn.Linear(emsize, emsize, bias=False)
        self.v_projection = nn.Linear(emsize, emsize, bias=False)
        self.out_projection = nn.Linear(emsize, emsize, bias=False)
        nn.init.xavier_uniform_(self.q_projection.weight)
        nn.init.xavier_uniform_(self.k_projection.weight)
        nn.init.xavier_uniform_(self.v_projection.weight)
        nn.init.zeros_(self.out_projection.weight)

    def forward(self, x_q_BQE: torch.Tensor,
                x_kv_BVE: torch.Tensor) -> torch.Tensor:
        B, Q, _ = x_q_BQE.shape
        V = x_kv_BVE.shape[1]
        q = self.q_projection(x_q_BQE).view(B, Q, self.nhead, self.head_dim)
        k = self.k_projection(x_kv_BVE).view(B, V, self.nhead, self.head_dim)
        v = self.v_projection(x_kv_BVE).view(B, V, self.nhead, self.head_dim)
        out = _sdpa(q, k, v, scaling=self.scaling).reshape(B, Q, -1)
        return self.out_projection(out)


class _CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block with MLP (v3 CrossAttentionBlock)."""

    def __init__(self, emsize: int, nhead: int, dim_feedforward: int,
                 scaling: Optional[_SoftmaxScalingMLP] = None) -> None:
        super().__init__()
        self.attn = _CrossAttention(emsize, nhead, scaling=scaling)
        self.mlp = _MLP(emsize, dim_feedforward)
        self.layernorm_q = _RMSNorm(emsize)
        self.layernorm_kv = _RMSNorm(emsize)
        self.layernorm2 = _RMSNorm(emsize)

    def forward(self, x_BQE: torch.Tensor,
                context_BVE: torch.Tensor) -> torch.Tensor:
        x_BQE = x_BQE + self.attn(
            self.layernorm_q(x_BQE), self.layernorm_kv(context_BVE))
        return x_BQE + self.mlp(self.layernorm2(x_BQE))


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block for the column aggregator (v3)."""

    def __init__(self, emsize: int, nhead: int,
                 dim_feedforward: int) -> None:
        super().__init__()
        self.attention = _Attention(emsize, nhead)
        self.layernorm = _RMSNorm(emsize)
        self.layernorm_mlp = _RMSNorm(emsize)
        self.mlp = _MLP(emsize, dim_feedforward)

    def forward(self, x_BSE: torch.Tensor,
                rope: Optional[_RotaryEmbedding] = None) -> torch.Tensor:
        x_BSE = x_BSE + self.attention(self.layernorm(x_BSE), rope=rope)
        return x_BSE + self.mlp(self.layernorm_mlp(x_BSE))

    def forward_cross(self, query_BQE: torch.Tensor,
                      context_BVE: torch.Tensor,
                      rope: Optional[_RotaryEmbedding] = None
                      ) -> torch.Tensor:
        out = query_BQE + self.attention.forward_cross(
            self.layernorm(query_BQE), self.layernorm(context_BVE),
            rope=rope)
        return out + self.mlp(self.layernorm_mlp(out))


class _InducedSelfAttentionBlock(nn.Module):
    """SetTransformer-style induced self-attention over rows (v3 ISAB).

    Stage 1: inducing points attend to CONTEXT rows (query-scaled).
    Stage 2: any rows attend to the inducing hidden states.
    The stage-1 hidden depends only on the context → cacheable.
    """

    def __init__(self, emsize: int, nhead: int, num_inducing_points: int,
                 dim_feedforward: int, scaling_hidden: int = 64) -> None:
        super().__init__()
        self.cross_attn_block1 = _CrossAttentionBlock(
            emsize, nhead, dim_feedforward,
            scaling=_SoftmaxScalingMLP(
                nhead, emsize // nhead, n_hidden=scaling_hidden))
        self.cross_attn_block2 = _CrossAttentionBlock(
            emsize, nhead, dim_feedforward)
        self.inducing_vectors = nn.Parameter(
            torch.empty(num_inducing_points, emsize))
        nn.init.trunc_normal_(self.inducing_vectors, std=0.02)

    def compute_hidden(self, ctx_CRE: torch.Tensor) -> torch.Tensor:
        """Stage 1 on context rows: (C, K, E) → (C, n_ind, E)."""
        C = ctx_CRE.shape[0]
        ind = self.inducing_vectors.unsqueeze(0).expand(C, -1, -1)
        return self.cross_attn_block1(ind, ctx_CRE)

    def read(self, rows_CRE: torch.Tensor,
             hidden_CIE: torch.Tensor) -> torch.Tensor:
        """Stage 2: rows attend to the inducing hidden states."""
        return self.cross_attn_block2(rows_CRE, hidden_CIE)


class _ICLBlock(nn.Module):
    """ICL transformer block: keys/values from context rows only (v3)."""

    def __init__(self, emsize: int, nhead: int, dim_feedforward: int,
                 scaling_hidden: int = 64) -> None:
        super().__init__()
        assert emsize % nhead == 0
        self.nhead = nhead
        self.head_dim = emsize // nhead
        self.scaling = _SoftmaxScalingMLP(
            nhead, self.head_dim, n_hidden=scaling_hidden)
        self.q_projection = nn.Linear(emsize, emsize, bias=False)
        self.k_projection = nn.Linear(emsize, emsize, bias=False)
        self.v_projection = nn.Linear(emsize, emsize, bias=False)
        self.out_projection = nn.Linear(emsize, emsize, bias=False)
        nn.init.xavier_uniform_(self.q_projection.weight)
        nn.init.xavier_uniform_(self.k_projection.weight)
        nn.init.xavier_uniform_(self.v_projection.weight)
        nn.init.zeros_(self.out_projection.weight)
        self.layernorm = _RMSNorm(emsize)
        self.layernorm_mlp = _RMSNorm(emsize)
        self.mlp = _MLP(emsize, dim_feedforward)

    def context_kv(self, ctx_1KW: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project the (normed) context sequence to this layer's K/V."""
        _, K, _ = ctx_1KW.shape
        normed = self.layernorm(ctx_1KW)
        k = self.k_projection(normed).view(1, K, self.nhead, self.head_dim)
        v = self.v_projection(normed).view(1, K, self.nhead, self.head_dim)
        return k, v

    def attend(self, rows_1RW: torch.Tensor, k: torch.Tensor,
               v: torch.Tensor) -> torch.Tensor:
        """Rows (context or query) attend to the context K/V + MLP."""
        _, R, _ = rows_1RW.shape
        q = self.q_projection(self.layernorm(rows_1RW)).view(
            1, R, self.nhead, self.head_dim)
        out = _sdpa(q, k, v, scaling=self.scaling).reshape(1, R, -1)
        rows_1RW = rows_1RW + self.out_projection(out)
        return rows_1RW + self.mlp(self.layernorm_mlp(rows_1RW))


# ---------------------------------------------------------------------------
# Context bank holder
# ---------------------------------------------------------------------------


class _SharedBank:
    """Read-only training-bank holder shared across module deep-copies.

    ``AveragedModel`` (EMA/SWA) deep-copies the wrapped module; without this
    the multi-hundred-MB training tensors would be duplicated. A plain
    attribute (not a buffer) so it never enters ``state_dict``.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor,
                 tc: torch.Tensor) -> None:
        self.x = x
        self.y = y
        self.tc = tc

    def __deepcopy__(self, memo) -> "_SharedBank":
        return self


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class TabPFNCompact(nn.Module):
    """TabPFN-v3-style in-context regressor with a frozen learned context.

    Args:
        input_dim:  7 — continuous features [V(4), NFIN_log, L, T].
        output_dim: Must be 13 (OUTPUT_COLUMN_ORDER).
        embed_dim:  Per-column embedding dim E.
        n_inducing: Inducing points per distribution-embedder block.
        dist_blocks / dist_heads: Distribution-embedder depth / heads.
        agg_blocks / agg_heads:   Column-aggregator depth / heads.
        n_cls_tokens: CLS tokens; ICL width W = n_cls_tokens * embed_dim.
        icl_num_blocks / icl_heads: ICL transformer depth / heads.
        ctx_len:    Context length K (episodic sample size == frozen size).
        num_tech_codes: Vocabulary size for the tech embedding.
        tech_embed_dropout: Train-time probability of replacing a tech code
            with ``unknown_code_id`` (Rule 16; applied to query AND sampled
            context codes).
        unknown_code_id: UNKNOWN slot; ``None`` derives num_tech_codes - 1.
        use_rope:   RoPE over the column-token axis in the aggregator (v3
            default on).
        ff_factor:  FFN expansion factor (v3 default 2).
        feature_group_size: Scalars per grouped column token (v3 default 3;
            circular shifts {1,2,4} form a perfect difference cover mod 7).
    """

    def __init__(
        self,
        input_dim: int = 7,
        output_dim: int = 13,
        embed_dim: int = 96,
        *,
        n_inducing: int = 32,
        dist_blocks: int = 3,
        dist_heads: int = 6,
        agg_blocks: int = 3,
        agg_heads: int = 6,
        n_cls_tokens: int = 2,
        icl_num_blocks: int = 4,
        icl_heads: int = 6,
        ctx_len: int = 2048,
        num_tech_codes: int = 18,
        tech_embed_dropout: float = 0.1,
        unknown_code_id: Optional[int] = None,
        use_rope: bool = True,
        ff_factor: int = 2,
        feature_group_size: int = 3,
    ) -> None:
        super().__init__()
        assert input_dim == 7, (
            "TabPFNCompact expects 7-column continuous input "
            f"[V(4), NFIN_log, L, T], got input_dim={input_dim}")
        assert output_dim == 13, (
            f"TabPFNCompact assumes OUTPUT_COLUMN_ORDER, got {output_dim}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.n_cls_tokens = n_cls_tokens
        self.icl_emsize = embed_dim * n_cls_tokens
        self.ctx_len = ctx_len
        self.num_tech_codes = num_tech_codes
        self.feature_group_size = feature_group_size
        # Feature-token count: input_dim grouped columns + 1 tech token.
        self.n_columns = input_dim + 1
        self._tech_embed_dropout = tech_embed_dropout
        # Rule 16: UNKNOWN sits at the tail of the (possibly local) vocab.
        self._unknown_code_id = (
            unknown_code_id if unknown_code_id is not None
            else num_tech_codes - 1)

        E = embed_dim
        W = self.icl_emsize

        # ---- Cell embedding: grouped raw values → E (v3 x_embed) ----
        self.x_embed = nn.Linear(feature_group_size, E)

        # ---- Tech token (bsimar convention, extra feature column) ----
        self.tech_embedding = nn.Embedding(num_tech_codes, E)

        # ---- Target-aware column embedding (context rows only) ----
        self.col_y_encoder = nn.Linear(output_dim, E)

        # ---- Stage 1: per-column distribution embedder ----
        self.dist_blocks = nn.ModuleList(
            _InducedSelfAttentionBlock(
                E, dist_heads, n_inducing, E * ff_factor)
            for _ in range(dist_blocks))

        # ---- Stage 2: column aggregator with CLS readout ----
        self.agg_blocks = nn.ModuleList(
            _TransformerBlock(E, agg_heads, E * ff_factor)
            for _ in range(agg_blocks))
        self.rope = (_RotaryEmbedding(E // agg_heads) if use_rope else None)
        self.cls_tokens = nn.Parameter(torch.empty(n_cls_tokens, E))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        self.agg_out_ln = _RMSNorm(E)

        # ---- Stage 3: ICL transformer ----
        self.icl_y_encoder = nn.Linear(output_dim, W)
        self.icl_blocks = nn.ModuleList(
            _ICLBlock(W, icl_heads, W * ff_factor)
            for _ in range(icl_num_blocks))

        # ---- Head: direct value regression (deviation from bar dist) ----
        self.output_norm = _RMSNorm(W)
        self.head = nn.Sequential(
            nn.Linear(W, W * ff_factor),
            nn.GELU(),
            nn.Linear(W * ff_factor, output_dim),
        )

        # ---- Frozen context buffers (float32; EMA of a constant float
        # buffer is exact, while integer buffers take the non-lerp EMA
        # branch and can silently corrupt — tech codes are cast back with
        # .long() in forward) ----
        self.register_buffer("ctx_x", torch.zeros(ctx_len, input_dim))
        self.register_buffer("ctx_y", torch.zeros(ctx_len, output_dim))
        self.register_buffer("ctx_tc", torch.zeros(ctx_len))

        # ---- Non-state attributes ----
        self._bank: Optional[_SharedBank] = None
        self._ctx_cache: Optional[Dict[str, list]] = None

    # ── Context management ────────────────────────────────────────────

    def set_context_bank(self, x: torch.Tensor, y: torch.Tensor,
                         tc: torch.Tensor) -> None:
        """Install the training bank for episodic context sampling."""
        self._bank = _SharedBank(x, y, tc)

    def set_frozen_context(self, x: torch.Tensor, y: torch.Tensor,
                           tc: torch.Tensor) -> None:
        """Fill the frozen-context buffers (call BEFORE the EMA wrap)."""
        assert x.shape == (self.ctx_len, self.input_dim), (
            f"frozen context x shape {tuple(x.shape)} != "
            f"({self.ctx_len}, {self.input_dim})")
        with torch.no_grad():
            self.ctx_x.copy_(x.float())
            self.ctx_y.copy_(y.float())
            self.ctx_tc.copy_(tc.float())
        self._ctx_cache = None

    def train(self, mode: bool = True) -> "TabPFNCompact":
        # Any train/eval flip may follow a weight update → drop the cache.
        self._ctx_cache = None
        return super().train(mode)

    def _apply(self, fn, recurse: bool = True):
        # .to(device)/.float() etc. relocate parameters → drop the cache.
        self._ctx_cache = None
        return super()._apply(fn, recurse)

    def _sample_context(
        self, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._bank is None:
            raise RuntimeError(
                "TabPFNCompact.training forward needs a context bank; "
                "call set_context_bank(train_x, train_y, train_tc) first")
        n = self._bank.x.shape[0]
        idx = torch.randint(0, n, (self.ctx_len,))
        return (self._bank.x[idx].to(device),
                self._bank.y[idx].to(device),
                self._bank.tc[idx].to(device))

    # ── Token building ────────────────────────────────────────────────

    def _dropout_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Rule-16 embedding dropout (train mode only)."""
        if self.training and self._tech_embed_dropout > 0.0:
            mask = (torch.rand(codes.size(0), device=codes.device)
                    < self._tech_embed_dropout)
            codes = codes.clone()
            codes[mask] = self._unknown_code_id
        return codes

    def _embed_rows(self, x: torch.Tensor, codes: torch.Tensor,
                    y: Optional[torch.Tensor]) -> torch.Tensor:
        """Rows (R, 7) + codes (R,) [+ y (R, 13)] → tokens (R, C, E).

        Circular-shift grouping (v3 _group_features): column token c holds
        the raw values of columns (c+1, c+2, c+4) mod 7. The tech embedding
        is appended as an extra column token; context rows additionally get
        the y embedding broadcast over all column tokens (v3 col_y stage).
        """
        grouped = torch.stack(
            [torch.roll(x, shifts=-(2 ** i), dims=1)
             for i in range(self.feature_group_size)],
            dim=-1)                                     # (R, 7, G)
        tok = self.x_embed(grouped)                     # (R, 7, E)
        tech_tok = self.tech_embedding(codes.long()).unsqueeze(1)
        tok = torch.cat([tok, tech_tok], dim=1)         # (R, C, E)
        if y is not None:
            tok = tok + self.col_y_encoder(y).unsqueeze(1)
        return tok

    # ── Stage runners ─────────────────────────────────────────────────

    def _aggregate(self, tok_RCE: torch.Tensor) -> torch.Tensor:
        """Column aggregator: (R, C, E) → flattened CLS (R, W)."""
        R = tok_RCE.shape[0]
        cls = self.cls_tokens.expand(R, -1, -1)
        seq = torch.cat([cls, tok_RCE], dim=1)          # (R, n_cls+C, E)
        for block in self.agg_blocks[:-1]:
            seq = block(seq, rope=self.rope)
        cls_out = self.agg_blocks[-1].forward_cross(
            seq[:, : self.n_cls_tokens], seq, rope=self.rope)
        return self.agg_out_ln(cls_out).reshape(R, self.icl_emsize)

    def _context_side(
        self, ctx_x: torch.Tensor, ctx_y: torch.Tensor,
        ctx_tc: torch.Tensor,
    ) -> Tuple[list, list]:
        """Run the context through all three stages.

        Returns ``(dist_hidden, icl_kv)`` — per-block inducing hidden
        states (C, n_ind, E) and per-layer ICL (k, v) pairs. Everything a
        query needs from the context.
        """
        ctx_tok = self._embed_rows(ctx_x, ctx_tc, ctx_y)   # (K, C, E)
        ctx_CRE = ctx_tok.transpose(0, 1)                  # (C, K, E)
        dist_hidden: list = []
        for block in self.dist_blocks:
            hidden = block.compute_hidden(ctx_CRE)
            dist_hidden.append(hidden)
            ctx_CRE = block.read(ctx_CRE, hidden)
        ctx_cls = self._aggregate(ctx_CRE.transpose(0, 1))  # (K, W)
        ctx_seq = (ctx_cls + self.icl_y_encoder(ctx_y)).unsqueeze(0)
        icl_kv: list = []
        for block in self.icl_blocks:
            k, v = block.context_kv(ctx_seq)
            icl_kv.append((k, v))
            ctx_seq = block.attend(ctx_seq, k, v)
        return dist_hidden, icl_kv

    def _get_ctx_cache(self, device: torch.device) -> Dict[str, list]:
        """Frozen-context cache for eval mode (built lazily, detached)."""
        if self._ctx_cache is None:
            with torch.no_grad():
                dist_hidden, icl_kv = self._context_side(
                    self.ctx_x, self.ctx_y, self.ctx_tc)
            self._ctx_cache = {
                "dist_hidden": [h.detach() for h in dist_hidden],
                "icl_kv": [(k.detach(), v.detach()) for k, v in icl_kv],
            }
        return self._ctx_cache

    # ── Forward ───────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        tech_codes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict all 13 outputs for query rows.

        Args:
            x: (B, 7) normalized input features.
            tech_codes: (B,) integer tech-variant codes (required).

        Returns:
            (B, 13) predictions in OUTPUT_COLUMN_ORDER (normalized space).
        """
        assert tech_codes is not None, "tech_codes is required"

        if self.training:
            ctx_x, ctx_y, ctx_tc = self._sample_context(x.device)
            ctx_tc = self._dropout_codes(ctx_tc)
            dist_hidden, icl_kv = self._context_side(ctx_x, ctx_y, ctx_tc)
        else:
            cache = self._get_ctx_cache(x.device)
            dist_hidden, icl_kv = cache["dist_hidden"], cache["icl_kv"]

        codes = self._dropout_codes(tech_codes)
        qry_tok = self._embed_rows(x, codes, None)          # (B, C, E)
        qry_CRE = qry_tok.transpose(0, 1)                   # (C, B, E)
        for block, hidden in zip(self.dist_blocks, dist_hidden):
            qry_CRE = block.read(qry_CRE, hidden)
        qry_seq = self._aggregate(
            qry_CRE.transpose(0, 1)).unsqueeze(0)           # (1, B, W)
        for block, (k, v) in zip(self.icl_blocks, icl_kv):
            qry_seq = block.attend(qry_seq, k, v)
        return self.head(self.output_norm(qry_seq)).squeeze(0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
