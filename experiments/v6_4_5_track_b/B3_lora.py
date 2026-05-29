"""B3 LoRA: low-rank fine-tune wrapper for DirectNet.

Provides:
    LoRALinear     — wraps a single nn.Linear with A/B low-rank deltas.
    wrap_trunk_with_lora(model, rank, alpha) — patches all Linear layers in
                                               model.net (the trunk) in-place.
    merge_lora(model) — folds LoRA deltas back into the base weights and
                        strips LoRA state so the result is a plain DirectNet
                        compatible with _build_from_state.

Design constraints (from plan):
* Base weights are frozen (requires_grad=False).
* Only A, B parameters are trained.
* B is zero-initialised so wrapped model == base at init (Rule: sanity check).
* tech_embedding is also kept frozen.
* merge_lora writes W' = W + (alpha/rank) * B @ A, then removes lora.* keys
  so the saved state_dict loads cleanly via the existing _build_from_state.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in wrapper that adds a low-rank delta to a frozen nn.Linear.

    Forward: y = base_linear(x) + x @ A.T @ B.T * (alpha / rank)
    A ~ N(0, 1/sqrt(rank)); B = 0 at init => delta = 0 at init.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        in_features = base.in_features
        out_features = base.out_features

        # Freeze the base layer.
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        # Detect the device the base layer lives on so LoRA params land there.
        dev = next(base.parameters()).device

        # Low-rank parameters on the same device as the base.
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=torch.float32, device=dev))

        # Kaiming init for A; B already zero.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # x @ A.T => (B, rank); then @ B.T => (B, out)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + lora_out * (self.alpha / self.rank)

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """Return W + (alpha/rank) * B @ A (same dtype as base.weight)."""
        delta = (self.alpha / self.rank) * (self.lora_B @ self.lora_A)
        return self.base.weight + delta


def wrap_trunk_with_lora(
    model: nn.Module,
    rank: int,
    alpha: Optional[float] = None,
) -> nn.Module:
    """Replace every nn.Linear in model.net (the trunk) with LoRALinear.

    The tech_embedding and mono (if present) are left untouched and frozen.
    Returns the mutated model (in-place).
    """
    if alpha is None:
        alpha = float(rank)  # standard LoRA convention: alpha == rank => scale=1

    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad = False

    # Walk trunk and swap Linear layers.
    trunk: nn.Sequential = model.net  # type: ignore[attr-defined]
    for i, layer in enumerate(trunk):
        if isinstance(layer, nn.Linear):
            trunk[i] = LoRALinear(layer, rank=rank, alpha=alpha)

    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold LoRA deltas into the base weights and return a plain DirectNet.

    Walks model.net, merges each LoRALinear back to a plain nn.Linear with
    W' = W + (alpha/rank)*B@A, then replaces the LoRALinear with the merged
    Linear. After this call all parameters are plain tensors with no lora.*
    keys — the state_dict loads via _build_from_state without modification.

    Note: the returned model's parameters all have requires_grad=False because
    the base was frozen. The caller should re-enable grad if further training
    is needed (not the case for checkpointing).
    """
    trunk: nn.Sequential = model.net  # type: ignore[attr-defined]
    for i, layer in enumerate(trunk):
        if isinstance(layer, LoRALinear):
            merged_w = layer.merged_weight().detach().clone()
            bias = (layer.base.bias.detach().clone()
                    if layer.base.bias is not None else None)
            new_linear = nn.Linear(
                layer.base.in_features,
                layer.base.out_features,
                bias=(bias is not None),
            )
            with torch.no_grad():
                new_linear.weight.copy_(merged_w)
                if bias is not None:
                    new_linear.bias.copy_(bias)
            # Freeze (consistent with the rest of the state for checkpointing).
            for p in new_linear.parameters():
                p.requires_grad = False
            trunk[i] = new_linear

    return model
