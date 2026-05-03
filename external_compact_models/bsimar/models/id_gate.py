"""Structural Vds gate on the Id output of NN compact models (B3).

Sprint S-ARCH-A (v5 Phase B). The gate multiplies the post-denormalisation
``id_phys`` by ``tanh(Vds_phys / VT_arch)`` and re-normalises in place. This
enforces ``Id(Vds=0) = 0`` *structurally* at training time, rather than via
the inference-time patch in ``_MOSFETNNBase._apply_vds_correction``.

Dual-head pattern (see plan §4 B3):
    * The model emits ``id_raw_norm`` (untrained at Vds=0) — what the AR
      conditioning sees.
    * ``apply_id_gate`` produces ``id_gated_norm`` — what the loss target
      and the simulator consume.

The function is stateless: it pulls fresh tensors from the normaliser at
call time so it stays consistent with whatever dtype/device ``x_norm``
lives on.

Math summary
------------
Let ``vd_idx, vs_idx`` be the input columns for Vd, Vs. The normalised
input layout is ``[Vd_n, Vg_n, Vs_n, Vb_n, NFIN_log_n, L_n, T_n]``, so
``vd_idx=0`` and ``vs_idx=2``.

Vds in physical space::

    Vd_phys = x_norm[:, vd_idx] * input_std[vd_idx] + input_mean[vd_idx]
    Vs_phys = x_norm[:, vs_idx] * input_std[vs_idx] + input_mean[vs_idx]
    Vds_phys = Vd_phys - Vs_phys
    gate = tanh(Vds_phys / vt_arch)

zscore mode::

    id_raw_phys   = id_raw_norm * out_std[id_idx] + out_mean[id_idx]
    id_gated_phys = id_raw_phys * gate
    id_gated_norm = (id_gated_phys - out_mean[id_idx]) / out_std[id_idx]

asinh mode (out_norm is in asinh+zscore space)::

    u             = id_raw_norm * out_std[id_idx] + out_mean[id_idx]
    id_raw_phys   = s_id * sinh(u)
    id_gated_phys = id_raw_phys * gate
    id_gated_norm = (asinh(id_gated_phys / s_id) - out_mean[id_idx])
                    / out_std[id_idx]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    # Type-only import to avoid circular import at module load time.
    from bsimar.data.normalize import BSIMARNormalizer


def apply_id_gate(
    x_norm: torch.Tensor,
    out_norm: torch.Tensor,
    normalizer: "BSIMARNormalizer",
    *,
    id_idx_in_output: int,
    id_idx_in_stats: int = 0,
    vd_idx_in_input: int = 0,
    vs_idx_in_input: int = 2,
    vt_arch: float = 0.04,
    eps_grad: float = 1e-3,  # reserved; tanh is already smooth at zero
) -> torch.Tensor:
    """Apply structural Vds gate on the id slot of ``out_norm``.

    Returns a NEW tensor of shape ``out_norm.shape``. Every column except
    ``id_idx_in_output`` is identical to the corresponding column of
    ``out_norm``; the id slot is replaced by the gated normalised value.

    The function distinguishes two indices that refer to *different* column
    orderings:

    * ``id_idx_in_output`` slices the model's output tensor — DirectNet uses
      ``OUTPUT_COLUMN_ORDER`` (id at 0); the BSIMAR Transformer uses
      ``BSIMAR_COLUMN_ORDER`` (id at 4).
    * ``id_idx_in_stats`` reads id's normalisation stats out of
      ``BSIMARNormStats``. **The stats always live in
      ``OUTPUT_COLUMN_ORDER``** regardless of the model's layout, so the
      correct value is **always 0**. The default of 0 reflects this
      invariant; passing anything else silently denormalises id with the
      wrong column's scale (e.g. qg's ~1e-16 asinh_scale, a ~10⁹× error).

    Args:
        x_norm: ``(B, F)`` normalised input tensor (autograd-friendly).
        out_norm: ``(B, K)`` model output in either OUTPUT_COLUMN_ORDER
            (DirectNet, K=13, id_idx_in_output=0) or BSIMAR_COLUMN_ORDER
            (Transformer, K=13, id_idx_in_output=4).
        normalizer: a fitted ``BSIMARNormalizer``. Must have
            ``normalizer.stats`` populated.
        id_idx_in_output: column index of id in ``out_norm``.
            **0 for DirectNet, 4 for BSIMAR.**
        id_idx_in_stats: column index of id in ``normalizer.stats``.
            **Always 0** because ``BSIMARNormStats`` always lives in
            ``OUTPUT_COLUMN_ORDER`` (default).
        vd_idx_in_input: column index of Vd in ``x_norm`` (default 0).
        vs_idx_in_input: column index of Vs in ``x_norm`` (default 2).
        vt_arch: thermal-voltage-like gate width in volts. ``0.04 V``
            is the architecture default from the plan.
        eps_grad: reserved for a future tanh dead-zone smoother. Currently
            unused — the tanh is C^∞ at zero so no extra smoothing is
            needed.

    Returns:
        ``(B, K)`` tensor with the id slot gated. The other columns are
        passed through unchanged. The returned tensor is a fresh
        allocation (no in-place mutation of ``out_norm``).
    """
    if normalizer.stats is None:
        raise ValueError("normalizer must be fitted (stats is None)")
    mode = normalizer.stats.mode
    if mode not in ("zscore", "asinh"):
        raise ValueError(f"unknown normaliser mode {mode!r}")

    device = x_norm.device
    dtype = x_norm.dtype

    # Pull stats as plain Python floats and lift to tensors on the right
    # device/dtype. This is cheap (a handful of scalar lookups per call)
    # and avoids stashing buffers that might drift out of sync with the
    # caller's device.
    in_mean_vd = float(normalizer.stats.input_mean[vd_idx_in_input])
    in_std_vd = float(normalizer.stats.input_std[vd_idx_in_input])
    in_mean_vs = float(normalizer.stats.input_mean[vs_idx_in_input])
    in_std_vs = float(normalizer.stats.input_std[vs_idx_in_input])
    out_mean_id = float(normalizer.stats.output_mean[id_idx_in_stats])
    out_std_id = float(normalizer.stats.output_std[id_idx_in_stats])

    # Vds in physical units, autograd-friendly.
    vd_phys = (x_norm[:, vd_idx_in_input] * in_std_vd) + in_mean_vd
    vs_phys = (x_norm[:, vs_idx_in_input] * in_std_vs) + in_mean_vs
    vds_phys = vd_phys - vs_phys
    gate = torch.tanh(vds_phys / vt_arch)  # (B,)

    # Extract the id column. Keep autograd graph intact.
    id_raw_norm = out_norm[:, id_idx_in_output]

    if mode == "zscore":
        id_raw_phys = id_raw_norm * out_std_id + out_mean_id
        id_gated_phys = id_raw_phys * gate
        id_gated_norm = (id_gated_phys - out_mean_id) / out_std_id
    else:
        # asinh + zscore: out_norm is in asinh+zscore space.
        s_id = float(normalizer.stats.asinh_scale[id_idx_in_stats])
        u = id_raw_norm * out_std_id + out_mean_id  # asinh-space
        id_raw_phys = s_id * torch.sinh(u)
        id_gated_phys = id_raw_phys * gate
        # Re-encode through asinh + zscore.
        id_gated_norm = (
            torch.asinh(id_gated_phys / s_id) - out_mean_id
        ) / out_std_id

    # Build output without in-place mutation: clone, then overwrite id slot.
    out_gated = out_norm.clone()
    # Use index_copy_ on the id column equivalent: we need an autograd-
    # friendly assignment. Direct slicing + assignment via cat is the
    # cleanest way.
    K = out_norm.shape[1]
    if id_idx_in_output == 0:
        out_gated = torch.cat(
            [id_gated_norm.unsqueeze(1), out_norm[:, 1:]], dim=1)
    elif id_idx_in_output == K - 1:
        out_gated = torch.cat(
            [out_norm[:, :K - 1], id_gated_norm.unsqueeze(1)], dim=1)
    else:
        out_gated = torch.cat(
            [
                out_norm[:, :id_idx_in_output],
                id_gated_norm.unsqueeze(1),
                out_norm[:, id_idx_in_output + 1:],
            ],
            dim=1,
        )

    # Cast back to the input dtype (concat under autograd may have
    # broadcast a different dtype; in practice both sides match, but
    # being explicit prevents surprises).
    if out_gated.dtype != dtype:
        out_gated = out_gated.to(dtype)
    if out_gated.device != device:
        out_gated = out_gated.to(device)

    return out_gated
