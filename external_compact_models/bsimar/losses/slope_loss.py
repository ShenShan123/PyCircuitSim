"""Slope-match auxiliary loss for B2 (v5 Phase B).

Penalises ``|d(pred_id_norm)/d(vg_norm) - d(target_id_norm)/d(vg_norm)|``
on a random subsample of grid-class rows in the batch.

Why slope (and not magnitude)?
The B1-regenerated dataset closes the magnitude gap on TSMC7 NMOS DC,
but the verifier's NRMSE plateau is dominated by *shape* error: the
model reproduces |Id| but mis-curves the saturation plateau. A penalty
on dId/dVg (which is gm) acts directly on shape error.

Why grid-class only?
Grid rows (sample_class == 4) are uniformly sampled in voltage; they
are the only class for which a uniform slope penalty has a well-defined
weight. Anchor / vds_zero / hot rows are concentrated near boundaries
or rails and would bias the slope statistic.

Contract for the trainer:
    The trainer MUST construct the input tensor ``x_norm`` with
    ``requires_grad_(True)`` BEFORE calling ``model(x_norm, ...)``.
    Otherwise the autograd-derived predicted slope cannot be computed
    and ``forward`` will raise.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from bsimar.data.normalize import (
    BSIMARNormalizer,
    OUTPUT_COLUMN_ORDER,
)


class SlopeMatchLoss(nn.Module):
    """Slope-match auxiliary loss in normalised space.

    For each batch:
      1. Filter to grid-class rows (``sample_class == 4``).
      2. Sub-sample at most ``max_samples`` rows.
      3. Compute predicted slope via ``torch.autograd.grad`` on the
         model's predicted normalised id w.r.t. the normalised Vg input.
      4. Compute target slope from the physical ``gm`` and ``id``
         recovered by denormalising the normalised target via the
         closed-form chain rule:

         - zscore mode:
               target_slope_norm = (vg_std / id_std) * gm_phys
         - asinh mode:
               target_slope_norm =
                   vg_std * gm_phys
                   / (id_sigma_zscore * sqrt(s_id**2 + id_phys**2))

      5. Return ``mean(|pred_slope - target_slope|)``.

    Args:
        normalizer: Fitted BSIMARNormalizer (any mode).
        mode: "zscore" or "asinh" — must match ``normalizer.stats.mode``.
        id_idx_in_output: Column index of ``id`` in the model's *output*
            tensor. DirectNet outputs OUTPUT_COLUMN_ORDER -> id is
            index 0. BSIMAR Transformer outputs BSIMAR_COLUMN_ORDER ->
            id is index 4.
        vg_idx_in_input: Column index of ``Vg`` in the model's *input*
            tensor. The normalised input layout is
            [Vd, Vg, Vs, Vb, NFIN_log, L, T] so Vg is index 1.
        max_samples: Cap on the per-batch sub-sample size. Slope-loss
            cost is dominated by the autograd backward, which scales
            linearly in this number.
    """

    GRID_CLASS_CODE = 4  # see meta_sample_class_names in the .npz

    def __init__(
        self,
        normalizer: BSIMARNormalizer,
        mode: str,
        id_idx_in_output: int,
        vg_idx_in_input: int = 1,
        max_samples: int = 256,
    ) -> None:
        super().__init__()
        if mode not in ("zscore", "asinh"):
            raise ValueError(f"mode must be zscore or asinh, got {mode!r}")
        if normalizer.stats is None:
            raise ValueError("normalizer must be fitted (stats is None)")
        if normalizer.stats.mode != mode:
            raise ValueError(
                f"normalizer.stats.mode={normalizer.stats.mode!r} does "
                f"not match SlopeMatchLoss mode={mode!r}")

        self.mode = mode
        self.id_idx_in_output = int(id_idx_in_output)
        self.vg_idx_in_input = int(vg_idx_in_input)
        self.max_samples = int(max_samples)

        # The normalised input layout is [Vd, Vg, Vs, Vb, NFIN_log, L, T].
        # The 4 voltage features come first (input_mean/std are per-feature),
        # so vg_std is input_std[vg_idx_in_input].
        vg_std = float(normalizer.stats.input_std[self.vg_idx_in_input])
        # In OUTPUT_COLUMN_ORDER, id is at index 0. The normalizer's
        # output stats are kept in OUTPUT_COLUMN_ORDER (the BSIMAR
        # reordering only happens at the model boundary; the
        # normalizer never sees BSIMAR_COLUMN_ORDER).
        id_idx_in_norm = OUTPUT_COLUMN_ORDER.index("id")
        id_std = float(normalizer.stats.output_std[id_idx_in_norm])

        # Register as buffers so .to(device) carries them.
        self.register_buffer(
            "vg_std",
            torch.tensor(vg_std, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "id_std",
            torch.tensor(id_std, dtype=torch.float32),
            persistent=False,
        )
        # gm in OUTPUT_COLUMN_ORDER (used for denormalising the target)
        self._gm_idx_in_norm = OUTPUT_COLUMN_ORDER.index("gm")
        self._id_idx_in_norm = id_idx_in_norm

        if mode == "asinh":
            s_id = float(normalizer.stats.asinh_scale[id_idx_in_norm])
            self.register_buffer(
                "s_id",
                torch.tensor(s_id, dtype=torch.float32),
                persistent=False,
            )
            # We also need the asinh output_mean for id to recover
            # id_phys from the normalised target. Cache it.
            id_mean = float(normalizer.stats.output_mean[id_idx_in_norm])
            gm_mean = float(normalizer.stats.output_mean[self._gm_idx_in_norm])
            gm_std = float(normalizer.stats.output_std[self._gm_idx_in_norm])
            s_gm = float(normalizer.stats.asinh_scale[self._gm_idx_in_norm])
            self.register_buffer(
                "id_mean", torch.tensor(id_mean, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "gm_mean", torch.tensor(gm_mean, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "gm_std", torch.tensor(gm_std, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "s_gm", torch.tensor(s_gm, dtype=torch.float32),
                persistent=False,
            )
        else:  # zscore
            id_mean = float(normalizer.stats.output_mean[id_idx_in_norm])
            gm_mean = float(normalizer.stats.output_mean[self._gm_idx_in_norm])
            gm_std = float(normalizer.stats.output_std[self._gm_idx_in_norm])
            self.register_buffer(
                "id_mean", torch.tensor(id_mean, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "gm_mean", torch.tensor(gm_mean, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "gm_std", torch.tensor(gm_std, dtype=torch.float32),
                persistent=False,
            )

    # ------------------------------------------------------------------
    # Helpers (pure-torch denormalisation of id and gm, in the model's
    # OUTPUT_COLUMN_ORDER frame).
    # ------------------------------------------------------------------

    def _denorm_id_phys(self, id_norm: torch.Tensor) -> torch.Tensor:
        """Recover physical id from normalised id (id slot in normalizer)."""
        if self.mode == "zscore":
            return id_norm * self.id_std + self.id_mean
        # asinh: id_phys = s_id * sinh(id_norm * id_std + id_mean)
        return self.s_id * torch.sinh(id_norm * self.id_std + self.id_mean)

    def _denorm_gm_phys(self, gm_norm: torch.Tensor) -> torch.Tensor:
        """Recover physical gm from normalised gm slot."""
        if self.mode == "zscore":
            return gm_norm * self.gm_std + self.gm_mean
        return self.s_gm * torch.sinh(gm_norm * self.gm_std + self.gm_mean)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_norm: torch.Tensor,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        sample_class: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the slope-match loss on grid rows.

        Args:
            x_norm: (B, F) normalised input tensor. MUST have
                ``requires_grad=True``; the trainer is responsible for
                this.
            pred_norm: (B, K) model output in either OUTPUT_COLUMN_ORDER
                (DirectNet, K=13) or BSIMAR_COLUMN_ORDER (Transformer,
                K=13). The id column index for this layout was passed
                at construction.
            target_norm: (B, K) ground-truth target in the same order
                as ``pred_norm``. The id column at ``id_idx_in_output``
                holds the normalised id; the gm column is at
                ``id_idx_in_output + 1`` for both layouts (id is always
                immediately followed by gm in both orders).
            sample_class: (B,) int tensor with sample class codes.
                Only rows where ``sample_class == GRID_CLASS_CODE``
                (4) are used.

        Returns:
            Scalar tensor. If no grid rows are present in the batch,
            returns ``torch.zeros((), device=x_norm.device)``.
        """
        device = x_norm.device

        if not x_norm.requires_grad:
            raise RuntimeError(
                "SlopeMatchLoss requires x_norm.requires_grad=True. "
                "The trainer must call x.requires_grad_(True) before "
                "the model forward when slope loss is active.")

        grid_mask = sample_class == self.GRID_CLASS_CODE
        n_grid = int(grid_mask.sum().item())
        if n_grid == 0:
            return torch.zeros((), device=device, dtype=pred_norm.dtype)

        grid_idx = torch.nonzero(grid_mask, as_tuple=False).squeeze(-1)

        # Sub-sample up to max_samples rows (deterministic per call:
        # uses torch's RNG so the trainer can seed if it wants
        # reproducibility).
        if grid_idx.numel() > self.max_samples:
            perm = torch.randperm(grid_idx.numel(), device=device)
            grid_idx = grid_idx[perm[: self.max_samples]]

        # Predicted id (normalised) on the sub-sample.
        pred_id_norm = pred_norm[grid_idx, self.id_idx_in_output]

        # Predicted slope d(pred_id_norm)/d(vg_norm).
        # The .sum() trick: grad of sum-over-rows w.r.t. x_norm is just
        # the per-row grad stacked, so we recover dId/dVg as the diagonal.
        # We grab column vg_idx_in_input from the resulting (B, F) grad.
        grad_x = torch.autograd.grad(
            outputs=pred_id_norm.sum(),
            inputs=x_norm,
            create_graph=True,
            retain_graph=True,
        )[0]
        pred_slope = grad_x[grid_idx, self.vg_idx_in_input]

        # Target slope: closed-form chain rule from gm_phys + id_phys.
        # In both DirectNet and BSIMAR layouts, the gm column sits at
        # id_idx_in_output + 1 (DirectNet: id=0,gm=1; BSIMAR:id=4,gm=5).
        gm_idx_in_pred = self.id_idx_in_output + 1
        target_id_norm_sub = target_norm[grid_idx, self.id_idx_in_output]
        target_gm_norm_sub = target_norm[grid_idx, gm_idx_in_pred]

        # Recover physical id/gm from normalised slots (no grad path
        # needed: target slope is a constant w.r.t. model params).
        with torch.no_grad():
            id_phys = self._denorm_id_phys(target_id_norm_sub)
            gm_phys = self._denorm_gm_phys(target_gm_norm_sub)

            if self.mode == "zscore":
                # d(id_norm)/d(vg_norm) = (vg_std / id_std) * (d id_phys / d vg_phys)
                target_slope = (self.vg_std / self.id_std) * gm_phys
            else:  # asinh
                # id_norm = (asinh(id_phys/s_id) - id_mean) / id_std
                # d(id_norm)/d(id_phys) = 1 / (id_std * sqrt(s_id^2 + id_phys^2))
                # d(vg_norm)/d(vg_phys) = 1 / vg_std
                # => d(id_norm)/d(vg_norm) = vg_std * gm_phys
                #                            / (id_std * sqrt(s_id^2 + id_phys^2))
                denom = self.id_std * torch.sqrt(
                    self.s_id ** 2 + id_phys ** 2)
                target_slope = self.vg_std * gm_phys / denom

        return torch.mean(torch.abs(pred_slope - target_slope))
