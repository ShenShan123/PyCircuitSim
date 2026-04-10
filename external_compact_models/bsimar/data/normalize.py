"""Normalization utilities for BSIMAR training data.

Handles the extreme dynamic range of MOSFET data:
- Currents: 1e-18 A (cutoff) to 1e-4 A (on) — 14 decades
- Conductances: 1e-8 to 1e-3 S — 5 decades
- Charges: 1e-18 to 1e-15 C — 3 decades
- Capacitances: 1e-20 to 1e-15 F — 5 decades

One normalizer class: ``BSIMARNormalizer`` with two modes:

- ``'asinh'``  (recommended, BSIMAR v3 default) — per-target
  ``arcsinh(y / s_k) + zscore``, where ``s_k`` is a per-target
  geometric-mean scale clamped at the floor. Compresses the 14-decade
  dynamic range without the error-amplification behaviour of
  ``inv_signed_log`` that the earlier signed-log normaliser exhibited.

- ``'zscore'`` (DirectNet baseline path) — plain z-score over raw
  outputs. Used by DirectNet; numerically stable because the DirectNet
  MLP is not autoregressive and absolute residuals dominate its loss.

The old signed-log normaliser and the DirectNet-specific ``Normalizer``
class were removed in the v3 sprint: the signed-log chain rule
amplified AR-accumulated errors catastrophically in physical space
(see ``docs/bsimar_improvement_plan_2026_04_08.md`` for the removal
rationale), and the legacy ``Normalizer`` was only used by the
now-deleted ``load_and_split`` path.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ── Floor values and column ordering ────────────────────────────────────────

# Per-target floor magnitudes used by the asinh scale fit. Any sample
# whose |y_k| falls below the corresponding floor contributes the floor
# (not its true value) to the geometric-mean scale ``s_k``, so a noisy
# near-zero sample cannot collapse the scale to zero.
OUTPUT_LOG_FLOORS = {
    # Group A: Currents and conductances
    "id": 1e-18, "gm": 1e-18, "gds": 1e-18, "gmb": 1e-18,
    # Group B: Charges
    "qg": 1e-19, "qd": 1e-19, "qs": 1e-19, "qb": 1e-19,
    # Group C: Capacitances
    "cgg": 1e-20, "cgd": 1e-20, "cgs": 1e-20, "cdg": 1e-20, "cdd": 1e-20,
}

# S2 — per-target asinh_scale lower bounds (2026-04-10 LOO sprint).
# The geometric-mean scale ``s_k`` fit on the training pool can be
# dramatically smaller for gmb / qb than the magnitudes seen in other
# techs: the ASAP7 FinFET modelcard decouples the body from the channel,
# so gmb and qb are 3-4 orders of magnitude smaller than the TSMC family
# (see ``docs/bsimar_loo_improvement_plan_2026_04_10.md``). Without a
# floor the resulting normalized scale is TSMC-dominated and ASAP7 gmb/qb
# map to asinh-space values far outside the training mean±std, producing
# astronomical relative errors on denormalisation. These per-target lower
# bounds pin gmb/qb to roughly linear z-score-like behaviour so the LOO
# error no longer blows up via scale mismatch. They are harmless on
# the in-distribution random split because the train pool still sees
# the full TSMC gmb/qb range.
OUTPUT_ASINH_SCALE_MIN = {
    "gmb": 1e-5,   # TSMC gmb max ≈ 1.07e-4, asinh(y/1e-5) gives [0, ~3.1].
    "qb":  1e-15,  # TSMC qb  max ≈ 5.15e-15, asinh(y/1e-15) gives [0, ~2.4].
}

OUTPUT_COLUMN_ORDER = [
    "id", "gm", "gds", "gmb",
    "qg", "qd", "qs", "qb",
    "cgg", "cgd", "cgs", "cdg", "cdd",
]

# BSIM-AR autoregressive order: paper §4.2 — Q-V → I-V → C-V.
# This matches the BSIM-CMG physical derivative chain (∂Q/∂V → I, then
# ∂I/∂V → g, then ∂Q/∂V → C). Currents and conductances are emitted before
# capacitances so each token conditions on its physical predecessor.
BSIMAR_COLUMN_ORDER = [
    "qg", "qb", "qd", "qs",                          # Q-V
    "id", "gm", "gds", "gmb",                        # I-V (currents + conductances)
    "cgg", "cgd", "cgs", "cdg", "cdd",               # C-V
]

# Permutation indices: BSIMAR_COLUMN_ORDER[i] == OUTPUT_COLUMN_ORDER[_REORDER_IDX[i]]
_REORDER_IDX = [OUTPUT_COLUMN_ORDER.index(c) for c in BSIMAR_COLUMN_ORDER]
_UNREORDER_IDX = [BSIMAR_COLUMN_ORDER.index(c) for c in OUTPUT_COLUMN_ORDER]


def reorder_outputs(arr: np.ndarray) -> np.ndarray:
    """Permute columns from OUTPUT_COLUMN_ORDER -> BSIMAR_COLUMN_ORDER."""
    return arr[:, _REORDER_IDX]


def unreorder_outputs(arr: np.ndarray) -> np.ndarray:
    """Permute columns from BSIMAR_COLUMN_ORDER -> OUTPUT_COLUMN_ORDER."""
    return arr[:, _UNREORDER_IDX]


# ── asinh-scaled transform ───────────────────────────────────────────────────

def asinh_scaled(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Per-target asinh(x / s).

    `s` may be a scalar or a (K,) vector broadcastable against `x`'s last
    axis. Sign-preserving and smooth through zero.
    """
    return np.arcsinh(x / s)


def inv_asinh_scaled(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Inverse of asinh_scaled: s * sinh(y)."""
    return s * np.sinh(y)


# ── Input geometry unpacking ─────────────────────────────────────────────────

def _build_combined_input(
    inputs: np.ndarray,
    geometry: np.ndarray,
) -> np.ndarray:
    """Combine voltage inputs with geometry features.

    Geometry column layouts (backward-compatible):
      (N,  2): [NFIN, T]                         — legacy
      (N,  3): [NFIN, T, PHIG]                   — Phase 13
      (N,  9): [NFIN, T, PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW]  — 7-param
      (N, 14): [NFIN, T, <12 process params>]     — 12-param (old universal)
      (N, 15): [NFIN, L, T, <12 process params>]  — 12-param + L (current)

    Returns combined feature matrix of shape (N, 4+k).
    """
    nfin_log = np.log2(np.clip(geometry[:, 0], 1.0, None))

    if geometry.shape[1] == 15:
        L_col = geometry[:, 1]
        temperature = geometry[:, 2]
        proc_params = geometry[:, 3:]
        return np.column_stack(
            [inputs, nfin_log, L_col, temperature, proc_params])

    temperature = geometry[:, 1]
    if geometry.shape[1] >= 9:
        proc_params = geometry[:, 2:]
        return np.column_stack(
            [inputs, nfin_log, temperature, proc_params])
    elif geometry.shape[1] >= 3:
        phig = geometry[:, 2]
        return np.column_stack([inputs, nfin_log, temperature, phig])
    return np.column_stack([inputs, nfin_log, temperature])


# ── BSIMARNormalizer ─────────────────────────────────────────────────────────

@dataclass
class BSIMARNormStats:
    """Normalization statistics for BSIMARNormalizer. Carries explicit mode.

    Persisted as ``<save_prefix>_norm.npz``. Always contains:
    ``mode``, ``output_mean``, ``output_std``, ``input_mean``,
    ``input_std``, and the training-set ``input_min`` / ``input_max``
    (recorded as metadata so downstream simulators can clamp inference
    inputs to the training domain; the normalisation math itself uses
    the mean/std fields). In ``asinh`` mode ``asinh_scale`` is also
    present.
    """
    mode: str  # "zscore" or "asinh"
    output_mean: np.ndarray
    output_std: np.ndarray
    input_mean: np.ndarray
    input_std: np.ndarray
    # Training-domain min/max (metadata for input clamping only)
    input_min: np.ndarray
    input_max: np.ndarray
    # asinh-mode per-target geometric-mean scale
    asinh_scale: Optional[np.ndarray] = None

    def save(self, path: str) -> None:
        data = {
            "mode": np.array(self.mode),
            "output_mean": self.output_mean,
            "output_std": self.output_std,
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "input_min": self.input_min,
            "input_max": self.input_max,
        }
        if self.asinh_scale is not None:
            data["asinh_scale"] = self.asinh_scale
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> "BSIMARNormStats":
        d = np.load(path, allow_pickle=True)
        mode = str(d["mode"])
        return cls(
            mode=mode,
            output_mean=d["output_mean"],
            output_std=d["output_std"],
            input_mean=d["input_mean"],
            input_std=d["input_std"],
            input_min=d["input_min"],
            input_max=d["input_max"],
            asinh_scale=d["asinh_scale"] if "asinh_scale" in d.files else None,
        )


class BSIMARNormalizer:
    """Unified normalizer with two modes.

    ``mode='zscore'``: inputs → z-score, outputs → z-score.
    ``mode='asinh'`` : inputs → z-score, outputs → arcsinh(y/s_k) + z-score,
    where per-target ``s_k`` is the geometric mean of ``|y|`` over the
    train split, masked at ``OUTPUT_LOG_FLOORS`` and clamped to the
    floor.
    """

    def __init__(self, mode: str = "asinh",
                 stats: Optional[BSIMARNormStats] = None) -> None:
        assert mode in ("zscore", "asinh"), f"Unknown mode: {mode}"
        self.mode = mode
        self.stats = stats

    def fit(self, inputs: np.ndarray, geometry: np.ndarray,
            outputs: np.ndarray) -> "BSIMARNormalizer":
        combined = _build_combined_input(inputs, geometry)

        # Inputs are voltages (~0.5V scale) and process params
        # (~1e-3 to ~1 scale). 1e-12 safely catches truly constant
        # columns without ever clipping a real std.
        input_mean = combined.mean(axis=0)
        input_std = combined.std(axis=0)
        input_std[input_std < 1e-12] = 1.0

        # Record the training-domain min/max as metadata (used by
        # downstream simulators to clamp inference-time inputs to the
        # training domain; the normalisation math itself uses
        # mean/std).
        input_min = combined.min(axis=0)
        input_max = combined.max(axis=0)

        if self.mode == "zscore":
            output_mean = outputs.mean(axis=0)
            output_std = outputs.std(axis=0)
            # CRITICAL: Outputs span 14+ decades. Charges (~1e-19 C) and
            # capacitances (~1e-20 F) have legitimate std values well
            # below 1e-12. Clipping them to 1.0 leaves the "normalized"
            # targets in physical units, breaks training, and produces
            # astronomical NRMSE / negative R² on denormalization. Use a
            # much smaller absolute floor — only truly degenerate
            # (numerically zero) columns should be replaced.
            output_std[output_std < 1e-30] = 1.0

            self.stats = BSIMARNormStats(
                mode="zscore",
                output_mean=output_mean, output_std=output_std,
                input_mean=input_mean, input_std=input_std,
                input_min=input_min, input_max=input_max,
            )
        else:  # asinh
            # Per-target geometric-mean scale s_k from |y|, masked by
            # OUTPUT_LOG_FLOORS and clamped at the floor.
            floors = np.array(
                [OUTPUT_LOG_FLOORS[c] for c in OUTPUT_COLUMN_ORDER],
                dtype=np.float64)
            abs_y = np.abs(outputs).astype(np.float64)
            mask = abs_y > floors[None, :]
            log_y = np.log(np.maximum(abs_y, floors[None, :]))
            denom = np.maximum(mask.sum(axis=0), 1)
            s_log = np.where(
                mask.sum(axis=0) > 0,
                np.sum(np.where(mask, log_y, 0.0), axis=0) / denom,
                np.log(floors),
            )
            asinh_scale = np.maximum(np.exp(s_log), floors)

            # S2 — apply per-target scale lower bounds so gmb/qb cannot
            # be over-compressed on a TSMC-dominated train pool (see
            # ``OUTPUT_ASINH_SCALE_MIN`` and the 2026-04-10 LOO plan).
            for col_name, lower in OUTPUT_ASINH_SCALE_MIN.items():
                col_idx = OUTPUT_COLUMN_ORDER.index(col_name)
                if asinh_scale[col_idx] < lower:
                    asinh_scale[col_idx] = lower

            outputs_t = np.arcsinh(
                outputs.astype(np.float64) / asinh_scale[None, :])
            output_mean = outputs_t.mean(axis=0)
            output_std = outputs_t.std(axis=0)
            # asinh-space stds are O(1) for non-constant columns; a
            # generous floor is safe here.
            output_std[output_std < 1e-12] = 1.0

            self.stats = BSIMARNormStats(
                mode="asinh",
                output_mean=output_mean, output_std=output_std,
                input_mean=input_mean, input_std=input_std,
                input_min=input_min, input_max=input_max,
                asinh_scale=asinh_scale,
            )
        return self

    def normalize_inputs(self, inputs: np.ndarray,
                         geometry: np.ndarray) -> np.ndarray:
        assert self.stats is not None, "Must call fit() first"
        combined = _build_combined_input(inputs, geometry)
        return (combined - self.stats.input_mean) / self.stats.input_std

    def normalize_outputs(self, outputs: np.ndarray) -> np.ndarray:
        assert self.stats is not None, "Must call fit() first"
        if self.stats.mode == "zscore":
            return (outputs - self.stats.output_mean) / self.stats.output_std
        # asinh
        outputs_t = np.arcsinh(
            outputs.astype(np.float64)
            / self.stats.asinh_scale[None, :])
        return (outputs_t - self.stats.output_mean) / self.stats.output_std

    def denormalize_outputs(self, outputs_norm: np.ndarray) -> np.ndarray:
        assert self.stats is not None, "Must call fit() first"
        if self.stats.mode == "zscore":
            return (
                outputs_norm * self.stats.output_std + self.stats.output_mean)
        # asinh
        outputs_t = (
            outputs_norm.astype(np.float64) * self.stats.output_std
            + self.stats.output_mean)
        return self.stats.asinh_scale[None, :] * np.sinh(outputs_t)
