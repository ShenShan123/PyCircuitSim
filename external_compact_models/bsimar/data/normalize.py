"""Normalization utilities for BSIMAR training data.

Handles the extreme dynamic range of MOSFET data:
- Currents: 1e-18 A (cutoff) to 1e-4 A (on) — 14 decades
- Conductances: 1e-8 to 1e-3 S — 5 decades
- Charges: 1e-18 to 1e-15 C — 3 decades
- Capacitances: 1e-20 to 1e-15 F — 5 decades

Two normalizers coexist for checkpoint compatibility:

- `Normalizer` — legacy signed-log + z-score output normalization with
  min-max input normalization. Used by the DirectNet baseline.
  Produces `NormStats` files.

- `BSIMARNormalizer` — unified class supporting two modes:
    * `'zscore'`   — z-score for inputs and outputs (paper's approach, default).
    * `'signedlog'` — min-max inputs, signed-log+z-score outputs (Normalizer compat).
  Produces `BSIMARNormStats` files (with explicit `mode` field).

The signed-log helpers (`signed_log`, `inv_signed_log`, `OUTPUT_LOG_FLOORS`,
`OUTPUT_COLUMN_ORDER`, `BSIMAR_COLUMN_ORDER`, `reorder_outputs`,
`unreorder_outputs`) are shared by both.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ── Floor values and column ordering ────────────────────────────────────────

# Floor values for each output group (below floor → treated as zero)
OUTPUT_LOG_FLOORS = {
    # Group A: Currents and conductances
    "id": 1e-18, "gm": 1e-18, "gds": 1e-18, "gmb": 1e-18,
    # Group B: Charges
    "qg": 1e-19, "qd": 1e-19, "qs": 1e-19, "qb": 1e-19,
    # Group C: Capacitances
    "cgg": 1e-20, "cgd": 1e-20, "cgs": 1e-20, "cdg": 1e-20, "cdd": 1e-20,
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


# ── Signed-log transform ─────────────────────────────────────────────────────

def signed_log(x: np.ndarray, floor: float = 1e-18) -> np.ndarray:
    """Map signed values to floor-relative log scale, preserving sign.

    For |x| > floor: sign(x) * log10(|x| / floor)
    For |x| <= floor: 0.0 (treat as zero)
    """
    abs_x = np.abs(x)
    result = np.zeros_like(x, dtype=np.float64)
    mask = abs_x > floor
    result[mask] = np.sign(x[mask]) * np.log10(abs_x[mask] / floor)
    return result


def inv_signed_log(y: np.ndarray, floor: float = 1e-18) -> np.ndarray:
    """Inverse of signed_log: recover original scale."""
    result = np.zeros_like(y, dtype=np.float64)
    mask = np.abs(y) > 0
    result[mask] = np.sign(y[mask]) * floor * (10.0 ** np.abs(y[mask]))
    return result


# ── Input geometry unpacking (shared by both normalizers) ────────────────────

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


# ── Legacy Normalizer (signed-log + z-score outputs, min-max inputs) ─────────

@dataclass
class NormStats:
    """Normalization statistics for the legacy Normalizer.

    Used by DirectNet checkpoints. Persisted as `_norm.npz`.
    """
    input_min: np.ndarray
    input_max: np.ndarray
    output_log_floors: np.ndarray  # (13,) per-column floor
    output_mean: np.ndarray        # (13,) mean in log-space
    output_std: np.ndarray         # (13,) std in log-space (clamped > 0)

    def save(self, path: str) -> None:
        np.savez(
            path,
            input_min=self.input_min,
            input_max=self.input_max,
            output_log_floors=self.output_log_floors,
            output_mean=self.output_mean,
            output_std=self.output_std,
        )

    @classmethod
    def load(cls, path: str) -> "NormStats":
        data = np.load(path)
        return cls(
            input_min=data["input_min"],
            input_max=data["input_max"],
            output_log_floors=data["output_log_floors"],
            output_mean=data["output_mean"],
            output_std=data["output_std"],
        )


class Normalizer:
    """DirectNet-style normalizer: min-max inputs + signed-log+z-score outputs."""

    def __init__(self, stats: Optional[NormStats] = None):
        self.stats = stats

    def fit(
        self,
        inputs: np.ndarray,
        geometry: np.ndarray,
        outputs: np.ndarray,
    ) -> "Normalizer":
        combined_input = _build_combined_input(inputs, geometry)

        input_min = combined_input.min(axis=0)
        input_max = combined_input.max(axis=0)
        range_vals = input_max - input_min
        range_vals[range_vals < 1e-10] = 1.0

        output_log_floors = np.array(
            [OUTPUT_LOG_FLOORS[col] for col in OUTPUT_COLUMN_ORDER],
            dtype=np.float64,
        )
        outputs_log = np.zeros_like(outputs)
        for i in range(outputs.shape[1]):
            outputs_log[:, i] = signed_log(
                outputs[:, i], floor=output_log_floors[i])

        output_mean = outputs_log.mean(axis=0)
        output_std = outputs_log.std(axis=0)
        output_std[output_std < 1e-10] = 1.0

        self.stats = NormStats(
            input_min=input_min,
            input_max=input_max,
            output_log_floors=output_log_floors,
            output_mean=output_mean,
            output_std=output_std,
        )
        return self

    def normalize_inputs(
        self,
        inputs: np.ndarray,
        geometry: np.ndarray,
    ) -> np.ndarray:
        assert self.stats is not None, "Must call fit() first"
        combined = _build_combined_input(inputs, geometry)
        range_vals = self.stats.input_max - self.stats.input_min
        range_vals[range_vals < 1e-10] = 1.0
        return (combined - self.stats.input_min) / range_vals

    def normalize_outputs(self, outputs: np.ndarray) -> np.ndarray:
        assert self.stats is not None, "Must call fit() first"
        outputs_log = np.zeros_like(outputs)
        for i in range(outputs.shape[1]):
            outputs_log[:, i] = signed_log(
                outputs[:, i], floor=self.stats.output_log_floors[i]
            )
        return (outputs_log - self.stats.output_mean) / self.stats.output_std

    def denormalize_outputs(self, outputs_norm: np.ndarray) -> np.ndarray:
        assert self.stats is not None, "Must call fit() first"
        outputs_log = (
            outputs_norm * self.stats.output_std + self.stats.output_mean)
        outputs_phys = np.zeros_like(outputs_log)
        for i in range(outputs_log.shape[1]):
            outputs_phys[:, i] = inv_signed_log(
                outputs_log[:, i], floor=self.stats.output_log_floors[i]
            )
        return outputs_phys


# ── BSIMARNormalizer (unified zscore / signedlog) ────────────────────────────

@dataclass
class BSIMARNormStats:
    """Normalization statistics for BSIMARNormalizer. Carries explicit mode."""
    mode: str  # "zscore" or "signedlog"

    # Common output stats
    output_mean: np.ndarray
    output_std: np.ndarray

    # zscore-mode input stats
    input_mean: Optional[np.ndarray] = None
    input_std: Optional[np.ndarray] = None

    # signedlog-mode input stats
    input_min: Optional[np.ndarray] = None
    input_max: Optional[np.ndarray] = None

    # signedlog-mode output floor values
    output_log_floors: Optional[np.ndarray] = None

    def save(self, path: str) -> None:
        data = {"mode": np.array(self.mode),
                "output_mean": self.output_mean,
                "output_std": self.output_std}
        if self.input_mean is not None:
            data["input_mean"] = self.input_mean
        if self.input_std is not None:
            data["input_std"] = self.input_std
        if self.input_min is not None:
            data["input_min"] = self.input_min
        if self.input_max is not None:
            data["input_max"] = self.input_max
        if self.output_log_floors is not None:
            data["output_log_floors"] = self.output_log_floors
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> "BSIMARNormStats":
        d = np.load(path, allow_pickle=True)
        mode = str(d["mode"])
        return cls(
            mode=mode,
            output_mean=d["output_mean"],
            output_std=d["output_std"],
            input_mean=d.get("input_mean"),
            input_std=d.get("input_std"),
            input_min=d.get("input_min"),
            input_max=d.get("input_max"),
            output_log_floors=d.get("output_log_floors"),
        )


class BSIMARNormalizer:
    """Unified normalizer with mode switching.

    mode='zscore':    inputs -> z-score,  outputs -> z-score
    mode='signedlog': inputs -> min-max [0,1], outputs -> signed_log + z-score
    """

    def __init__(self, mode: str = "zscore",
                 stats: Optional[BSIMARNormStats] = None) -> None:
        assert mode in ("zscore", "signedlog"), f"Unknown mode: {mode}"
        self.mode = mode
        self.stats = stats

    def fit(self, inputs: np.ndarray, geometry: np.ndarray,
            outputs: np.ndarray) -> "BSIMARNormalizer":
        combined = _build_combined_input(inputs, geometry)

        if self.mode == "zscore":
            input_mean = combined.mean(axis=0)
            input_std = combined.std(axis=0)
            input_std[input_std < 1e-12] = 1.0

            output_mean = outputs.mean(axis=0)
            output_std = outputs.std(axis=0)
            # Match Normalizer / input_std clip threshold. The previous
            # 1e-30 sentinel was a no-op for float64 std (lower bound
            # ~1e-154); a borderline-constant column with std~1e-20 would
            # blow up normalization. 1e-12 is the same threshold we use
            # for input_std a few lines above.
            output_std[output_std < 1e-12] = 1.0

            self.stats = BSIMARNormStats(
                mode="zscore",
                output_mean=output_mean, output_std=output_std,
                input_mean=input_mean, input_std=input_std,
            )
        else:  # signedlog
            input_min = combined.min(axis=0)
            input_max = combined.max(axis=0)
            input_range = input_max - input_min
            input_range[input_range < 1e-10] = 1.0

            output_log_floors = np.array(
                [OUTPUT_LOG_FLOORS[col] for col in OUTPUT_COLUMN_ORDER],
                dtype=np.float64)
            outputs_log = np.zeros_like(outputs)
            for i in range(outputs.shape[1]):
                outputs_log[:, i] = signed_log(
                    outputs[:, i], floor=output_log_floors[i])
            output_mean = outputs_log.mean(axis=0)
            output_std = outputs_log.std(axis=0)
            output_std[output_std < 1e-10] = 1.0

            self.stats = BSIMARNormStats(
                mode="signedlog",
                output_mean=output_mean, output_std=output_std,
                input_min=input_min, input_max=input_max,
                output_log_floors=output_log_floors,
            )
        return self

    def normalize_inputs(self, inputs: np.ndarray,
                         geometry: np.ndarray) -> np.ndarray:
        assert self.stats is not None, "Must call fit() first"
        combined = _build_combined_input(inputs, geometry)
        if self.mode == "zscore":
            return (combined - self.stats.input_mean) / self.stats.input_std
        else:
            input_range = self.stats.input_max - self.stats.input_min
            input_range[input_range < 1e-10] = 1.0
            return (combined - self.stats.input_min) / input_range

    def normalize_outputs(self, outputs: np.ndarray) -> np.ndarray:
        assert self.stats is not None, "Must call fit() first"
        if self.mode == "zscore":
            return (outputs - self.stats.output_mean) / self.stats.output_std
        else:
            outputs_log = np.zeros_like(outputs)
            for i in range(outputs.shape[1]):
                outputs_log[:, i] = signed_log(
                    outputs[:, i], floor=self.stats.output_log_floors[i])
            return (
                outputs_log - self.stats.output_mean) / self.stats.output_std

    def denormalize_outputs(self, outputs_norm: np.ndarray) -> np.ndarray:
        assert self.stats is not None, "Must call fit() first"
        if self.mode == "zscore":
            return (
                outputs_norm * self.stats.output_std + self.stats.output_mean)
        else:
            outputs_log = (
                outputs_norm * self.stats.output_std + self.stats.output_mean)
            outputs_phys = np.zeros_like(outputs_log)
            for i in range(outputs_log.shape[1]):
                outputs_phys[:, i] = inv_signed_log(
                    outputs_log[:, i],
                    floor=self.stats.output_log_floors[i])
            return outputs_phys
