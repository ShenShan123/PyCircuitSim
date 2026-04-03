"""Normalization utilities for NN-based compact model training.

Handles the extreme dynamic range of MOSFET data:
- Currents: 1e-18 A (cutoff) to 1e-4 A (on) — 14 decades
- Conductances: 1e-8 to 1e-3 S — 5 decades
- Charges: 1e-18 to 1e-15 C — 3 decades
- Capacitances: 1e-20 to 1e-15 F — 5 decades

Strategy: signed_log transform + z-score normalization.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


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

# BSIM-AR autoregressive order: easy targets first, hardest (id) last.
# Charges (well-behaved, smooth) -> capacitances -> conductances -> current.
BSIMAR_COLUMN_ORDER = [
    "qg", "qd", "qs", "qb",
    "cgg", "cgd", "cgs", "cdg", "cdd",
    "gm", "gds", "gmb",
    "id",
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


def signed_log(x: np.ndarray, floor: float = 1e-18) -> np.ndarray:
    """Map signed values to floor-relative log scale, preserving sign.

    For |x| > floor: sign(x) * log10(|x| / floor)
    For |x| <= floor: 0.0 (treat as zero)

    This maps floor → 0, and larger magnitudes to positive values.
    For floor=1e-18 and |x|=1e-4: result = sign(x) * 14.0
    Sign is always preserved (positive x → positive result).

    Args:
        x: Input array (any shape).
        floor: Values with |x| <= floor are mapped to 0.

    Returns:
        Transformed array, same shape as x.
    """
    abs_x = np.abs(x)
    result = np.zeros_like(x, dtype=np.float64)
    mask = abs_x > floor
    result[mask] = np.sign(x[mask]) * np.log10(abs_x[mask] / floor)
    return result


def inv_signed_log(y: np.ndarray, floor: float = 1e-18) -> np.ndarray:
    """Inverse of signed_log: recover original scale from floor-relative log.

    For |y| > 0: sign(y) * floor * 10^|y|
    For y == 0: 0.0

    Args:
        y: Log-transformed array.
        floor: The floor used in the forward transform.

    Returns:
        Original-scale array.
    """
    result = np.zeros_like(y, dtype=np.float64)
    mask = np.abs(y) > 0
    result[mask] = np.sign(y[mask]) * floor * (10.0 ** np.abs(y[mask]))
    return result


@dataclass
class NormStats:
    """Normalization statistics saved alongside the trained model.

    Stores everything needed to normalize raw data or denormalize NN outputs
    back to physical units.
    """
    # Input normalization (min-max to [0, 1])
    # shape (6,) legacy, (7,) Phase 13 PHIG, (13,) universal 7 params, or (18,) universal 12 params
    input_min: np.ndarray
    input_max: np.ndarray

    # Output normalization (signed_log → z-score)
    output_log_floors: np.ndarray  # shape (13,) — per-column floor for signed_log
    output_mean: np.ndarray        # shape (13,) — mean in log-space
    output_std: np.ndarray         # shape (13,) — std in log-space (clamped > 0)

    def save(self, path: str) -> None:
        """Save normalization stats to .npz file."""
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
        """Load normalization stats from .npz file."""
        data = np.load(path)
        return cls(
            input_min=data["input_min"],
            input_max=data["input_max"],
            output_log_floors=data["output_log_floors"],
            output_mean=data["output_mean"],
            output_std=data["output_std"],
        )


class Normalizer:
    """Handles normalization and denormalization of MOSFET training data.

    Input normalization:
        - Voltages (Vd, Vg, Vs, Vb): min-max → [0, 1]
        - NFIN: log2(NFIN) then min-max → [0, 1]
        - Temperature: min-max → [0, 1]

    Output normalization:
        - All 13 outputs: signed_log(x, floor) → z-score
    """

    def __init__(self, stats: Optional[NormStats] = None):
        self.stats = stats

    def fit(
        self,
        inputs: np.ndarray,
        geometry: np.ndarray,
        outputs: np.ndarray,
    ) -> "Normalizer":
        """Compute normalization statistics from training data.

        Args:
            inputs: (N, 4) — [Vd, Vg, Vs, Vb]
            geometry: (N, 2), (N, 3), (N, 9), or (N, 14) — see _build_combined_input()
            outputs: (N, 13) — 13 output columns

        Returns:
            self (for chaining).
        """
        # Build combined input: [Vd, Vg, Vs, Vb, log2(NFIN), T]
        combined_input = self._build_combined_input(inputs, geometry)

        # Input: min-max stats
        input_min = combined_input.min(axis=0)
        input_max = combined_input.max(axis=0)

        # Prevent division by zero for constant features (e.g., Vs=0, Vb=0)
        range_vals = input_max - input_min
        range_vals[range_vals < 1e-10] = 1.0  # Constant feature → map to 0

        # Output: signed_log transform, then compute mean/std
        output_log_floors = np.array(
            [OUTPUT_LOG_FLOORS[col] for col in OUTPUT_COLUMN_ORDER],
            dtype=np.float64,
        )

        # Apply signed_log per column
        outputs_log = np.zeros_like(outputs)
        for i in range(outputs.shape[1]):
            outputs_log[:, i] = signed_log(outputs[:, i], floor=output_log_floors[i])

        output_mean = outputs_log.mean(axis=0)
        output_std = outputs_log.std(axis=0)
        # Prevent division by zero
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
        """Normalize inputs to [0, 1] range.

        Args:
            inputs: (N, 4) — [Vd, Vg, Vs, Vb]
            geometry: (N, 2), (N, 3), (N, 9), or (N, 14) — see _build_combined_input()

        Returns:
            (N, 6), (N, 7), (N, 13), or (N, 18) normalized input array.
        """
        assert self.stats is not None, "Must call fit() first"
        combined = self._build_combined_input(inputs, geometry)
        range_vals = self.stats.input_max - self.stats.input_min
        range_vals[range_vals < 1e-10] = 1.0
        return (combined - self.stats.input_min) / range_vals

    def normalize_outputs(self, outputs: np.ndarray) -> np.ndarray:
        """Normalize outputs: signed_log → z-score.

        Args:
            outputs: (N, 13) raw physical outputs.

        Returns:
            (N, 13) normalized outputs.
        """
        assert self.stats is not None, "Must call fit() first"
        outputs_log = np.zeros_like(outputs)
        for i in range(outputs.shape[1]):
            outputs_log[:, i] = signed_log(
                outputs[:, i], floor=self.stats.output_log_floors[i]
            )
        return (outputs_log - self.stats.output_mean) / self.stats.output_std

    def denormalize_outputs(self, outputs_norm: np.ndarray) -> np.ndarray:
        """Denormalize outputs: z-score → inv_signed_log → physical units.

        Args:
            outputs_norm: (N, 13) normalized outputs.

        Returns:
            (N, 13) physical-unit outputs.
        """
        assert self.stats is not None, "Must call fit() first"
        # Undo z-score
        outputs_log = outputs_norm * self.stats.output_std + self.stats.output_mean

        # Undo signed_log
        outputs_phys = np.zeros_like(outputs_log)
        for i in range(outputs_log.shape[1]):
            outputs_phys[:, i] = inv_signed_log(
                outputs_log[:, i], floor=self.stats.output_log_floors[i]
            )
        return outputs_phys

    def _build_combined_input(
        self,
        inputs: np.ndarray,
        geometry: np.ndarray,
    ) -> np.ndarray:
        """Combine voltage inputs with geometry into feature vector.

        Args:
            inputs: (N, 4) — [Vd, Vg, Vs, Vb]
            geometry: (N, 2), (N, 3), (N, 9), or (N, 14):
                - (N, 2): [NFIN, T] — legacy
                - (N, 3): [NFIN, T, PHIG] — Phase 13
                - (N, 9): [NFIN, T, PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW] — universal (7 process params)
                - (N, 14): [NFIN, T, <12 process params>] — universal (12 process params)

        Returns:
            (N, 6), (N, 7), (N, 13), or (N, 18) normalized feature vector.
        """
        # Transform NFIN to log2 scale (captures roughly linear scaling)
        nfin_log = np.log2(np.clip(geometry[:, 0], 1.0, None))
        temperature = geometry[:, 1]
        if geometry.shape[1] >= 9:
            # Universal: all process params follow [NFIN, T]
            proc_params = geometry[:, 2:]  # (N, 12) — PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW, CFS, TOXP, CGSL, UA, EU
            return np.column_stack([inputs, nfin_log, temperature, proc_params])
        elif geometry.shape[1] >= 3:
            # Phase 13: PHIG only
            phig = geometry[:, 2]
            return np.column_stack([inputs, nfin_log, temperature, phig])
        return np.column_stack([inputs, nfin_log, temperature])


def test_round_trip() -> None:
    """Test that normalize → denormalize is approximately identity."""
    rng = np.random.default_rng(42)

    # Simulate realistic MOSFET data
    N = 1000
    inputs = rng.uniform(-0.1, 0.8, size=(N, 4))
    geometry = np.column_stack([
        rng.choice([1, 2, 5, 10, 15, 20], size=N).astype(float),
        np.full(N, 300.15),
    ])

    # Simulate outputs with realistic ranges
    outputs = np.zeros((N, 13))
    outputs[:, 0] = rng.uniform(-1e-4, 1e-4, N)      # id
    outputs[:, 1] = rng.uniform(0, 1e-3, N)           # gm
    outputs[:, 2] = rng.uniform(1e-8, 1e-4, N)        # gds
    outputs[:, 3] = rng.uniform(0, 1e-4, N)           # gmb
    outputs[:, 4] = rng.uniform(-1e-15, 1e-15, N)     # qg
    outputs[:, 5] = rng.uniform(-1e-15, 1e-15, N)     # qd
    outputs[:, 6] = rng.uniform(-1e-15, 1e-15, N)     # qs
    outputs[:, 7] = rng.uniform(-1e-16, 1e-16, N)     # qb
    outputs[:, 8] = rng.uniform(1e-17, 1e-15, N)      # cgg
    outputs[:, 9] = rng.uniform(-1e-15, 0, N)         # cgd
    outputs[:, 10] = rng.uniform(-1e-15, 0, N)        # cgs
    outputs[:, 11] = rng.uniform(-1e-15, 0, N)        # cdg
    outputs[:, 12] = rng.uniform(1e-18, 1e-15, N)     # cdd

    # Fit normalizer
    normalizer = Normalizer()
    normalizer.fit(inputs, geometry, outputs)

    # Round-trip: normalize → denormalize
    norm_out = normalizer.normalize_outputs(outputs)
    recon_out = normalizer.denormalize_outputs(norm_out)

    # Check relative error (only for non-tiny values)
    for i, name in enumerate(OUTPUT_COLUMN_ORDER):
        col_orig = outputs[:, i]
        col_recon = recon_out[:, i]
        floor = normalizer.stats.output_log_floors[i]

        # Only check values above floor
        mask = np.abs(col_orig) > floor * 10
        if mask.sum() == 0:
            continue

        rel_err = np.abs(col_orig[mask] - col_recon[mask]) / (np.abs(col_orig[mask]) + 1e-30)
        max_rel_err = rel_err.max()

        status = "PASS" if max_rel_err < 0.01 else "FAIL"
        print(f"  {name:>6s}: max_rel_err = {max_rel_err:.2e}  [{status}]")

    # Check normalized output stats
    print(f"\n  Normalized output mean: {norm_out.mean(axis=0).round(2)}")
    print(f"  Normalized output std:  {norm_out.std(axis=0).round(2)}")
    print(f"  (Should be ~0 mean, ~1 std)")


if __name__ == "__main__":
    print("Running round-trip normalization test...")
    test_round_trip()
