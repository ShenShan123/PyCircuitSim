"""Normalization for BSIMAR / DirectNet training data.

Two normalizers, one tiny protocol — every chain-rule conversion lives
here, so trainers and the simulator never reach into stats fields.

- ``ZScoreNormalizer``  — inputs and outputs use z-score. DirectNet baseline.
- ``AsinhNormalizer``   — inputs z-score; outputs ``arcsinh(y/s_k) + zscore``
  with a per-target geometric-mean scale ``s_k``. BSIMAR Transformer.

Both expose the same API:

    n.fit(inputs, geometry, outputs)        # train fit
    n.normalize_inputs(inputs, geometry)    # raw -> normalised
    n.normalize_outputs(outputs)            # raw -> normalised
    n.denormalize_outputs(y_norm)           # normalised -> raw
    n.denormalize_derivative(deriv_norm,    # ∂y_norm/∂x_norm at y_phys
                             out_idx, in_idx, y_phys)
                                            #   -> ∂y_phys/∂x_phys

Persistence is handled by ``NormStats`` which carries an explicit ``mode``
field; ``NormStats.load(path)`` returns the matching normalizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ── Output column ordering and floors ──────────────────────────────────────

OUTPUT_COLUMN_ORDER = [
    "id", "gm", "gds", "gmb",
    "qg", "qd", "qs", "qb",
    "cgg", "cgd", "cgs", "cdg", "cdd",
]

# BSIM-AR autoregressive order (paper §4.2): Q-V → I-V → C-V.
BSIMAR_COLUMN_ORDER = [
    "qg", "qb", "qd", "qs",
    "id", "gm", "gds", "gmb",
    "cgg", "cgd", "cgs", "cdg", "cdd",
]

# Per-target floor magnitudes used by the asinh scale fit. Samples with
# |y_k| below the floor contribute the floor (not the value) to the
# geometric-mean scale s_k, so a noisy near-zero sample cannot collapse
# the scale.
_OUTPUT_LOG_FLOORS = {
    "id": 1e-18, "gm": 1e-18, "gds": 1e-18, "gmb": 1e-18,
    "qg": 1e-19, "qd": 1e-19, "qs": 1e-19, "qb": 1e-19,
    "cgg": 1e-20, "cgd": 1e-20, "cgs": 1e-20, "cdg": 1e-20, "cdd": 1e-20,
}

# LOO sprint S2 (2026-04-10): pin gmb / qb scales so a TSMC-only train
# pool cannot collapse them below physically meaningful magnitudes.
_OUTPUT_ASINH_SCALE_MIN = {"gmb": 1e-5, "qb": 1e-15}

_REORDER_IDX = [OUTPUT_COLUMN_ORDER.index(c) for c in BSIMAR_COLUMN_ORDER]
_UNREORDER_IDX = [BSIMAR_COLUMN_ORDER.index(c) for c in OUTPUT_COLUMN_ORDER]


def reorder_outputs(arr: np.ndarray) -> np.ndarray:
    """OUTPUT_COLUMN_ORDER → BSIMAR_COLUMN_ORDER."""
    return arr[:, _REORDER_IDX]


def unreorder_outputs(arr: np.ndarray) -> np.ndarray:
    """BSIMAR_COLUMN_ORDER → OUTPUT_COLUMN_ORDER."""
    return arr[:, _UNREORDER_IDX]


# ── Geometry helper ────────────────────────────────────────────────────────

def _build_combined_input(
    inputs: np.ndarray, geometry: np.ndarray,
) -> np.ndarray:
    """Voltages + log-NFIN + L + T → 7-column normaliser input.

    geometry is (N, 15): [NFIN, L, T, 12 unused process params].
    """
    assert geometry.shape[1] == 15, (
        f"Expected 15-col geometry, got {geometry.shape[1]}")
    nfin_log = np.log2(np.clip(geometry[:, 0], 1.0, None))
    return np.column_stack([inputs, nfin_log, geometry[:, 1], geometry[:, 2]])


# ── NormStats: persistence ─────────────────────────────────────────────────

@dataclass
class NormStats:
    """Persisted normalisation statistics.

    ``mode``           — "zscore" | "asinh"
    ``input_mean/std`` — z-score over the 7-col continuous input.
    ``input_min/max``  — training-domain bounds (metadata only;
                         used by the simulator to clamp inference inputs).
    ``output_mean/std``— in raw space (zscore) or asinh-space (asinh).
    ``asinh_scale``    — per-target s_k (asinh mode only).
    ``phys_best_metric`` — which aggregator the trainer's phys-best
                         tracker used. ``"median"`` = trustworthy;
                         ``"legacy_mean"`` = bug-prone (see plan §2B).
    """
    mode: str
    input_mean: np.ndarray
    input_std: np.ndarray
    input_min: np.ndarray
    input_max: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray
    asinh_scale: Optional[np.ndarray] = None
    phys_best_metric: str = "median"

    def save(self, path: str) -> None:
        data = {
            "mode": np.array(self.mode),
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "input_min": self.input_min,
            "input_max": self.input_max,
            "output_mean": self.output_mean,
            "output_std": self.output_std,
            "phys_best_metric": np.array(str(self.phys_best_metric)),
        }
        if self.asinh_scale is not None:
            data["asinh_scale"] = self.asinh_scale
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> "NormStats":
        d = np.load(path, allow_pickle=True)
        mode = str(d["mode"])
        return cls(
            mode=mode,
            input_mean=d["input_mean"],
            input_std=d["input_std"],
            input_min=d["input_min"],
            input_max=d["input_max"],
            output_mean=d["output_mean"],
            output_std=d["output_std"],
            asinh_scale=d["asinh_scale"] if "asinh_scale" in d.files else None,
            phys_best_metric=(
                str(d["phys_best_metric"]) if "phys_best_metric" in d.files
                else "legacy_mean"),
        )


# Back-compat alias (older simulator code imports this name).
BSIMARNormStats = NormStats


# ── Common base ────────────────────────────────────────────────────────────

class _NormalizerBase:
    """Shared input-side z-score; subclasses define the output transform."""

    mode: str = ""  # override

    def __init__(self, stats: Optional[NormStats] = None) -> None:
        self.stats = stats

    # — public API used by trainers and the simulator —

    def fit(
        self,
        inputs: np.ndarray,
        geometry: np.ndarray,
        outputs: np.ndarray,
    ) -> "_NormalizerBase":
        combined = _build_combined_input(inputs, geometry)
        in_mean = combined.mean(axis=0)
        in_std = combined.std(axis=0)
        in_std[in_std < 1e-12] = 1.0
        in_min = combined.min(axis=0)
        in_max = combined.max(axis=0)
        out_mean, out_std, asinh_scale = self._fit_outputs(outputs)
        self.stats = NormStats(
            mode=self.mode,
            input_mean=in_mean, input_std=in_std,
            input_min=in_min, input_max=in_max,
            output_mean=out_mean, output_std=out_std,
            asinh_scale=asinh_scale,
        )
        return self

    def normalize_inputs(
        self, inputs: np.ndarray, geometry: np.ndarray,
    ) -> np.ndarray:
        s = self._require_stats()
        combined = _build_combined_input(inputs, geometry)
        return (combined - s.input_mean) / s.input_std

    def normalize_outputs(self, outputs: np.ndarray) -> np.ndarray:
        s = self._require_stats()
        u = self._to_inner(outputs)
        return (u - s.output_mean) / s.output_std

    def denormalize_outputs(self, y_norm: np.ndarray) -> np.ndarray:
        s = self._require_stats()
        u = y_norm.astype(np.float64) * s.output_std + s.output_mean
        return self._from_inner(u)

    def denormalize_derivative(
        self,
        deriv_norm: float,
        out_idx: int,
        in_idx: int,
        y_phys: float,
    ) -> float:
        """∂y_norm/∂x_norm  →  ∂y_phys/∂x_phys.

        Single source of truth for the chain rule. Subclasses provide
        the output-side jacobian factor.
        """
        s = self._require_stats()
        in_std = float(s.input_std[in_idx])
        if in_std < 1e-12:
            return 0.0
        out_std = float(s.output_std[out_idx])
        out_factor = self._output_jacobian_factor(out_idx, y_phys)
        return float(deriv_norm) * out_std * out_factor / in_std

    # — subclass hooks —

    def _fit_outputs(
        self, outputs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError

    def _to_inner(self, outputs: np.ndarray) -> np.ndarray:
        """Raw outputs → inner space (asinh / identity) before z-score."""
        raise NotImplementedError

    def _from_inner(self, inner: np.ndarray) -> np.ndarray:
        """Inverse of _to_inner."""
        raise NotImplementedError

    def _output_jacobian_factor(self, out_idx: int, y_phys: float) -> float:
        """d(y_phys)/d(y_inner): identity for zscore, sqrt(s²+y²) for asinh."""
        raise NotImplementedError

    # —

    def _require_stats(self) -> NormStats:
        assert self.stats is not None, "Must call fit() or load stats first"
        return self.stats


# ── ZScore normalizer (DirectNet) ──────────────────────────────────────────

class ZScoreNormalizer(_NormalizerBase):
    mode = "zscore"

    def _fit_outputs(self, outputs):
        out_mean = outputs.mean(axis=0)
        out_std = outputs.std(axis=0)
        # Charges and caps have legitimate std as low as 1e-20; only pure
        # constants get clipped, otherwise denormalisation explodes.
        out_std[out_std < 1e-30] = 1.0
        return out_mean, out_std, None

    def _to_inner(self, outputs):
        return outputs.astype(np.float64)

    def _from_inner(self, inner):
        return inner

    def _output_jacobian_factor(self, out_idx, y_phys):
        return 1.0


# ── Asinh + z-score normalizer (Transformer) ───────────────────────────────

class AsinhNormalizer(_NormalizerBase):
    mode = "asinh"

    def _fit_outputs(self, outputs):
        floors = np.array(
            [_OUTPUT_LOG_FLOORS[c] for c in OUTPUT_COLUMN_ORDER],
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
        for col_name, lower in _OUTPUT_ASINH_SCALE_MIN.items():
            i = OUTPUT_COLUMN_ORDER.index(col_name)
            if asinh_scale[i] < lower:
                asinh_scale[i] = lower

        u = np.arcsinh(outputs.astype(np.float64) / asinh_scale[None, :])
        out_mean = u.mean(axis=0)
        out_std = u.std(axis=0)
        out_std[out_std < 1e-12] = 1.0
        return out_mean, out_std, asinh_scale

    def _to_inner(self, outputs):
        s = self._require_stats()
        return np.arcsinh(
            outputs.astype(np.float64) / s.asinh_scale[None, :])

    def _from_inner(self, inner):
        s = self._require_stats()
        return s.asinh_scale[None, :] * np.sinh(inner)

    def _output_jacobian_factor(self, out_idx, y_phys):
        s = self._require_stats()
        scale = float(s.asinh_scale[out_idx])
        return float(np.sqrt(scale * scale + y_phys * y_phys))


# ── Factory + back-compat ──────────────────────────────────────────────────

def normalizer_for(mode: str) -> _NormalizerBase:
    """Return a fresh, unfitted normalizer of the given mode."""
    if mode == "zscore":
        return ZScoreNormalizer()
    if mode == "asinh":
        return AsinhNormalizer()
    raise ValueError(f"Unknown normalizer mode: {mode!r}")


def normalizer_from_stats(stats: NormStats) -> _NormalizerBase:
    """Wrap loaded stats in the matching normalizer instance."""
    n = normalizer_for(stats.mode)
    n.stats = stats
    return n


class BSIMARNormalizer:
    """Back-compat shim: ``BSIMARNormalizer(mode="asinh")`` returns a
    fitted ``AsinhNormalizer`` / ``ZScoreNormalizer`` instance.
    """

    def __new__(cls, mode: str = "asinh", stats: Optional[NormStats] = None):
        n = normalizer_for(mode)
        if stats is not None:
            n.stats = stats
        return n
