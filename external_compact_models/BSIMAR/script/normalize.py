"""BSIMAR normalizer supporting both z-score and signed-log modes.

Mode 'zscore' (paper): z-score for inputs and outputs. Linear denormalization.
Mode 'signedlog' (nn_model compat): min-max inputs, signed_log+z-score outputs.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from nn_model.data.normalize import (
    signed_log, inv_signed_log, OUTPUT_LOG_FLOORS, OUTPUT_COLUMN_ORDER,
)


@dataclass
class BSIMARNormStats:
    """Normalization statistics. Fields are Optional based on mode."""
    mode: str  # "zscore" or "signedlog"

    # Common output stats
    output_mean: np.ndarray   # (D_out,)
    output_std: np.ndarray    # (D_out,)

    # zscore-mode input stats
    input_mean: Optional[np.ndarray] = None  # (D_in,)
    input_std: Optional[np.ndarray] = None   # (D_in,)

    # signedlog-mode input stats
    input_min: Optional[np.ndarray] = None   # (D_in,)
    input_max: Optional[np.ndarray] = None   # (D_in,)

    # signedlog-mode output floor values
    output_log_floors: Optional[np.ndarray] = None  # (D_out,)

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
        combined = self._build_combined_input(inputs, geometry)

        if self.mode == "zscore":
            input_mean = combined.mean(axis=0)
            input_std = combined.std(axis=0)
            input_std[input_std < 1e-12] = 1.0

            output_mean = outputs.mean(axis=0)
            output_std = outputs.std(axis=0)
            output_std[output_std < 1e-30] = 1.0

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
                outputs_log[:, i] = signed_log(outputs[:, i],
                                               floor=output_log_floors[i])
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
        combined = self._build_combined_input(inputs, geometry)
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
            return (outputs_log - self.stats.output_mean) / self.stats.output_std

    def denormalize_outputs(self, outputs_norm: np.ndarray) -> np.ndarray:
        assert self.stats is not None, "Must call fit() first"
        if self.mode == "zscore":
            return outputs_norm * self.stats.output_std + self.stats.output_mean
        else:
            outputs_log = outputs_norm * self.stats.output_std + self.stats.output_mean
            outputs_phys = np.zeros_like(outputs_log)
            for i in range(outputs_log.shape[1]):
                outputs_phys[:, i] = inv_signed_log(
                    outputs_log[:, i], floor=self.stats.output_log_floors[i])
            return outputs_phys

    @staticmethod
    def _build_combined_input(inputs: np.ndarray,
                              geometry: np.ndarray) -> np.ndarray:
        """Combine voltage inputs with geometry features.

        Handles geometry column layouts:
          (N,  2): [NFIN, T]
          (N,  3): [NFIN, T, PHIG]
          (N,  9): [NFIN, T, 7 proc params]
          (N, 14): [NFIN, T, 12 proc params]
          (N, 15): [NFIN, L, T, 12 proc params]
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
