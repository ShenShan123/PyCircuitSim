"""PyTorch Dataset wrappers and loader functions for BSIMAR.

Provides two loader flavors that coexist for checkpoint compatibility:

- `load_and_split` — legacy DirectNet path. Uses `Normalizer` (signed-log+z-score
  outputs, min-max inputs). Returns `MOSFETDataset` objects.

- `load_and_split_bsimar` — Transformer path. Uses `BSIMARNormalizer` (zscore or
  signedlog modes), optionally filters tiny-magnitude samples before fitting.
  Returns `BSIMARDataset` objects.

Both loaders read the same .npz format produced by PyCMG's `generate_nn_data.py`.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from bsimar.data.normalize import Normalizer, BSIMARNormalizer


# ── MOSFETDataset (DirectNet, legacy) ────────────────────────────────────────

class MOSFETDataset(Dataset):
    """PyTorch Dataset wrapping normalized MOSFET I-V/Q-V data.

    Each sample provides:
        - inputs: (D,) tensor, D in {6, 7, 13, 18, 19}, normalized
        - outputs: (13,) tensor in signed_log + z-score normalized space
    """

    def __init__(
        self,
        inputs_norm: np.ndarray,
        outputs_norm: np.ndarray,
    ):
        self.inputs = torch.tensor(inputs_norm, dtype=torch.float32)
        self.outputs = torch.tensor(outputs_norm, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.outputs[idx]


# `BSIMARDataset` and `MOSFETDataset` share identical behavior — the second
# name is kept for clarity in BSIMAR training code and for backward compat.
BSIMARDataset = MOSFETDataset


# ── Loader: DirectNet path (legacy) ──────────────────────────────────────────

def load_and_split(
    data_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    normalizer: Optional[Normalizer] = None,
) -> Tuple[MOSFETDataset, MOSFETDataset, MOSFETDataset, Normalizer]:
    """Load .npz dataset, normalize with `Normalizer`, and split."""
    data = np.load(data_path, allow_pickle=True)
    inputs = data["inputs"]      # (N, 4)
    geometry = data["geometry"]  # (N, 2+)
    outputs = data["outputs"]    # (N, 13)

    N = len(inputs)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(N)

    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    if normalizer is None:
        normalizer = Normalizer()
        normalizer.fit(inputs[train_idx], geometry[train_idx], outputs[train_idx])

    train_in = normalizer.normalize_inputs(inputs[train_idx], geometry[train_idx])
    val_in = normalizer.normalize_inputs(inputs[val_idx], geometry[val_idx])
    test_in = normalizer.normalize_inputs(inputs[test_idx], geometry[test_idx])

    train_out = normalizer.normalize_outputs(outputs[train_idx])
    val_out = normalizer.normalize_outputs(outputs[val_idx])
    test_out = normalizer.normalize_outputs(outputs[test_idx])

    train_ds = MOSFETDataset(train_in, train_out)
    val_ds = MOSFETDataset(val_in, val_out)
    test_ds = MOSFETDataset(test_in, test_out)

    print(f"Dataset split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    return train_ds, val_ds, test_ds, normalizer


# ── Loader: Transformer / BSIMAR path ────────────────────────────────────────

# Per-output filtering thresholds (physical units).
# Samples where ANY target falls below its group threshold are removed.
DEFAULT_FILTER_THRESHOLDS: Dict[str, float] = {
    "id": 1e-12, "gm": 1e-12, "gds": 1e-12, "gmb": 1e-12,
    "qg": 1e-19, "qd": 1e-19, "qs": 1e-19, "qb": 1e-19,
    "cgg": 1e-19, "cgd": 1e-19, "cgs": 1e-19, "cdg": 1e-19, "cdd": 1e-19,
}


def filter_small_targets(
    outputs: np.ndarray,
    column_names: List[str],
    thresholds: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Boolean mask: True for samples where ALL targets exceed their threshold."""
    if thresholds is None:
        thresholds = DEFAULT_FILTER_THRESHOLDS
    mask = np.ones(len(outputs), dtype=bool)
    for i, name in enumerate(column_names):
        if name in thresholds:
            mask &= np.abs(outputs[:, i]) > thresholds[name]
    return mask


def load_and_split_bsimar(
    data_path: str,
    column_names: List[str],
    norm_mode: str = "zscore",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    apply_filter: bool = True,
    filter_thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[MOSFETDataset, MOSFETDataset, MOSFETDataset, BSIMARNormalizer]:
    """Load .npz, optionally filter tiny targets, split, normalize."""
    data = np.load(data_path, allow_pickle=True)
    inputs = data["inputs"]
    geometry = data["geometry"]
    outputs = data["outputs"]

    n_before = len(outputs)

    if apply_filter:
        keep = filter_small_targets(outputs, column_names, filter_thresholds)
        inputs = inputs[keep]
        geometry = geometry[keep]
        outputs = outputs[keep]
        n_after = len(outputs)
        pct = 100 * (n_before - n_after) / n_before if n_before > 0 else 0
        print(f"  Data filtering: {n_before} -> {n_after} "
              f"({n_before - n_after} removed, {pct:.1f}%)")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(outputs))
    n_train = int(len(idx) * train_ratio)
    n_val = int(len(idx) * val_ratio)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    normalizer = BSIMARNormalizer(mode=norm_mode)
    normalizer.fit(inputs[train_idx], geometry[train_idx], outputs[train_idx])

    def _make_ds(idxs: np.ndarray) -> MOSFETDataset:
        x = normalizer.normalize_inputs(inputs[idxs], geometry[idxs])
        y = normalizer.normalize_outputs(outputs[idxs])
        return MOSFETDataset(x, y)

    train_ds = _make_ds(train_idx)
    val_ds = _make_ds(val_idx)
    test_ds = _make_ds(test_idx)

    print(f"  Dataset split: train={len(train_ds)}, "
          f"val={len(val_ds)}, test={len(test_ds)}")

    return train_ds, val_ds, test_ds, normalizer
