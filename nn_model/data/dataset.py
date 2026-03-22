"""PyTorch Dataset for MOSFET training data."""

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from nn_model.data.normalize import Normalizer, NormStats
from nn_model.config import OUTPUT_COLUMNS


class MOSFETDataset(Dataset):
    """PyTorch Dataset wrapping normalized MOSFET I-V/Q-V data.

    Each sample provides:
        - inputs: (6,) tensor [Vd, Vg, Vs, Vb, log2(NFIN), T] normalized to [0,1]
        - outputs: (13,) tensor [id, gm, gds, gmb, qg, qd, qs, qb, cgg, cgd, cgs, cdg, cdd]
                   in signed_log + z-score normalized space
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


def load_and_split(
    data_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    normalizer: Optional[Normalizer] = None,
) -> Tuple[MOSFETDataset, MOSFETDataset, MOSFETDataset, Normalizer]:
    """Load .npz dataset, normalize, and split into train/val/test.

    Args:
        data_path: Path to .npz file from generate.py.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed for reproducible splits.
        normalizer: Pre-fitted normalizer (if None, fits from training data).

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, normalizer).
    """
    data = np.load(data_path, allow_pickle=True)
    inputs = data["inputs"]      # (N, 4)
    geometry = data["geometry"]  # (N, 2)
    outputs = data["outputs"]    # (N, 13)

    N = len(inputs)
    test_ratio = 1.0 - train_ratio - val_ratio

    # Shuffle with fixed seed
    rng = np.random.default_rng(seed)
    indices = rng.permutation(N)

    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Fit normalizer on training data only
    if normalizer is None:
        normalizer = Normalizer()
        normalizer.fit(inputs[train_idx], geometry[train_idx], outputs[train_idx])

    # Normalize all splits
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
