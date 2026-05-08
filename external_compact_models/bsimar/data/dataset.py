"""Dataset wrapper and loader for BSIMAR / DirectNet training.

One loader for both architectures. The caller picks ``norm_mode``:
``"zscore"`` for DirectNet, ``"asinh"`` for the Transformer.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from bsimar.data.normalize import _NormalizerBase, normalizer_for


class MOSFETDataset(Dataset):
    """(inputs, outputs, tech_codes) tuples in normalised space."""

    def __init__(
        self,
        inputs_norm: np.ndarray,
        outputs_norm: np.ndarray,
        tech_codes: np.ndarray,
    ) -> None:
        self.inputs = torch.tensor(inputs_norm, dtype=torch.float32)
        self.outputs = torch.tensor(outputs_norm, dtype=torch.float32)
        self.tech_codes = torch.tensor(tech_codes, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.outputs[idx], self.tech_codes[idx]


# Drop rows below the modelcard noise floor for Id. Charges and caps are
# absorbed by the asinh per-target scale, so the only useful filter is on
# Id (v5 plan §4-B4).
DEFAULT_FILTER_THRESHOLDS: Dict[str, float] = {"id": 1e-15}


def filter_small_targets(
    outputs: np.ndarray,
    column_names: List[str],
    thresholds: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    thresholds = thresholds or DEFAULT_FILTER_THRESHOLDS
    mask = np.ones(len(outputs), dtype=bool)
    for i, name in enumerate(column_names):
        if name in thresholds:
            mask &= np.abs(outputs[:, i]) > thresholds[name]
    return mask


def load_and_split_bsimar(
    data_path: str,
    column_names: List[str],
    device_type: str,
    norm_mode: str = "asinh",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    apply_filter: bool = True,
    filter_thresholds: Optional[Dict[str, float]] = None,
    exclude_techs: Optional[Set[str]] = None,
    max_rows: Optional[int] = None,
) -> Tuple[MOSFETDataset, MOSFETDataset, MOSFETDataset, _NormalizerBase]:
    """Load .npz, label, optionally filter / exclude techs / cap, split, normalise."""
    from bsimar.eval.loo_labels import get_or_build_tech_variant_labels

    data = np.load(data_path, allow_pickle=True)
    inputs = data["inputs"]
    geometry = data["geometry"]
    outputs = data["outputs"]
    tech_codes = get_or_build_tech_variant_labels(
        data_path, device_type, verbose=True)

    n0 = len(outputs)
    if apply_filter:
        keep = filter_small_targets(outputs, column_names, filter_thresholds)
        inputs, geometry, outputs = inputs[keep], geometry[keep], outputs[keep]
        tech_codes = tech_codes[keep]
        print(f"  Filter Id>1e-15: {n0} -> {len(outputs)}")

    if exclude_techs:
        from bsimar.config import TECH_VARIANT_CODES
        excl = {
            code for (tech, _), code in TECH_VARIANT_CODES.items()
            if tech in exclude_techs
        }
        keep = np.array(
            [int(c) not in excl for c in tech_codes], dtype=bool)
        inputs, geometry, outputs = inputs[keep], geometry[keep], outputs[keep]
        tech_codes = tech_codes[keep]
        print(f"  Excluded {exclude_techs}: kept {keep.sum()} samples")

    if max_rows is not None and len(outputs) > max_rows:
        rng_cap = np.random.default_rng(seed)
        idx = rng_cap.choice(len(outputs), size=max_rows, replace=False)
        inputs, geometry, outputs = inputs[idx], geometry[idx], outputs[idx]
        tech_codes = tech_codes[idx]
        print(f"  Capped to {max_rows} rows")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(outputs))
    n_train = int(len(perm) * train_ratio)
    n_val = int(len(perm) * val_ratio)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    normalizer = normalizer_for(norm_mode)
    normalizer.fit(
        inputs[train_idx], geometry[train_idx], outputs[train_idx])

    def _make(idxs: np.ndarray) -> MOSFETDataset:
        x = normalizer.normalize_inputs(inputs[idxs], geometry[idxs])
        y = normalizer.normalize_outputs(outputs[idxs])
        return MOSFETDataset(x, y, tech_codes[idxs])

    train_ds = _make(train_idx)
    val_ds = _make(val_idx)
    test_ds = _make(test_idx)

    print(f"  Split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    return train_ds, val_ds, test_ds, normalizer
