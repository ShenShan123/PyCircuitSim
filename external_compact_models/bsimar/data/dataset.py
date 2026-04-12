"""PyTorch Dataset wrapper and loader for BSIMAR + DirectNet training.

Both models now share the same loader (``load_and_split_bsimar``) and the
same normalizer (``BSIMARNormalizer``). BSIMAR trains under ``'asinh'``
mode; DirectNet trains under ``'zscore'`` mode.

v4 adds ``MOSFETDatasetV4`` (carries tech codes alongside features) and
``load_and_split_bsimar_v4`` (7-dim input, tech-variant codes, optional
held-out tech filtering).
"""

from typing import Tuple, Optional, Dict, List, Set

import numpy as np
import torch
from torch.utils.data import Dataset

from bsimar.data.normalize import BSIMARNormalizer


# ── Dataset ──────────────────────────────────────────────────────────────────

class MOSFETDataset(Dataset):
    """PyTorch Dataset wrapping normalized MOSFET I-V/Q-V data.

    Each sample provides:
        - inputs: (D,) tensor, D in {6, 7, 13, 18, 19}, normalized
        - outputs: (13,) tensor in normalized space
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


# ── Loader ───────────────────────────────────────────────────────────────────

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
    norm_mode: str = "asinh",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    apply_filter: bool = True,
    filter_thresholds: Optional[Dict[str, float]] = None,
    exclude_techs: Optional[Set[str]] = None,
    device_type: Optional[str] = None,
) -> Tuple[MOSFETDataset, MOSFETDataset, MOSFETDataset, BSIMARNormalizer,
           Optional[np.ndarray]]:
    """Load .npz, optionally filter tiny targets, split, normalize.

    Args:
        norm_mode: ``'asinh'`` (BSIMAR default) or ``'zscore'`` (DirectNet).
        apply_filter: Drop samples where any target falls below its floor.
        exclude_techs: Tech names to exclude entirely (e.g., {"asap7"}).
            Requires ``device_type`` to be set.
        device_type: "nmos" or "pmos" (needed for tech labeling when
            ``exclude_techs`` is set).

    Returns:
        (train_ds, val_ds, test_ds, normalizer, test_tech_codes).
        ``test_tech_codes`` is non-None only when ``exclude_techs`` is set
        or ``device_type`` is provided (used for per-tech reporting).
    """
    assert norm_mode in ("asinh", "zscore"), \
        f"Unknown norm_mode: {norm_mode!r}"

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

    # Tech-variant labeling (for exclude_techs and per-tech reporting).
    tech_codes: Optional[np.ndarray] = None
    if device_type is not None:
        from bsimar.eval.loo_labels import get_or_build_tech_variant_labels
        tech_codes = get_or_build_tech_variant_labels(
            data_path, device_type, verbose=True)
        if apply_filter:
            tech_codes = tech_codes[keep]

    if exclude_techs and tech_codes is not None:
        from bsimar.config import TECH_VARIANT_CODES
        exclude_code_set = {
            code for (tech, _), code in TECH_VARIANT_CODES.items()
            if tech in exclude_techs
        }
        keep_mask = np.array(
            [int(c) not in exclude_code_set for c in tech_codes], dtype=bool)
        n_total = len(keep_mask)
        inputs = inputs[keep_mask]
        geometry = geometry[keep_mask]
        outputs = outputs[keep_mask]
        tech_codes = tech_codes[keep_mask]
        print(f"  Excluded techs {exclude_techs}: "
              f"{keep_mask.sum()}/{n_total} samples kept")

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

    test_tech_codes = tech_codes[test_idx] if tech_codes is not None else None

    print(f"  Dataset split: train={len(train_ds)}, "
          f"val={len(val_ds)}, test={len(test_ds)}")

    return train_ds, val_ds, test_ds, normalizer, test_tech_codes


# ── v4 Dataset (with tech codes) ────────────────────────────────────────────

class MOSFETDatasetV4(Dataset):
    """PyTorch Dataset wrapping normalized MOSFET data + tech-variant codes.

    Each sample provides:
        - inputs: (7,) tensor [V(4), NFIN_log, L, T], normalized
        - outputs: (13,) tensor in normalized space
        - tech_code: scalar int64 tensor
    """

    def __init__(
        self,
        inputs_norm: np.ndarray,
        outputs_norm: np.ndarray,
        tech_codes: np.ndarray,
    ):
        self.inputs = torch.tensor(inputs_norm, dtype=torch.float32)
        self.outputs = torch.tensor(outputs_norm, dtype=torch.float32)
        self.tech_codes = torch.tensor(tech_codes, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.outputs[idx], self.tech_codes[idx]


def load_and_split_bsimar_v4(
    data_path: str,
    column_names: List[str],
    device_type: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    apply_filter: bool = True,
    filter_thresholds: Optional[Dict[str, float]] = None,
    exclude_techs: Optional[Set[str]] = None,
) -> Tuple[MOSFETDatasetV4, MOSFETDatasetV4, MOSFETDatasetV4, BSIMARNormalizer]:
    """Load .npz, compute tech-variant codes, split, normalize (v4 mode).

    v4 differences from ``load_and_split_bsimar``:
    - Input features are 7-dim [V(4), NFIN_log, L, T] (no process params).
    - Returns ``MOSFETDatasetV4`` with per-sample tech codes.
    - Normalizer is ``BSIMARNormalizer(mode='asinh', include_proc_params=False)``.
    - When ``exclude_techs`` is given (e.g., {"asap7"}), those techs'
      samples are dropped entirely before the train/val/test split.

    Args:
        data_path: Path to universal .npz dataset.
        column_names: Output column names for filtering.
        device_type: "nmos" or "pmos" (for tech labeling).
        exclude_techs: Tech names to exclude entirely from the dataset.
        Other args: same as ``load_and_split_bsimar``.
    """
    from bsimar.eval.loo_labels import get_or_build_tech_variant_labels

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

    # Get per-sample tech-variant codes (integer array).
    tech_codes = get_or_build_tech_variant_labels(
        data_path, device_type, verbose=True)
    if apply_filter:
        tech_codes = tech_codes[keep]

    rng = np.random.default_rng(seed)

    if exclude_techs:
        # Drop excluded techs entirely before splitting.
        from bsimar.config import TECH_VARIANT_CODES
        exclude_code_set = {
            code for (tech, _), code in TECH_VARIANT_CODES.items()
            if tech in exclude_techs
        }
        keep_mask = np.array(
            [int(c) not in exclude_code_set for c in tech_codes], dtype=bool)
        n_total = len(keep_mask)
        inputs = inputs[keep_mask]
        geometry = geometry[keep_mask]
        outputs = outputs[keep_mask]
        tech_codes = tech_codes[keep_mask]
        print(f"  Excluded techs {exclude_techs}: "
              f"{keep_mask.sum()}/{n_total} samples kept")

    # Standard 80/10/10 random split.
    idx = rng.permutation(len(outputs))
    n_train = int(len(idx) * train_ratio)
    n_val = int(len(idx) * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    # Fit normalizer on train split (7-dim, no process params).
    normalizer = BSIMARNormalizer(mode="asinh", include_proc_params=False)
    normalizer.fit(inputs[train_idx], geometry[train_idx], outputs[train_idx])

    def _make_ds(idxs: np.ndarray) -> MOSFETDatasetV4:
        x = normalizer.normalize_inputs(inputs[idxs], geometry[idxs])
        y = normalizer.normalize_outputs(outputs[idxs])
        return MOSFETDatasetV4(x, y, tech_codes[idxs])

    train_ds = _make_ds(train_idx)
    val_ds = _make_ds(val_idx)
    test_ds = _make_ds(test_idx)

    print(f"  Dataset split: train={len(train_ds)}, "
          f"val={len(val_ds)}, test={len(test_ds)}")

    return train_ds, val_ds, test_ds, normalizer
