"""PyTorch Dataset wrapper and loader for BSIMAR + DirectNet training.

Both models share the same loader (``load_and_split_bsimar``) and the
same normalizer (``BSIMARNormalizer``). BSIMAR trains under ``'asinh'``
mode; DirectNet trains under ``'zscore'`` mode.

Input: 7-dim continuous [V(4), NFIN_log, L, T] + per-sample discrete
tech-variant code (integer). Process parameters are not included as
continuous features — tech identity is carried by the discrete code.
"""

from typing import Tuple, Optional, Dict, List, Set

import numpy as np
import torch
from torch.utils.data import Dataset

from bsimar.data.normalize import BSIMARNormalizer


# ── Dataset ──────────────────────────────────────────────────────────────────

class MOSFETDataset(Dataset):
    """PyTorch Dataset wrapping normalized MOSFET data + tech-variant codes.

    Each sample provides:
        - inputs: (7,) tensor [V(4), NFIN_log, L, T], normalized
        - outputs: (13,) tensor in normalized space
        - tech_code: scalar int64 tensor
        - sample_class: scalar int8 tensor (only when ``sample_class``
          is supplied at construction; see B1 sprint dataset schema).

    When ``sample_class`` is None (legacy datasets), ``__getitem__``
    returns the original 3-tuple ``(inputs, outputs, tech_code)``.
    When supplied, ``__getitem__`` returns the 4-tuple
    ``(inputs, outputs, tech_code, sample_class)``. Trainers that opt
    into the slope-loss path must request the 4-tuple form.
    """

    def __init__(
        self,
        inputs_norm: np.ndarray,
        outputs_norm: np.ndarray,
        tech_codes: np.ndarray,
        sample_class: Optional[np.ndarray] = None,
    ):
        self.inputs = torch.tensor(inputs_norm, dtype=torch.float32)
        self.outputs = torch.tensor(outputs_norm, dtype=torch.float32)
        self.tech_codes = torch.tensor(tech_codes, dtype=torch.long)
        if sample_class is not None:
            self.sample_class = torch.tensor(sample_class, dtype=torch.int8)
        else:
            self.sample_class = None

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        if self.sample_class is None:
            return self.inputs[idx], self.outputs[idx], self.tech_codes[idx]
        return (self.inputs[idx], self.outputs[idx],
                self.tech_codes[idx], self.sample_class[idx])


# ── Loader ───────────────────────────────────────────────────────────────────

# Per-output filtering thresholds (physical units).
# Samples where ANY target falls below its group threshold are removed.
DEFAULT_FILTER_THRESHOLDS: Dict[str, float] = {
    "id": 1e-15, "gm": 1e-12, "gds": 1e-12, "gmb": 1e-12,
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
    device_type: str,
    norm_mode: str = "asinh",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    apply_filter: bool = True,
    filter_thresholds: Optional[Dict[str, float]] = None,
    exclude_techs: Optional[Set[str]] = None,
) -> Tuple["MOSFETDataset", "MOSFETDataset", "MOSFETDataset", BSIMARNormalizer]:
    """Load .npz, compute tech-variant codes, split, normalize.

    Input features are 7-dim [V(4), NFIN_log, L, T] (no process params).
    Returns ``MOSFETDataset`` with per-sample tech codes.

    When ``exclude_techs`` is given (e.g., {"asap7"}), those techs'
    samples are dropped entirely before the train/val/test split.

    If the npz contains a ``sample_class`` column (B1-regenerated
    datasets onward, see ``meta_sample_class_names``), it is carried
    through filtering / exclusion / splitting and attached to each
    ``MOSFETDataset``. Old datasets without this column work as before;
    ``Dataset.sample_class`` will be ``None`` and the 3-tuple
    ``__getitem__`` form is used.

    Args:
        data_path: Path to universal .npz dataset.
        column_names: Output column names for filtering.
        device_type: "nmos" or "pmos" (for tech labeling).
        norm_mode: ``'asinh'`` (BSIMAR default) or ``'zscore'`` (DirectNet).
        exclude_techs: Tech names to exclude entirely from the dataset.
    """
    from bsimar.eval.loo_labels import get_or_build_tech_variant_labels

    assert norm_mode in ("asinh", "zscore"), \
        f"Unknown norm_mode: {norm_mode!r}"

    data = np.load(data_path, allow_pickle=True)
    inputs = data["inputs"]
    geometry = data["geometry"]
    outputs = data["outputs"]
    sample_class = data["sample_class"] if "sample_class" in data.files else None

    n_before = len(outputs)

    if apply_filter:
        keep = filter_small_targets(outputs, column_names, filter_thresholds)
        inputs = inputs[keep]
        geometry = geometry[keep]
        outputs = outputs[keep]
        if sample_class is not None:
            sample_class = sample_class[keep]
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
        if sample_class is not None:
            sample_class = sample_class[keep_mask]
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
    normalizer = BSIMARNormalizer(mode=norm_mode)
    normalizer.fit(inputs[train_idx], geometry[train_idx], outputs[train_idx])

    def _make_ds(idxs: np.ndarray) -> MOSFETDataset:
        x = normalizer.normalize_inputs(inputs[idxs], geometry[idxs])
        y = normalizer.normalize_outputs(outputs[idxs])
        sc = sample_class[idxs] if sample_class is not None else None
        return MOSFETDataset(x, y, tech_codes[idxs], sample_class=sc)

    train_ds = _make_ds(train_idx)
    val_ds = _make_ds(val_idx)
    test_ds = _make_ds(test_idx)

    print(f"  Dataset split: train={len(train_ds)}, "
          f"val={len(val_ds)}, test={len(test_ds)}")

    return train_ds, val_ds, test_ds, normalizer
