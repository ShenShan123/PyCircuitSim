"""Deterministic stratified sampling and geometry-grouped dataset splits."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _groups(strata: np.ndarray) -> List[np.ndarray]:
    """Return row indices grouped by exact values in each stratum column."""
    buckets: Dict[Tuple[object, ...], List[int]] = defaultdict(list)
    values = np.asarray(strata)
    if values.ndim != 2:
        raise ValueError("strata must be a two-dimensional array")
    for index, row in enumerate(values):
        buckets[tuple(row.tolist())].append(index)
    return [np.asarray(indices, dtype=np.int64)
            for indices in buckets.values()]


def stratified_sample_indices(
    strata: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Sample proportionally while retaining every declared stratum."""
    groups = _groups(strata)
    n_total = sum(len(group) for group in groups)
    if n_samples >= n_total:
        return np.arange(n_total, dtype=np.int64)
    if n_samples < len(groups):
        raise ValueError(
            f"sample cap {n_samples} is smaller than {len(groups)} strata"
        )

    rng = np.random.default_rng(seed)
    shuffled = [rng.permutation(group) for group in groups]
    counts = np.asarray([len(group) for group in groups], dtype=np.int64)
    allocation = np.ones(len(groups), dtype=np.int64)
    remaining = n_samples - len(groups)
    capacity = counts - allocation
    while remaining:
        available = np.flatnonzero(capacity > 0)
        weights = capacity[available].astype(np.float64)
        raw = remaining * weights / weights.sum()
        add = np.minimum(np.floor(raw).astype(np.int64), capacity[available])
        if not np.any(add):
            order = available[np.argsort(-(raw - np.floor(raw)))]
            add = np.zeros_like(available)
            for index in range(min(remaining, len(order))):
                add[np.flatnonzero(available == order[index])[0]] = 1
        allocation[available] += add
        capacity[available] -= add
        remaining -= int(add.sum())

    selected = np.concatenate([
        group[:count] for group, count in zip(shuffled, allocation)
    ])
    return rng.permutation(selected)


def grouped_split_indices(
    strata: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split whole strata so a geometry/variant combination cannot leak."""
    if not 0.0 < train_ratio < 1.0 or not 0.0 <= val_ratio < 1.0:
        raise ValueError("invalid train/validation ratios")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be below one")
    groups = _groups(strata)
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(groups)))
    order.sort(key=lambda index: len(groups[int(index)]), reverse=True)
    targets = np.asarray([
        train_ratio, val_ratio, 1.0 - train_ratio - val_ratio,
    ]) * sum(len(group) for group in groups)
    active_splits = np.flatnonzero(targets > 0.0)
    if len(groups) < len(active_splits):
        raise ValueError(
            "combo split needs at least one distinct stratum per non-empty split"
        )
    assigned: List[List[np.ndarray]] = [[], [], []]
    totals = np.zeros(3, dtype=np.float64)

    # Largest groups are assigned first to the split with the greatest
    # proportional deficit. Whole groups are never divided.
    for group_index in order:
        group = groups[int(group_index)]
        deficits = np.full(3, -np.inf, dtype=np.float64)
        deficits[active_splits] = (
            (targets[active_splits] - totals[active_splits])
            / targets[active_splits]
        )
        split = int(np.argmax(deficits))
        assigned[split].append(group)
        totals[split] += len(group)

    def _flatten(parts: Sequence[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.empty(0, dtype=np.int64)
        values = np.concatenate(parts).astype(np.int64, copy=False)
        return rng.permutation(values)

    return tuple(_flatten(parts) for parts in assigned)  # type: ignore[return-value]
