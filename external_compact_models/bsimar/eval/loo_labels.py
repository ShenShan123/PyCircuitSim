"""Leave-one-out (by technology) helpers for BSIMAR datasets.

Builds per-sample technology / tech-variant label arrays for
``universal_{device}.npz`` by enumerating every
``(tech, variant, L, NFIN)`` bin in the PyCMG registry, reading its
process parameters from the resolved modelcard, and matching each
sample's geometry row to a known fingerprint.

Two labeling granularities:

- **Tech-level** (5 string labels): used by the v3 LOO experiments.
  ``get_or_build_tech_labels()`` returns ``(N,)`` str array.

- **Tech-variant-level** (21 integer codes): used by v4 training.
  ``get_or_build_tech_variant_labels()`` returns ``(N,)`` int array
  matching ``bsimar.config.TECH_VARIANT_CODES``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from bsimar.config import (
    DATA_DIR,
    TECH_CONFIGS,
    extract_process_params,
    TECH_VARIANT_CODES,
    tech_variant_to_code,
)
from bsimar.data.dataset import MOSFETDataset, filter_small_targets
from bsimar.data.normalize import BSIMARNormalizer

from pycmg.parser import parse_modelcard  # noqa: E402


TECH_ORDER: List[str] = ["asap7", "tsmc5", "tsmc7", "tsmc12", "tsmc16"]
DEFAULT_TEMPERATURE_K: float = 300.0
_FINGERPRINT_SIG_FIGS: int = 8


def _round_sig(x: float, sig: int = _FINGERPRINT_SIG_FIGS) -> float:
    """Round ``x`` to ``sig`` significant figures for stable hashing."""
    if x == 0.0 or not np.isfinite(x):
        return float(x)
    from math import floor, log10
    digits = sig - int(floor(log10(abs(x)))) - 1
    return float(round(x, digits))


def _fingerprint(
    NFIN: float, L: float, proc_array: List[float],
) -> Tuple[float, ...]:
    """Build a tuple fingerprint stable across small float drift."""
    vals = [NFIN, L] + list(proc_array)
    return tuple(_round_sig(float(v)) for v in vals)


# ── Tech-level labeling (v3 LOO) ────────────────────────────────────────────

def build_tech_fingerprint_map(
    device_type: str, verbose: bool = False,
) -> Dict[Tuple[float, ...], str]:
    """Return ``{fingerprint: tech_name}`` for every known bin."""
    t0 = time.time()
    out: Dict[Tuple[float, ...], str] = {}
    collisions: List[Tuple[Tuple[float, ...], str, str]] = []

    for tech_name in TECH_ORDER:
        tech = TECH_CONFIGS[tech_name]
        for variant in tech.variant_names:
            try:
                combos = tech.get_geometry_combos(device_type, variant)
            except Exception as exc:
                if verbose:
                    print(f"  skip {tech_name}:{variant} "
                          f"(get_geometry_combos: {exc})")
                continue
            model_name = tech.get_model_name(device_type, variant)
            for L, NFIN in combos:
                try:
                    modelcard_path = tech.resolve_modelcard(
                        device_type, variant, float(L), float(NFIN))
                    parsed = parse_modelcard(modelcard_path, model_name)
                    proc = extract_process_params(dict(parsed.params))
                except Exception as exc:
                    if verbose:
                        print(f"  skip {tech_name}:{variant} "
                              f"L={L*1e9:.1f}nm NFIN={NFIN:.0f}: {exc}")
                    continue
                fp = _fingerprint(float(NFIN), float(L), proc.as_array())
                if fp in out and out[fp] != tech_name:
                    collisions.append((fp, out[fp], tech_name))
                out[fp] = tech_name

    elapsed = time.time() - t0
    if verbose:
        print(f"  built {len(out)} fingerprints in {elapsed:.1f}s "
              f"across {len(TECH_ORDER)} techs")
    if collisions:
        raise RuntimeError(
            f"Fingerprint collisions across techs: {len(collisions)} "
            f"(first: {collisions[0]})")
    return out


def label_samples_by_tech(
    geometry: np.ndarray, device_type: str, verbose: bool = False,
) -> np.ndarray:
    """Return a ``(N,)`` string array with the tech name per sample."""
    assert geometry.ndim == 2 and geometry.shape[1] == 15
    fp_map = build_tech_fingerprint_map(device_type, verbose=verbose)
    n = geometry.shape[0]
    labels = np.empty(n, dtype=object)
    misses: List[int] = []
    for i in range(n):
        row = geometry[i]
        fp = _fingerprint(float(row[0]), float(row[1]),
                          [float(x) for x in row[3:15]])
        label = fp_map.get(fp)
        if label is None:
            misses.append(i)
        else:
            labels[i] = label
    if misses:
        raise AssertionError(
            f"Tech labeller missed {len(misses)} / {n} samples. "
            f"First miss idx={misses[0]}")
    return labels.astype("<U10")


def _cached_labels_path(data_path: Path) -> Path:
    return data_path.with_name(data_path.stem + "_tech_labels.npy")


def get_or_build_tech_labels(
    data_path: str, device_type: str,
    force_rebuild: bool = False, verbose: bool = True,
) -> np.ndarray:
    """Load cached tech labels or rebuild them."""
    data_path_p = Path(data_path)
    cache_path = _cached_labels_path(data_path_p)
    data = np.load(data_path_p, allow_pickle=True)
    geometry = data["geometry"]
    if cache_path.exists() and not force_rebuild:
        cached = np.load(cache_path, allow_pickle=True)
        if len(cached) == len(geometry):
            if verbose:
                print(f"  loaded cached labels from {cache_path.name}")
            return cached
    labels = label_samples_by_tech(geometry, device_type, verbose=verbose)
    np.save(cache_path, labels)
    if verbose:
        print(f"  cached labels to {cache_path.name}")
    return labels


# ── Tech-variant-level labeling (v4) ────────────────────────────────────────

def build_tech_variant_fingerprint_map(
    device_type: str, verbose: bool = False,
) -> Dict[Tuple[float, ...], Tuple[str, str]]:
    """Return ``{fingerprint: (tech_name, variant_name)}`` for every known bin."""
    t0 = time.time()
    out: Dict[Tuple[float, ...], Tuple[str, str]] = {}

    for tech_name in TECH_ORDER:
        tech = TECH_CONFIGS[tech_name]
        for variant in tech.variant_names:
            try:
                combos = tech.get_geometry_combos(device_type, variant)
            except Exception as exc:
                if verbose:
                    print(f"  skip {tech_name}:{variant} "
                          f"(get_geometry_combos: {exc})")
                continue
            model_name = tech.get_model_name(device_type, variant)
            for L, NFIN in combos:
                try:
                    modelcard_path = tech.resolve_modelcard(
                        device_type, variant, float(L), float(NFIN))
                    parsed = parse_modelcard(modelcard_path, model_name)
                    proc = extract_process_params(dict(parsed.params))
                except Exception:
                    continue
                fp = _fingerprint(float(NFIN), float(L), proc.as_array())
                out[fp] = (tech_name, variant)

    elapsed = time.time() - t0
    if verbose:
        print(f"  built {len(out)} tech-variant fingerprints in {elapsed:.1f}s")
    return out


def label_samples_by_tech_variant(
    geometry: np.ndarray, device_type: str, verbose: bool = False,
) -> np.ndarray:
    """Return ``(N,)`` int array of tech-variant codes per sample.

    Uses the code mapping from ``bsimar.config.TECH_VARIANT_CODES``.
    """
    assert geometry.ndim == 2 and geometry.shape[1] == 15
    fp_map = build_tech_variant_fingerprint_map(device_type, verbose=verbose)
    n = geometry.shape[0]
    codes = np.empty(n, dtype=np.int64)
    misses: List[int] = []
    for i in range(n):
        row = geometry[i]
        fp = _fingerprint(float(row[0]), float(row[1]),
                          [float(x) for x in row[3:15]])
        tv = fp_map.get(fp)
        if tv is None:
            misses.append(i)
        else:
            codes[i] = tech_variant_to_code(tv[0], tv[1])
    if misses:
        raise AssertionError(
            f"Tech-variant labeller missed {len(misses)} / {n} samples. "
            f"First miss idx={misses[0]}")
    return codes


def _cached_tv_labels_path(data_path: Path) -> Path:
    return data_path.with_name(data_path.stem + "_tech_variant_labels.npy")


def get_or_build_tech_variant_labels(
    data_path: str, device_type: str,
    force_rebuild: bool = False, verbose: bool = True,
) -> np.ndarray:
    """Load cached tech-variant integer labels or rebuild them."""
    data_path_p = Path(data_path)
    cache_path = _cached_tv_labels_path(data_path_p)
    data = np.load(data_path_p, allow_pickle=True)
    geometry = data["geometry"]
    if cache_path.exists() and not force_rebuild:
        cached = np.load(cache_path, allow_pickle=True)
        if len(cached) == len(geometry):
            if verbose:
                print(f"  loaded cached tech-variant labels from {cache_path.name}")
            return cached
    codes = label_samples_by_tech_variant(geometry, device_type, verbose=verbose)
    np.save(cache_path, codes)
    if verbose:
        print(f"  cached tech-variant labels to {cache_path.name}")
    return codes


def summarize_labels(labels: np.ndarray) -> Dict[str, int]:
    """Return counts per tech, sorted by TECH_ORDER."""
    out: Dict[str, int] = {}
    for t in TECH_ORDER:
        out[t] = int((labels == t).sum())
    return out
