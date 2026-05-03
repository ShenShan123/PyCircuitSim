"""Leave-one-out (by technology) helpers for BSIMAR datasets.

Builds per-sample technology / tech-variant label arrays for
``universal_{device}.npz`` by enumerating every
``(tech, variant, L, NFIN)`` bin in the PyCMG registry, reading its
process parameters from the resolved modelcard, and matching each
sample's geometry row to a known fingerprint.

Two labeling granularities:

- **Tech-level** (5 string labels): used by the v3 LOO experiments.
  ``get_or_build_tech_labels()`` returns ``(N,)`` str array.

- **Tech-variant-level** (21 integer codes): used by v4 / v4-re training.
  ``get_or_build_tech_variant_labels()`` returns ``(N,)`` int array
  matching ``bsimar.config.TECH_VARIANT_CODES``.

Key design points:

- Uses ``parse_modelcard`` + ``extract_process_params`` directly. No
  OSDI instantiation is needed, which makes the build fast (~seconds
  for ~950 bins once the TSMC naive modelcard cache is warm).

- TSMC12 and TSMC16 share identical ``(L, NFIN)`` grids and can only
  be distinguished by their 12-element process-param fingerprints, so
  the lookup key is the full ``(NFIN, L, 12 proc params)`` tuple.

- The label arrays are cached next to the dataset as
  ``<dataset>_tech_labels.npy`` / ``<dataset>_tech_variant_labels.npy``
  so subsequent runs skip the scan.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from bsimar.config import (
    DATA_DIR,
    TECH_CONFIGS,
    TECH_VARIANT_CODES,
    tech_variant_to_code,
)
from bsimar.data.dataset import MOSFETDataset, filter_small_targets
from bsimar.data.normalize import BSIMARNormalizer

# Process-param extraction lives in PyCMG (it's no longer re-exported
# through ``bsimar.config`` as of the v4-re trim). ``bsimar.config``
# already appended the PyCMG root to ``sys.path`` on import.
from pycmg.nn_config import extract_process_params  # noqa: E402
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
    """Build a tuple fingerprint stable across small float drift.

    Temperature is intentionally excluded from the fingerprint: the 12
    process parameters are temperature-independent (they come from the
    modelcard verbatim), so a ``(NFIN, L, proc_params)`` tuple already
    pins the (tech, variant, geometry bin) identity regardless of the
    T-sweep the dataset was generated at.
    """
    vals = [NFIN, L] + list(proc_array)
    return tuple(_round_sig(float(v)) for v in vals)


# ── Tech-level labeling (v3 LOO) ────────────────────────────────────────────

def build_tech_fingerprint_map(
    device_type: str, verbose: bool = False,
) -> Dict[Tuple[float, ...], str]:
    """Return ``{fingerprint: tech_name}`` for every known bin.

    Iterates the 5 base techs × their variants × every legal
    ``(L, NFIN)`` combo and reads the bin's process parameters by
    parsing the resolved modelcard directly (no OSDI load).
    """
    t0 = time.time()
    out: Dict[Tuple[float, ...], str] = {}
    collisions: List[Tuple[Tuple[float, ...], str, str]] = []

    for tech_name in TECH_ORDER:
        tech = TECH_CONFIGS[tech_name]
        for variant in tech.variant_names:
            try:
                combos = tech.get_geometry_combos(device_type, variant)
            except Exception as exc:  # pragma: no cover
                if verbose:
                    print(f"  skip {tech_name}:{variant} "
                          f"(get_geometry_combos: {exc})")
                continue
            model_name = tech.get_model_name(device_type, variant)
            for L, NFIN in combos:
                try:
                    modelcard_path = tech.resolve_modelcard(
                        device_type, variant, float(L), float(NFIN),
                    )
                    parsed = parse_modelcard(modelcard_path, model_name)
                    proc = extract_process_params(dict(parsed.params))
                except Exception as exc:  # pragma: no cover
                    if verbose:
                        print(f"  skip {tech_name}:{variant} "
                              f"L={L*1e9:.1f}nm NFIN={NFIN:.0f}: {exc}")
                    continue
                fp = _fingerprint(
                    float(NFIN), float(L), proc.as_array(),
                )
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
            f"(first: {collisions[0]}). Process params are not unique "
            f"per bin — cannot safely label samples by tech."
        )
    return out


def label_samples_by_tech(
    geometry: np.ndarray,
    device_type: str,
    verbose: bool = False,
) -> np.ndarray:
    """Return a ``(N,)`` string array with the tech name per sample.

    Geometry must be the 15-col layout ``[NFIN, L, T, 12 proc params]``
    produced by ``pycmg.nn_generate._assemble``. Any row whose
    fingerprint is not in the tech map raises ``AssertionError`` with a
    diagnostic (a silent "unknown" label is never acceptable).
    """
    assert geometry.ndim == 2 and geometry.shape[1] == 15, (
        f"Expected (N, 15) geometry, got {geometry.shape}")

    fp_map = build_tech_fingerprint_map(device_type, verbose=verbose)

    n = geometry.shape[0]
    labels = np.empty(n, dtype=object)
    misses: List[int] = []

    for i in range(n):
        row = geometry[i]
        NFIN, L = float(row[0]), float(row[1])
        proc = [float(x) for x in row[3:15]]
        fp = _fingerprint(NFIN, L, proc)
        label = fp_map.get(fp)
        if label is None:
            misses.append(i)
            if len(misses) <= 5 and verbose:
                print(f"  MISS row {i}: NFIN={NFIN} L={L*1e9:.1f}nm "
                      f"proc[:4]={proc[:4]}")
        else:
            labels[i] = label

    if misses:
        raise AssertionError(
            f"Tech labeller missed {len(misses)} / {n} samples. "
            f"First miss idx={misses[0]}, geometry row="
            f"{geometry[misses[0]].tolist()}. This usually means "
            f"a PDK revision or a rounding drift; bump "
            f"_FINGERPRINT_SIG_FIGS and retry."
        )
    return labels.astype("<U10")


def _cached_labels_path(data_path: Path) -> Path:
    return data_path.with_name(data_path.stem + "_tech_labels.npy")


def get_or_build_tech_labels(
    data_path: str, device_type: str,
    force_rebuild: bool = False, verbose: bool = True,
) -> np.ndarray:
    """Load cached tech labels for a universal dataset or rebuild them."""
    data_path_p = Path(data_path)
    cache_path = _cached_labels_path(data_path_p)
    data = np.load(data_path_p, allow_pickle=True)
    geometry = data["geometry"]
    if cache_path.exists() and not force_rebuild:
        cached = np.load(cache_path, allow_pickle=True)
        if len(cached) == len(geometry):
            if verbose:
                print(f"  loaded cached labels from {cache_path.name} "
                      f"({len(cached)} samples)")
            return cached
        if verbose:
            print(f"  cached labels stale "
                  f"({len(cached)} vs {len(geometry)}) — rebuilding")
    labels = label_samples_by_tech(geometry, device_type, verbose=verbose)
    np.save(cache_path, labels)
    if verbose:
        print(f"  cached labels to {cache_path.name}")
    return labels


# ── Tech-variant-level labeling (v4 / v4-re) ────────────────────────────────

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


def build_loo_splits(
    data_path: str,
    held_out_tech: str,
    device_type: str,
    val_frac: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[MOSFETDataset, MOSFETDataset, MOSFETDataset, BSIMARNormalizer]:
    """Build (train, val, test) datasets for a single LOO fold.

    - Applies the standard ``filter_small_targets`` sub-floor mask.
    - Holds out **all** samples of ``held_out_tech`` as the test set.
    - Splits the remaining 4 techs ``(1 - val_frac) / val_frac``
      randomly into train / val.
    - Fits ``BSIMARNormalizer(mode='asinh')`` on the **train** slice
      only (no leakage through val or test).
    - Returns three ``MOSFETDataset`` objects in the column order used
      by ``load_and_split_bsimar`` so ``train_transformer`` can apply
      ``reorder_outputs`` as usual.
    """
    from bsimar.config import OUTPUT_COLUMNS

    assert held_out_tech in TECH_ORDER, (
        f"Unknown held_out_tech {held_out_tech!r}; "
        f"expected one of {TECH_ORDER}")

    data = np.load(data_path, allow_pickle=True)
    inputs = np.asarray(data["inputs"])
    geometry = np.asarray(data["geometry"])
    outputs = np.asarray(data["outputs"])
    n_before = len(outputs)

    # Same filter as load_and_split_bsimar (apply_filter=True default).
    keep = filter_small_targets(outputs, OUTPUT_COLUMNS)
    inputs = inputs[keep]
    geometry = geometry[keep]
    outputs = outputs[keep]

    labels = get_or_build_tech_labels(
        data_path, device_type, verbose=verbose)
    labels = labels[keep]

    if verbose:
        pct = 100.0 * (n_before - len(outputs)) / max(n_before, 1)
        print(f"  Data filtering: {n_before} -> {len(outputs)} "
              f"({n_before - len(outputs)} removed, {pct:.1f}%)")

    test_mask = labels == held_out_tech
    train_pool_mask = ~test_mask
    n_test = int(test_mask.sum())
    n_train_pool = int(train_pool_mask.sum())
    assert n_test > 0, (
        f"No samples labelled {held_out_tech!r} — label/data mismatch.")
    assert n_train_pool > 0, (
        f"All samples were labelled {held_out_tech!r} — nothing to "
        f"train on.")

    rng = np.random.default_rng(seed)
    train_pool_idx = np.nonzero(train_pool_mask)[0]
    rng.shuffle(train_pool_idx)
    n_val = max(1, int(round(n_train_pool * val_frac)))
    val_idx = train_pool_idx[:n_val]
    train_idx = train_pool_idx[n_val:]
    test_idx = np.nonzero(test_mask)[0]

    # Fit normalizer on train only.
    normalizer = BSIMARNormalizer(mode="asinh")
    normalizer.fit(
        inputs[train_idx], geometry[train_idx], outputs[train_idx])

    def _make_ds(idx: np.ndarray) -> MOSFETDataset:
        x = normalizer.normalize_inputs(inputs[idx], geometry[idx])
        y = normalizer.normalize_outputs(outputs[idx])
        return MOSFETDataset(x, y)

    train_ds = _make_ds(train_idx)
    val_ds = _make_ds(val_idx)
    test_ds = _make_ds(test_idx)

    if verbose:
        print(f"  LOO split [{held_out_tech}]: "
              f"train={len(train_ds)}, val={len(val_ds)}, "
              f"test={len(test_ds)} (held out {held_out_tech}: {n_test})")

    return train_ds, val_ds, test_ds, normalizer


def get_test_slice(
    data_path: str, held_out_tech: str, device_type: str,
) -> Dict[str, np.ndarray]:
    """Return raw (unfiltered, unnormalised) inputs/geometry/outputs for
    the held-out tech. Used by the worst-case dumper so it can report
    bias points in physical units."""
    data = np.load(data_path, allow_pickle=True)
    inputs = np.asarray(data["inputs"])
    geometry = np.asarray(data["geometry"])
    outputs = np.asarray(data["outputs"])
    labels = get_or_build_tech_labels(data_path, device_type, verbose=False)
    mask = labels == held_out_tech
    return {
        "inputs": inputs[mask],
        "geometry": geometry[mask],
        "outputs": outputs[mask],
    }
