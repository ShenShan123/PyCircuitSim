"""Tech-variant labeling for BSIMAR ``universal_{device}.npz`` datasets.

Builds a per-sample integer code from ``bsimar.config.TECH_VARIANT_CODES``
by enumerating every ``(tech, variant, L, NFIN)`` bin in the PyCMG
registry, parsing each bin's modelcard for its 12 process parameters,
and matching every sample's geometry row to a known fingerprint.

TSMC12 and TSMC16 share identical ``(L, NFIN)`` grids and can only be
distinguished by the 12-parameter fingerprint, so the lookup key is the
full ``(NFIN, L, 12 proc params)`` tuple.

Labels are cached next to the dataset as
``<dataset>_tech_variant_labels.npy`` so subsequent runs skip the scan.
"""

from __future__ import annotations

import time
from math import floor, log10
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from bsimar.config import TECH_CONFIGS, tech_variant_to_code

# Process-param extraction lives in PyCMG. ``bsimar.config`` already
# appended the PyCMG root to ``sys.path`` on import.
from pycmg.nn_config import extract_process_params  # noqa: E402
from pycmg.parser import parse_modelcard  # noqa: E402


_TECH_ORDER: List[str] = ["asap7", "tsmc5", "tsmc7", "tsmc12", "tsmc16"]
_FINGERPRINT_SIG_FIGS: int = 8


def _round_sig(x: float, sig: int = _FINGERPRINT_SIG_FIGS) -> float:
    """Round ``x`` to ``sig`` significant figures for stable hashing."""
    if x == 0.0 or not np.isfinite(x):
        return float(x)
    digits = sig - int(floor(log10(abs(x)))) - 1
    return float(round(x, digits))


def _fingerprint(
    NFIN: float, L: float, proc_array: List[float],
) -> Tuple[float, ...]:
    """Stable ``(NFIN, L, 12 proc params)`` tuple.

    Temperature is excluded — process params are T-independent so the
    geometry+process tuple already pins the (tech, variant, bin) identity.
    """
    return tuple(_round_sig(float(v)) for v in [NFIN, L, *proc_array])


def _build_fingerprint_map(
    device_type: str, verbose: bool = False,
) -> Dict[Tuple[float, ...], Tuple[str, str]]:
    """Return ``{fingerprint: (tech, variant)}`` for every known bin."""
    t0 = time.time()
    out: Dict[Tuple[float, ...], Tuple[str, str]] = {}

    for tech_name in _TECH_ORDER:
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

    if verbose:
        print(f"  built {len(out)} tech-variant fingerprints in "
              f"{time.time() - t0:.1f}s")
    return out


def _label_samples(
    geometry: np.ndarray, device_type: str, verbose: bool = False,
) -> np.ndarray:
    """Return a ``(N,)`` int array of tech-variant codes."""
    assert geometry.ndim == 2 and geometry.shape[1] == 15, (
        f"Expected (N, 15) geometry, got {geometry.shape}")

    fp_map = _build_fingerprint_map(device_type, verbose=verbose)
    n = geometry.shape[0]
    codes = np.empty(n, dtype=np.int64)
    misses: List[int] = []

    for i in range(n):
        row = geometry[i]
        fp = _fingerprint(
            float(row[0]), float(row[1]),
            [float(x) for x in row[3:15]],
        )
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


def get_or_build_tech_variant_labels(
    data_path: str, device_type: str,
    force_rebuild: bool = False, verbose: bool = True,
) -> np.ndarray:
    """Load cached tech-variant labels for a universal dataset or rebuild them."""
    data_path_p = Path(data_path)
    cache_path = data_path_p.with_name(
        data_path_p.stem + "_tech_variant_labels.npy")
    data = np.load(data_path_p, allow_pickle=True)
    geometry = data["geometry"]

    if cache_path.exists() and not force_rebuild:
        cached = np.load(cache_path, allow_pickle=True)
        if len(cached) == len(geometry):
            if verbose:
                print(f"  loaded cached tech-variant labels from "
                      f"{cache_path.name}")
            return cached

    codes = _label_samples(geometry, device_type, verbose=verbose)
    np.save(cache_path, codes)
    if verbose:
        print(f"  cached tech-variant labels to {cache_path.name}")
    return codes
