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

V6.9.0 — per-tech scope filter + collision guard:

* TSMC6 core devices were copied from N7 by TSMC (153/204 core ``.model``
  blocks byte-identical to the N7 card), so tsmc6 and tsmc7 bins share
  identical fingerprints and are NOT distinguishable by this labeller.
  tsmc6 is therefore excluded from the no-filter (universal) fingerprint
  map (``_TECH_ORDER``) and is labelled only through the per-tech path:
  a dataset whose stem starts with a known tech scope (``tsmc6_nmos.npz``)
  is labelled against that tech's fingerprints alone. Universal datasets
  that mix tsmc6 must carry sidecars built by ``scripts/uni_concat_npz.py``
  (which concatenates per-tech sidecars), never by this fingerprint scan.
* ``_build_fingerprint_map`` now raises on a cross-(tech, variant)
  fingerprint collision instead of silently keeping the last writer.
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


# Universal (no-filter) fingerprint scan order. Deliberately EXCLUDES
# tsmc6: its N7-copied bins collide with tsmc7 fingerprints (see module
# docstring). Per-tech datasets are labelled via the tech_filter path.
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
    tech_filter: str | None = None,
) -> Dict[Tuple[float, ...], Tuple[str, str]]:
    """Return ``{fingerprint: (tech, variant)}`` for every known bin.

    ``tech_filter`` restricts the scan to a single tech (per-tech
    datasets). Cross-(tech, variant) fingerprint collisions raise —
    a silent last-writer-wins here would mislabel training data.
    """
    t0 = time.time()
    out: Dict[Tuple[float, ...], Tuple[str, str]] = {}

    tech_order = [tech_filter] if tech_filter is not None else _TECH_ORDER
    for tech_name in tech_order:
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
                prev = out.get(fp)
                if prev is not None and prev != (tech_name, variant):
                    raise AssertionError(
                        f"Fingerprint collision: {prev} and "
                        f"({tech_name!r}, {variant!r}) share the same "
                        f"(NFIN={NFIN}, L={L}, 12-param) fingerprint — "
                        f"these bins cannot be distinguished; label the "
                        f"dataset through the per-tech tech_filter path "
                        f"instead.")
                out[fp] = (tech_name, variant)

    if verbose:
        print(f"  built {len(out)} tech-variant fingerprints in "
              f"{time.time() - t0:.1f}s")
    return out


def _label_samples(
    geometry: np.ndarray, device_type: str, verbose: bool = False,
    tech_filter: str | None = None,
) -> np.ndarray:
    """Return a ``(N,)`` int array of tech-variant codes."""
    assert geometry.ndim == 2 and geometry.shape[1] == 15, (
        f"Expected (N, 15) geometry, got {geometry.shape}")

    fp_map = _build_fingerprint_map(
        device_type, verbose=verbose, tech_filter=tech_filter)
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


def _infer_tech_filter(stem: str) -> str | None:
    """Infer a per-tech scope from a dataset filename stem.

    ``tsmc6_nmos`` -> ``"tsmc6"``; anything whose leading token is not a
    known tech (``universal_nmos``, ``u716_...``) -> ``None`` (full scan).
    """
    head = stem.split("_", 1)[0].lower()
    return head if head in TECH_CONFIGS else None


def get_or_build_tech_variant_labels(
    data_path: str, device_type: str,
    force_rebuild: bool = False, verbose: bool = True,
) -> np.ndarray:
    """Load cached tech-variant labels for a dataset or rebuild them.

    Per-tech datasets (stem ``<tech>_<device>``) are labelled against
    their own tech's fingerprints only — required for tsmc6, whose
    N7-copied bins are indistinguishable from tsmc7 in a full scan.
    """
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

    tech_filter = _infer_tech_filter(data_path_p.stem)
    if verbose and tech_filter is not None:
        print(f"  labelling against tech scope {tech_filter!r} "
              f"(inferred from filename)")
    codes = _label_samples(
        geometry, device_type, verbose=verbose, tech_filter=tech_filter)
    np.save(cache_path, codes)
    if verbose:
        print(f"  cached tech-variant labels to {cache_path.name}")
    return codes
