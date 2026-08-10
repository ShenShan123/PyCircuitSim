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
A cached sidecar is validated before it is trusted (audit C6o) — scope,
dtype, row count, and, when the companion
``<dataset>_tech_variant_labels.meta.npz`` is present, a SHA-256 of the
geometry block the labels were built from.

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

import hashlib
import time
from math import floor, log10
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from bsimar.config import (
    CODE_TO_TECH_VARIANT, TECH_CONFIGS, tech_variant_to_code,
)

# Process-param extraction lives in PyCMG. ``bsimar.config`` already
# appended the PyCMG root to ``sys.path`` on import.
from pycmg.nn_config import extract_process_params  # noqa: E402
from pycmg.parser import parse_modelcard  # noqa: E402


# Universal (no-filter) fingerprint scan order. Deliberately EXCLUDES
# tsmc6: its N7-copied bins collide with tsmc7 fingerprints (see module
# docstring). Per-tech datasets are labelled via the tech_filter path.
_TECH_ORDER: List[str] = ["asap7", "tsmc5", "tsmc7", "tsmc12", "tsmc16"]
_FINGERPRINT_SIG_FIGS: int = 8

_SIDECAR_SUFFIX: str = "_tech_variant_labels.npy"
_SIDECAR_META_SUFFIX: str = "_tech_variant_labels.meta.npz"
# Keys every meta sidecar must carry; a meta missing any of them is treated
# as if it were absent (a foreign producer, not a corrupt dataset).
_META_KEYS: Tuple[str, ...] = (
    "geometry_sha256", "geometry_shape", "geometry_dtype",
    "n_rows", "labels_dtype",
)


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
    max_l_ratio: float | None = None,
) -> Dict[Tuple[float, ...], Tuple[str, str]]:
    """Return ``{fingerprint: (tech, variant)}`` for every known bin.

    ``tech_filter`` restricts the scan to a single tech (per-tech
    datasets). Cross-(tech, variant) fingerprint collisions raise —
    a silent last-writer-wins here would mislabel training data.

    ``max_l_ratio`` must match the value the dataset was generated with
    (V7.4.2 intra-bin L sampling). L is part of the fingerprint, so a map
    built on a coarser L grid than the data simply has no entry for the
    interior rows and every one of them becomes a miss. Callers read it
    from the dataset's own ``meta_max_l_ratio``.
    """
    t0 = time.time()
    out: Dict[Tuple[float, ...], Tuple[str, str]] = {}

    tech_order = [tech_filter] if tech_filter is not None else _TECH_ORDER
    for tech_name in tech_order:
        tech = TECH_CONFIGS[tech_name]
        for variant in tech.variant_names:
            try:
                combos = tech.get_geometry_combos(
                    device_type, variant, max_l_ratio=max_l_ratio)
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
    tech_filter: str | None = None, max_l_ratio: float | None = None,
) -> np.ndarray:
    """Return a ``(N,)`` int array of tech-variant codes."""
    assert geometry.ndim == 2 and geometry.shape[1] == 15, (
        f"Expected (N, 15) geometry, got {geometry.shape}")

    fp_map = _build_fingerprint_map(
        device_type, verbose=verbose, tech_filter=tech_filter,
        max_l_ratio=max_l_ratio)
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
        miss_l = sorted({float(geometry[i][1]) for i in misses})
        raise AssertionError(
            f"Tech-variant labeller missed {len(misses)} / {n} samples. "
            f"First miss idx={misses[0]}. Unmatched L values (nm): "
            f"{[round(v * 1e9, 3) for v in miss_l[:12]]}. If those are "
            f"intra-bin lengths, the dataset was generated with a "
            f"--max-l-ratio the fingerprint map was not built for; the "
            f"value is stored as meta_max_l_ratio in the .npz.")
    return codes


def _infer_tech_filter(stem: str) -> str | None:
    """Infer a per-tech scope from a dataset filename stem.

    ``tsmc6_nmos`` -> ``"tsmc6"``; anything whose leading token is not a
    known tech (``universal_nmos``, ``u716_...``) -> ``None`` (full scan).
    """
    head = stem.split("_", 1)[0].lower()
    return head if head in TECH_CONFIGS else None


def _geometry_digest(geometry: np.ndarray) -> str:
    """SHA-256 over the geometry block the labels were derived from.

    Row count alone cannot tell two datasets apart — ``tsmc12_nmos`` and
    ``tsmc16_nmos`` have identical row counts and file sizes yet disjoint
    code sets — so the sidecar fingerprint has to be content-based.
    ``memoryview`` hashes the buffer in place (no 262 MB ``tobytes`` copy);
    dtype and shape go in first so a reinterpreted buffer cannot collide.
    """
    arr = np.ascontiguousarray(geometry)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(memoryview(arr))
    return h.hexdigest()


def sidecar_paths(data_path: str | Path) -> Tuple[Path, Path]:
    """Return ``(<stem>_labels.npy, <stem>_labels.meta.npz)`` for a dataset."""
    p = Path(data_path)
    return (p.with_name(p.stem + _SIDECAR_SUFFIX),
            p.with_name(p.stem + _SIDECAR_META_SUFFIX))


def write_sidecar_meta(
    data_path: str | Path, geometry: np.ndarray, codes: np.ndarray,
) -> Path:
    """Write the companion fingerprint for a sidecar. Public on purpose.

    Every producer of a ``*_tech_variant_labels.npy`` — this module and the
    scripts that pre-seed concatenated / appended / subsampled sidecars —
    should call this right after ``np.save``. A sidecar without it still
    loads, but only under the weaker scope+row-count validation (audit C6o).
    """
    _, meta_path = sidecar_paths(data_path)
    np.savez(
        meta_path,
        geometry_sha256=np.array(_geometry_digest(geometry)),
        geometry_shape=np.asarray(geometry.shape, dtype=np.int64),
        geometry_dtype=np.array(str(geometry.dtype)),
        n_rows=np.array(len(codes), dtype=np.int64),
        labels_dtype=np.array(codes.dtype.str),
    )
    return meta_path


def _validate_cached_sidecar(
    cached: np.ndarray, geometry: np.ndarray, tech_filter: str | None,
    cache_path: Path, meta_path: Path, verbose: bool,
) -> None:
    """Raise unless the cached sidecar provably belongs to this dataset.

    audit C6o: the cache used to be accepted on row count alone, which
    accepts any same-length sidecar — a copy from a sibling tech, a
    sidecar left behind by a regenerated dataset (Rule 1 invites exactly
    that), or codes outside every vocabulary. Mislabelled tech codes do
    not crash; they train an embedding row that the parser will never
    address, so the failure is silent all the way to the gates.

    Checks, cheapest first: row count, integer dtype, per-tech scope, then
    the geometry SHA-256 when a meta sidecar exists. A missing meta is the
    legacy case (32 sidecars on disk predate it, and two external producer
    scripts still write bare ``.npy``) — warn and keep the weaker checks.
    Never rebuilds silently: a sidecar that fails here means someone's
    bookkeeping is broken, and some sidecars (the concatenated universal
    ones) cannot be rebuilt by this labeller at all.
    """
    def _reject(reason: str) -> None:
        raise ValueError(
            f"Tech-variant label sidecar {cache_path} does not match its "
            f"dataset: {reason}. Delete the sidecar (and its .meta.npz) or "
            f"call get_or_build_tech_variant_labels(..., "
            f"force_rebuild=True) to re-label from the geometry block.")

    if cached.ndim != 1 or len(cached) != len(geometry):
        _reject(f"sidecar holds {cached.shape} labels for "
                f"{len(geometry)} geometry rows")
    if cached.dtype.kind != "i":
        _reject(f"sidecar dtype {cached.dtype} is not an integer code array")

    if tech_filter is not None:
        techs = {
            CODE_TO_TECH_VARIANT.get(int(c), (None, None))[0]
            for c in np.unique(cached)
        }
        if techs != {tech_filter}:
            _reject(f"dataset stem scopes it to {tech_filter!r} but the "
                    f"codes decode to techs {sorted(map(str, techs))}")

    if not meta_path.exists():
        if verbose:
            print(f"  [warn] legacy unhashed sidecar {cache_path.name} — "
                  f"validated by scope+row-count only")
        return
    with np.load(meta_path) as meta:
        _validate_meta(meta, meta_path, cached, geometry, _reject, verbose)


def _validate_meta(
    meta: Any,
    meta_path: Path,
    cached: np.ndarray,
    geometry: np.ndarray,
    _reject: Callable[[str], None],
    verbose: bool,
) -> None:
    """Cross-check the sidecar against its ``.meta.npz`` (audit C6o).

    Split out of :func:`_validate_cached_sidecar` only so the ``np.load``
    context manager closes the NpzFile on every return path, including the
    early ``return`` for a meta file that predates a key.
    """
    missing = [k for k in _META_KEYS if k not in meta.files]
    if missing:
        if verbose:
            print(f"  [warn] {meta_path.name} lacks {missing} — validated "
                  f"by scope+row-count only")
        return
    if int(meta["n_rows"]) != len(cached):
        _reject(f"meta records {int(meta['n_rows'])} labels, sidecar has "
                f"{len(cached)}")
    if str(meta["labels_dtype"]) != cached.dtype.str:
        _reject(f"meta records label dtype {str(meta['labels_dtype'])!r}, "
                f"sidecar is {cached.dtype.str!r}")
    if tuple(int(v) for v in meta["geometry_shape"]) != geometry.shape:
        _reject(f"meta was built for geometry "
                f"{tuple(int(v) for v in meta['geometry_shape'])}, dataset "
                f"is {geometry.shape}")
    if str(meta["geometry_dtype"]) != str(geometry.dtype):
        _reject(f"meta was built for geometry dtype "
                f"{str(meta['geometry_dtype'])!r}, dataset is "
                f"{str(geometry.dtype)!r}")
    digest = _geometry_digest(geometry)
    if str(meta["geometry_sha256"]) != digest:
        _reject(f"geometry sha256 {digest[:16]}… != the "
                f"{str(meta['geometry_sha256'])[:16]}… the labels were "
                f"built from (dataset regenerated, or foreign sidecar)")


def get_or_build_tech_variant_labels(
    data_path: str, device_type: str,
    force_rebuild: bool = False, verbose: bool = True,
) -> np.ndarray:
    """Load cached tech-variant labels for a dataset or rebuild them.

    Per-tech datasets (stem ``<tech>_<device>``) are labelled against
    their own tech's fingerprints only, which keeps a per-tech run
    independent of whether any two techs share a fingerprint globally.
    That scoping is *required* for tsmc6, whose N7-copied bins are
    indistinguishable from tsmc7 in a full scan.

    A cached sidecar is validated against the dataset before it is
    trusted (``_validate_cached_sidecar``); a mismatch raises rather than
    relabelling silently.
    """
    data_path_p = Path(data_path)
    cache_path, meta_path = sidecar_paths(data_path_p)
    data = np.load(data_path_p, allow_pickle=True)
    geometry = data["geometry"]
    # V7.4.2: L is part of the fingerprint, so the map has to be built on
    # the same L grid the rows came from. 0.0 / absent = the legacy
    # lower-corner-only grid.
    _mlr = float(data["meta_max_l_ratio"]) if "meta_max_l_ratio" in data.files \
        else 0.0
    max_l_ratio = _mlr if _mlr > 1.0 else None

    # Inferred before the cache branch: the scope a filename implies is the
    # cheapest evidence that a sidecar belongs to this dataset (audit C6o).
    tech_filter = _infer_tech_filter(data_path_p.stem)

    if cache_path.exists() and not force_rebuild:
        # No allow_pickle: a label sidecar is always a plain int array, so
        # unpickling here would only add a code-execution surface.
        cached = np.load(cache_path)
        _validate_cached_sidecar(
            cached, geometry, tech_filter, cache_path, meta_path, verbose)
        if verbose:
            print(f"  loaded cached tech-variant labels from "
                  f"{cache_path.name}")
        return cached

    if verbose and tech_filter is not None:
        print(f"  labelling against tech scope {tech_filter!r} "
              f"(inferred from filename)")
    codes = _label_samples(
        geometry, device_type, verbose=verbose, tech_filter=tech_filter,
        max_l_ratio=max_l_ratio)
    np.save(cache_path, codes)
    write_sidecar_meta(data_path_p, geometry, codes)
    if verbose:
        print(f"  cached tech-variant labels to {cache_path.name}")
    return codes
