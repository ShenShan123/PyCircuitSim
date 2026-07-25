#!/usr/bin/env python
"""V6.7.0 universal-transfer campaign — stratified TSMC5 fine-tune tiers.

Subsamples ``tsmc5_corro_{dev}.npz`` into fixed-size tiers, stratified by
``sample_class`` (proportional allocation, >=1 row per non-empty class,
seeded, largest-remainder rounding) so the rare curriculum classes
(traj_corridor ~0.4%, inv_trip ~3%) survive even the 2k tier — the
uniform-random ``--max-rows`` cap would nearly lose them (plan
2026-07-02 §2/§4).

Outputs (into the standard datasets dir):
  tsmc5ft_n{N}_{dev}.npz + tsmc5ft_n{N}_{dev}_tech_variant_labels.npy
for N in {2000, 10000, 50000, 200000, 1000000}. The "full" tier is the
source file itself and is not duplicated.

All ``meta_*`` keys are copied verbatim (single-tech source, so they stay
valid); row arrays + sidecar are subset with one shared index vector, so
alignment is preserved by construction. Validation: exact tier size,
per-class floor, sidecar codes subset of the source's, seeded spot-check
rows bit-identical to the source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# audit C6o — sidecar fingerprints are written through bsimar's loo_labels, so
# put the package on the path the same way scripts/v6_4_7_s12_append_corridors.py
# does.
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "external_compact_models" / "PyCMG",
           _ROOT / "external_compact_models", _ROOT):
    _sp = str(_p)
    if _sp in sys.path:
        sys.path.remove(_sp)
    sys.path.insert(0, _sp)

from bsimar.eval.loo_labels import write_sidecar_meta  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "external_compact_models" / "bsimar" / "data" / "datasets"

ROW_KEYS = ("inputs", "geometry", "outputs", "sample_class")
DEFAULT_TIERS = (2_000, 10_000, 50_000, 200_000, 1_000_000)
SEED = 42
N_SPOT_ROWS = 100


def stratified_indices(
    sample_class: np.ndarray, n_target: int, rng: np.random.Generator
) -> np.ndarray:
    """Proportional per-class allocation, >=1 per non-empty class,
    largest-remainder rounding to hit n_target exactly."""
    classes, counts = np.unique(sample_class, return_counts=True)
    total = int(counts.sum())
    if n_target >= total:
        raise ValueError(f"tier {n_target} >= source rows {total}")

    exact = counts * (n_target / total)
    alloc = np.maximum(np.floor(exact).astype(np.int64), 1)
    alloc = np.minimum(alloc, counts)  # can't take more than a class has
    # largest-remainder fixup to hit n_target exactly
    while alloc.sum() != n_target:
        if alloc.sum() < n_target:
            room = counts - alloc
            frac = np.where(room > 0, exact - alloc, -np.inf)
            i = int(np.argmax(frac))
            alloc[i] += 1
        else:
            over = alloc - 1  # keep the >=1 floor
            frac = np.where(over > 0, exact - alloc, np.inf)
            i = int(np.argmin(frac))
            alloc[i] -= 1

    picks: list[np.ndarray] = []
    for cls, k in zip(classes, alloc):
        rows = np.nonzero(sample_class == cls)[0]
        picks.append(rng.choice(rows, size=int(k), replace=False))
    idx = np.sort(np.concatenate(picks))
    assert len(idx) == n_target and len(np.unique(idx)) == n_target
    return idx


def build_tiers(dev: str, tiers: tuple[int, ...]) -> bool:
    src_stem = f"tsmc5_corro_{dev}"
    src_npz = DS / f"{src_stem}.npz"
    src_sc_path = DS / f"{src_stem}_tech_variant_labels.npy"
    if not src_npz.exists() or not src_sc_path.exists():
        print(f"FAIL missing source {src_npz.name} / sidecar")
        return False
    data = np.load(src_npz, allow_pickle=True)
    sidecar = np.load(src_sc_path)
    n = len(data["outputs"])
    if len(sidecar) != n:
        print(f"FAIL {src_stem}: sidecar rows {len(sidecar)} != {n}")
        return False
    sample_class = np.asarray(data["sample_class"])
    names = [str(x) for x in data["meta_sample_class_names"]]
    src_codes = set(np.unique(sidecar).tolist())
    meta_keys = [k for k in data.files if k.startswith("meta_")]
    print(f"\n=== {src_stem}: {n} rows, sidecar codes {sorted(src_codes)} ===")

    for n_target in tiers:
        rng = np.random.default_rng(SEED)  # same seed per tier -> reproducible
        idx = stratified_indices(sample_class, n_target, rng)
        out_stem = f"tsmc5ft_n{n_target}_{dev}"
        payload = {k: data[k][idx] for k in ROW_KEYS}
        payload.update({k: data[k] for k in meta_keys})
        payload["meta_created_by"] = np.array(
            f"scripts/uni_subsample_npz.py seed={SEED} from {src_stem}.npz")
        out_npz = DS / f"{out_stem}.npz"
        np.savez(out_npz, **payload)
        np.save(DS / f"{out_stem}_tech_variant_labels.npy", sidecar[idx])
        # audit C6o — fingerprint the sidecar against the rows it labels.
        write_sidecar_meta(out_npz, payload["geometry"], sidecar[idx])

        # validation
        tier_sc = payload["sample_class"]
        cls, cnt = np.unique(tier_sc, return_counts=True)
        src_cls = set(np.unique(sample_class).tolist())
        if len(tier_sc) != n_target:
            print(f"  FAIL {out_stem}: {len(tier_sc)} != {n_target}")
            return False
        if set(cls.tolist()) != src_cls:
            print(f"  FAIL {out_stem}: lost classes {src_cls - set(cls.tolist())}")
            return False
        tier_codes = set(np.unique(sidecar[idx]).tolist())
        if not tier_codes <= src_codes:
            print(f"  FAIL {out_stem}: alien sidecar codes {tier_codes - src_codes}")
            return False
        spot = np.random.default_rng(SEED + 1).choice(
            n_target, size=min(N_SPOT_ROWS, n_target), replace=False)
        for j in spot:
            for k in ROW_KEYS:
                if not np.array_equal(payload[k][j], data[k][idx[j]]):
                    print(f"  FAIL {out_stem}: row {j} not bit-identical ({k})")
                    return False
        hist = {names[int(c)]: int(k) for c, k in zip(cls, cnt)}
        print(f"  {out_stem}: {n_target} rows  classes={hist}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--devices", default="nmos,pmos")
    ap.add_argument("--tiers", default=",".join(str(t) for t in DEFAULT_TIERS))
    args = ap.parse_args()
    tiers = tuple(int(t) for t in args.tiers.split(",") if t.strip())
    devs = [d.strip() for d in args.devices.split(",") if d.strip()]

    for dev in devs:
        if not build_tiers(dev, tiers):
            print("\nFAILED")
            sys.exit(1)
    print("\nALL TIER BUILDS PASS")


if __name__ == "__main__":
    main()
