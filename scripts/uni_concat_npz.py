#!/usr/bin/env python
"""V6.7.0 universal-transfer campaign — concat per-tech npz datasets.

Builds the universal-716 training sets by concatenating the per-tech
datasets (order: tsmc7, tsmc12, tsmc16) for the row arrays
``inputs/geometry/outputs/sample_class`` plus the aligned
``*_tech_variant_labels.npy`` sidecars (universal codes: tsmc7 4-6,
tsmc12 7-11, tsmc16 12-16). ``load_and_split_bsimar`` reads only the
row arrays + ``meta_sample_class_names`` and the sidecar is validated
by row count, so the concat file is exactly equivalent to a native
universal dataset (plan 2026-07-02 §2/§4).

Outputs (into the standard datasets dir):
  uni716_{nmos,pmos}.npz        from tsmc{7,12,16}_{dev}.npz
  uni716_corro_{nmos,pmos}.npz  from tsmc{7,12,16}_corro_{dev}.npz
  + <stem>_tech_variant_labels.npy sidecars

In-script validation (hard-fails on any miss):
  * output rows == sum of source rows (all arrays + sidecar)
  * sidecar codes are a subset of {4..16}
  * sample_class histogram == sum of the source histograms
  * meta_sample_class_names identical across sources
  * 100 seeded random rows bit-identical to their source file
    (checked on the RELOADED output, not the in-memory buffer)
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

TECH_ORDER = ("tsmc7", "tsmc12", "tsmc16")
ROW_KEYS = ("inputs", "geometry", "outputs", "sample_class")
UNIVERSAL_CODES_716 = set(range(4, 17))  # tsmc7 4-6, tsmc12 7-11, tsmc16 12-16
N_SPOT_ROWS = 100
SPOT_SEED = 42


def _source_stem(tech: str, variant: str, dev: str) -> str:
    return f"{tech}_{dev}" if variant == "base" else f"{tech}_{variant}_{dev}"


def _out_stem(variant: str, dev: str) -> str:
    return f"uni716_{dev}" if variant == "base" else f"uni716_{variant}_{dev}"


def build_one(variant: str, dev: str) -> bool:
    out_stem = _out_stem(variant, dev)
    print(f"\n=== {out_stem} (concat {' + '.join(TECH_ORDER)}, variant={variant}) ===")

    parts: dict[str, list[np.ndarray]] = {k: [] for k in ROW_KEYS}
    sidecars: list[np.ndarray] = []
    src_rows: list[int] = []
    src_files: list[str] = []
    class_names_ref: np.ndarray | None = None

    for tech in TECH_ORDER:
        stem = _source_stem(tech, variant, dev)
        npz_path = DS / f"{stem}.npz"
        sc_path = DS / f"{stem}_tech_variant_labels.npy"
        if not npz_path.exists() or not sc_path.exists():
            print(f"  FAIL missing source {npz_path.name} / {sc_path.name}")
            return False
        data = np.load(npz_path, allow_pickle=True)
        n = len(data["outputs"])
        for k in ROW_KEYS:
            arr = data[k]
            if len(arr) != n:
                print(f"  FAIL {stem}: len({k})={len(arr)} != {n}")
                return False
            parts[k].append(arr)
        sc = np.load(sc_path)
        if len(sc) != n:
            print(f"  FAIL {stem}: sidecar rows {len(sc)} != {n}")
            return False
        sidecars.append(np.asarray(sc, dtype=np.int64))
        names = data["meta_sample_class_names"]
        if class_names_ref is None:
            class_names_ref = names
        elif not np.array_equal(class_names_ref, names):
            print(f"  FAIL {stem}: meta_sample_class_names differ from {TECH_ORDER[0]}")
            return False
        src_rows.append(n)
        src_files.append(npz_path.name)
        print(f"  + {stem}: {n} rows, sidecar codes {sorted(set(np.unique(sc).tolist()))}")

    cat = {k: np.concatenate(parts[k], axis=0) for k in ROW_KEYS}
    sidecar = np.concatenate(sidecars, axis=0)
    n_total = sum(src_rows)

    # ── validations on the in-memory concat ─────────────────────────────
    ok = True
    for k in ROW_KEYS:
        if len(cat[k]) != n_total:
            print(f"  FAIL rowsum: len({k})={len(cat[k])} != {n_total}")
            ok = False
    if len(sidecar) != n_total:
        print(f"  FAIL rowsum: sidecar={len(sidecar)} != {n_total}")
        ok = False
    bad_codes = set(np.unique(sidecar).tolist()) - UNIVERSAL_CODES_716
    if bad_codes:
        print(f"  FAIL sidecar codes outside 4..16: {sorted(bad_codes)}")
        ok = False
    hist_out = np.bincount(cat["sample_class"].astype(np.int64), minlength=32)
    hist_sum = np.zeros(32, dtype=np.int64)
    for arr in parts["sample_class"]:
        hist_sum += np.bincount(arr.astype(np.int64), minlength=32)
    if not np.array_equal(hist_out, hist_sum):
        print("  FAIL class histogram != sum of parts")
        ok = False
    if not ok:
        return False

    # ── save npz + sidecar ───────────────────────────────────────────────
    out_npz = DS / f"{out_stem}.npz"
    out_sc = DS / f"{out_stem}_tech_variant_labels.npy"
    np.savez(
        out_npz,
        **cat,
        meta_sample_class_names=class_names_ref,
        meta_source_files=np.array(src_files),
        meta_source_rows=np.array(src_rows, dtype=np.int64),
        meta_concat_order=np.array(TECH_ORDER),
        meta_created_by=np.array("scripts/uni_concat_npz.py (V6.7.0 universal campaign)"),
    )
    np.save(out_sc, sidecar)
    # audit C6o — pair every sidecar with its geometry fingerprint, or the
    # loader can only validate it by scope + row count (and for a u716_* stem
    # the scope check is skipped too, leaving row count as the only guard).
    write_sidecar_meta(out_npz, cat["geometry"], sidecar)
    print(f"  saved {out_npz.name} ({n_total} rows) + sidecar + meta")

    # ── spot-check: reload output, compare seeded random rows to sources ─
    del cat, sidecar, parts, sidecars
    reloaded = np.load(out_npz, allow_pickle=True)
    re_sc = np.load(out_sc)
    offsets = np.cumsum([0] + src_rows)
    rng = np.random.default_rng(SPOT_SEED)
    idxs = rng.choice(n_total, size=min(N_SPOT_ROWS, n_total), replace=False)
    # group sampled rows by source to load each source once
    by_src: dict[int, list[int]] = {}
    for gi in idxs:
        si = int(np.searchsorted(offsets, gi, side="right") - 1)
        by_src.setdefault(si, []).append(int(gi))
    n_checked = 0
    for si, rows in sorted(by_src.items()):
        stem = _source_stem(TECH_ORDER[si], variant, dev)
        src = np.load(DS / f"{stem}.npz", allow_pickle=True)
        src_sidecar = np.load(DS / f"{stem}_tech_variant_labels.npy")
        for gi in rows:
            li = gi - int(offsets[si])
            for k in ROW_KEYS:
                a, b = reloaded[k][gi], src[k][li]
                if not (np.array_equal(a, b) and np.asarray(a).dtype == np.asarray(b).dtype):
                    print(f"  FAIL bit-identity {k} row {gi} (src {stem}:{li})")
                    return False
            if int(re_sc[gi]) != int(src_sidecar[li]):
                print(f"  FAIL sidecar identity row {gi} (src {stem}:{li})")
                return False
            n_checked += 1
    print(f"  spot-check: {n_checked}/{len(idxs)} random rows bit-identical  PASS")
    nz = [(class_names_ref[i], int(hist_out[i])) for i in np.nonzero(hist_out)[0]]
    print(f"  class histogram: {nz}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=["base", "corro", "all"], default="all")
    ap.add_argument("--devices", default="nmos,pmos")
    args = ap.parse_args()

    variants = ["base", "corro"] if args.variant == "all" else [args.variant]
    devs = [d.strip() for d in args.devices.split(",") if d.strip()]

    failures = []
    for variant in variants:
        for dev in devs:
            if not build_one(variant, dev):
                failures.append(_out_stem(variant, dev))
    if failures:
        print(f"\nFAILED: {failures}")
        sys.exit(1)
    print("\nALL CONCAT BUILDS PASS")


if __name__ == "__main__":
    main()
