"""Investigate TSMC7 NMOS coverage gap.

Both BSIMAR (probe + production) and DirectNet predict TSMC7 NMOS DC
NRMSE = 15-19% — much worse than TSMC5/12/16 NMOS (~5-10%). Both NN
families fail similarly, so this is data/coverage, not architecture.

This script slices universal_nmos.npz by tech-variant and asks:

  1. How many samples per tech-variant? Is TSMC7 underrepresented?
  2. Sample density vs Vgs (focus near Vth — subthreshold transition)?
  3. Geometry coverage (L, NFIN combos) per tech-variant?
  4. Where do the BSIMAR + DirectNet errors concentrate? Sweep
     Id-Vgs at the inverter operating point and compare to PyCMG.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT))

from bsimar.config import DATA_DIR, tech_variant_to_code, CODE_TO_TECH_VARIANT


def _label_to_str(c: int) -> str:
    if c in CODE_TO_TECH_VARIANT:
        t, v = CODE_TO_TECH_VARIANT[c]
        return f"{t}:{v}"
    return f"code{c}"

OUT_DIR = PROJECT_ROOT / "results" / "diag_tsmc7_nmos"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=== TSMC7 NMOS coverage diagnostic ===\n")

    npz_path = DATA_DIR / "universal_nmos.npz"
    labels_path = DATA_DIR / "universal_nmos_tech_variant_labels.npy"
    if not npz_path.exists() or not labels_path.exists():
        print(f"Data not found:\n  {npz_path}\n  {labels_path}")
        sys.exit(1)

    print(f"Loading {npz_path.name}...")
    d = np.load(npz_path)
    print(f"Loading {labels_path.name}...")
    labels = np.load(str(labels_path))   # one tech-variant string per sample

    inputs = d["inputs"]      # (N, 4): Vgs, Vds, Vbs, T (or similar)
    geometry = d["geometry"]  # (N, 15)
    outputs = d["outputs"]    # (N, 13): id, gm, gds, ...

    N = len(labels)
    print(f"Total samples: {N:,}")
    print(f"Inputs shape: {inputs.shape}, dtype {inputs.dtype}")
    print(f"Geometry shape: {geometry.shape}")
    print(f"Outputs shape: {outputs.shape}\n")

    # ── 1. Sample count per tech-variant ───────────────────────────────────
    uniq, counts = np.unique(labels, return_counts=True)
    sort_idx = np.argsort(-counts)
    print("Samples per tech-variant (sorted by count):")
    print(f"  {'tech:variant':<20s} {'count':>10s} {'%':>6s}")
    print("  " + "-" * 38)
    for i in sort_idx:
        pct = 100.0 * counts[i] / N
        tv = _label_to_str(int(uniq[i]))
        print(f"  {tv:<20s} {counts[i]:>10,d} {pct:>6.2f}%")
    print()

    # ── 2. Per-tech Vgs distribution near Vth ───────────────────────────────
    # Need to know which input column is Vgs. Try guessing: usually first.
    # From CLAUDE.md context: 7-dim continuous = (Vgs, Vds, Vbs, NFIN, L, T, tech_code)
    # but here inputs has only 4 cols (Vgs, Vds, Vbs, T?), geometry has rest.
    print("Input col[0] stats: min={:.3f}, max={:.3f}, mean={:.3f}".format(
        float(inputs[:, 0].min()), float(inputs[:, 0].max()),
        float(inputs[:, 0].mean())))
    print("Input col[1] stats: min={:.3f}, max={:.3f}, mean={:.3f}".format(
        float(inputs[:, 1].min()), float(inputs[:, 1].max()),
        float(inputs[:, 1].mean())))
    # Assume col 0 = Vgs, col 1 = Vds for NMOS (positive ranges).

    # ── 3. Per-tech 2D Vgs/Vds heatmap for the SVT variant ─────────────────
    print("\n--- Per-tech sample density on (Vgs, Vds) plane (SVT variant) ---")
    techs_to_plot = ["tsmc5:svt", "tsmc7:svt", "tsmc12:svt", "tsmc16:svt"]
    codes_to_plot = [tech_variant_to_code(t.split(":")[0], t.split(":")[1])
                     for t in techs_to_plot]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col_i, (t, code) in enumerate(zip(techs_to_plot, codes_to_plot)):
        mask = labels == code
        n_t = int(mask.sum())
        vgs = inputs[mask, 0]
        vds = inputs[mask, 1]
        id_vals = outputs[mask, 0]   # id
        # Top: 2D histogram
        h2 = axes[0, col_i].hist2d(vgs, vds, bins=60, cmap="viridis")
        axes[0, col_i].set_title(f"{t} | N={n_t:,}\nVgs/Vds density")
        axes[0, col_i].set_xlabel("Vgs [V]")
        axes[0, col_i].set_ylabel("Vds [V]")
        plt.colorbar(h2[3], ax=axes[0, col_i], label="count")
        # Bottom: |Id| vs Vgs at Vds in mid-saturation slice
        if n_t > 0:
            vds_med = np.median(vds[vds > 0])
            slice_mask = (vds >= 0.85 * vds_med) & (vds <= 1.15 * vds_med)
            order = np.argsort(vgs[slice_mask])
            axes[1, col_i].semilogy(
                vgs[slice_mask][order],
                np.abs(id_vals[slice_mask][order]) + 1e-15,
                ".", ms=2, alpha=0.4)
            axes[1, col_i].set_title(
                f"{t} | |Id| vs Vgs @ Vds≈{vds_med:.2f}V (slice ±15%)")
            axes[1, col_i].set_xlabel("Vgs [V]")
            axes[1, col_i].set_ylabel("|Id| [A]")
            axes[1, col_i].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "vgs_vds_density.png", dpi=120)
    plt.close()
    print(f"  saved {OUT_DIR / 'vgs_vds_density.png'}")

    # ── 4. Per-tech geometry coverage (L, NFIN combos) ──────────────────────
    print("\n--- Geometry coverage per tech-variant (SVT) ---")
    # geometry columns from CLAUDE.md: [NFIN, L, T, 12 process params]
    # so geometry[:, 0] = NFIN, geometry[:, 1] = L
    for t, code in zip(techs_to_plot, codes_to_plot):
        mask = labels == code
        if mask.sum() == 0:
            continue
        nfin = geometry[mask, 0]
        l_m = geometry[mask, 1]
        unique_combos = np.unique(np.column_stack([nfin, l_m]), axis=0)
        print(f"  {t}: {len(unique_combos)} unique (NFIN, L) combos, "
              f"NFIN range [{nfin.min():.0f}, {nfin.max():.0f}], "
              f"L range [{l_m.min()*1e9:.0f}nm, {l_m.max()*1e9:.0f}nm]")

    # ── 5. Per-tech Id distribution (negative tail = subthreshold? bug?) ───
    print("\n--- Per-tech Id sign distribution ---")
    print(f"  {'tech':<20s} {'frac id<0':>12s} {'frac id≈0':>12s} {'frac id>0':>12s}")
    print("  " + "-" * 60)
    for t, code in zip(techs_to_plot, codes_to_plot):
        mask = labels == code
        if mask.sum() == 0:
            continue
        ids = outputs[mask, 0]
        neg = float(np.mean(ids < -1e-15))
        zero = float(np.mean(np.abs(ids) <= 1e-15))
        pos = float(np.mean(ids > 1e-15))
        print(f"  {t:<20s} {neg:>12.4f} {zero:>12.4f} {pos:>12.4f}")


if __name__ == "__main__":
    main()
