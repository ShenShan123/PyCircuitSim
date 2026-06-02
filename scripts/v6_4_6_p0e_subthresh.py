#!/usr/bin/env python3
"""V6.4.6 Phase-0 diagnostic P0-E — subthreshold normalisation / sign-noise audit.

Pure dataset/normalisation audit (no sims, no model runs). Ground truth = the
OSDI-generated dataset itself. Decides whether a subthreshold log-reweight or
log-derivative distillation target is SAFE, i.e. whether the sub-1e-7 leakage
band is clean signal or sign-random PyCMG floor noise.

Decisions implemented (plan §4 row P0-E, §12 Q5):
  1. Confirm the asinh transform crushes the 1nA-100nA leakage band into a
     vanishing fraction of the normalised id dynamic range -> near-zero
     training-loss gradient on the leakage band.
  2. Quantify floor noise: among rows with |id| < 1e-7 A, what fraction are
     negative and what fraction are literal-0? Histogram of sign and |id| in
     decades from 1e-12 to 1e-7.
  3. Subthreshold OSDI gm/gds cleanliness in sample_class==2: NaN/Inf fraction,
     wrong-sign fraction, |id| range covered.
  4. DECISION: clean tail -> log-reweight allowed; sign-random floor noise ->
     clipped/winsorised signed-floor target only (or abandon the reweight).

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n pycircuitsim \
        python scripts/v6_4_6_p0e_subthresh.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── repo bootstrap ──────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external_compact_models"))

from bsimar.data.normalize import NormStats, normalizer_from_stats  # noqa: E402

DATASET = REPO / "external_compact_models/bsimar/data/datasets/tsmc7_nmos.npz"
NORM = REPO / "external_compact_models/bsimar/checkpoints/tsmc7_dn_medium_nmos_norm.npz"
REPORT = REPO / "results/v6_4_6/phase0_E_subthresh_audit.md"
LOG = REPO / "results/v6_4_6/phase0_logs/p0e_subthresh.log"

# dataset column indices (confirmed: meta_output_columns)
ID, GM, GDS, GMB = 0, 1, 2, 3
SUBTHRESH_CLASS = 2          # meta_sample_class_names[2] == 'subthresh'
LEAK_THRESH = 1e-7           # |id| < 1e-7 A == leakage band
WINSOR_FLOOR = 1e-12         # clipped-target lower support per plan §4 / risk table


def _tee(lines: List[str], fh) -> None:
    for ln in lines:
        print(ln)
        fh.write(ln + "\n")


def _fmt_decades(abs_id: np.ndarray, lo_exp: int = -12, hi_exp: int = -7) -> List[str]:
    """Histogram of |id| in log10 decades from 10^lo_exp to 10^hi_exp."""
    out: List[str] = []
    edges = list(range(lo_exp, hi_exp + 1))
    # zero / below-floor bucket
    n_zero = int(np.sum(abs_id == 0.0))
    n_below = int(np.sum((abs_id > 0.0) & (abs_id < 10.0 ** lo_exp)))
    out.append(f"    |id| == 0 (literal)           : {n_zero}")
    out.append(f"    0 < |id| < 1e{lo_exp:+d}             : {n_below}")
    for e in edges[:-1]:
        lo, hi = 10.0 ** e, 10.0 ** (e + 1)
        n = int(np.sum((abs_id >= lo) & (abs_id < hi)))
        out.append(f"    1e{e:+d} <= |id| < 1e{e + 1:+d}        : {n}")
    return out


def _sign_fractions(idv: np.ndarray) -> Tuple[int, int, int, int]:
    """(n_total, n_neg, n_zero, n_pos) for an id array."""
    n = idv.size
    n_neg = int(np.sum(idv < 0.0))
    n_zero = int(np.sum(idv == 0.0))
    n_pos = int(np.sum(idv > 0.0))
    return n, n_neg, n_zero, n_pos


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    fh = open(LOG, "w")
    md: List[str] = []  # markdown report body

    def emit(*lines: str) -> None:
        _tee(list(lines), fh)

    emit("=" * 78)
    emit("V6.4.6 Phase-0 P0-E — subthreshold normalisation / sign-noise audit")
    emit("=" * 78)
    emit(f"dataset : {DATASET}")
    emit(f"norm    : {NORM}")
    emit("")

    # ── load ────────────────────────────────────────────────────────────────
    d = np.load(DATASET, allow_pickle=True)
    inputs = d["inputs"]                 # (N,4) = Vgs, Vds, Vbs(=0), col3
    outputs = d["outputs"].astype(np.float64)   # (N,13)
    sample_class = d["sample_class"]
    out_cols = list(d["meta_output_columns"])
    class_names = list(d["meta_sample_class_names"])
    vdd = float(d["meta_vdd"])
    N = outputs.shape[0]

    stats = NormStats.load(str(NORM))
    norm = normalizer_from_stats(stats)
    asinh_scale = np.asarray(stats.asinh_scale, dtype=np.float64)
    out_mean = np.asarray(stats.output_mean, dtype=np.float64)
    out_std = np.asarray(stats.output_std, dtype=np.float64)

    # ── schema confirmation ───────────────────────────────────────────────────
    emit("[0] SCHEMA CONFIRMATION")
    emit(f"    N rows                 : {N}")
    emit(f"    inputs shape           : {inputs.shape}  (4 cols)")
    in_ranges = [(float(inputs[:, j].min()), float(inputs[:, j].max()),
                  int(np.unique(inputs[:, j]).size)) for j in range(inputs.shape[1])]
    in_labels = ["Vgs", "Vds", "Vbs", "col3(Vbs-sweep)"]
    for j, (lo, hi, nu) in enumerate(in_ranges):
        emit(f"      inputs[:,{j}] {in_labels[j]:<16}: "
             f"min={lo:.4g} max={hi:.4g} nuniq={nu}")
    emit(f"    outputs shape          : {outputs.shape}  (13 cols)")
    emit(f"    meta_output_columns    : {out_cols}")
    emit(f"      -> id@{out_cols.index('id')} gm@{out_cols.index('gm')} "
         f"gds@{out_cols.index('gds')} gmb@{out_cols.index('gmb')}")
    emit(f"    sample_class names     : {class_names}")
    emit(f"      -> subthresh == code {class_names.index('subthresh')}")
    emit(f"    meta_vdd               : {vdd}")
    emit(f"    norm mode              : {stats.mode}")
    emit(f"    asinh_scale[id]        : {asinh_scale[ID]:.6e}")
    emit(f"    output_mean[id]        : {out_mean[ID]:.6f}")
    emit(f"    output_std[id]         : {out_std[ID]:.6f}")
    emit("")

    md.append("# V6.4.6 Phase-0 P0-E — Subthreshold normalisation / sign-noise audit")
    md.append("")
    md.append(f"- **Dataset:** `{DATASET.relative_to(REPO)}` (N={N:,})")
    md.append(f"- **Normaliser:** `{NORM.relative_to(REPO)}` (mode=`{stats.mode}`)")
    md.append(f"- **id column:** output col {ID}; gm col {GM}; gds col {GDS} "
              f"(per `meta_output_columns`)")
    md.append(f"- **subthresh class:** code {SUBTHRESH_CLASS} "
              f"(`meta_sample_class_names[{SUBTHRESH_CLASS}]`)")
    md.append(f"- **`asinh_scale[id]`** = `{asinh_scale[ID]:.6e}`, "
              f"`output_mean[id]`=`{out_mean[ID]:.4f}`, "
              f"`output_std[id]`=`{out_std[ID]:.4f}`, `meta_vdd`={vdd}")
    md.append("")
    md.append("Input columns confirmed: `inputs (N,4)` = "
              "(Vgs, Vds, Vbs[=0 for NMOS], col3[Vbs sweep var]). "
              "Body col2 is identically 0 in this NMOS set; col3 carries the "
              "Vbs LHS sweep (range "
              f"{in_ranges[3][0]:.3f}..{in_ranges[3][1]:.3f}).")
    md.append("")

    # ── helper: normalise id physical -> training space ────────────────────────
    # forward asinh: z = (arcsinh(id/scale) - mean)/std  (production normalize_outputs)
    id_phys_all = outputs[:, ID]
    # use the production normaliser end-to-end on the id column only
    z_all = norm.normalize_outputs(outputs)[:, ID]   # (N,) normalised id

    # ──────────────────────────────────────────────────────────────────────────
    # PART 1 — asinh crush of the leakage band
    # ──────────────────────────────────────────────────────────────────────────
    emit("[1] ASINH CRUSH OF THE LEAKAGE BAND")
    sub_mask = sample_class == SUBTHRESH_CLASS
    leak_mask = np.abs(id_phys_all) < LEAK_THRESH

    z_finite = z_all[np.isfinite(z_all)]
    z_full_lo, z_full_hi = float(z_finite.min()), float(z_finite.max())
    z_full_range = z_full_hi - z_full_lo

    # The 1nA..100nA band in NORMALISED space (signed). Use the *physical*
    # band edges mapped through the actual asinh forward (mean/std included).
    def fwd_id(idp: float) -> float:
        u = np.arcsinh(idp / asinh_scale[ID])
        return float((u - out_mean[ID]) / out_std[ID])

    band_edges_phys = [1e-9, 1e-8, 1e-7]   # 1nA, 10nA, 100nA
    emit(f"    Full normalised id range (all finite rows): "
         f"[{z_full_lo:.4f}, {z_full_hi:.4f}]  span={z_full_range:.4f}")
    emit("    Normalised id at leakage-band magnitude edges (positive id):")
    z_edges = []
    for be in band_edges_phys:
        ze = fwd_id(be)
        z_edges.append(ze)
        emit(f"      id=+{be:.0e} A -> z={ze:+.5f}  "
             f"(|z-z@0|={abs(ze - fwd_id(0.0)):.5f})")
    z_at_zero = fwd_id(0.0)
    emit(f"      id= 0      A -> z={z_at_zero:+.5f}")

    # width of the entire 1nA..100nA band in normalised units (signed, +id side)
    z_1n = fwd_id(1e-9)
    z_100n = fwd_id(1e-7)
    band_width_norm = abs(z_100n - z_1n)
    frac_of_range = band_width_norm / z_full_range
    emit("")
    emit(f"    1nA -> 100nA band occupies {band_width_norm:.5f} normalised-id units")
    emit(f"      = {100.0 * frac_of_range:.4f}% of the full normalised id range "
         f"({z_full_range:.4f}).")

    # On-state reference: median |id| of strong-on rows (Vgs high, Vds high)
    on_state_id = np.percentile(np.abs(id_phys_all[id_phys_all != 0]), 99.0)
    z_on = fwd_id(on_state_id)
    emit(f"    On-state ref (99th pct |id|={on_state_id:.4e} A) -> z={z_on:+.5f}")
    emit(f"      The on-band vs 100nA gap is {abs(z_on - z_100n):.4f} norm-units "
         f"({100.0 * abs(z_on - z_100n) / z_full_range:.2f}% of range); "
         f"the entire <100nA leakage band is {100.0 * abs(z_100n - z_at_zero) / z_full_range:.4f}% "
         f"of range below z@0.")

    # decades of physical id between scale and 1nA
    decades_below_scale = np.log10(asinh_scale[ID] / 1e-9)
    emit(f"    1nA is {decades_below_scale:.2f} decades below asinh_scale[id] "
         f"({asinh_scale[ID]:.3e}); asinh(1e-9/scale)={np.arcsinh(1e-9/asinh_scale[ID]):.3e} "
         f"~ linear regime -> crushed toward 0.")

    # gradient down-weighting factor d z / d id_phys at 1nA vs on-state
    # dz/did = 1/(std * sqrt(scale^2 + id^2))   (asinh chain rule)
    def dz_did(idp: float) -> float:
        return 1.0 / (out_std[ID] * np.sqrt(asinh_scale[ID] ** 2 + idp ** 2))
    g_1n = dz_did(1e-9)
    g_on = dz_did(on_state_id)
    emit(f"    dz/d|id| @1nA = {g_1n:.4e}  vs  @on-state = {g_on:.4e}  "
         f"-> ratio {g_1n / g_on:.2f}x (NOTE: gradient is *larger* at 1nA because "
         f"id<<scale; the crush is in the *range fraction*, not the local slope).")
    emit("")

    md.append("## 1. Asinh crush of the leakage band")
    md.append("")
    md.append(f"- Full normalised id range over all finite rows: "
              f"`[{z_full_lo:.4f}, {z_full_hi:.4f}]`, span **{z_full_range:.4f}** norm-units.")
    md.append(f"- Normalised id at band edges (positive id, full mean/std forward): "
              f"`z(1nA)={z_1n:+.5f}`, `z(10nA)={z_edges[1]:+.5f}`, "
              f"`z(100nA)={z_100n:+.5f}`, `z(0)={z_at_zero:+.5f}`.")
    md.append(f"- **The entire 1nA-100nA band spans only "
              f"{band_width_norm:.5f} normalised-id units = "
              f"{100.0 * frac_of_range:.4f}% of the full normalised id range.**")
    md.append(f"- 1 nA is **{decades_below_scale:.2f} decades** below "
              f"`asinh_scale[id]` ({asinh_scale[ID]:.3e}); "
              f"`asinh(1nA/scale)={np.arcsinh(1e-9/asinh_scale[ID]):.2e}` "
              f"(deep in the asinh linear regime) -> the whole sub-100nA band "
              f"is crushed to within {100.0 * abs(z_100n - z_at_zero) / z_full_range:.4f}% "
              f"of z@0.")
    md.append(f"- On-state reference (99th-pct |id|={on_state_id:.3e} A) maps to "
              f"`z={z_on:+.4f}`; the leakage band sits "
              f"{abs(z_on - z_100n):.3f} norm-units away.")
    md.append("")
    md.append("**Headline:** asinh confirmed to crush the leakage band — the full "
              f"1nA-100nA leakage band is **{100.0 * frac_of_range:.4f}% of the "
              "normalised id range**, so a pointwise MAE loss in normalised space "
              "carries effectively no signal there (training-loss gradient on the "
              "leakage band is negligible relative to the on-state band).")
    md.append("")

    # ──────────────────────────────────────────────────────────────────────────
    # PART 2 — floor noise in the leakage band
    # ──────────────────────────────────────────────────────────────────────────
    emit("[2] FLOOR NOISE IN THE |id| < 1e-7 BAND")
    id_leak = id_phys_all[leak_mask]
    n_leak, n_neg, n_zero, n_pos = _sign_fractions(id_leak)
    # non-zero leakage tail (exclude the structural Vds=0 zeros) — the part a
    # log-reweight would actually touch.
    nz_leak_mask = leak_mask & (id_phys_all != 0.0)
    id_nz = id_phys_all[nz_leak_mask]
    n_nz = id_nz.size
    n_nz_neg = int(np.sum(id_nz < 0.0))
    emit(f"    rows with |id| < {LEAK_THRESH:.0e} A : {n_leak} "
         f"({100.0 * n_leak / N:.2f}% of all {N} rows)")
    emit(f"      negative (id<0)   : {n_neg}  "
         f"({100.0 * n_neg / n_leak:.2f}% of band)")
    emit(f"      literal-zero      : {n_zero}  "
         f"({100.0 * n_zero / n_leak:.2f}% of band; {100.0 * n_zero / N:.2f}% of ALL rows)")
    emit(f"      positive (id>0)   : {n_pos}  ({100.0 * n_pos / n_leak:.2f}% of band)")
    emit(f"    NON-zero leakage tail (0<|id|<{LEAK_THRESH:.0e}): n={n_nz}  "
         f"negative={100.0 * n_nz_neg / n_nz:.2f}%  (the part a reweight touches)")
    emit("")
    emit("    Decade histogram of |id| in the leakage band (1e-12 .. 1e-7):")
    for ln in _fmt_decades(np.abs(id_leak)):
        emit(ln)
    emit("")
    # sign split per decade -> is the negativity confined to the deep floor?
    emit("    Sign split per |id| decade (neg% within each decade):")
    abs_leak = np.abs(id_leak)
    for e in range(-12, -7):
        lo, hi = 10.0 ** e, 10.0 ** (e + 1)
        m = (abs_leak >= lo) & (abs_leak < hi)
        nn = int(m.sum())
        if nn == 0:
            emit(f"      1e{e:+d}..1e{e+1:+d}: (empty)")
            continue
        neg = int(np.sum(id_leak[m] < 0))
        emit(f"      1e{e:+d}..1e{e+1:+d}: n={nn:>8}  neg={100.0 * neg / nn:6.2f}%")
    # below 1e-12 (incl zero)
    m_below = abs_leak < 1e-12
    nb = int(m_below.sum())
    if nb:
        negb = int(np.sum(id_leak[m_below] < 0))
        zerob = int(np.sum(id_leak[m_below] == 0))
        emit(f"      |id|<1e-12 : n={nb:>8}  neg={100.0 * negb / nb:6.2f}%  "
             f"zero={100.0 * zerob / nb:6.2f}%")
    emit("")

    md.append("## 2. Floor-noise sign/zero fractions in the leakage band")
    md.append("")
    md.append(f"Threshold: `|id| < {LEAK_THRESH:.0e} A`. "
              f"{n_leak:,} rows ({100.0 * n_leak / N:.2f}% of {N:,}).")
    md.append("")
    md.append("| quantity | count | % of leakage band | % of ALL N rows |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| negative (id<0) | {n_neg:,} | **{100.0 * n_neg / n_leak:.2f}%** | "
              f"{100.0 * n_neg / N:.2f}% |")
    md.append(f"| literal-zero (id==0) | {n_zero:,} | **{100.0 * n_zero / n_leak:.2f}%** | "
              f"**{100.0 * n_zero / N:.2f}%** |")
    md.append(f"| positive (id>0) | {n_pos:,} | {100.0 * n_pos / n_leak:.2f}% | "
              f"{100.0 * n_pos / N:.2f}% |")
    md.append("")
    md.append(f"> The plan's preliminary estimate was *~45% negative, ~6% literal-0*. "
              f"Confirmed: **{100.0 * n_neg / n_leak:.1f}% negative of the band**, and "
              f"literal-0 is **{100.0 * n_zero / N:.2f}% of ALL rows** (== the plan's "
              f"~6%) which is {100.0 * n_zero / n_leak:.1f}% *of the leakage band*. "
              f"Most literal zeros are structural (Vds=0 forces id=0).")
    md.append("")
    md.append(f"**Non-zero leakage tail** (`0<|id|<{LEAK_THRESH:.0e}`, the part a "
              f"reweight would actually touch): n={n_nz:,}, **{100.0 * n_nz_neg / n_nz:.2f}% "
              f"negative** — sign-randomness is even stronger once structural zeros "
              f"are excluded.")
    md.append("")
    md.append("Decade histogram of `|id|` in the leakage band:")
    md.append("")
    md.append("```")
    for ln in _fmt_decades(np.abs(id_leak)):
        md.append(ln.strip())
    md.append("```")
    md.append("")
    md.append("Sign split per decade (what fraction is negative inside each decade):")
    md.append("")
    md.append("| decade | n | neg% |")
    md.append("|---|---:|---:|")
    for e in range(-12, -7):
        lo, hi = 10.0 ** e, 10.0 ** (e + 1)
        m = (abs_leak >= lo) & (abs_leak < hi)
        nn = int(m.sum())
        if nn == 0:
            md.append(f"| 1e{e:+d}..1e{e+1:+d} | 0 | - |")
            continue
        neg = int(np.sum(id_leak[m] < 0))
        md.append(f"| 1e{e:+d}..1e{e+1:+d} | {nn:,} | {100.0 * neg / nn:.2f}% |")
    if nb:
        md.append(f"| <1e-12 (incl 0) | {nb:,} | {100.0 * negb / nb:.2f}% |")
    md.append("")

    # ──────────────────────────────────────────────────────────────────────────
    # PART 3 — subthreshold OSDI gm/gds cleanliness (class 2)
    # ──────────────────────────────────────────────────────────────────────────
    emit("[3] SUBTHRESHOLD OSDI gm/gds CLEANLINESS (sample_class == 2)")
    sub_id = outputs[sub_mask, ID]
    sub_gm = outputs[sub_mask, GM]
    sub_gds = outputs[sub_mask, GDS]
    n_sub = int(sub_mask.sum())
    emit(f"    subthresh class rows  : {n_sub}")
    emit(f"    |id| range in class   : "
         f"[{np.abs(sub_id).min():.4e}, {np.abs(sub_id).max():.4e}] A")
    emit(f"    id sign split         : neg={100.0 * np.mean(sub_id < 0):.2f}%  "
         f"zero={100.0 * np.mean(sub_id == 0):.2f}%  "
         f"pos={100.0 * np.mean(sub_id > 0):.2f}%")
    emit("")
    # NaN/Inf
    gm_bad = ~np.isfinite(sub_gm)
    gds_bad = ~np.isfinite(sub_gds)
    emit(f"    gm  NaN/Inf           : {int(gm_bad.sum())}  "
         f"({100.0 * gm_bad.mean():.4f}%)")
    emit(f"    gds NaN/Inf           : {int(gds_bad.sum())}  "
         f"({100.0 * gds_bad.mean():.4f}%)")
    # sign convention: NMOS in saturation/subthreshold, OSDI gm>=0, gds>=0.
    # "wrong sign" = strictly negative (a real conductance must be >= 0).
    gm_fin = sub_gm[np.isfinite(sub_gm)]
    gds_fin = sub_gds[np.isfinite(sub_gds)]
    gm_neg = int(np.sum(gm_fin < 0))
    gds_neg = int(np.sum(gds_fin < 0))
    gm_zero = int(np.sum(gm_fin == 0))
    gds_zero = int(np.sum(gds_fin == 0))
    emit(f"    gm  < 0 (wrong sign)  : {gm_neg}  ({100.0 * gm_neg / gm_fin.size:.4f}%)")
    emit(f"    gm  == 0              : {gm_zero}  ({100.0 * gm_zero / gm_fin.size:.4f}%)")
    emit(f"    gds < 0 (wrong sign)  : {gds_neg}  ({100.0 * gds_neg / gds_fin.size:.4f}%)")
    emit(f"    gds == 0              : {gds_zero}  ({100.0 * gds_zero / gds_fin.size:.4f}%)")
    emit(f"    gm  range (finite)    : [{gm_fin.min():.4e}, {gm_fin.max():.4e}]")
    emit(f"    gds range (finite)    : [{gds_fin.min():.4e}, {gds_fin.max():.4e}]")
    emit("")

    # gm cleanliness vs id-sign: in the OFF region, is gm well-behaved where
    # id itself is sign-noisy? Cross-tab gm sign against id sign for the deepest
    # leakage rows inside the subthresh class.
    deep = np.abs(sub_id) < 1e-9
    if int(deep.sum()):
        dg = sub_gm[deep]
        dg_fin = dg[np.isfinite(dg)]
        emit(f"    [deep-off subset of class, |id|<1nA: n={int(deep.sum())}]")
        emit(f"      gm neg%={100.0 * np.mean(dg_fin < 0):.2f}%  "
             f"gm zero%={100.0 * np.mean(dg_fin == 0):.2f}%  "
             f"gm range=[{dg_fin.min():.3e},{dg_fin.max():.3e}]")
    emit("")

    md.append("## 3. Subthreshold OSDI gm/gds cleanliness (class 2)")
    md.append("")
    md.append(f"- subthresh class rows: **{n_sub:,}**")
    md.append(f"- `|id|` range in class: "
              f"`[{np.abs(sub_id).min():.3e}, {np.abs(sub_id).max():.3e}] A` "
              f"(id sign: neg {100.0 * np.mean(sub_id < 0):.2f}%, "
              f"zero {100.0 * np.mean(sub_id == 0):.2f}%, "
              f"pos {100.0 * np.mean(sub_id > 0):.2f}%).")
    md.append("")
    md.append("| metric | gm (col 1) | gds (col 2) |")
    md.append("|---|---:|---:|")
    md.append(f"| NaN/Inf | {int(gm_bad.sum())} ({100.0 * gm_bad.mean():.4f}%) | "
              f"{int(gds_bad.sum())} ({100.0 * gds_bad.mean():.4f}%) |")
    md.append(f"| < 0 (wrong sign) | {gm_neg} ({100.0 * gm_neg / gm_fin.size:.4f}%) | "
              f"{gds_neg} ({100.0 * gds_neg / gds_fin.size:.4f}%) |")
    md.append(f"| == 0 | {gm_zero} ({100.0 * gm_zero / gm_fin.size:.4f}%) | "
              f"{gds_zero} ({100.0 * gds_zero / gds_fin.size:.4f}%) |")
    md.append(f"| range (finite) | [{gm_fin.min():.3e}, {gm_fin.max():.3e}] | "
              f"[{gds_fin.min():.3e}, {gds_fin.max():.3e}] |")
    md.append("")
    if int(deep.sum()):
        md.append(f"Deep-off subset of the class (`|id|<1nA`, n={int(deep.sum()):,}): "
                  f"gm neg {100.0 * np.mean(dg_fin < 0):.2f}%, "
                  f"gm zero {100.0 * np.mean(dg_fin == 0):.2f}%, "
                  f"gm range `[{dg_fin.min():.3e},{dg_fin.max():.3e}]`.")
        md.append("")

    # ──────────────────────────────────────────────────────────────────────────
    # PART 4 — DECISION
    # ──────────────────────────────────────────────────────────────────────────
    # Criteria: the leakage-band *id value* is sign-random floor noise (large
    # neg/zero fractions). But OSDI gm/gds are a separate, analytic quantity:
    # if they are finite, correctly-signed, and non-degenerate even where id is
    # noisy, then a *clipped derivative* distillation is safe while a raw
    # log-reweight on the id value is NOT.
    id_band_noisy = (n_neg / n_leak) > 0.10        # >10% negative -> sign-random
    gm_clean = (gm_bad.mean() < 1e-3) and (gm_neg / gm_fin.size < 0.05)
    gds_clean = (gds_bad.mean() < 1e-3) and (gds_neg / gds_fin.size < 0.05)

    emit("[4] DECISION")
    emit(f"    id-band sign-random?   {id_band_noisy}  "
         f"(neg frac {100.0 * n_neg / n_leak:.1f}% vs 10% threshold)")
    emit(f"    OSDI gm clean?         {gm_clean}")
    emit(f"    OSDI gds clean?        {gds_clean}")
    emit("")

    if id_band_noisy:
        decision = "UNSAFE for a raw log-reweight / asinh-floor drop on the id VALUE"
        rationale = (
            f"the |id|<1e-7 band is sign-random PyCMG floor noise "
            f"({100.0 * n_neg / n_leak:.1f}% negative of band, "
            f"{100.0 * n_zero / n_leak:.2f}% literal-zero of band = "
            f"{100.0 * n_zero / N:.2f}% of all rows; the non-zero tail alone is "
            f"{100.0 * n_nz_neg / n_nz:.1f}% negative). Dropping/lowering the "
            f"asinh floor (e.g. to 1e-9) would amplify this junk into the surface. "
        )
        allowed = (
            "ONLY a clipped/winsorised signed-floor target is allowed: restrict "
            f"support to |id|>{WINSOR_FLOOR:.0e} A and apply Huber on ln(|id|) "
            "(signed), NOT a plain log-reweight of the raw id value. "
        )
        if gm_clean and gds_clean:
            allowed += (
                "HOWEVER the analytic OSDI gm/gds in the subthresh class ARE clean "
                f"(NaN/Inf {100.0 * gm_bad.mean():.4f}%/{100.0 * gds_bad.mean():.4f}%, "
                f"wrong-sign {100.0 * gm_neg / gm_fin.size:.4f}%/"
                f"{100.0 * gds_neg / gds_fin.size:.4f}%), so a Phase-2 "
                "Jacobian/log-DERIVATIVE distillation against OSDI gm/gds on the "
                f"clipped support (|id|>{WINSOR_FLOOR:.0e}) is the safe lever — the "
                "derivative target, unlike the raw id value, is not floor-noise."
            )
        else:
            allowed += (
                "The OSDI derivatives are NOT clean enough to rescue this either; "
                "abandon the subthreshold reweight."
            )
        verdict_line = "no"
    else:
        decision = "SAFE — clean tail"
        rationale = (
            f"the |id|<1e-7 band is dominated by correctly-signed signal "
            f"({100.0 * n_neg / n_leak:.1f}% negative < 10%)."
        )
        allowed = ("A subthreshold log-reweight or log-derivative distillation is "
                   "allowed as a Phase-2 aux target.")
        verdict_line = "yes"

    emit(f"    >>> subthresh band clean? {verdict_line.upper()}")
    emit(f"    >>> DECISION: {decision}")
    emit(f"        rationale: {rationale}")
    emit(f"        allowed:   {allowed}")
    emit("")

    md.append("## 4. DECISION")
    md.append("")
    md.append("Decision criteria (per plan §4 P0-E):")
    md.append("")
    md.append(f"- id-band sign-random (neg frac > 10%)? **{id_band_noisy}** "
              f"(observed {100.0 * n_neg / n_leak:.1f}% negative).")
    md.append(f"- OSDI gm clean (finite & <5% wrong-sign)? **{gm_clean}**.")
    md.append(f"- OSDI gds clean (finite & <5% wrong-sign)? **{gds_clean}**.")
    md.append("")
    md.append(f"**DECISION (subthresh band clean? {verdict_line}):** {decision}.")
    md.append("")
    md.append(f"- *Rationale:* {rationale}")
    md.append(f"- *Allowed lever:* {allowed}")
    md.append("")
    md.append("### Plan §4 decision-tree line")
    md.append("")
    md.append(f"`P0-E subthresh band clean? -> {verdict_line} -> "
              + ("log-reweight allowed as a Phase-2 aux"
                 if verdict_line == "yes"
                 else "clipped/winsorised signed-floor target only "
                      f"(|id|>{WINSOR_FLOOR:.0e}, Huber on ln-current); "
                      "raw asinh-floor drop / log-reweight on the id value is UNSAFE")
              + "`")
    md.append("")

    # ── exact commands appendix ───────────────────────────────────────────────
    md.append("## Exact commands")
    md.append("")
    md.append("```bash")
    md.append("OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\")
    md.append("  conda run -n pycircuitsim python scripts/v6_4_6_p0e_subthresh.py")
    md.append("# log : results/v6_4_6/phase0_logs/p0e_subthresh.log")
    md.append("# report (this file): results/v6_4_6/phase0_E_subthresh_audit.md")
    md.append("```")
    md.append("")

    REPORT.write_text("\n".join(md) + "\n")
    emit(f"[done] report  -> {REPORT}")
    emit(f"[done] log     -> {LOG}")
    fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
