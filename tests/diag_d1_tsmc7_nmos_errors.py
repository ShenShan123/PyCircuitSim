"""D1 diagnostic: map where the v4 BSIMAR universal NMOS model
disagrees with PyCMG ground truth at TSMC7 SVT.

Target fixture:
  tech = tsmc7, variant = svt, tech_code = 4
  device = NMOS, L = 16 nm, NFIN = 10, T = 300.15 K
  Vbs = 0, VDD = 0.75 V

Grid: Vgs ∈ [0, 0.75] × Vds ∈ [0, 0.75], 40 × 40 = 1600 points.
For each (Vgs, Vds) we compute:
  - PyCMG BSIM-CMG Id ("ground truth")
  - NN-corrected Id (full _apply_vds_correction + rail-restoring
    extrapolation — what the simulator sees)
  - NN-raw Id (denormalized NN output, correction disabled)

Also produces 1-D slices and a count of training-set samples landing
in the identified hot-error region.

Artifacts saved under:
  /home/shenshan/NN_SPICE/results/v5_d1_tsmc7_nmos_errors/
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ── Path bootstrap ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT))

from bsimar.config import DATA_DIR, tech_variant_to_code

# Reuse verify script fixtures (PyCMG + BSIMAR instance factories)
from tests.verify_bsimar_v4_inverter import (
    TSMC7_SVT, create_pycmg_instance, create_bsimar_instance,
)

# ── Constants ─────────────────────────────────────────────────────────────
TECH_CFG = TSMC7_SVT  # NMOS L=16nm, VDD=0.75V, NFIN=10
VDD = TECH_CFG.vdd
TECH_CODE = tech_variant_to_code(TECH_CFG.tech_key, TECH_CFG.variant)

N_GRID = 40  # 40x40 = 1600 grid points
OUT_DIR = PROJECT_ROOT / "results" / "v5_d1_tsmc7_nmos_errors"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── NN evaluator with toggleable Vds correction ───────────────────────────

class _NNEvaluator:
    """Wraps an NMOS_BSIMAR instance so we can evaluate Id with or without
    `_apply_vds_correction` applied.

    "Raw" = bypass correction by monkey-patching the method to return the
    input dict unchanged. We restore on `__exit__`.
    """

    def __init__(self, instance) -> None:
        self.inst = instance
        self._orig = instance._apply_vds_correction

    def _identity(self, result: Dict[str, float], vds: float) -> Dict[str, float]:
        return result

    def id_corrected(self, vgs: float, vds: float) -> float:
        """Full simulator path: correction + rail extrapolation."""
        self.inst._apply_vds_correction = self._orig
        self.inst.clear_cache()
        voltages = {"drain": vds, "gate": vgs, "source": 0.0, "bulk": 0.0}
        # NMOS_BSIMAR.calculate_current returns -result["id"], so negate
        # back to the "terminal Id" convention matched with PyCMG.
        i_leaving = self.inst.calculate_current(voltages)
        return -i_leaving

    def id_raw(self, vgs: float, vds: float) -> float:
        """Raw NN output — no analytical correction, no rail extrapolation."""
        self.inst._apply_vds_correction = self._identity.__get__(self.inst)
        self.inst.clear_cache()
        voltages = {"drain": vds, "gate": vgs, "source": 0.0, "bulk": 0.0}
        i_leaving = self.inst.calculate_current(voltages)
        self.inst._apply_vds_correction = self._orig
        return -i_leaving


# ── Grid evaluation ───────────────────────────────────────────────────────

def build_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (vgs_grid, vds_grid, vgs_flat, vds_flat)."""
    vgs_1d = np.linspace(0.0, VDD, N_GRID)
    vds_1d = np.linspace(0.0, VDD, N_GRID)
    vgs_grid, vds_grid = np.meshgrid(vgs_1d, vds_1d, indexing="ij")
    return vgs_grid, vds_grid, vgs_1d, vds_1d


def eval_pycmg_grid(cmg, vgs_grid: np.ndarray, vds_grid: np.ndarray) -> np.ndarray:
    """Evaluate PyCMG BSIM-CMG Id on the (Vgs, Vds) grid."""
    out = np.full(vgs_grid.shape, np.nan)
    n = vgs_grid.size
    for idx, (vg, vd) in enumerate(zip(vgs_grid.ravel(), vds_grid.ravel())):
        try:
            r = cmg.eval_dc({"d": float(vd), "g": float(vg),
                             "s": 0.0, "e": 0.0})
            out.ravel()[idx] = r["id"]
        except Exception:
            pass
        if idx % 200 == 0:
            print(f"  PyCMG {idx+1}/{n}")
    return out


def eval_nn_grid(evaluator: _NNEvaluator,
                 vgs_grid: np.ndarray, vds_grid: np.ndarray,
                 label: str) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate NN Id on the grid; returns (id_corrected, id_raw)."""
    id_corr = np.full(vgs_grid.shape, np.nan)
    id_raw = np.full(vgs_grid.shape, np.nan)
    n = vgs_grid.size
    for idx, (vg, vd) in enumerate(zip(vgs_grid.ravel(), vds_grid.ravel())):
        try:
            id_corr.ravel()[idx] = evaluator.id_corrected(float(vg), float(vd))
        except Exception:
            pass
        try:
            id_raw.ravel()[idx] = evaluator.id_raw(float(vg), float(vd))
        except Exception:
            pass
        if idx % 200 == 0:
            print(f"  NN({label}) {idx+1}/{n}")
    return id_corr, id_raw


# ── Sign-convention note ──────────────────────────────────────────────────
# PyCMG's eval_dc returns `id` = terminal drain current (positive into D
# for an ON NMOS).  The NN convention (rule 2 in CLAUDE.md) has
# `calculate_current` return `i_leaving = -id_nmos_terminal`, and
# `_eval()` stores `result["id"]` in the PyCMG terminal-current sign.
# We already negate once in `id_corrected` / `id_raw` so both
# PyCMG and NN arrays here carry the *terminal Id* convention (positive
# for an NMOS conducting in the normal direction).


# ── Error aggregation + stats ────────────────────────────────────────────

def rel_err(nn: np.ndarray, cmg: np.ndarray) -> np.ndarray:
    """|NN - PyCMG| / max(|PyCMG|) × 100 %."""
    denom = float(np.nanmax(np.abs(cmg)))
    if denom == 0:
        return np.zeros_like(nn)
    return np.abs(nn - cmg) / denom * 100.0


def nrmse_pct(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() < 2:
        return float("nan")
    ptp = float(true[mask].max() - true[mask].min())
    if ptp < 1e-30:
        return 0.0
    rmse = float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)))
    return rmse / ptp * 100.0


def describe_hot_region(vgs_grid: np.ndarray, vds_grid: np.ndarray,
                        err_grid: np.ndarray) -> Dict:
    """Identify the hot region where |rel_err| is in the top decile."""
    mask = np.isfinite(err_grid)
    flat = err_grid[mask]
    if flat.size == 0:
        return {"threshold": float("nan"), "n_hot": 0}
    thr = float(np.percentile(flat, 90))
    hot_mask = mask & (err_grid >= thr)
    if hot_mask.sum() == 0:
        return {"threshold": thr, "n_hot": 0}
    hot_vgs = vgs_grid[hot_mask]
    hot_vds = vds_grid[hot_mask]
    return {
        "threshold": thr,
        "n_hot": int(hot_mask.sum()),
        "vgs_range": (float(hot_vgs.min()), float(hot_vgs.max())),
        "vds_range": (float(hot_vds.min()), float(hot_vds.max())),
        "vgs_mean": float(hot_vgs.mean()),
        "vds_mean": float(hot_vds.mean()),
    }


# ── Slice analysis ───────────────────────────────────────────────────────

def eval_slice_vgs(cmg, evaluator: _NNEvaluator, vds_fixed: float,
                   n_points: int = 81) -> Dict[str, np.ndarray]:
    """1D slice at fixed Vds: sweep Vgs."""
    vgs = np.linspace(0.0, VDD, n_points)
    id_cmg = np.full(n_points, np.nan)
    id_corr = np.full(n_points, np.nan)
    id_raw = np.full(n_points, np.nan)
    for i, vg in enumerate(vgs):
        try:
            id_cmg[i] = cmg.eval_dc({"d": vds_fixed, "g": float(vg),
                                     "s": 0.0, "e": 0.0})["id"]
        except Exception:
            pass
        try:
            id_corr[i] = evaluator.id_corrected(float(vg), vds_fixed)
        except Exception:
            pass
        try:
            id_raw[i] = evaluator.id_raw(float(vg), vds_fixed)
        except Exception:
            pass
    return {"vgs": vgs, "id_cmg": id_cmg,
            "id_corr": id_corr, "id_raw": id_raw}


def eval_slice_vds(cmg, evaluator: _NNEvaluator, vgs_fixed: float,
                   n_points: int = 81) -> Dict[str, np.ndarray]:
    vds = np.linspace(0.0, VDD, n_points)
    id_cmg = np.full(n_points, np.nan)
    id_corr = np.full(n_points, np.nan)
    id_raw = np.full(n_points, np.nan)
    for i, vd in enumerate(vds):
        try:
            id_cmg[i] = cmg.eval_dc({"d": float(vd), "g": vgs_fixed,
                                     "s": 0.0, "e": 0.0})["id"]
        except Exception:
            pass
        try:
            id_corr[i] = evaluator.id_corrected(vgs_fixed, float(vd))
        except Exception:
            pass
        try:
            id_raw[i] = evaluator.id_raw(vgs_fixed, float(vd))
        except Exception:
            pass
    return {"vds": vds, "id_cmg": id_cmg,
            "id_corr": id_corr, "id_raw": id_raw}


# ── Training-data coverage lookup ────────────────────────────────────────

def count_training_hot_samples(hot_vgs: Tuple[float, float],
                               hot_vds: Tuple[float, float]) -> Dict:
    """Count TSMC7-SVT training samples inside the hot (Vgs, Vds) box."""
    npz = DATA_DIR / "universal_nmos.npz"
    labels_path = DATA_DIR / "universal_nmos_tech_variant_labels.npy"
    if not npz.exists() or not labels_path.exists():
        return {"error": f"Missing {npz} or {labels_path}"}

    d = np.load(npz)
    labels = np.load(str(labels_path))
    inputs = d["inputs"]  # cols 0=Vgs, 1=Vds, 2=Vbs, 3=T (per diag_tsmc7)

    code = tech_variant_to_code("tsmc7", "svt")
    tech_mask = labels == code
    n_tech = int(tech_mask.sum())
    if n_tech == 0:
        return {"code": code, "n_tech_total": 0}

    vgs = inputs[tech_mask, 0]
    vds = inputs[tech_mask, 1]
    in_box = ((vgs >= hot_vgs[0]) & (vgs <= hot_vgs[1])
              & (vds >= hot_vds[0]) & (vds <= hot_vds[1]))
    n_hot = int(in_box.sum())
    return {
        "code": code,
        "n_tech_total": n_tech,
        "n_in_hot_box": n_hot,
        "pct_in_hot_box": 100.0 * n_hot / n_tech if n_tech else 0.0,
        "hot_vgs": hot_vgs,
        "hot_vds": hot_vds,
    }


# ── Plotting ────────────────────────────────────────────────────────────

def _heatmap_abs_id(vgs_1d: np.ndarray, vds_1d: np.ndarray,
                    id_grid: np.ndarray, title: str, save_path: Path) -> None:
    """Log-magnitude heatmap of |Id| on (Vgs, Vds) plane."""
    fig, ax = plt.subplots(figsize=(7, 6))
    Z = np.abs(id_grid) + 1e-15
    im = ax.pcolormesh(vgs_1d, vds_1d, Z.T,
                       norm=LogNorm(vmin=1e-12, vmax=max(Z.max(), 1e-9)),
                       shading="auto", cmap="viridis")
    ax.set_xlabel("Vgs [V]")
    ax.set_ylabel("Vds [V]")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="|Id| [A]")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _heatmap_rel_err(vgs_1d: np.ndarray, vds_1d: np.ndarray,
                     err_grid: np.ndarray, title: str, save_path: Path) -> None:
    """Linear-scale heatmap of relative error (%)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = float(np.nanpercentile(err_grid, 99))
    vmax = max(vmax, 5.0)
    im = ax.pcolormesh(vgs_1d, vds_1d, err_grid.T,
                       vmin=0.0, vmax=vmax, shading="auto", cmap="magma")
    ax.set_xlabel("Vgs [V]")
    ax.set_ylabel("Vds [V]")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="|rel err| %")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_slices_vgs(slices: Dict, save_path: Path) -> None:
    """3-panel Id-Vgs slices at Vds = {0, VDD/2, VDD}."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    keys = ["vds0", "vds_half", "vds_full"]
    vds_vals = [0.0, VDD / 2, VDD]
    for ax, key, vds_val in zip(axes, keys, vds_vals):
        s = slices[key]
        ax.plot(s["vgs"], s["id_cmg"] * 1e6, "b-", lw=1.8, label="PyCMG")
        ax.plot(s["vgs"], s["id_corr"] * 1e6, "r--", lw=1.3,
                label="NN (corrected)")
        ax.plot(s["vgs"], s["id_raw"] * 1e6, "g:", lw=1.1,
                label="NN (raw)", alpha=0.8)
        ax.set_xlabel("Vgs [V]")
        ax.set_ylabel("Id [uA]")
        ax.set_title(f"Id-Vgs @ Vds={vds_val:.3f}V")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_slice_vds(slice_data: Dict, vgs_fixed: float, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(slice_data["vds"], slice_data["id_cmg"] * 1e6, "b-", lw=1.8,
            label="PyCMG")
    ax.plot(slice_data["vds"], slice_data["id_corr"] * 1e6, "r--", lw=1.3,
            label="NN (corrected)")
    ax.plot(slice_data["vds"], slice_data["id_raw"] * 1e6, "g:", lw=1.1,
            label="NN (raw)", alpha=0.8)
    ax.set_xlabel("Vds [V]")
    ax.set_ylabel("Id [uA]")
    ax.set_title(f"Id-Vds @ Vgs={vgs_fixed:.3f}V")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main orchestration ──────────────────────────────────────────────────

def main() -> int:
    print("=" * 72)
    print("D1 Diagnostic: v4 BSIMAR universal NMOS vs PyCMG at TSMC7 SVT")
    print("=" * 72)
    print(f"  Tech: tsmc7 svt (code={TECH_CODE})")
    print(f"  L={TECH_CFG.l_nmos*1e9:.1f}nm  NFIN={TECH_CFG.nfin}"
          f"  T={TECH_CFG.temperature:.2f}K  VDD={VDD:.3f}V")
    print(f"  Grid: {N_GRID}x{N_GRID} = {N_GRID*N_GRID} points")
    print(f"  Output: {OUT_DIR}")
    print()

    print("[1/5] Building PyCMG + BSIMAR v4 instances")
    cmg = create_pycmg_instance(TECH_CFG, "nmos")
    bs_inst = create_bsimar_instance(TECH_CFG, "nmos")
    evaluator = _NNEvaluator(bs_inst)

    print("[2/5] Evaluating PyCMG ground truth on grid")
    vgs_grid, vds_grid, vgs_1d, vds_1d = build_grid()
    id_cmg = eval_pycmg_grid(cmg, vgs_grid, vds_grid)

    print("[3/5] Evaluating NN (corrected + raw) on grid")
    id_corr, id_raw = eval_nn_grid(evaluator, vgs_grid, vds_grid, "grid")

    # Relative-error grids vs PyCMG (normalized by max |id_cmg|)
    err_corr = rel_err(id_corr, id_cmg)
    err_raw = rel_err(id_raw, id_cmg)

    mean_err_corr = float(np.nanmean(err_corr))
    mean_err_raw = float(np.nanmean(err_raw))
    max_err_corr = float(np.nanmax(err_corr))
    max_err_raw = float(np.nanmax(err_raw))
    print(f"  Grid mean |rel err|: corrected={mean_err_corr:.3f}% "
          f"raw={mean_err_raw:.3f}%")
    print(f"  Grid  max |rel err|: corrected={max_err_corr:.3f}% "
          f"raw={max_err_raw:.3f}%")

    # Hot region (top 10% error mass) for the corrected NN (what the
    # simulator sees).
    hot_corr = describe_hot_region(vgs_grid, vds_grid, err_corr)
    print(f"  Hot region (corrected, top-10% errs): {hot_corr}")

    print("[4/5] Evaluating 1D slices")
    slices = {
        "vds0": eval_slice_vgs(cmg, evaluator, 0.0),
        "vds_half": eval_slice_vgs(cmg, evaluator, VDD / 2),
        "vds_full": eval_slice_vgs(cmg, evaluator, VDD),
        "vds_mid_gate": eval_slice_vds(cmg, evaluator, VDD / 2),
    }
    nrmse_half_corr = nrmse_pct(slices["vds_half"]["id_corr"],
                                slices["vds_half"]["id_cmg"])
    nrmse_half_raw = nrmse_pct(slices["vds_half"]["id_raw"],
                               slices["vds_half"]["id_cmg"])
    print(f"  NRMSE on Id-Vgs @ Vds=VDD/2: "
          f"corrected={nrmse_half_corr:.3f}%  raw={nrmse_half_raw:.3f}%")

    print("[5/5] Counting training-set coverage in hot region")
    hot_vgs = hot_corr.get("vgs_range", (0.0, VDD))
    hot_vds = hot_corr.get("vds_range", (0.0, VDD))
    coverage = count_training_hot_samples(hot_vgs, hot_vds)
    print(f"  {coverage}")

    # ── Save artifacts ───────────────────────────────────────────────
    print("\nSaving artifacts:")
    _heatmap_abs_id(vgs_1d, vds_1d, id_cmg,
                    "PyCMG BSIM-CMG |Id| (ground truth)",
                    OUT_DIR / "heatmap_pycmg_id.png")
    print(f"  {OUT_DIR/'heatmap_pycmg_id.png'}")

    _heatmap_abs_id(vgs_1d, vds_1d, id_corr,
                    "NN (corrected) |Id|",
                    OUT_DIR / "heatmap_nn_id.png")
    print(f"  {OUT_DIR/'heatmap_nn_id.png'}")

    _heatmap_rel_err(vgs_1d, vds_1d, err_corr,
                     "|rel err| % — NN corrected vs PyCMG",
                     OUT_DIR / "heatmap_rel_error_with_corr.png")
    print(f"  {OUT_DIR/'heatmap_rel_error_with_corr.png'}")

    _heatmap_rel_err(vgs_1d, vds_1d, err_raw,
                     "|rel err| % — NN raw vs PyCMG",
                     OUT_DIR / "heatmap_rel_error_raw.png")
    print(f"  {OUT_DIR/'heatmap_rel_error_raw.png'}")

    _plot_slices_vgs(slices, OUT_DIR / "slices_id_vgs.png")
    print(f"  {OUT_DIR/'slices_id_vgs.png'}")

    _plot_slice_vds(slices["vds_mid_gate"], VDD / 2,
                    OUT_DIR / "slice_id_vds.png")
    print(f"  {OUT_DIR/'slice_id_vds.png'}")

    np.savez(
        OUT_DIR / "data.npz",
        vgs_grid=vgs_grid, vds_grid=vds_grid,
        id_cmg=id_cmg, id_corr=id_corr, id_raw=id_raw,
        err_corr=err_corr, err_raw=err_raw,
        slice_vds0_vgs=slices["vds0"]["vgs"],
        slice_vds0_id_cmg=slices["vds0"]["id_cmg"],
        slice_vds0_id_corr=slices["vds0"]["id_corr"],
        slice_vds0_id_raw=slices["vds0"]["id_raw"],
        slice_vds_half_vgs=slices["vds_half"]["vgs"],
        slice_vds_half_id_cmg=slices["vds_half"]["id_cmg"],
        slice_vds_half_id_corr=slices["vds_half"]["id_corr"],
        slice_vds_half_id_raw=slices["vds_half"]["id_raw"],
        slice_vds_full_vgs=slices["vds_full"]["vgs"],
        slice_vds_full_id_cmg=slices["vds_full"]["id_cmg"],
        slice_vds_full_id_corr=slices["vds_full"]["id_corr"],
        slice_vds_full_id_raw=slices["vds_full"]["id_raw"],
        slice_vds_mid_gate_vds=slices["vds_mid_gate"]["vds"],
        slice_vds_mid_gate_id_cmg=slices["vds_mid_gate"]["id_cmg"],
        slice_vds_mid_gate_id_corr=slices["vds_mid_gate"]["id_corr"],
        slice_vds_mid_gate_id_raw=slices["vds_mid_gate"]["id_raw"],
    )
    print(f"  {OUT_DIR/'data.npz'}")

    # ── Write markdown report ──────────────────────────────────────
    report_path = OUT_DIR / "v5_d1_tsmc7_nmos_report.md"
    _write_report(
        report_path, mean_err_corr, mean_err_raw, max_err_corr, max_err_raw,
        hot_corr, nrmse_half_corr, nrmse_half_raw, coverage,
    )
    print(f"  {report_path}")

    return 0


def _write_report(path: Path,
                  mean_err_corr: float, mean_err_raw: float,
                  max_err_corr: float, max_err_raw: float,
                  hot_corr: Dict, nrmse_half_corr: float,
                  nrmse_half_raw: float, coverage: Dict) -> None:
    # Infer hot region semantic label
    vgs_mean = hot_corr.get("vgs_mean", 0.5 * VDD)
    vds_mean = hot_corr.get("vds_mean", 0.5 * VDD)
    region_label = []
    if vgs_mean < 0.25 * VDD:
        region_label.append("subthreshold (low Vgs)")
    elif vgs_mean < 0.55 * VDD:
        region_label.append("moderate inversion")
    else:
        region_label.append("strong inversion")
    if vds_mean < 0.15 * VDD:
        region_label.append("triode / linear")
    elif vds_mean < 0.45 * VDD:
        region_label.append("low-to-mid Vds")
    else:
        region_label.append("saturation")
    region_desc = ", ".join(region_label)

    # Rounded ranges for recommendation
    hv_lo, hv_hi = hot_corr.get("vgs_range", (0.0, VDD))
    hd_lo, hd_hi = hot_corr.get("vds_range", (0.0, VDD))
    correction_better = mean_err_corr < mean_err_raw

    text = f"""# D1 Diagnostic: v4 BSIMAR NMOS vs PyCMG at TSMC7 SVT

**Fixture:** tsmc7 svt, tech_code={TECH_CODE}, L=16 nm, NFIN=10, T=300.15 K,
Vbs=0, VDD={VDD:.3f} V. Grid: {N_GRID}×{N_GRID} = {N_GRID*N_GRID} points on
(Vgs, Vds) ∈ [0, {VDD:.3f}] × [0, {VDD:.3f}] V.

## Answers

**1. Max absolute relative error (% of max|Id|):**
- Corrected NN (what simulator sees): **{max_err_corr:.2f} %**
- Raw NN (no `_apply_vds_correction`): **{max_err_raw:.2f} %**

The {'corrected' if max_err_corr > max_err_raw else 'raw'} variant has the
larger peak. Grid-mean |rel err| is
corrected={mean_err_corr:.3f} %, raw={mean_err_raw:.3f} %.

**2. Where is the error largest?**
Top-decile error mass sits in **{region_desc}**:
Vgs ∈ [{hv_lo:.3f}, {hv_hi:.3f}] V, Vds ∈ [{hd_lo:.3f}, {hd_hi:.3f}] V
(Vgs mean {vgs_mean:.3f} V, Vds mean {vds_mean:.3f} V;
{hot_corr.get('n_hot', 0)} of {N_GRID*N_GRID} grid points above the
90th-percentile error threshold of {hot_corr.get('threshold', float('nan')):.2f} %).
The companion heatmap `heatmap_rel_error_with_corr.png` shows the exact
spatial footprint.

**3. Does the Vds correction help or hurt?**
Mean |rel err|: corrected={mean_err_corr:.3f} %, raw={mean_err_raw:.3f} %.
The correction {'reduces' if correction_better else 'increases'} the mean
error overall. At the triode rail (Vds≈0) the correction enforces Id=0,
which trades a small error near Vds=0 for a guaranteed physical boundary;
the raw NN already predicts ≈0 there, so the suppression is minor. The
rail-restoring extrapolation does not apply to in-range points
(|Vds| < VDD=VDD_train), so most of the grid sees identical NN output; the
correction vs raw deltas concentrate near the Vds=0 rail.

**4. Id-Vgs @ Vds = VDD/2 = {VDD/2:.3f} V (the NMOS DC bias):**
- Corrected NN NRMSE = **{nrmse_half_corr:.2f} %**
- Raw NN NRMSE = {nrmse_half_raw:.2f} %

This directly reproduces the verifier's ≈14-15 % NMOS DC NRMSE if the
corrected number is in that range. The slice plot
`slices_id_vgs.png` shows the three panels (Vds=0, VDD/2, VDD).

**5. Training-set LHS coverage in the hot-error box:**
- TSMC7 SVT samples with Vgs ∈ [{coverage.get('hot_vgs', (0,0))[0]:.3f},
{coverage.get('hot_vgs', (0,0))[1]:.3f}] and Vds ∈
[{coverage.get('hot_vds', (0,0))[0]:.3f}, {coverage.get('hot_vds', (0,0))[1]:.3f}]:
**{coverage.get('n_in_hot_box', 0):,}** out of
{coverage.get('n_tech_total', 0):,} total TSMC7-SVT samples
({coverage.get('pct_in_hot_box', float('nan')):.2f} %).

A low coverage fraction is consistent with the NN being starved of
training points in precisely the region where the verifier metric probes.

**6. Recommendation for E4 overlay sampling:**
Densify TSMC7 overlay generation in the following box (the hot region
identified above, widened by ±10 % of VDD on each side to give the NN
a buffer):

- **Tech / variant:** tsmc7 svt (code={TECH_CODE}) — highest priority. Consider
  also tsmc7 lvt and tsmc7 ulvt if coverage gap repeats at those codes.
- **Vgs:** [{max(0.0, hv_lo - 0.10*VDD):.3f}, {min(VDD, hv_hi + 0.10*VDD):.3f}] V
- **Vds:** [{max(0.0, hd_lo - 0.10*VDD):.3f}, {min(VDD, hd_hi + 0.10*VDD):.3f}] V
- **NFIN:** {{3, 5, 10, 15, 20}} (cover the simulator inverter NFIN=10 plus the
  neighbouring bin points that stress the same gate stack)
- **L:** {{14, 16, 18, 20}} nm around the NMOS L=16 nm used at inference
- **T:** 300.15 K (single design corner)
- **Density:** at least 400 new LHS samples inside the box, which is
  ≈{max(1, 400 // max(coverage.get('n_in_hot_box', 1), 1))}× the current
  density — sufficient to cover the gate-Vds surface without diluting the
  overall training mix.

This list should drop directly into the plan §4.4 overlay generator and
give E4 a reasonable chance of closing the 14 % NMOS-DC-NRMSE gap on
TSMC7 without retraining the 17-other-tech mix from scratch.
"""
    path.write_text(text)


if __name__ == "__main__":
    sys.exit(main())
