"""Reproduce inverter DC + Transient simulations for TSMC12 / TSMC16 and
report a comprehensive metrics table for the V6.2.1 DirectNet checkpoints.

For each tech and each comparison (VTC, transient):
    - run NGSPICE BSIM-CMG ground truth
    - run PyCircuitSim DirectNet (LEVEL=73)
    - resample on a common grid and compute:
        NRMSE%, MRE% (excluding samples below an output-floor threshold),
        Max abs error, R², MAE, plus transient post-startup + region
        breakdown.

Outputs:
    results/v6_2_1_metrics_report/inverter_summary.csv
    results/v6_2_1_metrics_report/inverter_vtc_<tech>.csv      (trace data)
    results/v6_2_1_metrics_report/inverter_tran_<tech>.csv     (trace data)
    Plots: results/v6_2_1_metrics_report/inverter_<tech>_{vtc,tran}.png
    Updates results/v6_2_1_metrics_report/report.md with an Inverter section.
"""

from __future__ import annotations

import csv
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from tests.verify_nn_dc_tran import (  # noqa: E402
    ALL_TEST_TECHS,
    INV_CLOAD,
    INV_TRAN_TR,
    INV_TRAN_TF,
    INV_TRAN_TD,
    INV_TRAN_PW,
    INV_TRAN_TSTEP,
    INV_TRAN_TSTOP,
    TestTechConfig,
    compute_region_errors,
    run_ngspice_inverter_vtc,
    run_ngspice_inverter_tran,
    run_pycircuitsim_nn_inverter_vtc,
    run_pycircuitsim_nn_inverter_tran,
)

REPORT_DIR = PROJECT_ROOT / "results" / "v6_2_1_metrics_report"
TECHS = ("TSMC5", "TSMC7", "TSMC12", "TSMC16")
# Post-startup window: skip the first ~200 ps where the DC OP solver and
# the integrator settle.
POST_STARTUP_T = 2.0e-10
# MRE denominator floor: 0.5% of VDD. Below this the output is "rail" and
# the relative error reads astronomical even for sub-millivolt offsets.
MRE_VOUT_FLOOR_PCT_OF_VDD = 0.005


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _metrics_on_trace(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vdd: float,
    mre_floor_pct_of_vdd: float = MRE_VOUT_FLOOR_PCT_OF_VDD,
) -> Dict[str, float]:
    """Comprehensive metrics on an Vout trace pair (already on a common grid)."""
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    nrmse_pct = rmse / vdd * 100.0
    mae = float(np.mean(np.abs(diff)))
    max_err = float(np.max(np.abs(diff)))
    r2 = _r2(y_true, y_pred)

    # MRE uses a denominator floor so near-rail samples don't blow up.
    floor = vdd * mre_floor_pct_of_vdd
    mask = np.abs(y_true) > floor
    if mask.any():
        rel = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        mre_pct = float(np.mean(rel) * 100.0)
        max_mre_pct = float(np.max(rel) * 100.0)
    else:
        mre_pct = float("nan")
        max_mre_pct = float("nan")

    return {
        "n_points": int(len(y_true)),
        "NRMSE_vdd(%)": nrmse_pct,
        "MAE(V)": mae,
        "MaxErr(V)": max_err,
        "MaxErr_vdd(%)": max_err / vdd * 100.0,
        "MRE(%)": mre_pct,
        "MaxMRE(%)": max_mre_pct,
        "R2": r2,
    }


def _vtrip(sweep: np.ndarray, vout: np.ndarray, vdd: float) -> float:
    """Inverter trip point Vin where Vout=VDD/2 (linear interp)."""
    target = vdd * 0.5
    # vout is monotonically decreasing for a CMOS inverter; flip if needed.
    if vout[0] < vout[-1]:
        sweep_s, vout_s = sweep[::-1], vout[::-1]
    else:
        sweep_s, vout_s = sweep, vout
    # np.interp wants monotonically increasing x; here vout_s is decreasing.
    return float(np.interp(target, vout_s[::-1], sweep_s[::-1]))


def _save_trace_csv(
    path: Path, columns: Dict[str, np.ndarray],
) -> None:
    keys = list(columns.keys())
    n = len(next(iter(columns.values())))
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([f"{columns[k][i]:.6g}" for k in keys])


def _plot_vtc(
    tech: TestTechConfig,
    ng: Dict[str, np.ndarray],
    nn_interp: np.ndarray,
    metrics: Dict[str, float],
    path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
    ax1.plot(ng["sweep"], ng["vout"], "b-", lw=2, label="NGSPICE BSIM-CMG")
    ax1.plot(ng["sweep"], nn_interp, "r--", lw=1.4,
             label=f"DirectNet V6.2.1 (NRMSE={metrics['NRMSE_vdd(%)']:.2f}%)")
    ax1.set_ylabel("V(out) [V]")
    ax1.set_title(
        f"Inverter VTC — {tech.name} "
        f"L_n={tech.effective_inv_l_nmos*1e9:.0f}nm "
        f"L_p={tech.effective_inv_l_pmos*1e9:.0f}nm "
        f"NFIN={tech.effective_inv_nfin} VDD={tech.vdd}V")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.plot(ng["sweep"], (nn_interp - ng["vout"]) * 1e3, "r-", lw=1.2)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("V(in) [V]")
    ax2.set_ylabel("Error [mV]")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_tran(
    tech: TestTechConfig,
    ng: Dict[str, np.ndarray],
    nn: Dict[str, np.ndarray],
    metrics_full: Dict[str, float],
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(9, 7),
        gridspec_kw={"height_ratios": [0.5, 1.2, 0.6]}, sharex=True)
    axes[0].plot(ng["time"] * 1e9, ng["v(in)"], "b-", lw=1.2)
    axes[0].set_ylabel("V(in) [V]")
    axes[0].set_title(
        f"Inverter Transient — {tech.name} "
        f"NFIN={tech.effective_inv_nfin} Cload={INV_CLOAD*1e15:.0f}fF")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ng["time"] * 1e9, ng["v(out)"], "b-", lw=2,
                 label="NGSPICE BSIM-CMG")
    axes[1].plot(nn["time"] * 1e9, nn["v(out)"], "r--", lw=1.4,
                 label=f"DirectNet V6.2.1 "
                       f"(NRMSE={metrics_full['NRMSE_vdd(%)']:.2f}%)")
    axes[1].set_ylabel("V(out) [V]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    t_max = min(ng["time"][-1], nn["time"][-1])
    t_common = np.arange(0.0, t_max, INV_TRAN_TSTEP)
    err = (np.interp(t_common, nn["time"], nn["v(out)"])
           - np.interp(t_common, ng["time"], ng["v(out)"])) * 1e3
    axes[2].plot(t_common * 1e9, err, "r-", lw=1.0)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_xlabel("Time [ns]")
    axes[2].set_ylabel("Error [mV]")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_inverter(
    tech_name: str, work_dir: Path,
) -> Dict[str, Dict[str, float]]:
    tech = ALL_TEST_TECHS[tech_name]
    print(f"\n=== Inverter reproduction: {tech.name} "
          f"(VDD={tech.vdd}V, L_n={tech.effective_inv_l_nmos*1e9:.0f}nm, "
          f"L_p={tech.effective_inv_l_pmos*1e9:.0f}nm, "
          f"NFIN={tech.effective_inv_nfin}) ===")

    # ── DC sweep ─────────────────────────────────────────────────────
    print("  [VTC] NGSPICE BSIM-CMG ground truth…")
    ng_vtc = run_ngspice_inverter_vtc(tech, work_dir)
    print(f"    ref pts={len(ng_vtc['sweep'])}  "
          f"Vout range [{ng_vtc['vout'].min():.3f}, "
          f"{ng_vtc['vout'].max():.3f}] V")

    print("  [VTC] PyCircuitSim DirectNet V6.2.1 (LEVEL=73)…")
    nn_vtc = run_pycircuitsim_nn_inverter_vtc(
        tech, work_dir, level=73, model_name="directnet_v6_2_1")
    nn_vout_on_ref = np.interp(ng_vtc["sweep"], nn_vtc["sweep"], nn_vtc["vout"])

    vtc_metrics = _metrics_on_trace(ng_vtc["vout"], nn_vout_on_ref, tech.vdd)
    vtrip_ref = _vtrip(ng_vtc["sweep"], ng_vtc["vout"], tech.vdd)
    vtrip_nn = _vtrip(ng_vtc["sweep"], nn_vout_on_ref, tech.vdd)
    vtc_metrics["Vtrip_ref(V)"] = vtrip_ref
    vtc_metrics["Vtrip_nn(V)"] = vtrip_nn
    vtc_metrics["Vtrip_err(mV)"] = (vtrip_nn - vtrip_ref) * 1e3

    print(f"    VTC: NRMSE={vtc_metrics['NRMSE_vdd(%)']:.3f}%  "
          f"MRE={vtc_metrics['MRE(%)']:.2f}%  "
          f"MaxErr={vtc_metrics['MaxErr(V)']*1e3:.1f}mV  "
          f"R²={vtc_metrics['R2']:.4f}  "
          f"ΔVtrip={vtc_metrics['Vtrip_err(mV)']:+.1f}mV")

    _save_trace_csv(
        REPORT_DIR / f"inverter_vtc_{tech.name}.csv",
        {"Vin": ng_vtc["sweep"],
         "Vout_ngspice": ng_vtc["vout"],
         "Vout_nn": nn_vout_on_ref,
         "err_V": nn_vout_on_ref - ng_vtc["vout"]},
    )
    _plot_vtc(tech, ng_vtc, nn_vout_on_ref, vtc_metrics,
              REPORT_DIR / f"inverter_vtc_{tech.name}.png")

    # ── Transient ────────────────────────────────────────────────────
    print("  [Tran] NGSPICE BSIM-CMG ground truth…")
    ng_tr = run_ngspice_inverter_tran(tech, work_dir)
    print(f"    ref pts={len(ng_tr['time'])}  "
          f"V(out) range [{ng_tr['v(out)'].min():.4f}, "
          f"{ng_tr['v(out)'].max():.4f}] V")

    print("  [Tran] PyCircuitSim DirectNet V6.2.1 (LEVEL=73)…")
    nn_tr = run_pycircuitsim_nn_inverter_tran(
        tech, work_dir, level=73, model_name="directnet_v6_2_1")
    if nn_tr.get("_nr_partial"):
        print(f"    !! NR partial result: {nn_tr.get('_nr_error_msg')}")

    # Common grid for the full window and post-startup window.
    t_max = min(ng_tr["time"][-1], nn_tr["time"][-1])
    t_full = np.arange(0.0, t_max, INV_TRAN_TSTEP)
    t_post = np.arange(POST_STARTUP_T, t_max, INV_TRAN_TSTEP)
    ng_full = np.interp(t_full, ng_tr["time"], ng_tr["v(out)"])
    nn_full = np.interp(t_full, nn_tr["time"], nn_tr["v(out)"])
    ng_post = np.interp(t_post, ng_tr["time"], ng_tr["v(out)"])
    nn_post = np.interp(t_post, nn_tr["time"], nn_tr["v(out)"])

    tran_full = _metrics_on_trace(ng_full, nn_full, tech.vdd)
    tran_post = _metrics_on_trace(ng_post, nn_post, tech.vdd)

    region = compute_region_errors(ng_tr, nn_tr, tech.vdd, t_start=POST_STARTUP_T)
    print(f"    Tran (full):         NRMSE={tran_full['NRMSE_vdd(%)']:.3f}%  "
          f"MRE={tran_full['MRE(%)']:.2f}%  "
          f"MaxErr={tran_full['MaxErr(V)']*1e3:.1f}mV  "
          f"R²={tran_full['R2']:.4f}")
    print(f"    Tran (post-startup): NRMSE={tran_post['NRMSE_vdd(%)']:.3f}%  "
          f"MRE={tran_post['MRE(%)']:.2f}%  "
          f"MaxErr={tran_post['MaxErr(V)']*1e3:.1f}mV  "
          f"R²={tran_post['R2']:.4f}")
    print(f"    Tran region breakdown: high-rail={region['nrmse_high']:.2f}% "
          f"low-rail={region['nrmse_low']:.2f}% "
          f"transition={region['nrmse_trans']:.2f}%")

    _save_trace_csv(
        REPORT_DIR / f"inverter_tran_{tech.name}.csv",
        {"t": t_full,
         "Vin": np.interp(t_full, ng_tr["time"], ng_tr["v(in)"]),
         "Vout_ngspice": ng_full,
         "Vout_nn": nn_full,
         "err_V": nn_full - ng_full},
    )
    _plot_tran(tech, ng_tr, nn_tr, tran_full,
               REPORT_DIR / f"inverter_tran_{tech.name}.png")

    return {
        "vtc": vtc_metrics,
        "tran_full": tran_full,
        "tran_post": tran_post,
        "region": region,
    }


def _row_for_csv(
    tech: str, kind: str, m: Dict[str, float],
) -> List:
    def _g(k: str, fmt: str = ".4f") -> str:
        v = m.get(k, float("nan"))
        return "N/A" if (v is None or np.isnan(v)) else f"{v:{fmt}}"
    return [
        tech, kind,
        m.get("n_points", ""),
        _g("NRMSE_vdd(%)", ".4f"),
        _g("MRE(%)", ".3f"),
        _g("MaxMRE(%)", ".2f"),
        _g("MaxErr(V)", ".5f"),
        _g("MaxErr_vdd(%)", ".3f"),
        _g("MAE(V)", ".6f"),
        _g("R2", ".6f"),
    ]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logging.disable(logging.WARNING)

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    with tempfile.TemporaryDirectory(prefix="v6_2_1_inv_") as tmp:
        work_dir = Path(tmp)
        for t in TECHS:
            all_results[t] = evaluate_inverter(t, work_dir)

    # ── inverter_summary.csv ────────────────────────────────────────
    with (REPORT_DIR / "inverter_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tech", "comparison", "n_points",
            "NRMSE_vdd(%)", "MRE(%)", "MaxMRE(%)",
            "MaxErr(V)", "MaxErr_vdd(%)", "MAE(V)", "R2",
        ])
        for t, r in all_results.items():
            w.writerow(_row_for_csv(t, "VTC", r["vtc"]))
            w.writerow(_row_for_csv(t, "Tran_full", r["tran_full"]))
            w.writerow(_row_for_csv(t, "Tran_post_startup", r["tran_post"]))

    # ── Append section to report.md ─────────────────────────────────
    report_path = REPORT_DIR / "report.md"
    existing = report_path.read_text() if report_path.exists() else ""

    lines: List[str] = []
    if "## Inverter Reproduction" not in existing:
        lines.append("\n## Inverter Reproduction (DC + Transient)\n")
        lines.append(
            "Reproduced inverter simulations for the V6.2.1 production "
            "(`medium`) DirectNet checkpoints. NGSPICE BSIM-CMG is ground "
            "truth; PyCircuitSim DirectNet V6.2.1 (LEVEL=73) is the test "
            "model. Both VTC and transient run via "
            "`tests.verify_nn_dc_tran` machinery.\n")
        lines.append(
            "MRE is computed on samples where `|Vout_ref| > 0.5% × VDD` so "
            "near-rail samples don't dominate the relative error. "
            "Post-startup transient skips the first 200 ps "
            "(DC OP + integrator warm-up window).\n")

    lines.append("\n### Inverter VTC (DC sweep)\n")
    lines.append(
        "| Tech | n_pts | NRMSE% | MRE% | MaxMRE% | MaxErr (mV) | MAE (mV) | R² | Vtrip ref / nn (V) | ΔVtrip (mV) |\n"
        "|------|------:|-------:|-----:|--------:|------------:|---------:|---:|--------------------|------------:|")
    for t, r in all_results.items():
        m = r["vtc"]
        lines.append(
            f"| {t} | {m['n_points']} | "
            f"{m['NRMSE_vdd(%)']:.3f} | {m['MRE(%)']:.2f} | "
            f"{m['MaxMRE(%)']:.1f} | "
            f"{m['MaxErr(V)']*1e3:.2f} | {m['MAE(V)']*1e3:.3f} | "
            f"{m['R2']:.4f} | "
            f"{m['Vtrip_ref(V)']:.4f} / {m['Vtrip_nn(V)']:.4f} | "
            f"{m['Vtrip_err(mV)']:+.2f} |")

    lines.append("\n### Inverter Transient (full window 0–3 ns)\n")
    lines.append(
        "| Tech | n_pts | NRMSE% | MRE% | MaxErr (mV) | MAE (mV) | R² |\n"
        "|------|------:|-------:|-----:|------------:|---------:|---:|")
    for t, r in all_results.items():
        m = r["tran_full"]
        lines.append(
            f"| {t} | {m['n_points']} | "
            f"{m['NRMSE_vdd(%)']:.3f} | {m['MRE(%)']:.2f} | "
            f"{m['MaxErr(V)']*1e3:.2f} | {m['MAE(V)']*1e3:.3f} | "
            f"{m['R2']:.4f} |")

    lines.append("\n### Inverter Transient (post-startup, t > 200 ps)\n")
    lines.append(
        "| Tech | n_pts | NRMSE% | MRE% | MaxErr (mV) | MAE (mV) | R² | high-rail% | low-rail% | transition% |\n"
        "|------|------:|-------:|-----:|------------:|---------:|---:|-----------:|----------:|------------:|")
    for t, r in all_results.items():
        m = r["tran_post"]
        reg = r["region"]
        lines.append(
            f"| {t} | {m['n_points']} | "
            f"{m['NRMSE_vdd(%)']:.3f} | {m['MRE(%)']:.2f} | "
            f"{m['MaxErr(V)']*1e3:.2f} | {m['MAE(V)']*1e3:.3f} | "
            f"{m['R2']:.4f} | "
            f"{reg['nrmse_high']:.2f} | {reg['nrmse_low']:.2f} | "
            f"{reg['nrmse_trans']:.2f} |")

    lines.append("\n### Inverter trace + plot files\n")
    for t in TECHS:
        lines.append(f"- TSMC{t[-2:]}: "
                     f"`inverter_vtc_{t}.csv`, `inverter_vtc_{t}.png`, "
                     f"`inverter_tran_{t}.csv`, `inverter_tran_{t}.png`")
    lines.append(
        "\n- `inverter_summary.csv` — flat CSV (3 rows per tech: "
        "VTC / Tran_full / Tran_post_startup)\n")

    report_path.write_text(existing + "\n".join(lines) + "\n")
    print(f"\nReport updated: {report_path}")


if __name__ == "__main__":
    main()
