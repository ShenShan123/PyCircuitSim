#!/usr/bin/env python3
"""Benchmark 3b — two-stage Miller opamp: DirectNet vs NGSPICE BSIM-CMG.

Part of the DirectNet V6.4 sprint, Phase 3 benchmark harness
(docs/plans/2026-05-15-directnet-complex-circuits.md).

Runs a DC transfer sweep of a two-stage Miller opamp built from DirectNet
(LEVEL=73) transistors and compares against the NGSPICE BSIM-CMG (LEVEL=72)
ground truth. Extracts open-loop DC gain (peak |dVout/dVin|), trip point
(Vin where Vout = VDD/2) and the worst output-rail slew step.

Gates: open-loop DC gain within +/-10%; trip-point shift reported.

Ground truth is ALWAYS NGSPICE BSIM-CMG (CLAUDE.md Validation rule).
Rule 16: report MRE / R2 / NRMSE / MaxErr.

Usage:
    conda run -n pycircuitsim python tests/verify_complex_opamp.py
    conda run -n pycircuitsim python tests/verify_complex_opamp.py --tech TSMC12
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

print = functools.partial(print, flush=True)  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))

from tests.common.complex import (  # noqa: E402
    BENCH, BENCH_TECHS, RESULTS_BASE, BenchTech,
    get_baked_modelcard, run_ngspice_wrdata,
    render_directnet_netlist, run_directnet_dc_sweep, full_metrics, fmt_metrics,
)

GAIN_TOL = 0.10            # +/-10% open-loop DC gain gate
TEMPLATE = PROJECT_ROOT / "examples" / "complex" / "miller_opamp_directnet.sp"


def _bias(bt: BenchTech) -> Tuple[float, float, float]:
    """Return (Vcm, Vbn, Vbp) — common-mode + the two bias rails."""
    return (round(bt.vdd * 0.55, 3),
            round(bt.vdd * 0.45, 3),
            round(bt.vdd * 0.55, 3))


def _gain_trip(sweep: np.ndarray, vout: np.ndarray,
               vdd: float) -> Tuple[float, float, float]:
    """Peak |dVout/dVin| gain, trip point (Vout=VDD/2), worst slew step."""
    g = np.gradient(vout, sweep)
    gain = float(np.max(np.abs(g)))
    ix = int(np.argmin(np.abs(vout - vdd / 2.0)))
    trip = float(sweep[ix])
    slew = float(np.max(np.abs(np.diff(vout))))
    return gain, trip, slew


def run_ngspice_opamp(bt: BenchTech, work_dir: Path) -> Dict[str, np.ndarray]:
    baked = get_baked_modelcard(bt, bt.nfin, work_dir)
    vcm, vbn, vbp = _bias(bt)
    n, p = bt.nmos_model, bt.pmos_model
    body = [f'.include "{baked}"', ".temp 27", f"Vdd vdd 0 {bt.vdd}",
            f"Vbn vbn 0 {vbn}", f"Vbp vbp 0 {vbp}",
            f"Vinn inn 0 {vcm}", f"Vinp inp 0 {vcm}",
            f"Nn1 n1 inp vtail 0 {n}", f"Nn2 vo1i inn vtail 0 {n}",
            f"Np3 n1 n1 vdd vdd {p}", f"Np4 vo1i n1 vdd vdd {p}",
            f"Nn5 vtail vbn 0 0 {n}",
            f"Np6 vout vo1i vdd vdd {p}", f"Nn7 vout vbn 0 0 {n}",
            "Cc vo1i vout 20f", "CL vout 0 50f"]
    lo, hi = round(vcm - 0.15, 3), round(vcm + 0.15, 3)
    data = run_ngspice_wrdata("\n".join(body), "v(vout)", work_dir,
                              f"opamp_{bt.name}", f"dc Vinp {lo} {hi} 0.002")
    return {"sweep": data[:, 0], "vout": data[:, 1]}


def run_directnet_opamp(bt: BenchTech, work_dir: Path) -> Dict[str, np.ndarray]:
    vcm, vbn, vbp = _bias(bt)
    netlist = render_directnet_netlist(
        TEMPLATE, bt, work_dir / f"opamp_{bt.name}.sp")
    # the template ships TSMC12 0.80V bias rails; rewrite per tech
    text = netlist.read_text()
    text = text.replace("Vbn vbn 0 0.36", f"Vbn vbn 0 {vbn}")
    text = text.replace("Vbp vbp 0 0.44", f"Vbp vbp 0 {vbp}")
    text = text.replace("Vinn inn 0 0.44", f"Vinn inn 0 {vcm}")
    text = text.replace("Vinp inp 0 0.44", f"Vinp inp 0 {vcm}")
    lo, hi = round(vcm - 0.15, 3), round(vcm + 0.15, 3)
    text = text.replace(".dc Vinp 0.29 0.59 0.002",
                        f".dc Vinp {lo} {hi} 0.002")
    netlist.write_text(text)
    results = run_directnet_dc_sweep(netlist, work_dir, f"opamp_{bt.name}")
    return {"sweep": np.asarray(results["inp"]),
            "vout": np.asarray(results["vout"])}


def run_one(bt: BenchTech) -> Dict:
    work_dir = RESULTS_BASE / "opamp" / bt.name
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- {bt.name} (VDD={bt.vdd} VT={bt.vt}) ---")

    print("  NGSPICE BSIM-CMG reference ...")
    ng = run_ngspice_opamp(bt, work_dir)
    ng_gain, ng_trip, ng_slew = _gain_trip(ng["sweep"], ng["vout"], bt.vdd)
    print(f"    NGSPICE gain={ng_gain:.1f}  trip={ng_trip:.4f}V  "
          f"slew(step)={ng_slew*1e3:.2f}mV")

    print("  DirectNet (LEVEL=73) DC transfer ...")
    try:
        dn = run_directnet_opamp(bt, work_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"    DirectNet FAILED: {exc!r}")
        return {"tech": bt.name, "ng_gain": ng_gain, "ng_trip": ng_trip,
                "error": repr(exc)}

    dn_gain, dn_trip, dn_slew = _gain_trip(dn["sweep"], dn["vout"], bt.vdd)
    print(f"    DirectNet gain={dn_gain:.1f}  trip={dn_trip:.4f}V  "
          f"slew(step)={dn_slew*1e3:.2f}mV")

    # Vout-curve metrics on the common sweep window
    lo = max(ng["sweep"].min(), dn["sweep"].min())
    hi = min(ng["sweep"].max(), dn["sweep"].max())
    grid = np.linspace(lo, hi, 300)
    ng_i = np.interp(grid, ng["sweep"], ng["vout"])
    dn_i = np.interp(grid, dn["sweep"], dn["vout"])
    metrics = full_metrics(dn_i, ng_i)

    gain_err = abs(dn_gain - ng_gain) / ng_gain * 100.0 if ng_gain > 0 else float("nan")
    trip_shift = (dn_trip - ng_trip) * 1e3
    passed = np.isfinite(gain_err) and gain_err <= GAIN_TOL * 100
    print(f"    Vout curve: {fmt_metrics(metrics)}")
    print(f"    gain error = {gain_err:.2f}%  trip shift = {trip_shift:.2f}mV"
          f"  ->  {'PASS' if passed else 'FAIL'}")
    return {"tech": bt.name, "ng_gain": ng_gain, "dn_gain": dn_gain,
            "gain_err_pct": gain_err, "trip_shift_mV": trip_shift,
            "passed": passed, **metrics}


def main() -> int:
    ap = argparse.ArgumentParser(description="Miller-opamp benchmark 3b")
    ap.add_argument("--tech", default=",".join(BENCH_TECHS))
    args = ap.parse_args()
    techs = [t.strip() for t in args.tech.split(",")]

    print("=" * 78)
    print("Benchmark 3b — two-stage Miller opamp: DirectNet vs NGSPICE BSIM-CMG")
    print(f"  Gate: open-loop DC gain within +/-{GAIN_TOL*100:.0f}%")
    print("=" * 78)

    results: List[Dict] = []
    for name in techs:
        if name not in BENCH:
            print(f"  SKIP unknown tech {name}")
            continue
        try:
            results.append(run_one(BENCH[name]))
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: ERROR {exc!r}")
            results.append({"tech": name, "error": repr(exc)})

    print("\n" + "=" * 78)
    print("SUMMARY — Benchmark 3b Miller opamp")
    print("=" * 78)
    hdr = (f"{'Tech':8s} | {'NG gain':>9s} | {'DN gain':>9s} | "
           f"{'GainErr%':>9s} | {'TripShift':>10s} | {'NRMSE%':>7s} | "
           f"{'Status':>8s}")
    print(hdr)
    print("-" * len(hdr))
    n_pass = 0
    for r in results:
        if "error" in r:
            print(f"{r['tech']:8s} | ERROR — {r['error'][:54]}")
            continue
        status = "PASS" if r.get("passed") else "FAIL"
        n_pass += int(r.get("passed", False))
        print(f"{r['tech']:8s} | {r['ng_gain']:9.1f} | "
              f"{r.get('dn_gain', float('nan')):9.1f} | "
              f"{r['gain_err_pct']:9.2f} | "
              f"{r['trip_shift_mV']:8.2f}mV | {r['nrmse_pct']:7.2f} | "
              f"{status:>8s}")
    print(f"\n  {n_pass}/{len(results)} within +/-{GAIN_TOL*100:.0f}% gain gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
