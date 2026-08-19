#!/usr/bin/env python3
"""Benchmark 3b — two-stage Miller opamp: DirectNet vs NGSPICE BSIM-CMG.

Part of the DirectNet V6.4 sprint, Phase 3 benchmark harness.

Runs a DC transfer sweep of a two-stage Miller opamp built from DirectNet
(LEVEL=73) transistors and compares against the NGSPICE BSIM-CMG (LEVEL=72)
ground truth. Extracts open-loop DC gain (peak |dVout/dVin|), trip point
(Vin where Vout = VDD/2) and the worst output-rail slew step.

Gates: open-loop DC gain within +/-10%; trip-point shift reported.

Ground truth is ALWAYS NGSPICE BSIM-CMG (AGENTS.md Validation rule).
Report MRE / R2 / NRMSE / MaxErr.

Usage:
    conda run -n pycircuitsim python tests/simple_circuits/verify_complex_opamp.py
    conda run -n pycircuitsim python tests/simple_circuits/verify_complex_opamp.py --tech TSMC12
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

print = functools.partial(print, flush=True)  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))

from tests.common.base import SIMPLE_DECKS, render_reference_deck  # noqa: E402
from tests.common.complex import (  # noqa: E402
    BENCH, BENCH_TECHS, RESULTS_BASE, BenchTech,
    get_baked_modelcard, run_ngspice_wrdata,
    render_directnet_text, run_directnet_dc_sweep, full_metrics, fmt_metrics,
)

GAIN_TOL = 0.10            # +/-10% open-loop DC gain gate
# audit B5c: an NGSPICE reference gain below this V/V means the cell is biased
# out of its amplifying region — it is not an opamp, so no DirectNet run can
# certify it. Mirrors the parametric twin (tests/common/complex_sweep.py
# OPAMP_MIN_GAIN). The shipped cells sit at 160-190 V/V, ~30x above the floor.
OPAMP_MIN_GAIN = 5.0
TEMPLATE = SIMPLE_DECKS / "directnet_opamp_miller_dc.sp"
NG_TEMPLATE = SIMPLE_DECKS / "bsimcmg_opamp_miller_dc.cir"


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


def _region_error(ng_gain: float) -> str:
    """Reason string when the NGSPICE reference is not a usable opamp, else "".

    audit B5c: the only reference-side guard used to be ``ng_gain > 0``, so a
    mis-biased cell (reference gain ~0.3 V/V) was certified by any DirectNet run
    that reproduced the same non-amplifying bias to within 10%.
    """
    if not np.isfinite(ng_gain) or ng_gain < OPAMP_MIN_GAIN:
        return f"out-of-region (ng_gain={ng_gain:.2f}<{OPAMP_MIN_GAIN})"
    return ""


def _verdict(ng_gain: float, dn_gain: float) -> Tuple[float, bool, str]:
    """(gain error %, pass flag, reason) for one opamp cell.

    Pure + importable so the region guard and the tolerance can be exercised
    without NGSPICE (audit B5c).
    """
    reason = _region_error(ng_gain)
    if reason:
        return float("nan"), False, reason
    err = abs(dn_gain - ng_gain) / ng_gain * 100.0
    return err, bool(err <= GAIN_TOL * 100), ""


def ngspice_opamp_body(bt: BenchTech, baked: Path) -> Dict[str, str]:
    """Single-point NGSPICE opamp ground-truth deck body (no .control/.end).

    The topology is NOT here — it is ``examples/simple_circuits/
    bsimcmg_opamp_miller_dc.cir``, rendered per tech. This function owns the
    bias and the sweep window, nothing else.

    Pure (returns text) so verify_complex_sweep_canaries can diff it against the
    parametric ``tests.common.complex.ngspice_opamp`` builder (bug report B8).
    """
    vcm, vbn, vbp = _bias(bt)
    body = render_reference_deck(NG_TEMPLATE, {
        "BAKED_LIB": str(baked),
        "VDD": f"{bt.vdd}",
        "VBN": f"{vbn}",
        "VBP": f"{vbp}",
        "VCM": f"{vcm}",
        "NMOS": bt.nmos_model,
        "PMOS": bt.pmos_model,
    }, body_only=True)
    lo, hi = round(vcm - 0.15, 3), round(vcm + 0.15, 3)
    return {"body": body, "signals": "v(vout)",
            "analysis": f"dc Vinp {lo} {hi} 0.002"}


def directnet_opamp_deck(bt: BenchTech) -> str:
    """Single-point DirectNet opamp ship-gate deck text (render + per-tech
    bias rewrites). Pure text so the equivalence canary diffs the REAL deck
    (not a hand-copied replica) against the sweep builder (bug report B8)."""
    vcm, vbn, vbp = _bias(bt)
    # the template ships TSMC12 0.80V bias rails; rewrite per tech
    text = render_directnet_text(TEMPLATE.read_text(), bt)
    text = text.replace("Vbn vbn 0 0.36", f"Vbn vbn 0 {vbn}")
    text = text.replace("Vbp vbp 0 0.44", f"Vbp vbp 0 {vbp}")
    text = text.replace("Vinn inn 0 0.44", f"Vinn inn 0 {vcm}")
    text = text.replace("Vinp inp 0 0.44", f"Vinp inp 0 {vcm}")
    lo, hi = round(vcm - 0.15, 3), round(vcm + 0.15, 3)
    return text.replace(".dc Vinp 0.29 0.59 0.002",
                        f".dc Vinp {lo} {hi} 0.002")


def run_ngspice_opamp(bt: BenchTech, work_dir: Path) -> Dict[str, np.ndarray]:
    baked = get_baked_modelcard(bt, bt.nfin, work_dir)
    spec = ngspice_opamp_body(bt, baked)
    data = run_ngspice_wrdata(spec["body"], spec["signals"], work_dir,
                              f"opamp_{bt.name}", spec["analysis"])
    return {"sweep": data[:, 0], "vout": data[:, 1]}


def run_directnet_opamp(bt: BenchTech, work_dir: Path) -> Dict[str, np.ndarray]:
    netlist = work_dir / f"opamp_{bt.name}.sp"
    netlist.parent.mkdir(parents=True, exist_ok=True)
    netlist.write_text(directnet_opamp_deck(bt))
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
    region_err = _region_error(ng_gain)
    if region_err:
        # audit B5c: bail before spending the DirectNet run — an out-of-region
        # reference cannot certify anything. Ordering mirrors the parametric
        # twin's `_err` in tests/common/complex_sweep.py.
        print(f"    {region_err} — not a usable opamp reference")
        return {"tech": bt.name, "ng_gain": ng_gain, "ng_trip": ng_trip,
                "error": region_err}

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

    gain_err, passed, _ = _verdict(ng_gain, dn_gain)
    trip_shift = (dn_trip - ng_trip) * 1e3
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
    # audit B5l: an unknown tech used to print SKIP and never enter `results`,
    # so `--tech TSMC5,TSMC7X` reported 1/1 and exited 0. Reject up front
    # (same pattern as tests/common/nn_gate.py).
    unknown = [t for t in techs if t not in BENCH]
    if unknown:
        print(f"ERROR: unknown tech(s) {unknown}. Available: {list(BENCH)}")
        return 1

    print("=" * 78)
    print("Benchmark 3b — two-stage Miller opamp: DirectNet vs NGSPICE BSIM-CMG")
    print(f"  Gate: open-loop DC gain within +/-{GAIN_TOL*100:.0f}%")
    print("=" * 78)

    results: List[Dict] = []
    for name in techs:
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
    # B10: surface the verdict in the exit code (consumers also parse stdout).
    # empty results (all techs skipped) must not exit green
    return 0 if (results and n_pass == len(results)) else 1


if __name__ == "__main__":
    sys.exit(main())
