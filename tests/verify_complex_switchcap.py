#!/usr/bin/env python3
"""Benchmark 3d — switched-capacitor unit cell: DirectNet vs NGSPICE BSIM-CMG.

Part of the DirectNet V6.4 sprint, Phase 3 benchmark harness
(docs/plans/2026-05-15-directnet-complex-circuits.md).

A CMOS transmission gate samples a DC input onto a hold capacitor under a
PULSE clock. The harness measures:
  * charge-transfer accuracy  — Vsamp at the end of a sample window vs the
    NGSPICE ground truth (gate +/-5%, expressed as a fraction of VDD),
  * hold-phase droop          — Vsamp decay across a hold window (gate +/-10%
    of the NGSPICE droop).

The DirectNet (LEVEL=73) transmission gate runs in PyCircuitSim transient;
NGSPICE BSIM-CMG (LEVEL=72) is the ground truth.

Ground truth is ALWAYS NGSPICE BSIM-CMG (CLAUDE.md Validation rule).
Rule 16: report MRE / R2 / NRMSE / MaxErr.

Usage:
    conda run -n pycircuitsim python tests/verify_complex_switchcap.py
    conda run -n pycircuitsim python tests/verify_complex_switchcap.py --tech TSMC7
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
    render_directnet_netlist, run_directnet_transient, full_metrics, fmt_metrics,
)

CHARGE_TOL = 0.05          # +/-5% of VDD on charge-transfer level
DROOP_TOL = 0.10           # +/-10% of NGSPICE hold droop
TRAN_TSTEP = 5e-12
TRAN_TSTOP = 12e-9
CLK_PER = 4e-9
TEMPLATE = PROJECT_ROOT / "examples" / "complex" / "switchcap_unitcell_directnet.sp"

# sample-window end (just before the 1st sample phase closes) and a hold-window
# pair to measure droop. The clock: td=0.5n, sample (phi high) pw=1.9n.
SAMPLE_END = 2.3e-9        # end of 1st sample phase
HOLD_START = 2.6e-9        # into the 1st hold phase
HOLD_END = 4.3e-9          # just before the 2nd sample phase opens


def _vin(bt: BenchTech) -> float:
    return round(bt.vdd * 0.6, 3)


def _at(t: np.ndarray, v: np.ndarray, t0: float) -> float:
    return float(np.interp(t0, t, v))


def run_ngspice_sc(bt: BenchTech, work_dir: Path) -> Dict[str, np.ndarray]:
    baked = get_baked_modelcard(bt, bt.nfin, work_dir)
    n, p = bt.nmos_model, bt.pmos_model
    vin = _vin(bt)
    body = [f'.include "{baked}"', ".temp 27", f"Vdd vdd 0 {bt.vdd}",
            f"Vin vin 0 {vin}",
            f"Vphi phi 0 PULSE(0 {bt.vdd} 0.5n 0.1n 0.1n 1.9n 4n)",
            f"Npc phib phi vdd vdd {p}", f"Nnc phib phi 0 0 {n}",
            f"Nnt vin phi vsamp 0 {n}", f"Npt vin phib vsamp vdd {p}",
            "Csample vsamp 0 100f", f".ic v(vsamp)=0 v(phib)={bt.vdd}"]
    data = run_ngspice_wrdata("\n".join(body), "v(vsamp)", work_dir,
                              f"sc_{bt.name}",
                              f"tran {TRAN_TSTEP:.0e} {TRAN_TSTOP:.0e} uic")
    return {"time": data[:, 0], "vsamp": data[:, 1]}


def run_directnet_sc(bt: BenchTech, work_dir: Path):
    netlist = render_directnet_netlist(
        TEMPLATE, bt, work_dir / f"switchcap_{bt.name}.sp")
    # template ships Vin at 0.48 (0.6*0.80); rewrite per tech
    text = netlist.read_text()
    text = text.replace("Vin vin 0 0.48", f"Vin vin 0 {_vin(bt)}")
    netlist.write_text(text)
    results, partial, err = run_directnet_transient(netlist)
    return {"time": np.asarray(results["time"]),
            "vsamp": np.asarray(results["vsamp"])}, partial, err


def run_one(bt: BenchTech) -> Dict:
    work_dir = RESULTS_BASE / "switchcap" / bt.name
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- {bt.name} (VDD={bt.vdd} VT={bt.vt}) ---")
    vin = _vin(bt)

    print("  NGSPICE BSIM-CMG reference ...")
    ng = run_ngspice_sc(bt, work_dir)
    ng_charge = _at(ng["time"], ng["vsamp"], SAMPLE_END)
    ng_droop = (_at(ng["time"], ng["vsamp"], HOLD_START)
                - _at(ng["time"], ng["vsamp"], HOLD_END))
    print(f"    NGSPICE: charge level={ng_charge:.4f}V (Vin={vin}V)  "
          f"hold droop={ng_droop*1e3:.3f}mV")

    print("  DirectNet (LEVEL=73) transient ...")
    try:
        dn, partial, err = run_directnet_sc(bt, work_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"    DirectNet FAILED: {exc!r}")
        return {"tech": bt.name, "ng_charge": ng_charge,
                "ng_droop": ng_droop, "error": repr(exc)}

    note = " (partial — NR truncated)" if partial else ""
    dn_charge = _at(dn["time"], dn["vsamp"], SAMPLE_END)
    dn_droop = (_at(dn["time"], dn["vsamp"], HOLD_START)
                - _at(dn["time"], dn["vsamp"], HOLD_END))
    print(f"    DirectNet: charge level={dn_charge:.4f}V  "
          f"hold droop={dn_droop*1e3:.3f}mV{note}")

    # waveform metrics on the common window
    t_hi = min(ng["time"][-1], dn["time"][-1])
    grid = np.arange(0.0, t_hi, TRAN_TSTEP)
    ng_i = np.interp(grid, ng["time"], ng["vsamp"])
    dn_i = np.interp(grid, dn["time"], dn["vsamp"])
    metrics = full_metrics(dn_i, ng_i)

    charge_err = abs(dn_charge - ng_charge) / bt.vdd * 100.0
    droop_err = (abs(dn_droop - ng_droop) / abs(ng_droop) * 100.0
                 if abs(ng_droop) > 1e-6 else float("nan"))
    charge_ok = charge_err <= CHARGE_TOL * 100
    droop_ok = (not np.isfinite(droop_err)) or droop_err <= DROOP_TOL * 100
    passed = charge_ok and droop_ok
    print(f"    waveform: {fmt_metrics(metrics)}")
    print(f"    charge err={charge_err:.2f}% of VDD  droop err={droop_err:.1f}%"
          f"  ->  {'PASS' if passed else 'FAIL'}")
    return {"tech": bt.name, "ng_charge": ng_charge, "dn_charge": dn_charge,
            "charge_err_pct": charge_err, "ng_droop": ng_droop,
            "dn_droop": dn_droop, "droop_err_pct": droop_err,
            "partial": partial, "passed": passed, **metrics}


def main() -> int:
    ap = argparse.ArgumentParser(description="Switched-cap benchmark 3d")
    ap.add_argument("--tech", default=",".join(BENCH_TECHS))
    args = ap.parse_args()
    techs = [t.strip() for t in args.tech.split(",")]

    print("=" * 78)
    print("Benchmark 3d — switched-cap unit cell: DirectNet vs NGSPICE BSIM-CMG")
    print(f"  Gates: charge transfer +/-{CHARGE_TOL*100:.0f}% of VDD,"
          f" hold droop +/-{DROOP_TOL*100:.0f}%")
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
    print("SUMMARY — Benchmark 3d switched-cap unit cell")
    print("=" * 78)
    hdr = (f"{'Tech':8s} | {'NG chg V':>9s} | {'DN chg V':>9s} | "
           f"{'ChgErr%':>8s} | {'DroopErr%':>10s} | {'NRMSE%':>7s} | "
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
        print(f"{r['tech']:8s} | {r['ng_charge']:9.4f} | "
              f"{r.get('dn_charge', float('nan')):9.4f} | "
              f"{r['charge_err_pct']:8.2f} | {r['droop_err_pct']:10.1f} | "
              f"{r['nrmse_pct']:7.2f} | {status:>8s}")
    print(f"\n  {n_pass}/{len(results)} passed both charge + droop gates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
