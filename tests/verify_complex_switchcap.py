#!/usr/bin/env python3
"""Benchmark 3d — switched-capacitor unit cell: DirectNet vs NGSPICE BSIM-CMG.

Part of the DirectNet V6.4 sprint, Phase 3 benchmark harness.

A CMOS transmission gate samples a DC input onto a hold capacitor under a
PULSE clock. The harness measures:
  * charge-transfer accuracy  — Vsamp at the end of a sample window vs the
    NGSPICE ground truth (gate +/-5%, expressed as a fraction of VDD),
  * hold-phase droop          — Vsamp decay across a hold window. Gate:
    |dn-ng| <= max(10% of the NGSPICE droop, 0.1% of VDD). The absolute
    floor (V6.4.7 R0.1) reflects the solvers' two-point difference
    tolerance (~2*(RELTOL*V_hold+VNTOL) ~ 0.6-0.9 mV); the old pure-
    relative gate was unpassable when the NG droop sat at sub-uV levels,
    and its nan-guard auto-passed ANY DirectNet droop when
    |ng_droop| <= 1 uV. Droop%alw in the summary is the disagreement as
    % of the allowance (<=100 passes).

The DirectNet (LEVEL=73) transmission gate runs in PyCircuitSim transient;
NGSPICE BSIM-CMG (LEVEL=72) is the ground truth.

Ground truth is ALWAYS NGSPICE BSIM-CMG (CLAUDE.md Validation rule).
Report MRE / R2 / NRMSE / MaxErr.

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
    render_directnet_text, run_directnet_transient, full_metrics, fmt_metrics,
)

CHARGE_TOL = 0.05          # +/-5% of VDD on charge-transfer level
DROOP_TOL = 0.10           # +/-10% of NGSPICE hold droop (relative part)
DROOP_FLOOR_FRAC = 1e-3    # absolute floor on the droop gate: 0.1% of VDD
TRAN_TSTEP = 5e-12
TRAN_TSTOP = 12e-9
CLK_PER = 4e-9
TEMPLATE = PROJECT_ROOT / "examples" / "complex" / "directnet_switchcap_tran.sp"

# sample-window end (just before the 1st sample phase closes) and a hold-window
# pair to measure droop. The clock: td=0.5n, sample (phi high) pw=1.9n.
SAMPLE_END = 2.3e-9        # end of 1st sample phase
HOLD_START = 2.6e-9        # into the 1st hold phase
HOLD_END = 4.3e-9          # just before the 2nd sample phase opens


def _vin(bt: BenchTech) -> float:
    return round(bt.vdd * 0.6, 3)


def _at(t: np.ndarray, v: np.ndarray, t0: float) -> float:
    return float(np.interp(t0, t, v))


def ngspice_sc_body(bt: BenchTech, baked: Path) -> Dict[str, str]:
    """Single-point NGSPICE switched-cap ground-truth deck body.

    Pure (returns text) so verify_complex_sweep_canaries can diff it against the
    parametric ``tests.common.complex.ngspice_switchcap`` builder (B8)."""
    n, p = bt.nmos_model, bt.pmos_model
    vin = _vin(bt)
    body = [f'.include "{baked}"', ".temp 27", f"Vdd vdd 0 {bt.vdd}",
            f"Vin vin 0 {vin}",
            f"Vphi phi 0 PULSE(0 {bt.vdd} 0.5n 0.1n 0.1n 1.9n 4n)",
            f"Npc phib phi vdd vdd {p}", f"Nnc phib phi 0 0 {n}",
            f"Nnt vin phi vsamp 0 {n}", f"Npt vin phib vsamp vdd {p}",
            "Csample vsamp 0 100f", f".ic v(vsamp)=0 v(phib)={bt.vdd}"]
    return {"body": "\n".join(body), "signals": "v(vsamp)",
            "analysis": f"tran {TRAN_TSTEP:.1e} {TRAN_TSTOP:.1e} uic"}


def directnet_sc_deck(bt: BenchTech) -> str:
    """Single-point DirectNet switched-cap ship-gate deck text (render + Vin
    rewrite). Pure text so the canary diffs the REAL deck against the sweep
    builder rather than a hand-copied replica (B8). The template ships Vin at
    0.48 (0.6*0.80); rewrite per tech."""
    text = render_directnet_text(TEMPLATE.read_text(), bt)
    return text.replace("Vin vin 0 0.48", f"Vin vin 0 {_vin(bt)}")


def run_ngspice_sc(bt: BenchTech, work_dir: Path) -> Dict[str, np.ndarray]:
    baked = get_baked_modelcard(bt, bt.nfin, work_dir)
    spec = ngspice_sc_body(bt, baked)
    data = run_ngspice_wrdata(spec["body"], spec["signals"], work_dir,
                              f"sc_{bt.name}", spec["analysis"])
    return {"time": data[:, 0], "vsamp": data[:, 1]}


def run_directnet_sc(bt: BenchTech, work_dir: Path):
    netlist = work_dir / f"switchcap_{bt.name}.sp"
    netlist.parent.mkdir(parents=True, exist_ok=True)
    netlist.write_text(directnet_sc_deck(bt))
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
    # Droop gate: |dn-ng| <= max(DROOP_TOL*|ng|, DROOP_FLOOR_FRAC*VDD).
    # The droop is the difference of two solver points, each honest only to
    # RELTOL*V + VNTOL (NGSPICE defaults 1e-3*V_hold + 1e-6 ~ 0.43 mV here),
    # so the two-point difference carries a ~2*(RELTOL*V_hold+VNTOL)
    # ~ 0.6-0.9 mV tolerance ~ 1e-3*VDD. A pure-relative gate against a
    # sub-uV NG droop demands agreement far below both solvers' tolerances
    # (unpassable), while a nan-guard that skips tiny |ng| auto-passes ANY
    # DN droop. NOTE: the floor admits up to ~50 nA of off-state leakage
    # error over the 1.7 ns hold — this is a circuit-level gate, not a
    # subthreshold-fidelity gate.
    droop_abs = abs(dn_droop - ng_droop)
    droop_allow = max(DROOP_TOL * abs(ng_droop), DROOP_FLOOR_FRAC * bt.vdd)
    droop_err = droop_abs / droop_allow * 100.0
    charge_ok = charge_err <= CHARGE_TOL * 100
    droop_ok = droop_abs <= droop_allow
    # A transient that diverged mid-run (NR truncated) leaves the sample/hold
    # reads on a clamped np.interp tail and can land in tolerance by accident —
    # fail loud instead (bug report B9).
    passed = charge_ok and droop_ok and not partial
    print(f"    waveform: {fmt_metrics(metrics)}")
    print(f"    charge err={charge_err:.2f}% of VDD  "
          f"droop |dn-ng|={droop_abs*1e3:.3f}mV "
          f"({droop_err:.0f}% of allowance {droop_allow*1e3:.3f}mV)"
          f"  ->  {'PASS' if passed else 'FAIL'}")
    return {"tech": bt.name, "ng_charge": ng_charge, "dn_charge": dn_charge,
            "charge_err_pct": charge_err, "ng_droop": ng_droop,
            "dn_droop": dn_droop, "droop_pct_of_allowance": droop_err,
            "partial": partial, "passed": passed, **metrics}


def main() -> int:
    ap = argparse.ArgumentParser(description="Switched-cap benchmark 3d")
    ap.add_argument("--tech", default=",".join(BENCH_TECHS))
    args = ap.parse_args()
    techs = [t.strip() for t in args.tech.split(",")]
    # audit B5l: an unknown tech used to print SKIP and never enter `results`,
    # so `--tech TSMC5,TSMC7X` reported 1/1 and exited 0. Reject up front
    # (same pattern as verify_nn_dc_tran.py).
    unknown = [t for t in techs if t not in BENCH]
    if unknown:
        print(f"ERROR: unknown tech(s) {unknown}. Available: {list(BENCH)}")
        return 1

    print("=" * 78)
    print("Benchmark 3d — switched-cap unit cell: DirectNet vs NGSPICE BSIM-CMG")
    print(f"  Gates: charge transfer +/-{CHARGE_TOL*100:.0f}% of VDD,"
          f" hold droop |dn-ng| <= max({DROOP_TOL*100:.0f}% of NG droop,"
          f" {DROOP_FLOOR_FRAC*100:.1f}% of VDD)")
    print("=" * 78)

    results: List[Dict] = []
    for name in techs:
        try:
            results.append(run_one(BENCH[name]))
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: ERROR {exc!r}")
            results.append({"tech": name, "error": repr(exc)})

    print("\n" + "=" * 78)
    print("SUMMARY — Benchmark 3d switched-cap unit cell")
    print("=" * 78)
    hdr = (f"{'Tech':8s} | {'NG chg V':>9s} | {'DN chg V':>9s} | "
           f"{'ChgErr%':>8s} | {'Droop%alw':>10s} | {'NRMSE%':>7s} | "
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
              f"{r['charge_err_pct']:8.2f} | "
              f"{r['droop_pct_of_allowance']:10.1f} | "
              f"{r['nrmse_pct']:7.2f} | {status:>8s}")
    print(f"\n  {n_pass}/{len(results)} passed both charge + droop gates")
    # B10: surface the verdict in the exit code (consumers also parse stdout).
    # empty results (all techs skipped) must not exit green
    return 0 if (results and n_pass == len(results)) else 1


if __name__ == "__main__":
    sys.exit(main())
