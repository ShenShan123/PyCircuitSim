#!/usr/bin/env python3
"""DirectNet V6.4.6 — Phase-0G: RO integration study (TSMC7).

INSTRUMENTATION / DIAGNOSTIC DRIVER (not a shipped solver change). Probes the
ONE axis P0-C left open for the TSMC7 ring-oscillator period gap (DN 50.82 ps vs
NG 46.64 ps, need <=48.97 ps): the **time-integration** side.

P0-C proved gds/caps (the NR Jacobian surfaces) are causally inert on the
period (swapping exact OSDI gds/caps moved it <=0.01 ps) because they cancel at
the converged NR fixed point. The remaining owners are the id-VALUE + charge
trajectories AND the BE/Trap/BDF-2 *truncation*. This driver isolates the
truncation owner two ways:

  R1 (integrator SELECTION): re-run the TSMC7 RO forcing a single integrator
     throughout (Trap / BE / BDF-2) via the env var RO_INTEG_FORCE that the
     instrumented solver reads. BDF-2 is dissipative (lengthens an oscillator
     period); Trap is not. If the baseline's auto-switch ever fires BDF-2,
     forcing pure-Trap should shorten DN toward NG.

  R2 (uniform-tstep CONVERGENCE study): re-run at a halving sequence of the
     base tstep (x1, /2, /4) holding the integrator at its baseline setting,
     and Richardson-extrapolate period(tstep -> 0). Converges toward ~46.6 ps
     => truncation-dominated (a solver fix CLOSES the gate at 0 GPU). Plateaus
     at ~50 ps => model-owned id/charge VALUE error, confirming P0-H's bucket.

Ground truth is ALWAYS NGSPICE BSIM-CMG (CLAUDE.md Validation rule). Run with
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1. No retrain, no checkpoint mutation. The
solver instrumentation it relies on (RO_INTEG_PROBE / RO_INTEG_FORCE) is
reverted after the study; this driver is the deliverable.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      conda run -n pycircuitsim python scripts/v6_4_6_p0g_integration_study.py \
        [--mode r1|r2|both] [--max-refine 4]
"""
from __future__ import annotations

import argparse
import os
import sys
import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tests.verify_complex_ring_osc as ro  # noqa: E402
from tests.common.complex import (  # noqa: E402
    BENCH, full_metrics, render_directnet_netlist, run_directnet_transient)

NG_REF_PS = 46.64           # NGSPICE BSIM-CMG ground-truth period (pinned, P0-F)
GATE_PS = 48.97             # <=5% gate boundary
BASE_TSTEP = 2e-12          # the gate's default tstep
TSTOP = 1.2e-9              # the gate's default window


def _run_dn_ro_precise(bt, tstep: float, tstop: float, work_dir: Path):
    """Render + run a DirectNet RO transient at an ARBITRARY (sub-ps) tstep.

    The gate's own `run_directnet_ro` formats the .tran tstep as
    f"{tstep*1e12:.0f}p" — which truncates any sub-ps step to "0p" (dt=0,
    ValueError). This bypasses that by writing the .tran line in seconds with
    scientific notation (the parser's _parse_value handles a bare float), so
    the convergence study can refine below 1 ps. Everything else (template,
    per-tech checkpoint resolution, solver settings) is identical to the gate.
    """
    netlist = render_directnet_netlist(
        ro.TEMPLATE, bt, work_dir / f"ring_osc_{bt.name}.sp")
    text = netlist.read_text().replace(
        ".tran 1p 5n", f".tran {tstep:.6e} {tstop:.6e}")
    netlist.write_text(text)
    return run_directnet_transient(netlist)


def _measure_dn_period(bt, tstep: float, tstop: float,
                       work_subdir: str) -> Tuple[float, Dict[str, float], bool]:
    """Run a single DirectNet RO transient at (tstep, tstop); return (period_ps,
    waveform-metrics-vs-NG, partial_flag). Reuses the gate's own machinery by
    overriding the module globals so the period estimator is byte-identical."""
    work_dir = ro.RESULTS_BASE / "ring_osc" / f"{bt.name}_{work_subdir}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # NGSPICE reference at the SAME tstep (its own truncation is tiny, but keep
    # the comparison honest). Period is read at VDD/2.
    old_tstep, old_tstop = ro.TRAN_TSTEP, ro.TRAN_TSTOP
    ro.TRAN_TSTEP = tstep
    ro.TRAN_TSTOP = tstop
    try:
        ng = ro.run_ngspice_ro(bt, work_dir)
        mid = bt.vdd / 2.0
        ng_per = ro._period_from_wave(ng["time"], ng["v(n5)"], mid, ro.SETTLE)

        t0 = _time.time()
        dn_raw, partial, err = _run_dn_ro_precise(bt, tstep, tstop, work_dir)
        dn = {"time": np.asarray(dn_raw["time"]),
              "v(n5)": np.asarray(dn_raw["n5"])}
        wall = _time.time() - t0
        dn_per = ro._period_from_wave(dn["time"], dn["v(n5)"], mid, ro.SETTLE)

        # waveform metrics on the common post-settle grid vs NG
        t_lo, t_hi = ro.SETTLE, min(ng["time"][-1], dn["time"][-1])
        metrics = {"mre_pct": float("nan"), "r2": float("nan"),
                   "nrmse_pct": float("nan"), "max_err": float("nan")}
        if t_hi > t_lo and len(dn["time"]) > 3:
            grid = np.arange(t_lo, t_hi, tstep)
            ng_i = np.interp(grid, ng["time"], ng["v(n5)"])
            dn_i = np.interp(grid, dn["time"], dn["v(n5)"])
            metrics = full_metrics(dn_i, ng_i)
        metrics["ng_per_ps"] = ng_per * 1e12
        metrics["wall_s"] = wall
        if partial:
            print(f"      [WARN] partial waveform (NR truncated): {err}")
        return dn_per * 1e12, metrics, partial
    finally:
        ro.TRAN_TSTEP, ro.TRAN_TSTOP = old_tstep, old_tstop


def _perr(period_ps: float) -> float:
    return abs(period_ps - NG_REF_PS) / NG_REF_PS * 100.0


def run_r1(bt) -> List[Dict]:
    """Integrator-selection sweep at the base tstep."""
    print("\n" + "=" * 72)
    print("R1 — integrator selection (TSMC7 RO, base tstep = %.0f fs)"
          % (BASE_TSTEP * 1e15))
    print("=" * 72)
    variants = [
        ("baseline (BE1->Trap->BDF2auto)", {}),
        ("force Trapezoidal",             {"RO_INTEG_FORCE": "trap"}),
        ("force Backward-Euler",          {"RO_INTEG_FORCE": "be"}),
        ("force BDF-2",                   {"RO_INTEG_FORCE": "bdf2"}),
    ]
    rows: List[Dict] = []
    for label, env in variants:
        # set/clear env for this variant
        os.environ.pop("RO_INTEG_FORCE", None)
        os.environ["RO_INTEG_PROBE"] = "1"
        os.environ.update(env)
        tag = "r1_" + (env.get("RO_INTEG_FORCE") or "baseline")
        print(f"\n  >>> {label}  (RO_INTEG_FORCE={env.get('RO_INTEG_FORCE','(none)')})")
        per, m, partial = _measure_dn_period(bt, BASE_TSTEP, TSTOP, tag)
        e = _perr(per)
        rows.append({"variant": label, "period_ps": per, "perr_pct": e,
                     "partial": partial, **m})
        print(f"      DN period = {per:.2f} ps   perErr = {e:.2f}%   "
              f"NRMSE={m['nrmse_pct']:.2f}% R2={m['r2']:.4f}   "
              f"{'PASS' if per <= GATE_PS else 'FAIL'}")
    os.environ.pop("RO_INTEG_FORCE", None)
    os.environ.pop("RO_INTEG_PROBE", None)
    return rows


def run_r2(bt, max_refine: int, force: str = "") -> List[Dict]:
    """Uniform-tstep convergence study. With force='' the baseline integrator
    (BE1->Trap, no BDF-2 since it never fires) is used; with force in
    {'trap','be','bdf2'} the named integrator is pinned every step (the R3
    BE-convergence study uses force='be')."""
    label = force or "baseline (Trap)"
    print("\n" + "=" * 72)
    print(f"R2/R3 — uniform-tstep convergence study (TSMC7 RO, integrator={label})")
    print("=" * 72)
    os.environ.pop("RO_INTEG_FORCE", None)
    if force in ("trap", "be", "bdf2"):
        os.environ["RO_INTEG_FORCE"] = force
    os.environ["RO_INTEG_PROBE"] = "1"
    rows: List[Dict] = []
    for k in range(max_refine):
        div = 2 ** k
        tstep = BASE_TSTEP / div
        print(f"\n  >>> tstep = base/{div} = {tstep*1e15:.2f} fs  "
              f"(num_steps ~ {int(np.ceil(TSTOP/tstep))+1})")
        tag = (force or "trap")
        per, m, partial = _measure_dn_period(bt, tstep, TSTOP, f"r2_{tag}_div{div}")
        e = _perr(per)
        rows.append({"div": div, "tstep_ps": tstep * 1e12, "period_ps": per,
                     "perr_pct": e, "partial": partial, **m})
        print(f"      DN period = {per:.2f} ps   perErr = {e:.2f}%   "
              f"NRMSE={m['nrmse_pct']:.2f}%   wall={m['wall_s']:.0f}s   "
              f"{'PASS' if per <= GATE_PS else 'FAIL'}")
    os.environ.pop("RO_INTEG_PROBE", None)
    os.environ.pop("RO_INTEG_FORCE", None)

    # Richardson extrapolation tstep->0. The baseline path is Trap (order 2)
    # unless BDF-2 fires; report both an order-2 and an order-1 estimate from
    # the two finest steps so the reader can see the integrator-order sensitivity.
    if len(rows) >= 2:
        p_coarse = rows[-2]["period_ps"]
        p_fine = rows[-1]["period_ps"]
        # order-p Richardson: P0 = (2^p * p_fine - p_coarse)/(2^p - 1)
        rich2 = (4.0 * p_fine - p_coarse) / 3.0       # order 2 (Trap)
        rich1 = 2.0 * p_fine - p_coarse               # order 1 (BE/BDF2 eff.)
        print(f"\n  Richardson extrapolation (tstep->0) from the two finest steps "
              f"({rows[-2]['tstep_ps']:.2f} & {rows[-1]['tstep_ps']:.2f} ps):")
        print(f"      order-2 (Trap)  P0 = {rich2:.2f} ps   perErr = {_perr(rich2):.2f}%")
        print(f"      order-1 (BE)    P0 = {rich1:.2f} ps   perErr = {_perr(rich1):.2f}%")
        rows.append({"div": "richardson_o2", "period_ps": rich2,
                     "perr_pct": _perr(rich2)})
        rows.append({"div": "richardson_o1", "period_ps": rich1,
                     "perr_pct": _perr(rich1)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["r1", "r2", "both"], default="both")
    ap.add_argument("--max-refine", type=int, default=3,
                    help="number of tstep halvings for R2 (x1,/2,/4,...)")
    ap.add_argument("--force", choices=["", "trap", "be", "bdf2"], default="",
                    help="pin the integrator for the R2/R3 convergence study")
    args = ap.parse_args()

    bt = BENCH["TSMC7"]
    print(f"TSMC7 RO integration study  (VDD={bt.vdd} VT={bt.vt})")
    print(f"  NG reference = {NG_REF_PS} ps   gate <=5% => DN must reach <={GATE_PS} ps")

    r1_rows: List[Dict] = []
    r2_rows: List[Dict] = []
    if args.mode in ("r1", "both"):
        r1_rows = run_r1(bt)
    if args.mode in ("r2", "both"):
        r2_rows = run_r2(bt, args.max_refine, force=args.force)

    print("\n" + "=" * 72)
    print("PHASE-0G SUMMARY")
    print("=" * 72)
    if r1_rows:
        print("\nR1 integrator selection:")
        print(f"  {'variant':34s} {'period(ps)':>11s} {'perErr%':>8s} {'NRMSE%':>8s} {'verdict':>8s}")
        for r in r1_rows:
            print(f"  {r['variant']:34s} {r['period_ps']:11.2f} {r['perr_pct']:8.2f} "
                  f"{r.get('nrmse_pct', float('nan')):8.2f} "
                  f"{'PASS' if r['period_ps'] <= GATE_PS else 'FAIL':>8s}")
    if r2_rows:
        print("\nR2 tstep convergence:")
        print(f"  {'div/extrap':16s} {'tstep(ps)':>10s} {'period(ps)':>11s} {'perErr%':>8s} {'wall(s)':>8s}")
        for r in r2_rows:
            print(f"  {str(r['div']):16s} {r.get('tstep_ps', float('nan')):10.3f} "
                  f"{r['period_ps']:11.2f} {r['perr_pct']:8.2f} "
                  f"{r.get('wall_s', float('nan')):8.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
