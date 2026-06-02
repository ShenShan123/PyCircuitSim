#!/usr/bin/env python3
"""DirectNet V6.4.6 — Diagnostic P0-B: RO-trip Jacobian overlay (TSMC7).

INSTRUMENTATION-ONLY. Determines WHICH model surface (gds vs caps) owns the
4.2 ps TSMC7 ring-oscillator phase walk, by overlaying the *post-Rule-15*
DirectNet autograd Jacobian (gds/gm) and charge-Jacobian (caps) against the
analytic OSDI ground truth at the SAME bias points along one real RO
oscillation cycle.

Ground truth is ALWAYS the OSDI binary via PyCMG (CLAUDE.md Validation rule).

Sign / convention alignment (verified by finite-difference against the OSDI
instance — see results/v6_4_6/phase0_B_ro_jacobian_overlay.md "Conventions"):

  id    : same sign  (NMOS conducting id<0; PMOS id>0)            -> direct
  gm    : NN result["gm"] = -d(id)/dVg ; OSDI gm = -d(id)/dVg     -> direct
  gds   : NN result["gds"] = +d(id)/dVd, FLOORED to >=|id|*0.5;
          OSDI gds = -d(id)/dVd (true physical, positive).
          Both positive -> direct compare; we FLAG where the floor binds
          (NN gds == max(|id|*0.5, 1e-12) within 1%).
  cgg   : NN = +d(qg)/dVg ; OSDI cgg = +d(qg)/dVg                 -> same sign
  cdd   : NN = +d(qd)/dVd ; OSDI cdd = +d(qd)/dVd                 -> same sign
  cgd   : NN = +d(qg)/dVd ; OSDI cgd = -d(qg)/dVd  => NN = -OSDI  -> negate OSDI
  cdg   : NN = +d(qd)/dVg ; OSDI cdg = -d(qd)/dVg  => NN = -OSDI  -> negate OSDI
  cgs   : grounded-source NMOS => not NR-load-bearing; reported, flagged.

For every surface we report the Rule-16 quartet (MRE%, R2, NRMSE%, MaxErr)
of NN-vs-OSDI over the cycle's bias points, separately for the NMOS and PMOS
of the stage that drives n5.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      conda run -n pycircuitsim python scripts/v6_4_6_p0b_ro_overlay.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Insert order matters: PyCMG has its OWN `tests/` dir, so `tests.common`
# must resolve to ROOT/tests. Insert ROOT LAST (insert(0)) so it wins the
# `tests` package lookup, while PyCMG stays on-path for `pycmg`.
for p in (ROOT / "external_compact_models" / "PyCMG" / "tests",
          ROOT / "external_compact_models" / "PyCMG",
          ROOT / "external_compact_models",
          ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tests.common.complex import BENCH  # noqa: E402
from tests.verify_complex_ring_osc import (  # noqa: E402
    run_directnet_ro, _period_from_wave, SETTLE,
)
from tests.common.complex import render_directnet_netlist, parse_netlist  # noqa: E402
from pycmg.nn_generate import _create_model_and_instance, eval_single_point  # noqa: E402
from pycmg.nn_config import TECH_CONFIGS  # noqa: E402

TEMPLATE = ROOT / "examples" / "complex" / "ring_osc_5stage_directnet.sp"
CAP_KEYS = ("cgg", "cgd", "cgs", "cdg", "cdd")
SURFACES = ("gds", "gm", "cgd", "cgg", "cdg", "cdd")


def rule16(nn: np.ndarray, osdi: np.ndarray) -> Dict[str, float]:
    """Rule-16 quartet of nn-vs-osdi. MRE/NRMSE robust to near-zero true."""
    nn = np.asarray(nn, float)
    osdi = np.asarray(osdi, float)
    diff = nn - osdi
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((osdi - osdi.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-300 else float("nan")
    ptp = float(np.ptp(osdi))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    nrmse = rmse / ptp * 100.0 if ptp > 1e-300 else float("nan")
    # MRE: relative to |osdi|, guarded with a per-surface scale floor so a
    # single near-zero true value does not blow up the mean.
    scale = max(float(np.max(np.abs(osdi))), 1e-30)
    denom = np.maximum(np.abs(osdi), 1e-3 * scale)
    mre = float(np.mean(np.abs(diff) / denom)) * 100.0
    return {"mre_pct": mre, "r2": r2, "nrmse_pct": nrmse,
            "max_err": float(np.max(np.abs(diff)))}


def build_overlay(tech_name: str = "TSMC7"):
    bt = BENCH[tech_name]
    VDD = bt.vdd
    work_dir = Path(tempfile.mkdtemp(prefix="p0b_"))

    # --- 1. Run the real DirectNet RO transient (same path as the gate) ---
    dn, partial, err = run_directnet_ro(bt, work_dir)
    t = dn["time"]
    v5 = dn["v(n5)"]
    mid = VDD / 2.0
    dn_per = _period_from_wave(t, v5, mid, SETTLE)
    print(f"[P0-B] DN RO period = {dn_per*1e12:.2f} ps  "
          f"(partial={partial}, err={err!r})")

    # We also need n4 (the predecessor / gate node) on the same time grid.
    # Re-parse the SAME rendered netlist that run_directnet_ro produced so the
    # device objects and node waveforms are consistent. run_directnet_ro wrote
    # ring_osc_<name>.sp into work_dir and rewrote the .tran line.
    nl_path = work_dir / f"ring_osc_{bt.name}.sp"
    parser = parse_netlist(nl_path)
    comps = {c.name: c for c in parser.circuit.components}
    mn = comps["Mn5"]   # NMOS: d=n5 g=n4 s=0  b=0
    mp = comps["Mp5"]   # PMOS: d=n5 g=n4 s=vdd b=vdd
    assert not mn._is_pmos and mp._is_pmos

    # Re-run the transient ourselves so we capture ALL node waveforms (the
    # gate verifier only returns v(n5)). Reuse run_directnet_transient on the
    # already-rewritten netlist -> identical numbers to run_directnet_ro.
    from tests.common.complex import run_directnet_transient
    results, partial2, err2 = run_directnet_transient(nl_path)
    tt = np.asarray(results["time"])
    n5 = np.asarray(results["n5"])
    n4 = np.asarray(results["n4"])
    per_check = _period_from_wave(tt, n5, mid, SETTLE)
    print(f"[P0-B] re-run period (all nodes) = {per_check*1e12:.2f} ps")

    # --- 2. Select ONE full post-settle oscillation cycle of timesteps ---
    keep = tt >= SETTLE
    tk, n5k, n4k = tt[keep], n5[keep], n4[keep]
    sign = np.sign(n5k - mid)
    rising = np.where((sign[:-1] < 0) & (sign[1:] >= 0))[0]
    if len(rising) < 2:
        raise RuntimeError("fewer than 2 rising crossings post-settle")
    i0, i1 = int(rising[0]), int(rising[1])   # one full cycle [i0, i1]
    idx = np.arange(i0, i1 + 1)
    cyc_t = tk[idx]
    cyc_n5 = n5k[idx]
    cyc_n4 = n4k[idx]
    print(f"[P0-B] cycle window: {len(idx)} timesteps "
          f"({cyc_t[0]*1e12:.2f} -> {cyc_t[-1]*1e12:.2f} ps), "
          f"span {(cyc_t[-1]-cyc_t[0])*1e12:.2f} ps")

    # --- 3. OSDI instances (one per device_type, cached) ---
    tech = TECH_CONFIGS[bt.nn_tech]
    _, inst_n, _ = _create_model_and_instance(
        tech, "nmos", bt.vt, bt.l_nmos, float(bt.nfin), 300.15)
    _, inst_p, _ = _create_model_and_instance(
        tech, "pmos", bt.vt, bt.l_pmos, float(bt.nfin), 300.15)

    # accumulators
    nn_acc: Dict[str, Dict[str, List[float]]] = {
        "nmos": {s: [] for s in SURFACES + ("id", "cgs")},
        "pmos": {s: [] for s in SURFACES + ("id", "cgs")}}
    os_acc: Dict[str, Dict[str, List[float]]] = {
        "nmos": {s: [] for s in SURFACES + ("id", "cgs")},
        "pmos": {s: [] for s in SURFACES + ("id", "cgs")}}
    floor_hits = {"nmos": 0, "pmos": 0}
    osdi_fail = {"nmos": 0, "pmos": 0}
    n_pts = 0

    # cap off-diagonals: NN = -OSDI ; diagonals same sign.
    # We align by mapping OSDI -> NN convention so the overlay is apples-apples
    # in the SAME convention the NR solver consumes (NN convention).
    def osdi_in_nn_conv(o: Dict[str, float], key: str) -> float:
        if key in ("cgd", "cdg"):
            return -o[key]
        return o[key]

    for vn5, vn4 in zip(cyc_n5, cyc_n4):
        volt = {"n5": float(vn5), "n4": float(vn4), "0": 0.0,
                "vdd": float(VDD)}
        # NMOS: d=n5 g=n4 s=0 b=0
        mn.clear_cache()
        rn = mn._eval(volt)
        on = eval_single_point(inst_n, float(vn5), float(vn4), 0.0, 0.0)
        # PMOS: d=n5 g=n4 s=vdd b=vdd
        mp.clear_cache()
        rp = mp._eval(volt)
        op = eval_single_point(inst_p, float(vn5), float(vn4),
                               float(VDD), float(VDD))
        if on is None or op is None:
            if on is None:
                osdi_fail["nmos"] += 1
            if op is None:
                osdi_fail["pmos"] += 1
            continue
        n_pts += 1
        for dev, r, o in (("nmos", rn, on), ("pmos", rp, op)):
            for s in SURFACES + ("id", "cgs"):
                nn_acc[dev][s].append(r[s])
                os_acc[dev][s].append(osdi_in_nn_conv(o, s))
            # floor-binding flag: NN gds within 1% of the |id|*0.5 floor
            floor_val = max(abs(r["id"]) * 0.5, 1e-12)
            if abs(r["gds"] - floor_val) <= 0.01 * floor_val:
                floor_hits[dev] += 1

    print(f"[P0-B] bias points used = {n_pts}  "
          f"(OSDI fails: NMOS {osdi_fail['nmos']}, PMOS {osdi_fail['pmos']})")
    print(f"[P0-B] gds-floor binds: NMOS {floor_hits['nmos']}/{n_pts}, "
          f"PMOS {floor_hits['pmos']}/{n_pts}")

    # --- 4. Rule-16 quartet per surface, per device ---
    report: Dict[str, Dict[str, Dict[str, float]]] = {"nmos": {}, "pmos": {}}
    for dev in ("nmos", "pmos"):
        for s in SURFACES + ("id", "cgs"):
            report[dev][s] = rule16(
                np.array(nn_acc[dev][s]), np.array(os_acc[dev][s]))

    return {
        "tech": tech_name, "vdd": VDD, "dn_period_ps": dn_per * 1e12,
        "rerun_period_ps": per_check * 1e12,
        "n_cycle_steps": len(idx), "n_pts": n_pts,
        "cycle_span_ps": (cyc_t[-1] - cyc_t[0]) * 1e12,
        "floor_hits": floor_hits, "osdi_fail": osdi_fail,
        "report": report,
        "nn_acc": nn_acc, "os_acc": os_acc,
    }


def fmt_q(m: Dict[str, float], scale: float, unit: str) -> str:
    return (f"MRE={m['mre_pct']:8.2f}%  R2={m['r2']:9.4f}  "
            f"NRMSE={m['nrmse_pct']:8.2f}%  "
            f"MaxErr={m['max_err']*scale:.4g}{unit}")


def main() -> int:
    out = build_overlay("TSMC7")
    print("\n" + "=" * 78)
    print("P0-B SUMMARY — TSMC7 RO-trip Jacobian overlay (NN post-Rule-15 vs OSDI)")
    print("=" * 78)
    # surface display scale/unit for MaxErr
    units = {"gds": (1e6, "uS"), "gm": (1e6, "uS"), "id": (1e6, "uA"),
             "cgd": (1e18, "aF"), "cgg": (1e18, "aF"), "cdg": (1e18, "aF"),
             "cdd": (1e18, "aF"), "cgs": (1e18, "aF")}
    for dev in ("nmos", "pmos"):
        print(f"\n--- {dev.upper()} (stage-5 device driving n5) ---")
        for s in ("gds", "gm", "id", "cgd", "cgg", "cdg", "cdd", "cgs"):
            sc, un = units[s]
            print(f"  {s:4s}: {fmt_q(out['report'][dev][s], sc, un)}")
    print(f"\n  gds-floor binds: NMOS {out['floor_hits']['nmos']}/{out['n_pts']}"
          f", PMOS {out['floor_hits']['pmos']}/{out['n_pts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
