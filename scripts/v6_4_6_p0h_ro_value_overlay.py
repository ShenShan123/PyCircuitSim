#!/usr/bin/env python3
"""DirectNet V6.4.6 — Diagnostic P0-H: RO-trip VALUE overlay (TSMC7).

INSTRUMENTATION-ONLY. The 0-GPU diagnostic that P0-C *motivated* but nobody had
run. P0-B overlaid the model's *Jacobian* (gm/gds, caps) and P0-C proved those
surfaces are causally INERT on the RO period (they are Jacobian-only and cancel
at the converged NR fixed point). P0-C's conclusion: the 4.18 ps TSMC7 RO gap
(DN 50.82 ps vs NG 46.64 ps) is owned by the **id-VALUE + charge-VALUE (qg/qd)
trajectories + BE/Trap/BDF-2 truncation** — none of which P0-B/P0-C measured.

This script overlays exactly those VALUES the transient companion model stamps:

  resistive companion current  <- id VALUE     (_stamp_mosfet_dc:304,
                                                 i_eq = i_leaving - g_ds*v_ds-...)
  capacitive companion current <- qg/qd VALUES (_stamp_mosfet_transient:1772,
                                                 i_g_cap = coeff*charges["qg"]-h_g)

Unlike the Jacobian (P0-B/P0-C), these VALUES do NOT cancel at the fixed point,
so if they diverge from analytic OSDI along the trajectory they directly shift
the period.

Surfaces overlaid (NN post-Rule-15 VALUE vs analytic OSDI VALUE), per device:
  id : the post-Rule-15 forward-pass current value the resistive stamp consumes.
  qg : the post-Rule-15 gate-charge value the capacitive gate stamp consumes.
  qd : the post-Rule-15 drain-charge value the capacitive drain stamp consumes.
  qs : qs = -(qg+qd+qb), enforced analytically by the simulator (Rule 14), for
       BOTH the NN and OSDI, then overlaid.

Sign / convention (VALUES, not derivatives — so NO off-diagonal negation):
  - id  : trained directly on OSDI col 0 (NMOS<0 conducting, PMOS>0) -> direct.
  - qg/qd/qb : trained directly on OSDI cols 4/5/7 (dataset stores result[k]
    straight from eval_single_point, nn_generate.py:657) -> SAME convention,
    direct compare. (Only the cap *derivatives* needed negation in P0-B.)
  - PMOS bias frame: the NN evaluates internally source-shifted (Vs=0); OSDI is
    queried at the absolute PMOS terminal bias (vn5, vn4, VDD, VDD), exactly as
    P0-B did. The device charge state at a given physical terminal bias is
    frame-independent; the dataset trained on the vs=0 frame == OSDI at the
    physical bias because the internal shift is identical at train and inference.

Ground truth is ALWAYS the OSDI binary via PyCMG (CLAUDE.md Validation rule).

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      conda run -n pycircuitsim python scripts/v6_4_6_p0h_ro_value_overlay.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, List

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
from tests.common.complex import parse_netlist, run_directnet_transient  # noqa: E402
from pycmg.nn_generate import _create_model_and_instance, eval_single_point  # noqa: E402
from pycmg.nn_config import TECH_CONFIGS  # noqa: E402

# The VALUE surfaces the transient companion model stamps directly.
VALUE_SURFACES = ("id", "qg", "qd", "qs")


def rule16(nn: np.ndarray, osdi: np.ndarray) -> Dict[str, float]:
    """Rule-16 quartet of nn-vs-osdi. MRE/NRMSE robust to near-zero true.

    Identical to the P0-B estimator so the two overlays are directly
    comparable (derivative overlay there, value overlay here).
    """
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
            "max_err": float(np.max(np.abs(diff))),
            "osdi_ptp": ptp, "osdi_absmax": scale, "rmse": rmse}


def build_overlay(tech_name: str = "TSMC7"):
    bt = BENCH[tech_name]
    VDD = bt.vdd
    work_dir = Path(tempfile.mkdtemp(prefix="p0h_"))

    # --- 1. Run the real DirectNet RO transient (same path as the gate) ---
    dn, partial, err = run_directnet_ro(bt, work_dir)
    t = dn["time"]
    v5 = dn["v(n5)"]
    mid = VDD / 2.0
    dn_per = _period_from_wave(t, v5, mid, SETTLE)
    print(f"[P0-H] DN RO period = {dn_per * 1e12:.2f} ps  "
          f"(partial={partial}, err={err!r})")

    # Re-parse the SAME rendered netlist the gate produced, and re-run the
    # transient ourselves to capture ALL node waveforms (n4 = the gate node).
    nl_path = work_dir / f"ring_osc_{bt.name}.sp"
    parser = parse_netlist(nl_path)
    comps = {c.name: c for c in parser.circuit.components}
    mn = comps["Mn5"]   # NMOS: d=n5 g=n4 s=0  b=0
    mp = comps["Mp5"]   # PMOS: d=n5 g=n4 s=vdd b=vdd
    assert not mn._is_pmos and mp._is_pmos

    results, partial2, err2 = run_directnet_transient(nl_path)
    tt = np.asarray(results["time"])
    n5 = np.asarray(results["n5"])
    n4 = np.asarray(results["n4"])
    per_check = _period_from_wave(tt, n5, mid, SETTLE)
    print(f"[P0-H] re-run period (all nodes) = {per_check * 1e12:.2f} ps")

    # --- 2. Select ONE full post-settle oscillation cycle of timesteps ---
    # (identical window selection to P0-B so the bias points coincide.)
    keep = tt >= SETTLE
    tk, n5k, n4k = tt[keep], n5[keep], n4[keep]
    sign = np.sign(n5k - mid)
    rising = np.where((sign[:-1] < 0) & (sign[1:] >= 0))[0]
    if len(rising) < 2:
        raise RuntimeError("fewer than 2 rising crossings post-settle")
    i0, i1 = int(rising[0]), int(rising[1])
    idx = np.arange(i0, i1 + 1)
    cyc_t = tk[idx]
    cyc_n5 = n5k[idx]
    cyc_n4 = n4k[idx]
    print(f"[P0-H] cycle window: {len(idx)} timesteps "
          f"({cyc_t[0] * 1e12:.2f} -> {cyc_t[-1] * 1e12:.2f} ps), "
          f"span {(cyc_t[-1] - cyc_t[0]) * 1e12:.2f} ps")

    # --- 3. OSDI instances (one per device_type, cached) ---
    tech = TECH_CONFIGS[bt.nn_tech]
    _, inst_n, _ = _create_model_and_instance(
        tech, "nmos", bt.vt, bt.l_nmos, float(bt.nfin), 300.15)
    _, inst_p, _ = _create_model_and_instance(
        tech, "pmos", bt.vt, bt.l_pmos, float(bt.nfin), 300.15)

    nn_acc: Dict[str, Dict[str, List[float]]] = {
        "nmos": {s: [] for s in VALUE_SURFACES},
        "pmos": {s: [] for s in VALUE_SURFACES}}
    os_acc: Dict[str, Dict[str, List[float]]] = {
        "nmos": {s: [] for s in VALUE_SURFACES},
        "pmos": {s: [] for s in VALUE_SURFACES}}
    osdi_fail = {"nmos": 0, "pmos": 0}
    n_pts = 0

    def qs_of(d: Dict[str, float]) -> float:
        """qs = -(qg+qd+qb), the Rule-14 analytic reconstruction."""
        return -(d["qg"] + d["qd"] + d["qb"])

    for vn5, vn4 in zip(cyc_n5, cyc_n4):
        volt = {"n5": float(vn5), "n4": float(vn4), "0": 0.0,
                "vdd": float(VDD)}
        # NMOS: d=n5 g=n4 s=0 b=0
        mn.clear_cache()
        rn = mn._eval(volt)
        on = eval_single_point(inst_n, float(vn5), float(vn4), 0.0, 0.0)
        # PMOS: d=n5 g=n4 s=vdd b=vdd (absolute frame, same as P0-B)
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
            # id / qg / qd are direct VALUE comparisons (same convention).
            nn_acc[dev]["id"].append(r["id"])
            os_acc[dev]["id"].append(o["id"])
            nn_acc[dev]["qg"].append(r["qg"])
            os_acc[dev]["qg"].append(o["qg"])
            nn_acc[dev]["qd"].append(r["qd"])
            os_acc[dev]["qd"].append(o["qd"])
            # qs reconstructed the same way (Rule 14) for both NN and OSDI.
            nn_acc[dev]["qs"].append(qs_of(r))
            os_acc[dev]["qs"].append(qs_of(o))

    print(f"[P0-H] bias points used = {n_pts}  "
          f"(OSDI fails: NMOS {osdi_fail['nmos']}, PMOS {osdi_fail['pmos']})")

    # --- 4. Rule-16 quartet per VALUE surface, per device ---
    report: Dict[str, Dict[str, Dict[str, float]]] = {"nmos": {}, "pmos": {}}
    for dev in ("nmos", "pmos"):
        for s in VALUE_SURFACES:
            report[dev][s] = rule16(
                np.array(nn_acc[dev][s]), np.array(os_acc[dev][s]))

    return {
        "tech": tech_name, "vdd": VDD, "dn_period_ps": dn_per * 1e12,
        "rerun_period_ps": per_check * 1e12,
        "n_cycle_steps": len(idx), "n_pts": n_pts,
        "cycle_span_ps": (cyc_t[-1] - cyc_t[0]) * 1e12,
        "osdi_fail": osdi_fail, "report": report,
        "nn_acc": nn_acc, "os_acc": os_acc,
    }


def fmt_q(m: Dict[str, float], scale: float, unit: str) -> str:
    return (f"MRE={m['mre_pct']:8.2f}%  R2={m['r2']:9.4f}  "
            f"NRMSE={m['nrmse_pct']:8.2f}%  "
            f"MaxErr={m['max_err'] * scale:.4g}{unit}  "
            f"(OSDI ptp={m['osdi_ptp'] * scale:.4g}{unit})")


def main() -> int:
    out = build_overlay("TSMC7")
    # display scale/unit for MaxErr: current in uA, charge in aC (1e-18 C).
    units = {"id": (1e6, "uA"),
             "qg": (1e18, "aC"), "qd": (1e18, "aC"), "qs": (1e18, "aC")}
    print("\n" + "=" * 78)
    print("P0-H SUMMARY — TSMC7 RO-trip VALUE overlay "
          "(NN post-Rule-15 VALUE vs analytic OSDI)")
    print("=" * 78)
    print(f"DN period {out['dn_period_ps']:.2f} ps vs NG 46.64 ps "
          f"(gap 4.18 ps, 8.97%); cycle = {out['n_pts']} bias points")
    for dev in ("nmos", "pmos"):
        print(f"\n--- {dev.upper()} (stage-5 device driving n5) ---")
        for s in VALUE_SURFACES:
            sc, un = units[s]
            print(f"  {s:3s}: {fmt_q(out['report'][dev][s], sc, un)}")

    # Per-cycle charge-swing context (per-stage load ~0.5 fF, like P0-B).
    print("\n--- per-cycle charge swing context (qg/qd VALUE) ---")
    for dev in ("nmos", "pmos"):
        for s in ("qg", "qd"):
            os_arr = np.array(out["os_acc"][dev][s])
            nn_arr = np.array(out["nn_acc"][dev][s])
            swing_os = float(np.ptp(os_arr)) * 1e18
            swing_nn = float(np.ptp(nn_arr)) * 1e18
            maxerr = float(np.max(np.abs(nn_arr - os_arr))) * 1e18
            pct = (maxerr / swing_os * 100.0) if swing_os > 1e-30 else float("nan")
            print(f"  {dev} {s}: OSDI swing {swing_os:7.2f} aC, "
                  f"NN swing {swing_nn:7.2f} aC, MaxErr {maxerr:7.2f} aC "
                  f"= {pct:6.1f}% of OSDI swing  "
                  f"(stage Cload 0.5 fF -> {0.5e3:.0f} aC per 1 V swing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
