"""V6.4.5 Phase-3 multi-circuit candidate evaluator.

Scores ONE (tech, nmos-stem, pmos-stem) candidate against the V6.4.5
multi-circuit selection vector (plan §Phase 3):

    inv_vtc_nrmse        — inverter VTC NRMSE %  (hard gate <= 5 %)
    inv_tran_post_nrmse  — inverter post-startup transient NRMSE %  (hard gate <= 5 %)
    ring_osc_period_err  — 5-stage RO period error %  (Pareto objective, TSMC7 gate <= 5 %)
    sram_rail_snap_resid — force_ic state-1 rail residual max(|q-VDD|,|qb-0|)/VDD  (Pareto objective)
    opamp_flat_flag      — 1 if |Vout(center bias) - VDD/2| > 0.3*VDD  (hard gate == 0)

Mechanism: instead of swapping the shared canonical slots (which would
serialise the search and risk corrupting the canonical checkpoints), this
evaluator runs in an ISOLATED checkpoint dir. It mkdtemp's a private dir,
copies the two candidate checkpoints into it UNDER the canonical
``tsmc{X}_dn_medium_{dev}`` stem names — so the parser's vocab-scope
detection (Rule 19, keys off the ``tsmc{5,7}_dn_*`` stem) still applies the
correct local embedding — points ``BSIMAR_CHECKPOINT_DIR`` at it (read by
``bsimar.config`` at import time), runs all four micro-benchmarks in THIS
process, prints a machine-parseable RESULT line, and removes the temp dir.
The real ``checkpoints/`` directory is NEVER mutated, so many candidates can
run concurrently.

The env var MUST be set before the first ``bsimar.config`` import; all
bsimar imports in this module are lazy (inside the _score_* helpers) and
fire only after main() has set it.

Usage:
    conda run -n pycircuitsim python scripts/eval_v6_4_5_candidate.py \
        --tech TSMC7 --nmos v6_4_2_p7_tsmc7_stock_s42_nmos \
        --pmos v6_4_2_p7_tsmc7_stock_s42_pmos --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))

# The REAL on-disk checkpoint directory (candidate stems live here). This is
# resolved from PROJECT_ROOT, NOT from BSIMAR_CHECKPOINT_DIR — the env var we
# set below redirects only where the parser/model LOAD from.
CKPT_DIR = PROJECT_ROOT / "external_compact_models" / "bsimar" / "checkpoints"


def _files(stem: str) -> tuple[Path, Path]:
    return CKPT_DIR / f"{stem}_best.pt", CKPT_DIR / f"{stem}_norm.npz"


# ---------------------------------------------------------------------------
# Per-circuit micro-benchmarks (all DirectNet-side; NGSPICE refs recomputed
# but cheap relative to the DirectNet transient).
# ---------------------------------------------------------------------------
def _score_inverter(tech: str) -> dict:
    from scripts.eval_v6_3_1_inverter import evaluate_inverter
    with tempfile.TemporaryDirectory(prefix="v645_inv_") as tmp:
        res = evaluate_inverter(tech, Path(tmp))
    v, tp = res["vtc"], res["tran_post"]
    return {
        "inv_vtc_nrmse": v["NRMSE_vdd(%)"],
        "inv_vtc_maxerr_mv": v["MaxErr(V)"] * 1e3,
        "inv_vtc_r2": v["R2"],
        "inv_vtc_dvtrip_mv": v.get("Vtrip_err(mV)", float("nan")),
        "inv_tran_post_nrmse": tp["NRMSE_vdd(%)"],
        "inv_tran_post_maxerr_mv": tp["MaxErr(V)"] * 1e3,
    }


def _score_ring_osc(tech: str) -> dict:
    from tests.common.complex import BENCH, RESULTS_BASE
    from tests.verify_complex_ring_osc import (
        run_ngspice_ro, run_directnet_ro, _period_from_wave,
        SETTLE, TRAN_TSTEP, full_metrics,
    )
    bt = BENCH[tech]
    work_dir = RESULTS_BASE / "ring_osc" / bt.name
    work_dir.mkdir(parents=True, exist_ok=True)
    mid = bt.vdd / 2.0
    ng = run_ngspice_ro(bt, work_dir)
    ng_per = _period_from_wave(ng["time"], ng["v(n5)"], mid, SETTLE)
    out = {"ro_ng_period_ps": ng_per * 1e12,
           "ring_osc_period_err": float("nan"),
           "ro_nrmse": float("nan"), "ro_r2": float("nan"),
           "ro_partial": False}
    try:
        dn, partial, _ = run_directnet_ro(bt, work_dir)
    except Exception as exc:  # noqa: BLE001
        out["ro_error"] = repr(exc)
        return out
    dn_per = _period_from_wave(dn["time"], dn["v(n5)"], mid, SETTLE)
    out["ro_dn_period_ps"] = dn_per * 1e12
    out["ro_partial"] = partial
    if np.isfinite(dn_per) and ng_per > 0:
        out["ring_osc_period_err"] = abs(dn_per - ng_per) / ng_per * 100.0
    t_lo, t_hi = SETTLE, min(ng["time"][-1], dn["time"][-1])
    if t_hi > t_lo and len(dn["time"]) > 3:
        grid = np.arange(t_lo, t_hi, TRAN_TSTEP)
        m = full_metrics(np.interp(grid, dn["time"], dn["v(n5)"]),
                         np.interp(grid, ng["time"], ng["v(n5)"]))
        out["ro_nrmse"] = m["nrmse_pct"]
        out["ro_r2"] = m["r2"]
    return out


def _score_sram_rail(tech: str) -> dict:
    """force_ic state-1 (q=VDD, qb=0) rail-snap residual."""
    from pycircuitsim.solver import DCSolver
    from tests.common.complex import BENCH, RESULTS_BASE, parse_netlist
    from tests.verify_complex_sram_snm import _directnet_6t_netlist
    bt = BENCH[tech]
    work_dir = RESULTS_BASE / "sram_snm" / bt.name
    work_dir.mkdir(parents=True, exist_ok=True)
    netlist = _directnet_6t_netlist(
        bt, bt.vdd, 0.0, work_dir / f"sram6t_{bt.name}_score_state1.sp")
    logging.disable(logging.CRITICAL)
    q_v = qb_v = float("nan")
    try:
        parser = parse_netlist(netlist)
        circuit = parser.circuit
        guess = circuit.initial_conditions or None
        solver = DCSolver(circuit, initial_guess=guess,
                          use_source_stepping=True, force_ic=True)
        sol = solver.solve()
        q_v = float(sol.get("q", float("nan")))
        qb_v = float(sol.get("qb", float("nan")))
    except Exception as exc:  # noqa: BLE001
        logging.disable(logging.NOTSET)
        return {"sram_rail_snap_resid": float("nan"),
                "sram_q": q_v, "sram_qb": qb_v, "sram_error": repr(exc)}
    finally:
        logging.disable(logging.NOTSET)
    resid = max(abs(q_v - bt.vdd), abs(qb_v - 0.0)) / bt.vdd
    return {"sram_rail_snap_resid": resid, "sram_q": q_v, "sram_qb": qb_v}


def _ng_memo(tech: str, kind: str, runner) -> dict:
    """File-memoized NGSPICE reference (candidate-independent, so it is
    computed once per (tech, circuit) and reused across the campaign)."""
    from tests.common.complex import RESULTS_BASE
    memo = RESULTS_BASE / f"ng_ref_{kind}_{tech}.json"
    if memo.exists():
        return json.loads(memo.read_text())
    val = runner()
    memo.parent.mkdir(parents=True, exist_ok=True)
    memo.write_text(json.dumps(val))
    return val


def _score_opamp(tech: str) -> dict:
    """Opamp gain error vs the NGSPICE reference (V6.4.7 S8: the scorer
    previously returned only flat-flag + raw gain — blind to the ±10 %
    gain gate) + the V6.4.5-recalibrated flat flag (gain < 10)."""
    from tests.common.complex import BENCH, RESULTS_BASE
    from tests.verify_complex_opamp import (
        run_directnet_opamp, run_ngspice_opamp, _bias, _gain_trip)
    bt = BENCH[tech]
    work_dir = RESULTS_BASE / "opamp" / bt.name
    work_dir.mkdir(parents=True, exist_ok=True)
    vcm, _, _ = _bias(bt)

    def _ng() -> dict:
        ng = run_ngspice_opamp(bt, work_dir)
        g, _, _ = _gain_trip(ng["sweep"], ng["vout"], bt.vdd)
        return {"gain": float(g)}

    ng_gain = float(_ng_memo(tech, "opamp", _ng)["gain"])
    try:
        dn = run_directnet_opamp(bt, work_dir)
    except Exception as exc:  # noqa: BLE001
        return {"opamp_flat_flag": 1, "opamp_gain": float("nan"),
                "opamp_ng_gain": ng_gain, "opamp_gain_err": float("nan"),
                "opamp_vout_center": float("nan"), "opamp_error": repr(exc)}
    ix = int(np.argmin(np.abs(dn["sweep"] - vcm)))
    vout_center = float(dn["vout"][ix])
    gain, _, _ = _gain_trip(dn["sweep"], dn["vout"], bt.vdd)
    # V6.4.5 re-calibration (CHANGELOG "V6.4.5"): vout-at-center flagged
    # even the passing TSMC5 opamp; flat == collapsed gain.
    flat = float(gain) < 10.0
    gain_err = (abs(float(gain) - ng_gain) / ng_gain * 100.0
                if ng_gain > 0 else float("nan"))
    return {"opamp_flat_flag": int(flat), "opamp_gain": float(gain),
            "opamp_ng_gain": ng_gain, "opamp_gain_err": gain_err,
            "opamp_vout_center": vout_center}


def _score_switchcap(tech: str) -> dict:
    """Switchcap charge-transfer + droop vs the repaired S3/R0.1 gate
    (V6.4.7 S8: the scorer previously had no SC cells at all)."""
    from tests.common.complex import BENCH, RESULTS_BASE
    from tests.verify_complex_switchcap import (
        run_ngspice_sc, run_directnet_sc, _at,
        SAMPLE_END, HOLD_START, HOLD_END, CHARGE_TOL, DROOP_TOL,
        DROOP_FLOOR_FRAC)
    bt = BENCH[tech]
    work_dir = RESULTS_BASE / "switchcap" / bt.name
    work_dir.mkdir(parents=True, exist_ok=True)

    def _ng() -> dict:
        ng = run_ngspice_sc(bt, work_dir)
        return {"charge": _at(ng["time"], ng["vsamp"], SAMPLE_END),
                "droop": (_at(ng["time"], ng["vsamp"], HOLD_START)
                          - _at(ng["time"], ng["vsamp"], HOLD_END))}

    ref = _ng_memo(tech, "switchcap", _ng)
    try:
        dn, partial, err = run_directnet_sc(bt, work_dir)
        if partial:
            raise RuntimeError(f"partial transient: {err}")
    except Exception as exc:  # noqa: BLE001
        return {"sc_charge_err_pct": float("nan"),
                "sc_droop_abs_mv": float("nan"),
                "sc_pass": 0, "sc_error": repr(exc)}
    t = np.asarray(dn["time"])
    v = np.asarray(dn["vsamp"])
    dn_charge = _at(t, v, SAMPLE_END)
    dn_droop = _at(t, v, HOLD_START) - _at(t, v, HOLD_END)
    charge_err = abs(dn_charge - ref["charge"]) / bt.vdd * 100.0
    droop_abs = abs(dn_droop - ref["droop"])
    allow = max(DROOP_TOL * abs(ref["droop"]), DROOP_FLOOR_FRAC * bt.vdd)
    ok = charge_err <= CHARGE_TOL * 100 and droop_abs <= allow
    return {"sc_charge_err_pct": charge_err,
            "sc_droop_abs_mv": droop_abs * 1e3,
            "sc_droop_pct_of_allow": droop_abs / allow * 100.0,
            "sc_pass": int(ok)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", required=True)
    ap.add_argument("--nmos", required=True, help="nmos checkpoint stem")
    ap.add_argument("--pmos", required=True, help="pmos checkpoint stem")
    ap.add_argument("--json", action="store_true",
                    help="emit one RESULT JSON line for machine parsing")
    ap.add_argument("--skip", default="",
                    help="comma-separated benchmarks to skip "
                         "(inv,ro,sram,opamp) — for fast debugging only")
    args = ap.parse_args()
    tech = args.tech
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    for dev, stem in (("nmos", args.nmos), ("pmos", args.pmos)):
        pt, nz = _files(stem)
        if not pt.exists() or not nz.exists():
            sys.exit(f"missing candidate: {pt} / {nz}")

    # Isolated checkpoint dir: copy candidates in under the canonical stem
    # names so vocab-scope detection fires, then redirect loading via env.
    iso = Path(tempfile.mkdtemp(prefix="v645_ckpt_"))
    os.environ["BSIMAR_CHECKPOINT_DIR"] = str(iso)
    try:
        for dev, stem in (("nmos", args.nmos), ("pmos", args.pmos)):
            spt, snz = _files(stem)
            slot = f"{tech.lower()}_dn_medium_{dev}"
            shutil.copy2(spt, iso / f"{slot}_best.pt")
            shutil.copy2(snz, iso / f"{slot}_norm.npz")

        # Redirect harness output dirs into the isolated dir so concurrent
        # candidates don't race on shared netlist / NGSPICE temp / trace files.
        # (Done after BSIMAR_CHECKPOINT_DIR is set; these imports pull
        # bsimar.config, which reads the env var at import.)
        import tests.common.complex as _cplx
        _cplx.RESULTS_BASE = iso / "work"
        import scripts.eval_v6_3_1_inverter as _ev
        _ev.REPORT_DIR = iso / "work" / "inv_report"
        _ev.REPORT_DIR.mkdir(parents=True, exist_ok=True)

        out = {"tech": tech, "nmos": args.nmos, "pmos": args.pmos}
        if "inv" not in skip:
            out.update(_score_inverter(tech))
        if "ro" not in skip:
            out.update(_score_ring_osc(tech))
        if "sram" not in skip:
            out.update(_score_sram_rail(tech))
        if "opamp" not in skip:
            out.update(_score_opamp(tech))
        if "switchcap" not in skip and "sc" not in skip:
            out.update(_score_switchcap(tech))

        if args.json:
            print("RESULT " + json.dumps(out))
        else:
            print(f"\n=== {tech}  nmos={args.nmos}  pmos={args.pmos} ===")
            print(f"  inv  VTC  NRMSE={out.get('inv_vtc_nrmse', float('nan')):.3f}%  "
                  f"MaxErr={out.get('inv_vtc_maxerr_mv', float('nan')):.1f}mV")
            print(f"  inv  tran NRMSE={out.get('inv_tran_post_nrmse', float('nan')):.3f}%")
            print(f"  RO   period_err={out.get('ring_osc_period_err', float('nan')):.2f}%  "
                  f"(NG {out.get('ro_ng_period_ps', float('nan')):.1f}ps / "
                  f"DN {out.get('ro_dn_period_ps', float('nan')):.1f}ps)")
            print(f"  SRAM rail_resid={out.get('sram_rail_snap_resid', float('nan')):.3f}  "
                  f"(q={out.get('sram_q', float('nan')):.3f} "
                  f"qb={out.get('sram_qb', float('nan')):.3f})")
            print(f"  opamp flat_flag={out.get('opamp_flat_flag', '?')}  "
                  f"gain={out.get('opamp_gain', float('nan')):.1f}  "
                  f"vout_c={out.get('opamp_vout_center', float('nan')):.3f}")
    finally:
        shutil.rmtree(iso, ignore_errors=True)
        print(f"Cleaned isolated checkpoint dir for {tech}", file=sys.stderr)


if __name__ == "__main__":
    main()
