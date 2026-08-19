#!/usr/bin/env python3
"""Complex-circuit AC gate — two-stage Miller opamp open-loop frequency response.

DirectNet (LEVEL=73) vs NGSPICE BSIM-CMG (LEVEL=72) ground truth on the IDENTICAL
opamp topology used by the DC opamp gate (tests/simple_circuits/verify_complex_opamp.py). Measures
the small-signal open-loop response: DC gain, gain-bandwidth / unity-gain
frequency, phase margin, −3 dB bandwidth.

Operating point: AC linearizes about the DC OP, so the gain you measure is the
small-signal gain about whatever OP the opamp converges to. Each side is biased at
ITS OWN DC trip point (Vinp giving Vout≈VDD/2 — the max-gain bias, exactly the
point the DC gate measures), found from that side's own DC transfer sweep. This is
the fair AC analog of the DC opamp gain gate; the known opamp value-surface
fragility (REPORT.md finding #3 — gain collapses to ~0 at small/medium) shows up
here as an honest low/zero AC gain, NOT as a bias artifact. The DC OP node
voltages are printed so a low gain is attributable to OP drift vs derivative error.

Ground truth is ALWAYS NGSPICE BSIM-CMG (AGENTS.md Validation rule).

Run CPU-pinned, repo ngspice:

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        NGSPICE_BIN="$PWD/tools/ngspice-45.2/bin/ngspice" \\
        python tests/simple_circuits/verify_complex_opamp_ac.py --tech TSMC12
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

from tests.common.complex import (  # noqa: E402
    BENCH, BENCH_TECHS, RESULTS_BASE, BenchTech, OpAmpParams, opamp_bias,
    get_baked_modelcard, run_ngspice_wrdata, run_directnet_dc_sweep,
    ngspice_opamp, directnet_opamp,
)
from tests.common.complex_ac import (  # noqa: E402
    ac_freq_grid, run_ngspice_ac_baked, run_directnet_ac, ac_metrics_extended,
    fmt_hz, fmt_ratio,
)

# First-cut PASS thresholds (loose-then-tighten — measurement campaign).
DC_GAIN_DB_ERR_TOL = 3.0          # dB
GBW_RATIO_LO, GBW_RATIO_HI = 0.6, 1.67
PM_ERR_TOL = 15.0                 # degrees (mag_nrmse reported, not gated)


def _peak_gain_bias(vinp: np.ndarray, vout: np.ndarray) -> Tuple[float, float]:
    """Vinp at the peak-|dVout/dVin| point (the max-gain bias), and Vout there.

    Matches the DC opamp gate's gain definition (verify_complex_opamp._gain_trip)
    so the AC linearizes about each opamp's own steepest, highest-gain bias —
    not merely where Vout≈VDD/2, which diverges for an offset/distorted curve.
    """
    g = np.abs(np.gradient(vout, vinp))
    ix = int(np.argmax(g))
    return round(float(vinp[ix]), 4), float(vout[ix])


def eval_opamp_ac(tech: str) -> Dict[str, object]:
    bt: BenchTech = BENCH[tech]
    p = OpAmpParams()
    vcm, _vbn, _vbp = opamp_bias(bt, p)
    vcm_tok = f"{vcm:g}"
    work_dir = RESULTS_BASE / "opamp_ac" / tech
    work_dir.mkdir(parents=True, exist_ok=True)
    baked = get_baked_modelcard(bt, bt.nfin, work_dir, nfin_p=bt.effective_nfin_p)
    freqs, ac_card = ac_freq_grid("opamp")

    # --- DC transfer sweeps → each side's own trip bias ---
    ng = ngspice_opamp(bt, p, baked)
    ng_dc = run_ngspice_wrdata(ng["body"], ng["signals"], work_dir,
                               f"{tech.lower()}_dcsweep", ng["analysis"])
    ng_trip, ng_vout = _peak_gain_bias(ng_dc[:, 0], ng_dc[:, 1])

    nn_deck_dc = directnet_opamp(bt, p)
    nn_dc_path = work_dir / f"{tech.lower()}_opamp_dc.sp"
    nn_dc_path.write_text(nn_deck_dc)
    nn_dc = run_directnet_dc_sweep(nn_dc_path, work_dir, f"{tech.lower()}_dc")
    nn_trip, nn_vout = _peak_gain_bias(np.asarray(nn_dc["inp"]),
                                       np.asarray(nn_dc["vout"]))

    # --- NGSPICE AC at its own trip ---
    ng_body = ng["body"].replace(
        f"Vinp inp 0 {vcm_tok}", f"Vinp inp 0 dc {ng_trip:g} ac 1")
    body_lines = ng_body.splitlines()
    v_ng = run_ngspice_ac_baked(body_lines, work_dir, f"{tech.lower()}_ac",
                                ac_card, "vout", freqs)

    # --- DirectNet AC about its own (freshly re-solved) trip OP ---
    nn_ac_deck = _to_ac_deck(nn_deck_dc, vcm_tok, nn_trip, ac_card)
    v_dn, op_ok, dc_op = run_directnet_ac(nn_ac_deck, work_dir,
                                          f"{tech.lower()}_ac", freqs, "vout")

    # Valid amplifying OP only if the NN output sits off the rails at the trip.
    nn_vout_op = dc_op.get("vout", float("nan"))
    op_valid = (np.isfinite(nn_vout_op)
                and 0.15 * bt.vdd < nn_vout_op < 0.85 * bt.vdd)

    m = ac_metrics_extended(freqs, v_dn, v_ng)
    # Gate on the three physically meaningful opamp FOMs: DC-gain level (dB),
    # gain-bandwidth (ratio) and phase margin (stability). Linear mag_nrmse is
    # NOT a gate criterion for a 40+ dB opamp — it is dominated by the passband
    # plateau and just re-expresses the dB-gain offset; it is reported for info.
    passed = (
        op_valid
        and m["gain0_db_err"] <= DC_GAIN_DB_ERR_TOL
        and np.isfinite(m["gbw_ratio"])
        and GBW_RATIO_LO <= m["gbw_ratio"] <= GBW_RATIO_HI
        and np.isfinite(m["pm_err"]) and m["pm_err"] <= PM_ERR_TOL
    )
    return dict(tech=tech, ng_trip=ng_trip, nn_trip=nn_trip,
                ng_vout=ng_vout, nn_vout=nn_vout, op_ok=op_ok,
                op_valid=bool(op_valid), dc_op=dc_op, m=m, passed=bool(passed))


def _to_ac_deck(dc_deck: str, vcm_tok: str, trip: float, ac_card: str) -> str:
    """Transform the DC-sweep DirectNet opamp deck into an open-loop AC deck:
    drive Vinp with an AC stimulus at its trip bias and swap .dc for .ac."""
    out = []
    for ln in dc_deck.splitlines():
        if ln.startswith(f"Vinp inp 0 {vcm_tok}"):
            out.append(f"Vinp inp 0 DC={trip:g} AC=1 0")
        elif ln.startswith(".dc Vinp"):
            out.append(f".{ac_card}")
        else:
            out.append(ln)
    return "\n".join(out) + "\n"


def _print_result(r: Dict[str, object]) -> None:
    m = r["m"]
    tech = r["tech"]
    verdict = "PASS" if r["passed"] else "FAIL"
    if not r.get("op_valid", True):
        op_note = "  [OP-MISBIAS: NN opamp output railed → value-surface collapse]"
    elif not r["op_ok"]:
        op_note = "  [OP-high-gain-NR-soft]"
    else:
        op_note = ""
    op = r["dc_op"]
    op_str = "  ".join(
        f"{k}={op[k]:.3f}" for k in ("vout", "vo1i", "vtail") if k in op)
    print(f"    NN DC OP: {op_str}")
    print(
        f"    dc_gain DN={m['gain0_db']:.2f}dB NG={m['gain0_db_ref']:.2f}dB "
        f"(err={m['gain0_db_err']:.2f}dB)  "
        f"GBW DN={fmt_hz(m['gbw_test'])} NG={fmt_hz(m['gbw_ref'])} "
        f"(ratio={fmt_ratio(m['gbw_ratio'])})  "
        f"PM DN={fmt_ratio(m['pm_test'])} NG={fmt_ratio(m['pm_ref'])} "
        f"(err={fmt_ratio(m['pm_err'])}deg)  "
        f"magNRMSE={m['mag_nrmse']*100:.2f}%")
    # parseable gate line (benchmark_collect.parse_ac_complex_log reads this)
    print(
        f"  opamp AC {tech}: dc_gain_err={m['gain0_db_err']:.2f}dB  "
        f"gbw_ratio={fmt_ratio(m['gbw_ratio'])}  "
        f"pm_err={fmt_ratio(m['pm_err'])}deg  "
        f"magNRMSE={m['mag_nrmse']*100:.2f}%  -> {verdict}{op_note}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tech", default=",".join(BENCH_TECHS))
    args = ap.parse_args()
    techs = [t.strip().upper() for t in args.tech.split(",") if t.strip()]
    # audit B5l — a typo'd tech used to SKIP silently, so `--tech TSMC5,TSMC7X`
    # scored 1/1 and exited 0. Reject up front instead of shrinking the matrix.
    unknown = [t for t in techs if t not in BENCH]
    if unknown:
        print(f"ERROR: unknown tech(s) {unknown}. Available: {list(BENCH)}")
        return 1

    print("=" * 78)
    print("Complex AC — two-stage Miller opamp open-loop: DirectNet vs NGSPICE")
    print("=" * 78)

    rows: List[Dict[str, object]] = []
    for tech in techs:
        print(f"\n--- {tech} (VDD={BENCH[tech].vdd} VT={BENCH[tech].vt}) ---")
        try:
            r = eval_opamp_ac(tech)
            print(f"    peak-gain bias: NG Vinp={r['ng_trip']:.4f}V "
                  f"(Vout={r['ng_vout']:.3f}V)  NN Vinp={r['nn_trip']:.4f}V "
                  f"(Vout={r['nn_vout']:.3f}V)")
            _print_result(r)
            rows.append(r)
        except Exception as exc:  # noqa: BLE001 — fail loud per tech
            print(f"  opamp AC {tech}: ERROR {exc}")
            rows.append(dict(tech=tech, error=str(exc), passed=False))

    print("\n" + "=" * 78)
    print("SUMMARY — opamp open-loop AC")
    print("=" * 78)
    n_pass = sum(1 for r in rows if r.get("passed"))
    for r in rows:
        if "error" in r:
            print(f"  {r['tech']:8s} | ERROR — {str(r['error'])[:50]}")
            continue
        m = r["m"]
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['tech']:8s} | dc_gain_err={m['gain0_db_err']:6.2f}dB | "
              f"gbw_ratio={fmt_ratio(m['gbw_ratio']):>6s} | "
              f"pm_err={fmt_ratio(m['pm_err']):>6s}deg | "
              f"magNRMSE={m['mag_nrmse']*100:5.1f}% | {status}")
    print(f"\nRESULT: {n_pass}/{len(rows)} opamp AC gates passed")
    return 0 if (rows and n_pass == len(rows)) else 1


if __name__ == "__main__":
    sys.exit(main())
