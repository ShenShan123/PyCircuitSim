#!/usr/bin/env python3
"""DirectNet V6.4.7 — S5 = R0.3: switchcap per-device id/charge/cap dump.

INSTRUMENTATION-ONLY (no production code touched). Decides the per-device,
per-window ownership split of the switched-cap failure between
  (a) the Rule-15 reverse-Vds clamp freezing conduction near Vds~0 (plan P2),
  (b) charge/cap VALUE errors (plan P5/P7),
  (c) anything else (id error in the conduction window, solver numerics).

Method (P0-D bias-dump + P0-H value-overlay plumbing, merged):
  1. Run the DN switchcap transient exactly as the benchmark does
     (render_directnet_netlist + Vin rewrite + run_directnet_transient),
     keeping ALL node waveforms.
  2. In three windows — SAMPLE (phi high, 0.6-2.3 ns), EDGE (phi falling,
     2.40-2.70 ns), HOLD (2.6-4.3 ns) — evaluate every MOSFET with the NN
     (``mosfet._eval``, post-Rule-15, exactly what the stamps consume) AND the
     analytic OSDI truth at the SAME absolute terminal bias
     (``eval_single_point``; shift invariance, P0-D convention).
  3. Rule-16 quartet + signed mean per (device, window, quantity), with the
     id error split by Rule-15 clamp class (REV / NEAR0 / FREE).
  4. Integral KCL decomposition of the vsamp charge per window. The stamps
     give, exactly (Trap companion telescopes):
       C*dv = Q_res + Q_cap + Q_num
       Q_res = trapz( -id(Mnt) - id(Mpt) ) dt    (resistive, into vsamp)
       Q_cap = sum_TG [ (qg+qd)(end) - (qg+qd)(start) ]  (charge-VALUE pump:
               the solver stamps the source companion as -(i_g+i_d))
       Q_num = remainder = solver/NR-tolerance channel (what symcaps
               modulates; not a model VALUE error)
     Computed with NN values and with OSDI values at the same trajectory.

Sign / convention (P0-B FD-verified; cgs added here because the TG source
node vsamp is NOT grounded):
  id/qg/qd/qb : same convention as training data    -> direct compare.
  qs          : -(qg+qd+qb) reconstructed for BOTH  (Rule 14).
  cgg/cdd     : diagonals, same sign                -> direct.
  cgd/cdg/cgs : NN = +dQ/dV raw autograd; OSDI returns SPICE convention
                (off-diagonals negated)             -> negate OSDI.
                An in-script FD check re-verifies cgs (never load-bearing
                before S5) and fails loud on mismatch.

Ground truth is ALWAYS the OSDI binary via PyCMG (CLAUDE.md Validation rule).

Usage:
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      conda run -n pycircuitsim python scripts/v6_4_7_s5_sc_dump.py \
      [--tech TSMC16,TSMC12,TSMC7,TSMC5] \
      > results/v6_4_7/s5_logs/s5_sc_dump.log 2>&1
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Insert order matters: PyCMG has its OWN `tests/` dir, so `tests.common`
# must resolve to ROOT/tests (ROOT inserted last -> wins).
for p in (ROOT / "external_compact_models" / "PyCMG" / "tests",
          ROOT / "external_compact_models" / "PyCMG",
          ROOT / "external_compact_models",
          ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tests.common.complex import (  # noqa: E402
    BENCH, BenchTech, render_directnet_netlist, parse_netlist,
    run_directnet_transient)
from tests.verify_complex_switchcap import (  # noqa: E402
    TEMPLATE, SAMPLE_END, HOLD_START, HOLD_END, _vin, _at, run_ngspice_sc)
from pycircuitsim.solver import _is_mosfet  # noqa: E402
from pycmg.nn_generate import (  # noqa: E402
    _create_model_and_instance, eval_single_point)
from pycmg.nn_config import TECH_CONFIGS  # noqa: E402

CSAMPLE = 100e-15            # Csample in the netlist
TEMP_K = 300.15              # .temp 27
QTYS = ("id", "qg", "qd", "qs", "qb", "cgg", "cgd", "cgs", "cdg", "cdd")
NEG_OSDI = ("cgd", "cdg", "cgs")   # OSDI SPICE-convention -> NN raw dQ/dV
# clock: PULSE td=0.5n tr=tf=0.1n pw=1.9n -> high 0.6-2.5n, falls 2.5-2.6n
WINDOWS: Tuple[Tuple[str, float, float], ...] = (
    ("RISE",   0.40e-9, 0.62e-9),         # phi rising edge (overshoot inject)
    ("SAMPLE", 0.60e-9, SAMPLE_END),      # conduction, phi high
    ("EDGE",   2.40e-9, 2.70e-9),         # phi falling edge +/-0.15n
    ("HOLD",   HOLD_START, HOLD_END),     # droop window (harness definition)
)
OUT_DIR = ROOT / "results" / "v6_4_7"
MD_PATH = OUT_DIR / "S5_R03_sc_device_dump.md"


def rule16(nn: np.ndarray, osdi: np.ndarray) -> Dict[str, float]:
    """Rule-16 quartet + signed mean error (P0-H estimator + mean_err)."""
    nn = np.asarray(nn, float)
    osdi = np.asarray(osdi, float)
    diff = nn - osdi
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((osdi - osdi.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-300 else float("nan")
    ptp = float(np.ptp(osdi))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    nrmse = rmse / ptp * 100.0 if ptp > 1e-300 else float("nan")
    scale = max(float(np.max(np.abs(osdi))), 1e-30)
    denom = np.maximum(np.abs(osdi), 1e-3 * scale)
    mre = float(np.mean(np.abs(diff) / denom)) * 100.0
    return {"mre_pct": mre, "r2": r2, "nrmse_pct": nrmse,
            "max_err": float(np.max(np.abs(diff))),
            "mean_err": float(np.mean(diff)), "osdi_absmax": scale}


def clamp_class(is_pmos: bool, vdd_train: float, vds: float) -> str:
    """Rule-15 region of the id correction (_apply_vds_correction semantics).

    REV   : reverse/zero Vds -> f_id = 0, id forced to 0 (plan-P2 region).
    NEAR0 : normal direction, |Vds| <= 4.6*VT -> 1-exp factor attenuates
            id by >1% (the clamp materially reshapes conduction).
    FREE  : correction inactive (<1% effect) -> raw network id.
    """
    vt = max(0.06 * vdd_train, 0.026)
    normal = (vds < 0.0) if is_pmos else (vds > 0.0)
    if not normal:
        return "REV"
    return "NEAR0" if abs(vds) <= 4.6 * vt else "FREE"


def qs_of(d: Dict[str, float]) -> float:
    return -(d["qg"] + d["qd"] + d["qb"])


def osdi_nn_conv(o: Dict[str, float], key: str) -> float:
    if key == "qs":
        return qs_of(o)
    return -o[key] if key in NEG_OSDI else o[key]


def nn_val(r: Dict[str, float], key: str) -> float:
    return qs_of(r) if key == "qs" else r[key]


def fd_check_caps(inst, vd: float, vg: float, vs: float, vb: float,
                  delta: float = 1e-3) -> List[str]:
    """FD-verify the OSDI->NN sign mapping for cgd/cdg/cgs at one bias."""
    lines: List[str] = []
    base = eval_single_point(inst, vd, vg, vs, vb)
    if base is None:
        return ["  FD check: OSDI eval failed at base point — SKIPPED (loud)"]
    for key, term in (("cgd", "d"), ("cgs", "s"), ("cdg", "g")):
        qty = "qg" if key.startswith("cg") else "qd"
        bias = {"d": vd, "g": vg, "s": vs, "b": vb}
        bias[term] += delta
        hi = eval_single_point(inst, bias["d"], bias["g"], bias["s"], bias["b"])
        bias[term] -= 2 * delta
        lo = eval_single_point(inst, bias["d"], bias["g"], bias["s"], bias["b"])
        if hi is None or lo is None:
            lines.append(f"  FD {key}: OSDI eval failed — SKIPPED (loud)")
            continue
        fd = (hi[qty] - lo[qty]) / (2 * delta)      # raw dQ/dV (NN convention)
        mapped = osdi_nn_conv(base, key)
        rel = abs(mapped - fd) / max(abs(fd), 1e-21)
        ok = "OK" if rel < 0.2 else "MISMATCH — convention suspect!"
        lines.append(f"  FD {key}: raw dQ/dV={fd:+.3e}  -OSDI {key}={mapped:+.3e}"
                     f"  rel={rel:.2%}  {ok}")
    return lines


def run_dn_sc_full(bt: BenchTech, work_dir: Path):
    """Benchmark-identical DN transient, but keep ALL node waveforms."""
    netlist = render_directnet_netlist(
        TEMPLATE, bt, work_dir / f"switchcap_{bt.name}.sp")
    text = netlist.read_text().replace("Vin vin 0 0.48",
                                       f"Vin vin 0 {_vin(bt)}")
    netlist.write_text(text)
    results, partial, err = run_directnet_transient(netlist)
    return netlist, results, partial, err


def analyze_tech(tech: str, scratch: Path) -> Dict[str, object]:
    bt = BENCH[tech]
    vin = _vin(bt)
    work = scratch / tech
    work.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 78}\n--- {tech} (VDD={bt.vdd} VT={bt.vt} Vin={vin}) ---")

    ng = run_ngspice_sc(bt, work)
    ng_chg = _at(ng["time"], ng["vsamp"], SAMPLE_END)
    ng_droop = (_at(ng["time"], ng["vsamp"], HOLD_START)
                - _at(ng["time"], ng["vsamp"], HOLD_END))

    netlist, dn, partial, err = run_dn_sc_full(bt, work)
    t = np.asarray(dn["time"])
    vsamp = np.asarray(dn["vsamp"])
    if partial or t[-1] < HOLD_END:
        raise RuntimeError(f"{tech}: DN transient partial/short "
                           f"(partial={partial}, t_max={t[-1]:.3e}, {err!r})")
    dn_chg = _at(t, vsamp, SAMPLE_END)
    dn_droop = (_at(t, vsamp, HOLD_START) - _at(t, vsamp, HOLD_END))
    print(f"  NG chg={ng_chg:.4f}V droop={ng_droop * 1e3:.3f}mV | "
          f"DN chg={dn_chg:.4f}V droop={dn_droop * 1e3:.3f}mV | "
          f"gate gap C*dV={(dn_chg - ng_chg) * CSAMPLE * 1e15:+.2f}fC")

    # devices from the SAME rendered netlist (fresh parse, same checkpoints)
    parser = parse_netlist(netlist)
    mosfets = {c.name: c for c in parser.circuit.components if _is_mosfet(c)}
    dev_names = sorted(mosfets)            # Mnc, Mnt, Mpc, Mpt
    tg_names = [n for n, m in mosfets.items() if m.nodes[2] == "vsamp"]
    nodes = [k for k in dn.keys() if k != "time"]

    cfg = TECH_CONFIGS[bt.nn_tech]
    _, inst_n, _ = _create_model_and_instance(
        cfg, "nmos", bt.vt, bt.l_nmos, float(bt.nfin), TEMP_K)
    _, inst_p, _ = _create_model_and_instance(
        cfg, "pmos", bt.vt, bt.l_pmos, float(bt.nfin), TEMP_K)
    inst = {"nmos": inst_n, "pmos": inst_p}

    # FD convention check at one mid-sample TG bias (cgs is load-bearing here)
    i_mid = int(np.searchsorted(t, 1.2e-9))
    vmid = {n: float(dn[n][i_mid]) for n in nodes}
    mnt = mosfets[tg_names[0] if not mosfets[tg_names[0]]._is_pmos
                  else tg_names[1]]
    d0, g0, s0, b0 = (vmid.get(n, 0.0) for n in mnt.nodes)
    print("  OSDI cap-sign FD check (NMOS pass device, mid-sample bias):")
    for ln in fd_check_caps(inst["nmos"], d0, g0, s0, b0):
        print(ln)

    # dense per-window, per-device dump
    rows: List[Dict[str, object]] = []
    acc: Dict[Tuple[str, str], Dict[str, Dict[str, List[float]]]] = {}
    osdi_fail = 0
    for wname, t0, t1 in WINDOWS:
        idx = np.where((t >= t0) & (t <= t1))[0]
        for i in idx:
            volt = {n: float(dn[n][i]) for n in nodes}
            for name in dev_names:
                m = mosfets[name]
                dtype = "pmos" if m._is_pmos else "nmos"
                vd, vg, vs, vb = (volt.get(n, 0.0) for n in m.nodes)
                m.clear_cache()
                r = m._eval(volt)
                o = eval_single_point(inst[dtype], vd, vg, vs, vb,
                                      _silent=True)
                if o is None:
                    osdi_fail += 1
                    continue
                cls = clamp_class(m._is_pmos, m._vdd_estimate, vd - vs)
                a = acc.setdefault((name, wname), {
                    "nn": {q: [] for q in QTYS},
                    "os": {q: [] for q in QTYS},
                    "t": [], "cls": []})
                a["t"].append(float(t[i]))
                a["cls"].append(cls)
                for q in QTYS:
                    a["nn"][q].append(nn_val(r, q))
                    a["os"][q].append(osdi_nn_conv(o, q))
                rows.append({
                    "tech": tech, "window": wname, "time": float(t[i]),
                    "device": name, "dtype": dtype, "vd": vd, "vg": vg,
                    "vs": vs, "vb": vb, "vgs": vg - vs, "vds": vd - vs,
                    "vbs": vb - vs, "clamp_class": cls,
                    **{f"nn_{q}": nn_val(r, q) for q in QTYS},
                    **{f"os_{q}": osdi_nn_conv(o, q) for q in QTYS}})
    if osdi_fail:
        print(f"  WARNING: {osdi_fail} OSDI eval failures (points dropped)")

    # metrics + clamp-split + KCL decomposition
    metrics: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    clamp_split: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    decomp: Dict[str, Dict[str, float]] = {}
    for (name, wname), a in acc.items():
        ta = np.array(a["t"])
        cls = np.array(a["cls"])
        for q in QTYS:
            metrics[(name, wname, q)] = rule16(
                np.array(a["nn"][q]), np.array(a["os"][q]))
        # id-error charge split by clamp class (trapezoid weights), into vsamp
        w = np.gradient(ta) if len(ta) > 2 else np.full_like(ta, 5e-12)
        d_i = -(np.array(a["nn"]["id"]) - np.array(a["os"]["id"]))  # into src
        split = {}
        for c in ("REV", "NEAR0", "FREE"):
            msk = cls == c
            split[c] = {"n": int(msk.sum()),
                        "dq_fc": float(np.sum(d_i[msk] * w[msk]) * 1e15)}
        clamp_split[(name, wname)] = split
    for wname, t0, t1 in WINDOWS:
        q_res_nn = q_res_os = q_cap_nn = q_cap_os = 0.0
        for name in tg_names:
            a = acc[(name, wname)]
            ta = np.array(a["t"])
            q_res_nn += float(np.trapezoid(-np.array(a["nn"]["id"]), ta))
            q_res_os += float(np.trapezoid(-np.array(a["os"]["id"]), ta))
            qgd_nn = np.array(a["nn"]["qg"]) + np.array(a["nn"]["qd"])
            qgd_os = np.array(a["os"]["qg"]) + np.array(a["os"]["qd"])
            q_cap_nn += float(qgd_nn[-1] - qgd_nn[0])
            q_cap_os += float(qgd_os[-1] - qgd_os[0])
        a0 = acc[(tg_names[0], wname)]
        dv = float(np.interp(a0["t"][-1], t, vsamp)
                   - np.interp(a0["t"][0], t, vsamp))
        c_dv = CSAMPLE * dv
        decomp[wname] = {
            "c_dv_fc": c_dv * 1e15,
            "q_res_nn_fc": q_res_nn * 1e15, "q_res_os_fc": q_res_os * 1e15,
            "q_cap_nn_fc": q_cap_nn * 1e15, "q_cap_os_fc": q_cap_os * 1e15,
            "q_num_fc": (c_dv - q_res_nn - q_cap_nn) * 1e15,
            "dq_id_fc": (q_res_nn - q_res_os) * 1e15,
            "dq_q_fc": (q_cap_nn - q_cap_os) * 1e15,
        }
        print(f"  [{wname:6s}] C*dV={decomp[wname]['c_dv_fc']:+8.2f}fC | "
              f"Q_res nn/os={q_res_nn * 1e15:+8.2f}/{q_res_os * 1e15:+8.2f} | "
              f"Q_cap nn/os={q_cap_nn * 1e15:+8.3f}/{q_cap_os * 1e15:+8.3f} | "
              f"Q_num={decomp[wname]['q_num_fc']:+8.2f}fC")

    # CSV
    csv_path = OUT_DIR / f"s5_sc_dump_{tech}.csv"
    with csv_path.open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"  CSV: {csv_path}  ({len(rows)} rows)")

    return {"tech": tech, "bt": bt, "vin": vin, "ng_chg": ng_chg,
            "dn_chg": dn_chg, "ng_droop": ng_droop, "dn_droop": dn_droop,
            "dev_names": dev_names, "tg_names": tg_names,
            "metrics": metrics, "clamp_split": clamp_split,
            "decomp": decomp, "osdi_fail": osdi_fail}


def fmt_m(m: Dict[str, float], sc: float, un: str) -> str:
    return (f"MRE={m['mre_pct']:9.2f}% | R2={m['r2']:8.4f} | "
            f"NRMSE={m['nrmse_pct']:8.2f}% | MaxErr={m['max_err'] * sc:9.3g}{un}"
            f" | mean(nn-os)={m['mean_err'] * sc:+9.3g}{un}")


def print_tables(res: Dict[str, object]) -> None:
    units = {"id": (1e6, "uA")}
    for q in QTYS[1:5]:
        units[q] = (1e18, "aC")
    for q in QTYS[5:]:
        units[q] = (1e18, "aF")
    print(f"\n  ===== {res['tech']} per-device Rule-16 tables "
          f"(NN post-Rule-15 vs OSDI, NN convention) =====")
    for name in res["dev_names"]:
        for wname, _, _ in WINDOWS:
            key0 = (name, wname, "id")
            if key0 not in res["metrics"]:
                continue
            print(f"  --- {name} / {wname} ---")
            for q in QTYS:
                sc, un = units[q]
                print(f"    {q:3s}: {fmt_m(res['metrics'][(name, wname, q)], sc, un)}")
            sp = res["clamp_split"][(name, wname)]
            print("    id clamp split (dQ into source, fC): " + "  ".join(
                f"{c}: n={sp[c]['n']:3d} dq={sp[c]['dq_fc']:+8.3f}"
                for c in ("REV", "NEAR0", "FREE")))


def write_md(all_res: List[Dict[str, object]]) -> None:
    L: List[str] = []
    L.append("# S5 = R0.3 — switchcap per-device id/charge/cap dump along the "
             "live DN transient (V6.4.7, 2026-06-10)")
    L.append("")
    L.append("**Status:** data sections auto-generated by "
             "`scripts/v6_4_7_s5_sc_dump.py`; verdicts hand-written below.")
    L.append("**Ground truth:** OSDI BSIM-CMG via PyCMG `eval_single_point` "
             "at identical absolute terminal bias (P0-D convention).")
    L.append("")
    L.append("## Commands")
    L.append("")
    L.append("```")
    L.append('CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\')
    L.append("  conda run -n pycircuitsim python scripts/v6_4_7_s5_sc_dump.py \\")
    L.append("  > results/v6_4_7/s5_logs/s5_sc_dump.log 2>&1")
    L.append("```")
    L.append("")
    L.append("Windows: SAMPLE 0.60–2.30 ns (phi high), EDGE 2.40–2.70 ns "
             "(phi fall 2.5→2.6 ns ±0.1), HOLD 2.60–4.30 ns (harness droop "
             "window). Conventions: off-diagonal caps compared in NN raw "
             "dQ/dV convention (OSDI cgd/cdg/cgs negated; FD-verified in "
             "the log). Clamp classes: REV (Rule-15 f_id=0), NEAR0 "
             "(|Vds|≤4.6·VT, >1 % attenuation), FREE.")
    L.append("")
    L.append("## Benchmark anchor (per-tech, post-P0 code)")
    L.append("")
    L.append("| tech | Vin | NG chg @2.3n | DN chg | gap (C·ΔV) | NG droop | DN droop |")
    L.append("|------|-----|--------------|--------|-----------|----------|----------|")
    for r in all_res:
        L.append(f"| {r['tech']} | {r['vin']:.2f} | {r['ng_chg']:.4f} | "
                 f"{r['dn_chg']:.4f} | "
                 f"{(r['dn_chg'] - r['ng_chg']) * CSAMPLE * 1e15:+.2f} fC | "
                 f"{r['ng_droop'] * 1e3:.3f} mV | {r['dn_droop'] * 1e3:.3f} mV |")
    L.append("")
    L.append("## vsamp charge decomposition per window (fC) — "
             "C·ΔV = Q_res + Q_cap + Q_num (exact for Trap companion)")
    L.append("")
    L.append("Q_res = resistive id into vsamp (TG devices); Q_cap = Δ(qg+qd) "
             "charge-VALUE pump via the source companion; Q_num = remainder "
             "= solver/NR-tolerance channel. `ΔQ_id = Q_res(NN) − Q_res(OSDI)` "
             "and `ΔQ_q = Q_cap(NN) − Q_cap(OSDI)` are the model-VALUE error "
             "injections along the same trajectory.")
    L.append("")
    L.append("| tech | window | C·ΔV | Q_res NN | Q_res OSDI | **ΔQ_id** | "
             "Q_cap NN | Q_cap OSDI | **ΔQ_q** | **Q_num** |")
    L.append("|------|--------|------|----------|------------|-----------|"
             "----------|------------|----------|-----------|")
    for r in all_res:
        for wname, _, _ in WINDOWS:
            d = r["decomp"][wname]
            L.append(
                f"| {r['tech']} | {wname} | {d['c_dv_fc']:+.2f} | "
                f"{d['q_res_nn_fc']:+.2f} | {d['q_res_os_fc']:+.2f} | "
                f"**{d['dq_id_fc']:+.2f}** | {d['q_cap_nn_fc']:+.3f} | "
                f"{d['q_cap_os_fc']:+.3f} | **{d['dq_q_fc']:+.3f}** | "
                f"**{d['q_num_fc']:+.2f}** |")
    L.append("")
    L.append("## Rule-16 quartets — transmission-gate devices (per window)")
    L.append("")
    units = {"id": (1e6, "uA")}
    for q in QTYS[1:5]:
        units[q] = (1e18, "aC")
    for q in QTYS[5:]:
        units[q] = (1e18, "aF")
    for r in all_res:
        L.append(f"### {r['tech']}")
        L.append("")
        for name in r["tg_names"]:
            L.append(f"**{name}** (pass "
                     f"{'PMOS' if name.lower().endswith('pt') else 'NMOS'})")
            L.append("")
            L.append("| window | qty | MRE% | R² | NRMSE% | MaxErr | "
                     "mean(NN−OSDI) | OSDI max |")
            L.append("|--------|-----|------|----|--------|--------|"
                     "---------------|----------|")
            for wname, _, _ in WINDOWS:
                for q in QTYS:
                    m = r["metrics"].get((name, wname, q))
                    if m is None:
                        continue
                    sc, un = units[q]
                    L.append(f"| {wname} | {q} | {m['mre_pct']:.1f} | "
                             f"{m['r2']:.3f} | {m['nrmse_pct']:.1f} | "
                             f"{m['max_err'] * sc:.3g} {un} | "
                             f"{m['mean_err'] * sc:+.3g} {un} | "
                             f"{m['osdi_absmax'] * sc:.3g} {un} |")
            L.append("")
            L.append("Clamp split of the id-error charge (into vsamp, fC):")
            L.append("")
            L.append("| window | REV n / ΔQ | NEAR0 n / ΔQ | FREE n / ΔQ |")
            L.append("|--------|------------|--------------|-------------|")
            for wname, _, _ in WINDOWS:
                sp = r["clamp_split"].get((name, wname))
                if sp is None:
                    continue
                L.append("| {} | {} / {:+.3f} | {} / {:+.3f} | {} / {:+.3f} |"
                         .format(wname,
                                 sp['REV']['n'], sp['REV']['dq_fc'],
                                 sp['NEAR0']['n'], sp['NEAR0']['dq_fc'],
                                 sp['FREE']['n'], sp['FREE']['dq_fc']))
            L.append("")
    L.append("## Conclusion / ownership verdict")
    L.append("")
    L.append("(hand-written after data review — see final report)")
    L.append("")
    MD_PATH.write_text("\n".join(L))
    print(f"\nMD draft: {MD_PATH}")


def main() -> int:
    ap = argparse.ArgumentParser(description="S5 SC per-device dump")
    ap.add_argument("--tech", default="TSMC16,TSMC12,TSMC7,TSMC5")
    args = ap.parse_args()
    scratch = OUT_DIR / "s5_logs" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    all_res: List[Dict[str, object]] = []
    for tech in [s.strip() for s in args.tech.split(",")]:
        res = analyze_tech(tech, scratch)
        print_tables(res)
        all_res.append(res)
    write_md(all_res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
