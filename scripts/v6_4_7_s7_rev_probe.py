#!/usr/bin/env python3
"""V6.4.7 S7 — P2 pre-build probe: is the NN's RAW (pre-clamp) reverse-Vds
id surface usable?

``_apply_vds_correction`` (pycircuitsim/models/mosfet_nn.py) hard-zeroes ALL
reverse conduction: step (b) sets ``f_id = 0`` when Vds is reverse, and the
wrong-sign clamp (d) zeroes whatever survives. Yet the V6.3 ``reverse_vds``
training corridor (7.48 % of rows, Vd in [-0.30, -0.01]*VDD source-referenced
for NMOS, mirrored for PMOS) was added so the network could LEARN reverse
conduction. Plan P2 wants to relax (b)+(d) — but only if the raw surface
under the clamp is trained, not garbage.

This probe, per tech (TSMC5/7/12/16) x device type (NMOS/PMOS):

  1. RAW tap: identity-monkeypatch ``_MOSFETNNBase._apply_vds_correction``
     (in-script only, always restored) so ``_eval`` returns the pure
     network + denorm output. The Rule-5 gds floor upstream in
     ``_unpack_eval`` still applies to gds, but ``id`` is untouched.
  2. POST tap: the unpatched production ``_eval`` at the same bias.
  3. Ground truth: OSDI via PyCMG ``eval_single_point`` at the SAME absolute
     bias (training-data convention == raw NN frame; sign-consistency is
     verified at one forward-bias point per tech/device first).

Probe set:
  (a) live SRAM ``force_ic`` state1 attractor (re-solved in-script with the
      production solver, cross-checked against results/v6_4_7/s2_logs/
      sram_snm.log) — all six 6T devices, the ON-PMOS restoring device, the
      pinning NMOS and the access transistors in particular;
  (b) the trained reverse corridor grid: Vds in -[0.30..0.01]*VDD (12 pts,
      linear) x Vgs in {0.2,0.4,0.6,0.8,1.0}*VDD, Vbs=0 (PMOS mirrored);
  (c) beyond-corridor: Vds in -{0.4, 0.5}*VDD at Vgs=0.6*VDD (mirrored).

Metrics: Rule-16 quartet (MRE/R2/NRMSE/MaxErr) on the corridor grid,
sign-agreement rate, and magnitude-agreement on the meaningful subset
(|id_OSDI| > 1 uA — the plan's go/no-go region).

Run:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    conda run -n pycircuitsim python -u scripts/v6_4_7_s7_rev_probe.py \
    2>&1 | tee results/v6_4_7/s7_logs/s7_rev_probe.log

Outputs: stdout tables, results/v6_4_7/s7_rev_probe_corridor.csv,
results/v6_4_7/S7_P2_rev_probe.md. Ground truth is ALWAYS the OSDI binary.
"""
from __future__ import annotations

import csv
import logging
import math
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# P0-D path setup: PyCMG paths appended (low precedence), PROJECT_ROOT at the
# FRONT so `tests.common` resolves to the project's tests package.
sys.path.append(str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
sys.path.append(str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.complex import BENCH, BenchTech, parse_netlist  # noqa: E402
from tests.common.nn import nrmse, mre  # noqa: E402
from tests.verify_complex_sram_snm import _directnet_6t_netlist  # noqa: E402
from pycircuitsim.solver import DCSolver, _is_mosfet  # noqa: E402
import pycircuitsim.models.mosfet_nn as mnn  # noqa: E402

from pycmg.nn_config import TECH_CONFIGS  # noqa: E402
from pycmg.nn_generate import (  # noqa: E402
    _create_model_and_instance, eval_single_point)

TECHS = ["TSMC5", "TSMC7", "TSMC12", "TSMC16"]
L_NMOS = 16e-9
L_PMOS = 20e-9
TEMP_K = 300.15
MEANINGFUL_A = 1e-6     # plan's "OSDI conducts meaningfully" floor
SIGN_FLOOR_A = 1e-9     # below this OSDI |id|, sign is numerically moot

RESULTS_DIR = PROJECT_ROOT / "results" / "v6_4_7"
LOG_DIR = RESULTS_DIR / "s7_logs"
SCRATCH = LOG_DIR / "scratch"
CSV_PATH = RESULTS_DIR / "s7_rev_probe_corridor.csv"
MD_PATH = RESULTS_DIR / "S7_P2_rev_probe.md"
S2_LOG = RESULTS_DIR / "s2_logs" / "sram_snm.log"


# ── RAW tap: identity-monkeypatch of the Vds correction ────────────────────
@contextmanager
def raw_tap():
    """Temporarily no-op ``_apply_vds_correction`` on the base class.

    ``_unpack_eval`` then returns the pure post-denorm network output
    (id/gm/gmb untouched; gds already passed the Rule-5 floor upstream).
    Always restored.
    """
    orig = mnn._MOSFETNNBase._apply_vds_correction

    def _identity(self, result, vds):  # noqa: ANN001
        return result

    mnn._MOSFETNNBase._apply_vds_correction = _identity
    try:
        yield
    finally:
        mnn._MOSFETNNBase._apply_vds_correction = orig


def nn_eval(m, vd: float, vg: float, vs: float, vb: float) -> Dict[str, float]:
    """One production-path ``_eval`` at absolute terminal voltages."""
    if m.nodes[2] == m.nodes[3] and abs(vs - vb) > 1e-15:
        raise ValueError(f"{m.name}: s/b share node {m.nodes[2]}, vs!=vb")
    volt = {m.nodes[0]: vd, m.nodes[1]: vg, m.nodes[2]: vs, m.nodes[3]: vb}
    m.clear_cache()
    res = dict(m._eval(volt))
    m.clear_cache()
    return res


# ── OSDI ground truth ───────────────────────────────────────────────────────
_OSDI_KEEP: Dict[Tuple[str, str], tuple] = {}
_OSDI_INST: Dict[Tuple[str, str], object] = {}


def osdi_inst(bt: BenchTech, dtype: str):
    key = (bt.name, dtype)
    if key not in _OSDI_INST:
        L = L_NMOS if dtype == "nmos" else L_PMOS
        made = _create_model_and_instance(
            TECH_CONFIGS[bt.nn_tech], dtype, bt.vt, L, float(bt.nfin), TEMP_K)
        if made is None:
            raise RuntimeError(f"OSDI instance build failed for {key}")
        _OSDI_KEEP[key] = made
        _OSDI_INST[key] = made[1]
    return _OSDI_INST[key]


def osdi_eval(bt: BenchTech, dtype: str, vd: float, vg: float,
              vs: float, vb: float) -> Optional[Dict[str, float]]:
    return eval_single_point(
        osdi_inst(bt, dtype), vd, vg, vs, vb, _silent=True)


# ── SRAM attractor: re-solve + log cross-check ──────────────────────────────
def parse_log_attractor() -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    tech = None
    for line in S2_LOG.read_text().splitlines():
        mh = re.match(r"^--- (TSMC\d+) ", line)
        if mh:
            tech = mh.group(1)
        mf = re.search(r"force_ic state1: q=([\d.]+) qb=([\d.]+)", line)
        if mf and tech and tech not in out:
            out[tech] = (float(mf.group(1)), float(mf.group(2)))
    return out


def solve_state1(bt: BenchTech):
    """Production force_ic solve of the 6T cell, state1 seed (q=VDD, qb=0)."""
    netlist = _directnet_6t_netlist(
        bt, bt.vdd, 0.0, SCRATCH / f"s7_{bt.name}_state1.sp")
    parser = parse_netlist(netlist)
    circuit = parser.circuit
    logging.disable(logging.CRITICAL)
    try:
        guess = circuit.initial_conditions or None
        solver = DCSolver(circuit, initial_guess=guess,
                          use_source_stepping=True, force_ic=True)
        sol = solver.solve()
    finally:
        logging.disable(logging.NOTSET)
    return circuit, sol


# ── metrics ──────────────────────────────────────────────────────────────────
def r_squared(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def corridor_metrics(raw: np.ndarray, osdi: np.ndarray) -> Dict[str, float]:
    """Rule-16 quartet + sign/magnitude agreement on the meaningful subset."""
    m: Dict[str, float] = {}
    m["n"] = len(osdi)
    m["mre_pct"] = mre(raw, osdi)
    m["r2"] = r_squared(raw, osdi)
    m["nrmse_pct"] = nrmse(raw, osdi)
    m["maxerr_uA"] = float(np.max(np.abs(raw - osdi))) * 1e6

    nz = np.abs(osdi) > SIGN_FLOOR_A
    m["sign_all"] = (float(np.mean(np.sign(raw[nz]) == np.sign(osdi[nz])))
                     if nz.any() else float("nan"))
    mf = np.abs(osdi) > MEANINGFUL_A
    m["n_meaningful"] = int(mf.sum())
    if mf.any():
        ratio = np.abs(raw[mf]) / np.abs(osdi[mf])
        m["sign_meaningful"] = float(
            np.mean(np.sign(raw[mf]) == np.sign(osdi[mf])))
        m["ratio_median"] = float(np.median(ratio))
        m["frac_2x"] = float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))
        m["frac_10x"] = float(np.mean((ratio >= 0.1) & (ratio <= 10.0)))
    else:
        m["sign_meaningful"] = float("nan")
        m["ratio_median"] = float("nan")
        m["frac_2x"] = float("nan")
        m["frac_10x"] = float("nan")
    return m


def verdict(m: Dict[str, float]) -> str:
    if m["n_meaningful"] == 0:
        return "NO-CONDUCTION"
    if (m["sign_meaningful"] >= 0.95 and 0.5 <= m["ratio_median"] <= 2.0
            and m["frac_10x"] >= 0.9):
        return "USABLE"
    if m["sign_meaningful"] >= 0.80 and m["frac_10x"] >= 0.70:
        return "MIXED"
    return "GARBAGE"


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    csv_rows: List[Dict[str, object]] = []
    md_lines: List[str] = []
    log_attr = parse_log_attractor()

    print("=" * 100)
    print("S7 — P2 pre-build probe: RAW (pre-clamp) reverse-Vds id surface "
          "vs OSDI ground truth")
    print(f"  raw tap = identity-monkeypatched _apply_vds_correction "
          f"(post-denorm, pre-correction; id untouched)")
    print(f"  meaningful-conduction floor = {MEANINGFUL_A*1e6:.1f} uA "
          f"(plan go/no-go region)")
    print("=" * 100)

    # ── 0) sign-consistency check at one forward-bias point ────────────────
    print("\n### 0) Sign-consistency check (forward bias, vs=vb=0 frame)")
    sign_ok = True
    nn_devs: Dict[Tuple[str, str], object] = {}
    sols: Dict[str, Dict[str, float]] = {}
    circuits: Dict[str, object] = {}
    for tech in TECHS:
        bt = BENCH[tech]
        circuit, sol = solve_state1(bt)
        circuits[tech] = circuit
        sols[tech] = sol
        devs = {c.name: c for c in circuit.components if _is_mosfet(c)}
        nn_devs[(tech, "nmos")] = devs["Mnr"]
        nn_devs[(tech, "pmos")] = devs["Mpr"]
        for dtype in ("nmos", "pmos"):
            s = -1.0 if dtype == "pmos" else 1.0
            vd, vg = s * bt.vdd, s * bt.vdd
            m = nn_devs[(tech, dtype)]
            with raw_tap():
                nn_raw = nn_eval(m, vd, vg, 0.0, 0.0)
            o = osdi_eval(bt, dtype, vd, vg, 0.0, 0.0)
            same = (o is not None
                    and math.copysign(1, nn_raw["id"]) ==
                    math.copysign(1, o["id"]))
            sign_ok &= bool(same)
            print(f"  {tech:7s} {dtype:5s} fwd bias vd=vg={vd:+.2f}: "
                  f"raw NN id={nn_raw['id']:+.4e}  OSDI id="
                  f"{o['id'] if o else float('nan'):+.4e}  "
                  f"same sign={'YES' if same else 'NO'}")
    print(f"  => sign convention {'CONSISTENT' if sign_ok else 'MISMATCH'} "
          f"(raw NN vs eval_single_point)")
    if not sign_ok:
        print("  !! ABORTING comparisons would be invalid")
        return 1

    # ── 1) live SRAM force_ic attractor (state1), all 6 devices ────────────
    print("\n### 1) Live SRAM force_ic state1 attractor — per-device bias, "
          "raw NN vs post-clamp NN vs OSDI")
    sram_md: List[str] = []
    for tech in TECHS:
        bt = BENCH[tech]
        sol = sols[tech]
        q_v, qb_v = sol.get("q", float("nan")), sol.get("qb", float("nan"))
        lq, lqb = log_attr.get(tech, (float("nan"), float("nan")))
        xchk = (abs(q_v - lq) < 2e-3 and abs(qb_v - lqb) < 2e-3)
        print(f"\n  --- {tech} (VDD={bt.vdd}) — solved q={q_v:.4f} "
              f"qb={qb_v:.4f}  (s2 log: q={lq:.3f} qb={lqb:.3f}  "
              f"cross-check {'OK' if xchk else 'DIVERGED'})")
        hdr = (f"    {'dev':4s} {'type':4s} {'dir':3s} "
               f"{'Vgs(mV)':>8s} {'Vds(mV)':>8s} {'Vbs(mV)':>8s} "
               f"{'raw NN id(A)':>13s} {'post-clamp(A)':>13s} "
               f"{'OSDI id(A)':>13s} {'raw/OSDI':>9s} {'sign':>5s} "
               f"{'raw gds(S)':>11s} {'OSDI gds(S)':>11s}")
        print(hdr)
        devs = {c.name: c for c in circuits[tech].components
                if _is_mosfet(c)}
        for name in ("Mpl", "Mnl", "Mpr", "Mnr", "Mal", "Mar"):
            m = devs[name]
            dtype = "pmos" if m._is_pmos else "nmos"
            d, g, s_, b = m.nodes
            vd = sol.get(d, 0.0)
            vg = sol.get(g, 0.0)
            vs = sol.get(s_, 0.0)
            vb = sol.get(b, 0.0)
            vds, vgs, vbs = vd - vs, vg - vs, vb - vs
            normal = (vds < 0.0) if m._is_pmos else (vds > 0.0)
            with raw_tap():
                nn_raw = nn_eval(m, vd, vg, vs, vb)
            nn_post = nn_eval(m, vd, vg, vs, vb)
            o = osdi_eval(bt, dtype, vd, vg, vs, vb)
            oid = o["id"] if o else float("nan")
            ogds = o["gds"] if o else float("nan")
            ratio = (abs(nn_raw["id"]) / abs(oid)
                     if (o and abs(oid) > 0) else float("nan"))
            same = ("Y" if (o and math.copysign(1, nn_raw["id"]) ==
                            math.copysign(1, oid)) else "n")
            print(f"    {name:4s} {dtype:4s} "
                  f"{'fwd' if normal else 'REV':3s} "
                  f"{vgs*1e3:8.1f} {vds*1e3:8.1f} {vbs*1e3:8.1f} "
                  f"{nn_raw['id']:+13.4e} {nn_post['id']:+13.4e} "
                  f"{oid:+13.4e} {ratio:9.3f} {same:>5s} "
                  f"{nn_raw['gds']:11.4e} {ogds:11.4e}")
            csv_rows.append({
                "region": "sram_state1", "tech": tech, "device": name,
                "dtype": dtype, "vgs_V": vgs, "vds_V": vds, "vbs_V": vbs,
                "id_raw_nn_A": nn_raw["id"], "id_postclamp_A": nn_post["id"],
                "id_osdi_A": oid,
                "ratio_raw_over_osdi": ratio, "sign_match": same})
            if not normal:
                sram_md.append(
                    f"| {tech} | {name} ({dtype}, REV) | {vgs*1e3:.1f} | "
                    f"{vds*1e3:.1f} | {nn_raw['id']:+.3e} | "
                    f"{nn_post['id']:+.3e} | {oid:+.3e} | {ratio:.3f} | "
                    f"{same} |")

    # ── 2) reverse corridor grid ────────────────────────────────────────────
    print("\n### 2) Trained reverse-Vds corridor grid — "
          "Vds in -[0.30..0.01]*VDD (12) x Vgs in {0.2..1.0}*VDD (5), Vbs=0")
    summary: Dict[Tuple[str, str], Dict[str, float]] = {}
    for tech in TECHS:
        bt = BENCH[tech]
        for dtype in ("nmos", "pmos"):
            sgn = -1.0 if dtype == "pmos" else 1.0
            m = nn_devs[(tech, dtype)]
            vds_grid = sgn * np.linspace(-0.30, -0.01, 12) * bt.vdd
            vgs_grid = sgn * np.array([0.2, 0.4, 0.6, 0.8, 1.0]) * bt.vdd
            raws, posts, osdis, pts = [], [], [], []
            for vgs in vgs_grid:
                for vds in vds_grid:
                    with raw_tap():
                        nn_raw = nn_eval(m, vds, vgs, 0.0, 0.0)
                    nn_post = nn_eval(m, vds, vgs, 0.0, 0.0)
                    o = osdi_eval(bt, dtype, vds, vgs, 0.0, 0.0)
                    if o is None:
                        print(f"    [WARN] OSDI eval failed {tech} {dtype} "
                              f"vgs={vgs:+.3f} vds={vds:+.3f} — skipped")
                        continue
                    raws.append(nn_raw["id"])
                    posts.append(nn_post["id"])
                    osdis.append(o["id"])
                    pts.append((vgs, vds))
                    ratio = (abs(nn_raw["id"]) / abs(o["id"])
                             if abs(o["id"]) > 0 else float("nan"))
                    same = ("Y" if math.copysign(1, nn_raw["id"]) ==
                            math.copysign(1, o["id"]) else "n")
                    csv_rows.append({
                        "region": "corridor", "tech": tech, "device": m.name,
                        "dtype": dtype, "vgs_V": vgs, "vds_V": vds,
                        "vbs_V": 0.0, "id_raw_nn_A": nn_raw["id"],
                        "id_postclamp_A": nn_post["id"],
                        "id_osdi_A": o["id"],
                        "ratio_raw_over_osdi": ratio, "sign_match": same})
            raw_a = np.array(raws)
            osdi_a = np.array(osdis)
            post_a = np.array(posts)
            met = corridor_metrics(raw_a, osdi_a)
            met["post_max_abs_uA"] = float(np.max(np.abs(post_a))) * 1e6
            met["osdi_max_abs_uA"] = float(np.max(np.abs(osdi_a))) * 1e6
            met["verdict"] = verdict(met)
            summary[(tech, dtype)] = met
            print(f"\n  {tech} {dtype.upper()} (N={met['n']:.0f}, "
                  f"meaningful |OSDI id|>1uA: {met['n_meaningful']}):")
            print(f"    Rule-16:  MRE={met['mre_pct']:.2f}%  "
                  f"R2={met['r2']:.4f}  NRMSE={met['nrmse_pct']:.2f}%  "
                  f"MaxErr={met['maxerr_uA']:.3f} uA   "
                  f"(OSDI max |id|={met['osdi_max_abs_uA']:.2f} uA)")
            print(f"    sign-agreement: all(|OSDI|>1nA)={met['sign_all']:.3f}"
                  f"  meaningful={met['sign_meaningful']:.3f}")
            print(f"    magnitude (meaningful): median raw/OSDI="
                  f"{met['ratio_median']:.3f}  within-2x={met['frac_2x']:.3f}"
                  f"  within-10x={met['frac_10x']:.3f}")
            print(f"    post-clamp max |id| = {met['post_max_abs_uA']:.3e} uA"
                  f" (production zeroes the corridor)")
            print(f"    VERDICT: {met['verdict']}")

    # ── 3) beyond-corridor extrapolation ────────────────────────────────────
    print("\n### 3) Beyond-corridor extrapolation — Vds in -{0.4, 0.5}*VDD "
          "at Vgs=0.6*VDD (mirrored for PMOS)")
    beyond_md: List[str] = []
    for tech in TECHS:
        bt = BENCH[tech]
        for dtype in ("nmos", "pmos"):
            sgn = -1.0 if dtype == "pmos" else 1.0
            m = nn_devs[(tech, dtype)]
            vgs = sgn * 0.6 * bt.vdd
            for mag in (0.4, 0.5):
                vds = sgn * (-mag) * bt.vdd
                with raw_tap():
                    nn_raw = nn_eval(m, vds, vgs, 0.0, 0.0)
                nn_post = nn_eval(m, vds, vgs, 0.0, 0.0)
                o = osdi_eval(bt, dtype, vds, vgs, 0.0, 0.0)
                oid = o["id"] if o else float("nan")
                ratio = (abs(nn_raw["id"]) / abs(oid)
                         if (o and abs(oid) > 0) else float("nan"))
                same = ("Y" if (o and math.copysign(1, nn_raw["id"]) ==
                                math.copysign(1, oid)) else "n")
                print(f"  {tech:7s} {dtype:5s} vgs={vgs:+.3f} "
                      f"vds={vds:+.3f} ({-mag:+.1f}*VDD): "
                      f"raw={nn_raw['id']:+.4e}  OSDI={oid:+.4e}  "
                      f"raw/OSDI={ratio:8.3f}  sign={same}")
                csv_rows.append({
                    "region": "beyond", "tech": tech, "device": m.name,
                    "dtype": dtype, "vgs_V": vgs, "vds_V": vds, "vbs_V": 0.0,
                    "id_raw_nn_A": nn_raw["id"],
                    "id_postclamp_A": nn_post["id"], "id_osdi_A": oid,
                    "ratio_raw_over_osdi": ratio, "sign_match": same})
                beyond_md.append(
                    f"| {tech} | {dtype} | {-mag:+.1f}*VDD | "
                    f"{nn_raw['id']:+.3e} | {oid:+.3e} | {ratio:.3f} | "
                    f"{same} |")

    # ── overall verdict ──────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("### OVERALL CORRIDOR VERDICT TABLE")
    print(f"  {'tech':7s} {'dev':5s} {'MRE%':>8s} {'R2':>8s} {'NRMSE%':>8s} "
          f"{'MaxErr(uA)':>11s} {'sign(mean.)':>11s} {'med ratio':>10s} "
          f"{'<2x':>6s} {'<10x':>6s}  verdict")
    for (tech, dtype), met in summary.items():
        print(f"  {tech:7s} {dtype:5s} {met['mre_pct']:8.2f} "
              f"{met['r2']:8.4f} {met['nrmse_pct']:8.2f} "
              f"{met['maxerr_uA']:11.3f} {met['sign_meaningful']:11.3f} "
              f"{met['ratio_median']:10.3f} {met['frac_2x']:6.3f} "
              f"{met['frac_10x']:6.3f}  {met['verdict']}")

    # ── artifacts ────────────────────────────────────────────────────────────
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nCSV written: {CSV_PATH}")

    md_lines.append("# S7_P2_rev_probe — raw reverse-Vds surface vs OSDI "
                    "(P2 pre-build go/no-go)\n")
    md_lines.append("Raw tap: identity-monkeypatched "
                    "`_MOSFETNNBase._apply_vds_correction` (post-denorm, "
                    "pre-correction). Ground truth: OSDI via PyCMG "
                    "`eval_single_point` at identical absolute bias "
                    "(sign-consistency verified at forward bias, all 8 "
                    "tech/device cells). Corridor grid: Vds "
                    "-[0.30..0.01]*VDD x Vgs {0.2..1.0}*VDD, Vbs=0, "
                    "mirrored for PMOS; 60 pts per cell.\n")
    md_lines.append("## Corridor Rule-16 + agreement (raw NN id vs OSDI id)\n")
    md_lines.append("| tech | dev | MRE % | R2 | NRMSE % | MaxErr uA | "
                    "sign-agree (meaningful) | median ratio | <2x | <10x | "
                    "verdict |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for (tech, dtype), met in summary.items():
        md_lines.append(
            f"| {tech} | {dtype} | {met['mre_pct']:.2f} | {met['r2']:.4f} | "
            f"{met['nrmse_pct']:.2f} | {met['maxerr_uA']:.3f} | "
            f"{met['sign_meaningful']:.3f} (n={met['n_meaningful']}) | "
            f"{met['ratio_median']:.3f} | {met['frac_2x']:.3f} | "
            f"{met['frac_10x']:.3f} | **{met['verdict']}** |")
    md_lines.append("\n## SRAM force_ic state1 attractor — reverse-biased "
                    "devices (the P2 beneficiaries)\n")
    md_lines.append("| tech | device | Vgs mV | Vds mV | raw NN id | "
                    "post-clamp id | OSDI id | raw/OSDI | sign |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|")
    md_lines.extend(sram_md)
    md_lines.append("\n## Beyond-corridor (Vgs=0.6*VDD)\n")
    md_lines.append("| tech | dev | Vds | raw NN id | OSDI id | raw/OSDI | "
                    "sign |")
    md_lines.append("|---|---|---|---|---|---|---|")
    md_lines.extend(beyond_md)
    md_lines.append("\nFull per-point data: `s7_rev_probe_corridor.csv`; "
                    "log: `s7_logs/s7_rev_probe.log`.\n")
    MD_PATH.write_text("\n".join(md_lines))
    print(f"MD summary written: {MD_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
