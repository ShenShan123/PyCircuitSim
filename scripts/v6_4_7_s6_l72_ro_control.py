#!/usr/bin/env python3
"""DirectNet V6.4.7 — S6 control: TSMC7 5-stage RO through PyCircuitSim's
NATIVE LEVEL=72 (BSIM-CMG via PyCMG/OSDI) device path vs NGSPICE.

WHY: the S6=P1 injection (exact OSDI id+charges monkeypatched into the live
LEVEL=73 RO) measured ~93 ps vs NGSPICE 46.65 ps — a suspicious 2.00x with
half-periods equal to NG's FULL period. Before indicting the solver, run the
clean control: the production LEVEL=72 model (DC <0.1 %, tran ~0.20 % NRMSE
vs NGSPICE) on the IDENTICAL RO circuit, NO monkeypatch of any device path.

DECISION:
  * L72 period ≈ NG (ratio ~1.0)  → solver/harness EXONERATED; the 93 ps of
    S6=P1 (and P0-I's 92 ps) is an injection-scheme artifact.
  * L72 period ≈ 2x NG (ratio ~2) → the solver genuinely runs OSDI-class RO
    dynamics 2x slow — indicts the solver/harness (huge finding).

LIKE-FOR-LIKE GUARANTEES
  * Circuit: identical to verify_complex_ring_osc.py — 5 stages, per stage
    PMOS L=20n NFIN=2 + NMOS L=16n NFIN=2 + 0.5 fF to ground, alternating
    .ic, VDD=0.75, TSMC7 ULVT (BENCH["TSMC7"]).
  * Modelcards: BOTH sides consume the same two resolved naive cards
    (profile.get_{n,p}mos_modelcard → pycmg.tech.resolve_modelcard).
    NGSPICE gets the baked merge (L/NFIN/TFIN/DEVTYPE injected — OSDI
    rejects instance params); PyCircuitSim gets the unbaked merge +
    L/NFIN/TFIN on the M lines (tests/common/bsimcmg_tran.py pattern).
    Source-card sha256s are printed as proof.
  * Runner: tests.common.complex.run_directnet_transient VERBATIM (S5b
    constrained-.ic uic-equivalent start, same solver settings as every S6
    run). Only the module-global `parse_netlist` is swapped for one that
    passes modelcard_path/model_name_map to Parser — harness plumbing; the
    device model is the untouched production NMOS_CMG/PMOS_CMG.
  * Window/estimator: 0.6 ns / settle 0.3 ns / tstep 2 ps and
    _period_from_wave (zero-crossing period estimator).

Usage:
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      conda run -n pycircuitsim python scripts/v6_4_7_s6_l72_ro_control.py \
      2>&1 | tee results/v6_4_7/s6_logs/s6_l72_control.log
Smoke (wall-time projection only, period will be NaN):
    ... --tstop-ns 0.05 --settle-ns 0.01
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

print = functools.partial(print, flush=True)  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "external_compact_models" / "bsim_cmg" / "tests",
          ROOT / "external_compact_models" / "bsim_cmg",
          ROOT / "external_compact_models",
          ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tests.common.complex as cx  # noqa: E402
import tests.simple_circuits.verify_complex_ring_osc as ro_mod  # noqa: E402
from tests.common.complex import (  # noqa: E402
    BENCH, BenchTech, full_metrics, run_directnet_transient,
)
from tests.simple_circuits.verify_complex_ring_osc import (  # noqa: E402
    _period_from_wave, run_ngspice_ro,
)

OUT_DIR = ROOT / "results" / "v6_4_7" / "s6_logs"
S6_INJECTION_PS = 93.01   # S6=P1 id+q N+P injection (s6_p1_id_q_np_600ps.md)
S6_BASELINE_PS = None     # printed from the baseline md if present


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def render_l72_netlist(bt: BenchTech, tstep: float, tstop: float,
                       out_path: Path) -> Path:
    """The verify_complex_ring_osc RO, with LEVEL=72 .model lines.

    Topology/geometry identical to both the NGSPICE deck (run_ngspice_ro) and
    the LEVEL=73 template: stage i drives nd[i] from nd[i-1] (n5 wraps to
    stage 1). TFIN is pinned on the M lines to the value the NGSPICE bake
    injects (bsimcmg_tran.py L72 pattern) so geometry cannot confound.
    """
    nd = ["n1", "n2", "n3", "n4", "n5"]
    ln_nm = bt.l_nmos * 1e9
    lp_nm = bt.l_pmos * 1e9
    tfin_nm = bt.tfin * 1e9
    lines = [
        "* 5-stage CMOS ring oscillator -- BSIM-CMG LEVEL=72 control "
        "(V6.4.7 S6)",
        f"* {bt.name} VT={bt.vt} VDD={bt.vdd} -- same circuit as "
        "verify_complex_ring_osc.py",
        "",
        f"Vdd vdd 0 {bt.vdd}",
        f".ic V(n1)=0.0 V(n2)={bt.vdd} V(n3)=0.0 V(n4)={bt.vdd} V(n5)=0.0",
        "",
    ]
    for i in range(5):
        lines += [
            f"Mp{i+1} {nd[i]} {nd[i-1]} vdd vdd {bt.pmos_model} "
            f"L={lp_nm:.0f}n NFIN={bt.nfin} TFIN={tfin_nm:.1f}n",
            f"Mn{i+1} {nd[i]} {nd[i-1]} 0   0   {bt.nmos_model} "
            f"L={ln_nm:.0f}n NFIN={bt.nfin} TFIN={tfin_nm:.1f}n",
            f"Cl{i+1} {nd[i]} 0 0.5f",
            "",
        ]
    lines += [
        f".model {bt.nmos_model} NMOS (LEVEL=72)",
        f".model {bt.pmos_model} PMOS (LEVEL=72)",
        "",
        f".tran {tstep*1e12:.0f}p {tstop*1e9:.4f}n",
        "",
        ".end",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


def build_merged_card(bt: BenchTech, work_dir: Path
                      ) -> Tuple[Path, Path, Path]:
    """Unbaked NMOS+PMOS merge for PyCircuitSim — same sources NGSPICE bakes."""
    nmos_src = bt.profile.get_nmos_modelcard(bt.vt_pair, bt.l_nmos)
    pmos_src = bt.profile.get_pmos_modelcard(bt.vt_pair, bt.l_pmos)
    if not nmos_src.exists() or not pmos_src.exists():
        raise FileNotFoundError(f"modelcard missing: {nmos_src} / {pmos_src}")
    merged = work_dir / f"merged_{bt.name}_{bt.vt}_l72.lib"
    merged.write_text(nmos_src.read_text() + "\n" + pmos_src.read_text())
    return merged, nmos_src, pmos_src


def make_l72_parse(merged: Path, bt: BenchTech):
    """parse_netlist drop-in that binds the merged TSMC card for LEVEL=72.

    Harness plumbing ONLY (Parser ctor args) — the device path is the
    production NMOS_CMG/PMOS_CMG, no monkeypatch anywhere near a model.
    """
    def parse(netlist_path: Path):
        from pycircuitsim.parser import Parser
        parser = Parser(
            modelcard_path=str(merged),
            model_name_map={"NMOS": bt.nmos_model, "PMOS": bt.pmos_model},
        )
        parser.parse_file(str(netlist_path))
        return parser
    return parse


def half_periods(t: np.ndarray, v: np.ndarray, mid: float, settle: float
                 ) -> Tuple[np.ndarray, int, int, float, float]:
    """All midpoint crossings (rise+fall, linear-interp) -> half-period list."""
    keep = t >= settle
    t, v = t[keep], v[keep]
    if len(v) < 4:
        return np.array([]), 0, 0, float("nan"), float("nan")
    sgn = np.sign(v - mid)
    rises = np.where((sgn[:-1] < 0) & (sgn[1:] >= 0))[0]
    falls = np.where((sgn[:-1] > 0) & (sgn[1:] <= 0))[0]

    def interp(idx: np.ndarray) -> list:
        out = []
        for i in idx:
            v0, v1 = v[i], v[i + 1]
            frac = (mid - v0) / (v1 - v0) if v1 != v0 else 0.0
            out.append(t[i] + frac * (t[i + 1] - t[i]))
        return out

    xt = np.sort(np.array(interp(rises) + interp(falls)))
    hp = np.diff(xt) if len(xt) > 1 else np.array([])
    return hp, len(rises), len(falls), float(v.min()), float(v.max())


def verdict_line(per_ps: float, ng_ps: float) -> str:
    if not np.isfinite(per_ps):
        return ("VERDICT: L72 period not measurable (NaN) — no oscillation "
                "in window or non-periodic; judge from waveform/half-periods")
    ratio = per_ps / ng_ps
    if abs(ratio - 1.0) <= 0.05:
        return (f"VERDICT: L72 {per_ps:.2f} ps vs NG {ng_ps:.2f} ps "
                f"(ratio {ratio:.3f}) — solver/harness EXONERATED; the "
                f"~93 ps S6=P1 (and P0-I ~92 ps) injection numbers are "
                f"injection-scheme artifacts")
    if ratio >= 1.8:
        return (f"VERDICT: L72 {per_ps:.2f} ps vs NG {ng_ps:.2f} ps "
                f"(ratio {ratio:.3f}) — solver genuinely runs OSDI-class RO "
                f"dynamics ~2x slow; INDICTS the solver/harness")
    return (f"VERDICT: L72 {per_ps:.2f} ps vs NG {ng_ps:.2f} ps "
            f"(ratio {ratio:.3f}) — INTERMEDIATE, judge manually")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="S6 control: native LEVEL=72 TSMC7 RO vs NGSPICE")
    ap.add_argument("--tstop-ns", type=float, default=0.6,
                    help="transient window in ns (S6 causal window: 0.6)")
    ap.add_argument("--settle-ns", type=float, default=0.3,
                    help="period-measurement settle in ns (S6: 0.3)")
    ap.add_argument("--skip-ngspice", action="store_true",
                    help="smoke: skip the NGSPICE reference run")
    args = ap.parse_args()

    tstop = args.tstop_ns * 1e-9
    settle = args.settle_ns * 1e-9
    tstep = ro_mod.TRAN_TSTEP                       # 2 ps, benchmark verbatim
    ro_mod.TRAN_TSTOP = tstop                       # NG side uses the global
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bt = BENCH["TSMC7"]
    mid = bt.vdd / 2.0
    work_dir = Path(tempfile.mkdtemp(prefix="s6_l72_ctrl_"))

    print(f"[S6-L72] TSMC7 RO control  TSTOP={tstop*1e9:.3f} ns  "
          f"SETTLE={settle*1e9:.3f} ns  tstep={tstep*1e12:.0f} ps  "
          f"(start-state: S5b constrained-.ic uic-equivalent, both sides)")
    print(f"[S6-L72] geometry: NMOS {bt.nmos_model} L={bt.l_nmos*1e9:.0f}n  "
          f"PMOS {bt.pmos_model} L={bt.l_pmos*1e9:.0f}n  NFIN={bt.nfin}  "
          f"TFIN={bt.tfin*1e9:.1f}n  VDD={bt.vdd}  Cload=0.5f")

    # --- modelcards: prove like-for-like -----------------------------------
    merged, nmos_src, pmos_src = build_merged_card(bt, work_dir)
    print(f"[S6-L72] NMOS source card: {nmos_src}")
    print(f"         sha256={_sha256(nmos_src)}")
    print(f"[S6-L72] PMOS source card: {pmos_src}")
    print(f"         sha256={_sha256(pmos_src)}")
    print(f"[S6-L72] PyCircuitSim merged (unbaked): {merged}")
    baked = cx.get_baked_modelcard(bt, bt.nfin, work_dir)
    print(f"[S6-L72] NGSPICE baked (same sources + L/NFIN/TFIN/DEVTYPE "
          f"injected): {baked}")

    # --- NGSPICE reference --------------------------------------------------
    ng_per = float("nan")
    ng: Dict[str, np.ndarray] = {}
    if not args.skip_ngspice:
        ng = run_ngspice_ro(bt, work_dir)
        ng_per = _period_from_wave(ng["time"], ng["v(n5)"], mid, settle)
        print(f"[S6-L72] NGSPICE period = {ng_per*1e12:.2f} ps")

    # --- PyCircuitSim native LEVEL=72 ---------------------------------------
    netlist = render_l72_netlist(bt, tstep, tstop,
                                 work_dir / "ring_osc_TSMC7_l72.sp")
    print(f"[S6-L72] L72 netlist: {netlist}")

    orig_parse = cx.parse_netlist
    cx.parse_netlist = make_l72_parse(merged, bt)
    t0 = time.time()
    try:
        results, partial, err = run_directnet_transient(netlist)
    finally:
        cx.parse_netlist = orig_parse
    wall = time.time() - t0

    t_a = np.asarray(results["time"])
    v_a = np.asarray(results["n5"])
    reached = t_a[-1] * 1e9 if len(t_a) else float("nan")
    s_per_ns = wall / reached if reached and np.isfinite(reached) else float("nan")
    print(f"[S6-L72] transient done: wall={wall:.0f}s  reached={reached:.4f}ns "
          f"({s_per_ns:.0f} s/ns; 0.6 ns ≈ {s_per_ns*0.6/60:.1f} min)  "
          f"partial={partial} err={err!r}")

    per = _period_from_wave(t_a, v_a, mid, settle)
    ratio = per / ng_per if np.isfinite(per) and np.isfinite(ng_per) else float("nan")
    per_err = (abs(per - ng_per) / ng_per * 100.0
               if np.isfinite(per) and np.isfinite(ng_per) else float("nan"))

    m = {"mre_pct": float("nan"), "r2": float("nan"),
         "nrmse_pct": float("nan"), "max_err": float("nan")}
    if ng and len(t_a) > 3:
        t_lo, t_hi = settle, min(ng["time"][-1], t_a[-1])
        if t_hi > t_lo:
            grid = np.arange(t_lo, t_hi, tstep)
            ng_i = np.interp(grid, ng["time"], ng["v(n5)"])
            l72_i = np.interp(grid, t_a, v_a)
            m = full_metrics(l72_i, ng_i)

    hp, n_rise, n_fall, vmin, vmax = half_periods(t_a, v_a, mid, settle)
    hp_ps = hp * 1e12
    print(f"[S6-L72] L72 period = {per*1e12:.2f} ps  NG = {ng_per*1e12:.2f} ps"
          f"  ratio = {ratio:.3f}  err = {per_err:.2f}%")
    print(f"         wave: vmin={vmin:.3f} vmax={vmax:.3f} "
          f"swing={vmax-vmin:.3f}V  n_rise={n_rise} n_fall={n_fall}")
    print(f"         half-periods(ps)="
          f"{np.array2string(hp_ps[:14], precision=1, separator=',')}")
    print(f"         waveform vs NG: NRMSE={m['nrmse_pct']:.2f}%  "
          f"R2={m['r2']:.4f}")

    wave_path = OUT_DIR / f"wave_l72_control_{tstop*1e12:.0f}ps.npz"
    np.savez(wave_path, time=t_a, vn5=v_a,
             ng_time=ng.get("time", np.array([])),
             ng_vn5=ng.get("v(n5)", np.array([])))
    print(f"         saved waveform -> {wave_path}")

    verdict = verdict_line(per * 1e12, ng_per * 1e12)
    print(f"[S6-L72] {verdict}")

    md = OUT_DIR / f"s6_l72_control_{tstop*1e12:.0f}ps.md"
    md.write_text(
        f"# S6 control — native LEVEL=72 TSMC7 RO vs NGSPICE "
        f"(window {tstop*1e9:.3f} ns, settle {settle*1e9:.3f} ns, "
        f"tstep {tstep*1e12:.0f} ps)\n\n"
        f"Circuit: 5-stage RO, PMOS {bt.pmos_model} L={bt.l_pmos*1e9:.0f}n + "
        f"NMOS {bt.nmos_model} L={bt.l_nmos*1e9:.0f}n, NFIN={bt.nfin}, "
        f"TFIN={bt.tfin*1e9:.1f}n, 0.5 fF/stage, VDD={bt.vdd}, alternating "
        f".ic, S5b constrained-.ic start. NO device monkeypatch — production "
        f"NMOS_CMG/PMOS_CMG.\n\n"
        f"Modelcard sources (both sides):\n"
        f"- NMOS `{nmos_src}` sha256 `{_sha256(nmos_src)}`\n"
        f"- PMOS `{pmos_src}` sha256 `{_sha256(pmos_src)}`\n\n"
        f"| run | period (ps) | ratio vs NG | err% | NRMSE% | R2 | partial | "
        f"reached (ns) | wall (s) |\n"
        f"|-----|------------:|------------:|-----:|-------:|---:|:-------:|"
        f"-------------:|---------:|\n"
        f"| PyCircuitSim L72 | {per*1e12:.2f} | {ratio:.3f} | {per_err:.2f} | "
        f"{m['nrmse_pct']:.2f} | {m['r2']:.4f} | {partial} | {reached:.4f} | "
        f"{wall:.0f} |\n"
        f"| NGSPICE (truth) | {ng_per*1e12:.2f} | 1.000 | — | — | — | — | "
        f"{tstop*1e9:.3f} | — |\n"
        f"| S6=P1 id+q injection (context) | {S6_INJECTION_PS:.2f} | "
        f"{S6_INJECTION_PS/(ng_per*1e12):.3f} | — | — | — | — | — | — |\n\n"
        f"L72 wave: vmin={vmin:.3f} vmax={vmax:.3f} swing={vmax-vmin:.3f} V, "
        f"n_rise={n_rise}, n_fall={n_fall}\n\n"
        f"half-periods (ps): "
        f"{np.array2string(hp_ps[:14], precision=1, separator=',')}\n\n"
        f"**{verdict}**\n")
    print(f"[S6-L72] results table -> {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
