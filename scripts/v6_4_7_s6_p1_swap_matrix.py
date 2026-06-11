#!/usr/bin/env python3
"""DirectNet V6.4.7 — S6 = P1: complete the causal swap matrix — exact OSDI
`id` AND charges (qg/qd/qs/qb + cap Jacobian) TOGETHER in the live TSMC7 RO.

INSTRUMENTATION-ONLY (no production code touched). The missing cell of the
V6.4.6 P0-I matrix:

    | device model                   | id   | charge/caps | RO period (pre-S5b) |
    |--------------------------------|------|-------------|---------------------|
    | baseline (full NN)             | NN   | NN          | 50.83 ps (+9 %)     |
    | P0-I id-only swap              | OSDI | NN          | 92.30 ps (+98 %)    |
    | **THIS SCRIPT (id+q swap)**    | OSDI | OSDI        | ?                   |
    | NGSPICE (full OSDI, truth)     | OSDI | OSDI        | 46.64 ps            |

DECISION CRITERIA (vs the NGSPICE period measured in the same run):
  * period within ~1.5 % of NG (≈46.6–47.3 ps) → **EXONERATED**: the
    PyCircuitSim solver/harness integrates a faithful OSDI device to the
    NGSPICE period; the RO gap is confirmed as JOINT (id, charge) model error.
  * period ≥ 75 ps or ≥ 40 % off NG (e.g. the ~92 ps of P0-I persists even
    with exact charges) → **INDICTS the solver/harness** — the biggest find
    of the iteration.
  * anything else → INTERMEDIATE, judge manually from the waveform diag.

CONSISTENCY SCHEME (the P0-I v2 lesson — value/Jacobian must come from the
SAME OSDI evaluation at the SAME live bias, or NR diverges / measures an
artifact). At each device eval we run the exact-bias analytic OSDI op-point
(`eval_single_point`) and inject, in the NN frame the stamps consume
(`_stamp_mosfet_dc` + `_stamp_mosfet_transient`, solver.py:1728-1844):
  id    : OSDI col, direct (training-data convention — P0-D shift invariance)
  gm    : OSDI gm  = -∂id/∂Vg == NN gm   (direct; v2 mapping, FD-probed)
  gmb   : OSDI gmb = -∂id/∂Vb == NN gmb  (direct; Vbs≡0 in the RO ⇒ inert)
  gds   : NN frame stamps floor(+∂id/∂Vd) = floor(-OSDI_gds) = |id|·0.5
          (v2 verbatim; P0-C proved the gds Jacobian is period-inert)
  qg/qd/qb : OSDI charge VALUES, direct (same convention as training data;
          these feed the transient companion currents i_cap = coeff·q − h)
  qs    : -(qg+qd+qb) reconstructed (Rule 14, simulator convention)
  cgg/cdd  : OSDI diagonals, direct
  cgd/cdg/cgs : OSDI returns SPICE convention (off-diagonals negated);
          the NN frame is raw +dQ/dV → negate OSDI (S5 NEG_OSDI; FD-verified
          at a mid-swing bias before the run, fail-loud on mismatch).

START-STATE CONVENTION: both modes run through run_directnet_ro →
tests.common.complex.run_directnet_transient, which at commit 7454034 (S5b)
starts the transient from a CONSTRAINED `.ic` operating point (uic-equivalent,
matching the NGSPICE `tran ... uic` side). Baseline and injection therefore
share the identical start state; the baseline must be re-anchored here because
the P0-I numbers above were measured pre-S5b.

Usage (full causal runs — same 0.6 ns / settle 0.3 ns window P0-I used):
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      conda run -n pycircuitsim python scripts/v6_4_7_s6_p1_swap_matrix.py \
      --mode baseline   > results/v6_4_7/s6_logs/s6_p1_baseline.log 2>&1
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      conda run -n pycircuitsim python scripts/v6_4_7_s6_p1_swap_matrix.py \
      --mode id_q_np    > results/v6_4_7/s6_logs/s6_p1_id_q_np.log 2>&1

Smoke (first ~25 transient steps = 50 ps, proves NR convergence + plumbing):
    ... --mode id_q_np --max-steps 25 --settle-ns 0.01
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Stream progress even under `conda run` file redirection.
print = functools.partial(print, flush=True)  # type: ignore[assignment]

# Force the per-device eval path so the monkeypatch is the single source of
# truth (the batched path bypasses _eval via _unpack_eval). Set BEFORE any
# solver import so the module-level flag read picks it up. (P0-I v2 verbatim.)
os.environ["NN_BATCHED_EVAL"] = "0"

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "external_compact_models" / "PyCMG" / "tests",
          ROOT / "external_compact_models" / "PyCMG",
          ROOT / "external_compact_models",
          ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tests.verify_complex_ring_osc as ro_mod  # noqa: E402
from tests.common.complex import BENCH, full_metrics  # noqa: E402
from tests.verify_complex_ring_osc import (  # noqa: E402
    run_directnet_ro, run_ngspice_ro, _period_from_wave,
)
from pycircuitsim.models.mosfet_nn import _MOSFETNNBase  # noqa: E402
from pycmg.nn_generate import _create_model_and_instance, eval_single_point  # noqa: E402
from pycmg.nn_config import TECH_CONFIGS  # noqa: E402

_ORIG_EVAL = _MOSFETNNBase._eval
OUT_DIR = ROOT / "results" / "v6_4_7" / "s6_logs"
NEG_OSDI = ("cgd", "cdg", "cgs")   # OSDI SPICE convention -> NN raw +dQ/dV
P0I_BASELINE_PS = 50.83            # pre-S5b anchors, for the comparison row
P0I_ID_ONLY_PS = 92.30


def _floor_gds(id_phys: float, gds_phys: float) -> float:
    """Mirror `_MOSFETNNBase._floor_gds` (Rule 5): max(|id|·0.5, 1e-12)."""
    return max(gds_phys, max(abs(id_phys) * 0.5, 1e-12))


class OsdiCache:
    """One PyCMG instance per (device_type, L) with an exact-bias memo cache.

    P0-I v2 verbatim, extended: `consistent_op` returns the FULL OSDI
    operating point mapped into the NN frame (id/gm/gds/gmb + qg/qd/qs/qb +
    cgg/cgd/cgs/cdg/cdd) instead of the 4-tuple, so value AND Jacobian of
    both the resistive and the charge path come from one OSDI evaluation.
    Memo only on a truly-identical re-query (bias rounded to 1e-10 V — six
    orders below the NR tolerance, cannot staircase the surface).
    """

    def __init__(self, bt) -> None:
        self.bt = bt
        self.tech = TECH_CONFIGS[bt.nn_tech]
        self._inst: Dict[Tuple[str, float], object] = {}
        self._cache: Dict[Tuple, Optional[Dict[str, float]]] = {}
        self.n_calls = 0
        self.n_hits = 0
        self.n_fail = 0

    def inst_for(self, device_type: str, L: float):
        key = (device_type, round(L, 12))
        if key not in self._inst:
            made = _create_model_and_instance(
                self.tech, device_type, self.bt.vt, L, float(self.bt.nfin),
                300.15)
            if made is None:
                raise RuntimeError(
                    f"OSDI instance build failed for {device_type} L={L}")
            self._inst[key] = made[1]
        return self._inst[key]

    def consistent_op(self, is_pmos: bool, L: float, vd: float, vg: float,
                      vs: float, vb: float) -> Optional[Dict[str, float]]:
        """Full OSDI op-point at the live bias, in the NN frame. None on fail."""
        dt = "pmos" if is_pmos else "nmos"
        rk = (dt, round(L, 12), round(vd, 10), round(vg, 10),
              round(vs, 10), round(vb, 10))
        cached = self._cache.get(rk)
        if cached is not None:
            self.n_hits += 1
            return cached
        self.n_calls += 1
        inst = self.inst_for(dt, L)
        o = eval_single_point(inst, vd, vg, vs, vb, _silent=True)
        if o is None:
            self.n_fail += 1
            self._cache[rk] = None
            return None
        id_phys = float(o["id"])
        qg, qd, qb = float(o["qg"]), float(o["qd"]), float(o["qb"])
        op: Dict[str, float] = {
            # resistive path (P0-I v2 mapping, verbatim)
            "id": id_phys,
            "gm": float(o["gm"]),                       # == NN gm
            "gmb": float(o["gmb"]),                     # == NN gmb (inert)
            "gds": _floor_gds(id_phys, -float(o["gds"])),
            # charge path (S5 conventions: q direct, off-diag caps negated)
            "qg": qg, "qd": qd, "qb": qb,
            "qs": -(qg + qd + qb),                      # Rule 14
            "cgg": float(o["cgg"]), "cdd": float(o["cdd"]),
            "cgd": -float(o["cgd"]), "cdg": -float(o["cdg"]),
            "cgs": -float(o["cgs"]),
        }
        self._cache[rk] = op
        return op


class SwapStats:
    """Running NN-vs-OSDI aggregates at injected biases (no per-call storage)."""

    def __init__(self, spot_n: int) -> None:
        self.spot_n = spot_n
        self.n = 0
        self.n_fallback = 0
        self.max_dqg = 0.0          # aC-comparable: |qg_nn - qg_osdi|
        self.max_dqd = 0.0
        self.max_did = 0.0          # A
        self.max_rel_id = 0.0       # among |id_osdi| > 1 uA
        self.max_rel_id_bias: Tuple[float, ...] = ()

    def update(self, dev_name: str, nn: Dict[str, float],
               op: Dict[str, float], bias: Tuple[float, float, float, float],
               ) -> None:
        self.n += 1
        dqg = abs(nn["qg"] - op["qg"])
        dqd = abs(nn["qd"] - op["qd"])
        did = abs(nn["id"] - op["id"])
        self.max_dqg = max(self.max_dqg, dqg)
        self.max_dqd = max(self.max_dqd, dqd)
        self.max_did = max(self.max_did, did)
        if abs(op["id"]) > 1e-6:
            rel = did / abs(op["id"])
            if rel > self.max_rel_id:
                self.max_rel_id = rel
                self.max_rel_id_bias = bias
        if self.n <= self.spot_n:
            vd, vg, vs, vb = bias
            print(f"    [spot {self.n}] {dev_name} "
                  f"(vd={vd:+.3f} vg={vg:+.3f} vs={vs:+.3f} vb={vb:+.3f})  "
                  f"id NN={nn['id']:+.4e} OSDI={op['id']:+.4e} A "
                  f"(Δ={nn['id'] - op['id']:+.2e})  "
                  f"qg NN={nn['qg'] * 1e18:+.3f} OSDI={op['qg'] * 1e18:+.3f} aC "
                  f"(Δ={dqg * 1e18:.3f} aC)")

    def summary(self) -> str:
        return (f"injected_evals={self.n} fallbacks={self.n_fallback}  "
                f"max|Δqg|={self.max_dqg * 1e18:.3f} aC  "
                f"max|Δqd|={self.max_dqd * 1e18:.3f} aC  "
                f"max|Δid|={self.max_did * 1e6:.3f} uA  "
                f"max rel id (|id|>1uA)={self.max_rel_id * 100:.1f}% "
                f"at bias={tuple(round(v, 3) for v in self.max_rel_id_bias)}")


def fd_check(inst, vd: float, vg: float, vs: float, vb: float,
             delta: float = 1e-3) -> None:
    """FD-verify the OSDI→NN sign mapping at one mid-swing bias; fail loud.

    Caps (cgd/cgs/cdg): NN frame = raw +dQ/dV; mapped OSDI = -OSDI value
    (S5 fd_check_caps pattern). gm: NN gm == OSDI gm == -∂id/∂Vg.
    """
    base = eval_single_point(inst, vd, vg, vs, vb, _silent=True)
    if base is None:
        raise RuntimeError("FD check: OSDI eval failed at base bias")
    failures: List[str] = []
    for key, term in (("cgd", "d"), ("cgs", "s"), ("cdg", "g")):
        qty = "qg" if key.startswith("cg") else "qd"
        bias = {"d": vd, "g": vg, "s": vs, "b": vb}
        bias[term] += delta
        hi = eval_single_point(inst, bias["d"], bias["g"], bias["s"],
                               bias["b"], _silent=True)
        bias[term] -= 2 * delta
        lo = eval_single_point(inst, bias["d"], bias["g"], bias["s"],
                               bias["b"], _silent=True)
        if hi is None or lo is None:
            raise RuntimeError(f"FD check: OSDI eval failed near base ({key})")
        fd = (hi[qty] - lo[qty]) / (2 * delta)       # raw dQ/dV (NN frame)
        mapped = -base[key]                          # NEG_OSDI mapping
        rel = abs(mapped - fd) / max(abs(fd), 1e-21)
        ok = "OK" if rel < 0.2 else "MISMATCH"
        print(f"  FD {key}: raw dQ/dV={fd:+.3e}  -OSDI {key}={mapped:+.3e}"
              f"  rel={rel:.2%}  {ok}")
        if rel >= 0.2:
            failures.append(key)
    # gm: central FD of OSDI id wrt Vg; NN gm = OSDI gm = -∂id/∂Vg
    hi = eval_single_point(inst, vd, vg + delta, vs, vb, _silent=True)
    lo = eval_single_point(inst, vd, vg - delta, vs, vb, _silent=True)
    if hi is None or lo is None:
        raise RuntimeError("FD check: OSDI eval failed near base (gm)")
    fd_gm = -(hi["id"] - lo["id"]) / (2 * delta)
    rel = abs(base["gm"] - fd_gm) / max(abs(fd_gm), 1e-21)
    ok = "OK" if rel < 0.2 else "MISMATCH"
    print(f"  FD gm : -d(id)/dVg={fd_gm:+.3e}  OSDI gm={base['gm']:+.3e}"
          f"  rel={rel:.2%}  {ok}")
    if rel >= 0.2:
        failures.append("gm")
    if failures:
        raise RuntimeError(
            f"FD convention check FAILED for {failures} — convention suspect, "
            f"aborting (fail loud)")


def make_patch(cache: OsdiCache, inject_pmos: bool, stats: SwapStats):
    """`_eval` replacement: inject the full consistent OSDI op-point (id +
    gm/gds/gmb + qg/qd/qs/qb + caps) on NMOS (and PMOS iff `inject_pmos`)."""

    def patched(self: _MOSFETNNBase, voltages: Dict[str, float]):
        # A warm cache already holds the INJECTED dict, so the NN-vs-OSDI
        # stats are only genuine on a fresh eval (cache miss).
        fresh = (self._cache_voltages != self._v_tuple(voltages)
                 or self._eval_cache is None)
        r = _ORIG_EVAL(self, voltages)            # NN result (spot compare)
        if self._is_pmos and not inject_pmos:
            return r                              # NMOS-only: PMOS stays NN
        vd = voltages.get(self.nodes[0], 0.0)
        vg = voltages.get(self.nodes[1], 0.0)
        vs = voltages.get(self.nodes[2], 0.0)
        vb = voltages.get(self.nodes[3], 0.0)
        op = cache.consistent_op(self._is_pmos, self.L, vd, vg, vs, vb)
        if op is None:
            stats.n_fallback += 1
            return r                              # OSDI failed → keep NN (rare)
        if fresh:
            stats.update(self.name, r, op, (vd, vg, vs, vb))
        r = dict(r)                               # don't mutate the cached dict
        r.update(op)
        self._eval_cache = r
        self._cache_voltages = self._v_tuple(voltages)
        return r

    return patched


def run_variant(bt, label: str, inject_pmos: Optional[bool],
                cache: OsdiCache, stats: SwapStats):
    """Run one RO transient under the given injection mode; return waveform."""
    work_dir = Path(tempfile.mkdtemp(prefix=f"s6p1_{label}_"))
    if inject_pmos is None:
        _MOSFETNNBase._eval = _ORIG_EVAL
    else:
        _MOSFETNNBase._eval = make_patch(cache, inject_pmos, stats)
    try:
        dn, partial, err = run_directnet_ro(bt, work_dir)
    finally:
        _MOSFETNNBase._eval = _ORIG_EVAL          # ALWAYS restore
    return dn, partial, err


def verdict_line(mode: str, per_ps: float, ng_ps: float) -> str:
    """One-line verdict vs the S6=P1 decision thresholds."""
    if mode == "baseline":
        return (f"VERDICT(baseline): anchor under S5b uic-start = {per_ps:.2f} ps "
                f"(pre-S5b anchor {P0I_BASELINE_PS:.2f} ps; NG {ng_ps:.2f} ps)")
    if not np.isfinite(per_ps):
        return ("VERDICT: period not measurable in this window (NaN) — "
                "extend tstop or judge from half-period diagnostics")
    err = abs(per_ps - ng_ps) / ng_ps
    if err <= 0.015:
        return (f"VERDICT: {per_ps:.2f} ps within 1.5% of NG {ng_ps:.2f} ps — "
                f"simulator/harness EXONERATED; RO gap is JOINT (id,q) "
                f"model error")
    if per_ps >= 75.0 or err >= 0.40:
        return (f"VERDICT: {per_ps:.2f} ps vs NG {ng_ps:.2f} ps "
                f"({err * 100:.1f}% off; P0-I id-only gave "
                f"{P0I_ID_ONLY_PS:.2f} ps) — INDICTS solver/harness")
    return (f"VERDICT: {per_ps:.2f} ps vs NG {ng_ps:.2f} ps "
            f"({err * 100:.1f}% off) — INTERMEDIATE, judge manually")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="S6=P1 causal swap matrix: exact OSDI id+charges in the "
                    "live TSMC7 RO")
    ap.add_argument("--mode", required=True,
                    choices=["baseline", "id_q_np", "id_q_nmos_only"])
    ap.add_argument("--tstop-ns", type=float, default=0.6,
                    help="transient window in ns (P0-I causal window: 0.6)")
    ap.add_argument("--settle-ns", type=float, default=0.3,
                    help="period-measurement settle in ns (P0-I: 0.3)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="smoke: limit to N nominal steps "
                         "(tstop = N * 2 ps, overrides --tstop-ns)")
    ap.add_argument("--spot-n", type=int, default=6,
                    help="print the first N NN-vs-OSDI spot comparisons")
    args = ap.parse_args()

    if args.max_steps is not None:
        ro_mod.TRAN_TSTOP = args.max_steps * ro_mod.TRAN_TSTEP
    else:
        ro_mod.TRAN_TSTOP = args.tstop_ns * 1e-9
    settle = args.settle_ns * 1e-9
    tstop = ro_mod.TRAN_TSTOP
    tstep = ro_mod.TRAN_TSTEP
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inject_pmos: Optional[bool] = {
        "baseline": None, "id_q_np": True, "id_q_nmos_only": False,
    }[args.mode]

    print(f"[S6-P1] mode={args.mode}  TSTOP={tstop * 1e9:.3f} ns  "
          f"SETTLE={settle * 1e9:.3f} ns  tstep={tstep * 1e12:.0f} ps  "
          f"(start-state: S5b constrained-.ic uic-equivalent, both sides)")

    bt = BENCH["TSMC7"]
    mid = bt.vdd / 2.0
    cache = OsdiCache(bt)
    stats = SwapStats(args.spot_n)

    # FD convention check (fail loud) at a mid-swing bias before any run.
    if inject_pmos is not None:
        print(f"[S6-P1] FD convention check — NMOS @ "
              f"(vd={mid:.3f}, vg={mid:.3f}, vs=0, vb=0):")
        fd_check(cache.inst_for("nmos", bt.l_nmos), mid, mid, 0.0, 0.0)
        if inject_pmos:
            print(f"[S6-P1] FD convention check — PMOS @ "
                  f"(vd={mid:.3f}, vg={mid:.3f}, vs={bt.vdd:.3f}, "
                  f"vb={bt.vdd:.3f}):")
            fd_check(cache.inst_for("pmos", bt.l_pmos), mid, mid,
                     bt.vdd, bt.vdd)

    # NGSPICE ground truth (same window)
    ng_dir = Path(tempfile.mkdtemp(prefix="s6p1_ng_"))
    ng = run_ngspice_ro(bt, ng_dir)
    ng_per = _period_from_wave(ng["time"], ng["v(n5)"], mid, settle)
    print(f"[S6-P1] NGSPICE ground truth period = {ng_per * 1e12:.2f} ps")

    t0 = time.time()
    dn, partial, err = run_variant(bt, args.mode, inject_pmos, cache, stats)
    wall = time.time() - t0

    per = _period_from_wave(dn["time"], dn["v(n5)"], mid, settle)
    per_err = (abs(per - ng_per) / ng_per * 100.0
               if np.isfinite(per) and ng_per > 0 else float("nan"))
    t_lo, t_hi = settle, min(ng["time"][-1], dn["time"][-1])
    m = {"nrmse_pct": float("nan"), "r2": float("nan")}
    if t_hi > t_lo and len(dn["time"]) > 3:
        grid = np.arange(t_lo, t_hi, tstep)
        ng_i = np.interp(grid, ng["time"], ng["v(n5)"])
        dn_i = np.interp(grid, dn["time"], dn["v(n5)"])
        m = full_metrics(dn_i, ng_i)
    reached = dn["time"][-1] * 1e9 if len(dn["time"]) else float("nan")
    s_per_ns = wall / reached if reached and np.isfinite(reached) else float("nan")

    print(f"[S6-P1] {args.mode:14s} period = {per * 1e12:7.2f} ps  "
          f"err vs NG = {per_err:6.2f}%  "
          f"Δvs pre-S5b baseline {P0I_BASELINE_PS} = "
          f"{per * 1e12 - P0I_BASELINE_PS:+6.2f} ps  "
          f"NRMSE = {m['nrmse_pct']:6.2f}%  R2 = {m['r2']:8.4f}")
    print(f"        partial={partial} reached={reached:.4f}ns err={err!r}")
    print(f"        wall={wall:.0f}s  ({s_per_ns:.0f} s/ns; 0.6 ns full run "
          f"≈ {s_per_ns * 0.6 / 3600:.2f} h)  osdi_calls={cache.n_calls} "
          f"hits={cache.n_hits} osdi_fail={cache.n_fail}")
    if inject_pmos is not None:
        print(f"        NN-vs-OSDI at injected biases: {stats.summary()}")

    # Waveform diagnostics (P0-I §6 pattern): distinguish a genuine period
    # from a sub-harmonic artifact — all midpoint crossings + swing.
    t_a, v_a = np.asarray(dn["time"]), np.asarray(dn["v(n5)"])
    keep = t_a >= settle
    ta, va = t_a[keep], v_a[keep]
    if len(va) > 3:
        sgn = np.sign(va - mid)
        rises = np.where((sgn[:-1] < 0) & (sgn[1:] >= 0))[0]
        falls = np.where((sgn[:-1] > 0) & (sgn[1:] <= 0))[0]
        xt = np.sort(np.concatenate([ta[rises], ta[falls]]))
        half = np.diff(xt) * 1e12 if len(xt) > 1 else np.array([])
        print(f"        wave: vmin={va.min():.3f} vmax={va.max():.3f} "
              f"swing={va.max() - va.min():.3f}V  n_rise={len(rises)} "
              f"n_fall={len(falls)}  half-periods(ps)="
              f"{np.array2string(half[:12], precision=1, separator=',')}")

    wave_path = OUT_DIR / f"wave_{args.mode}_{tstop * 1e12:.0f}ps.npz"
    np.savez(wave_path, time=t_a, vn5=v_a,
             ng_time=ng["time"], ng_vn5=ng["v(n5)"])
    print(f"        saved waveform -> {wave_path}")

    verdict = verdict_line(args.mode, per * 1e12, ng_per * 1e12)
    print(f"[S6-P1] {verdict}")

    # Results table (one file per invocation; parallel-safe)
    md = OUT_DIR / f"s6_p1_{args.mode}_{tstop * 1e12:.0f}ps.md"
    md.write_text(
        f"# S6=P1 swap matrix — mode `{args.mode}` "
        f"(TSMC7 RO, window {tstop * 1e9:.3f} ns, settle {settle * 1e9:.3f} ns)\n\n"
        f"Start-state: S5b constrained-.ic (uic-equivalent), commit 7454034.\n\n"
        f"| mode | period (ps) | NG (ps) | err% | Δ vs pre-S5b 50.83 | NRMSE% | "
        f"R2 | partial | reached (ns) | wall (s) | osdi calls/hits/fail |\n"
        f"|------|------------:|--------:|-----:|-------------------:|-------:|"
        f"---:|:-------:|-------------:|---------:|---------------------:|\n"
        f"| {args.mode} | {per * 1e12:.2f} | {ng_per * 1e12:.2f} | "
        f"{per_err:.2f} | {per * 1e12 - P0I_BASELINE_PS:+.2f} | "
        f"{m['nrmse_pct']:.2f} | {m['r2']:.4f} | {partial} | {reached:.4f} | "
        f"{wall:.0f} | {cache.n_calls}/{cache.n_hits}/{cache.n_fail} |\n\n"
        f"NN-vs-OSDI at injected biases: {stats.summary() if inject_pmos is not None else 'n/a (baseline)'}\n\n"
        f"**{verdict}**\n")
    print(f"[S6-P1] results table -> {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
