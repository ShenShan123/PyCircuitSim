#!/usr/bin/env python3
"""DirectNet V6.4.6 — Diagnostic P0-I (v2): consistent causal OSDI-`id`-VALUE
injection into the live TSMC7 ring oscillator.

INSTRUMENTATION-ONLY. The decisive 0-GPU CAUSAL test for the open TSMC7
ring-oscillator gate — the analogue of P0-C, but for the `id` VALUE instead of
the gds/cap Jacobian surfaces.

WHY v2 (the v1 divergence was an instrumentation artifact, not a verdict):
  v1 (`scripts/v6_4_6_p0i_id_injection.py`) injected the OSDI `id` VALUE and the
  OSDI `gds`, but KEPT the NN's autograd `gm`/`gmb`. That left the NR Jacobian's
  gate-slope (gm) inconsistent with the injected current, AND the OSDI `id` was
  served piecewise-CONSTANT from a 1 mV bias cache while the Jacobian claimed a
  finite slope. NR limit-cycled below the cache cell width (stuck max-delta
  0.137 mV < 1 mV) and failed to converge at the first switching edge
  (t=2.6e-11 s NMOS-only / 6.0e-11 s N+P). Neither variant produced a period.
  See `results/v6_4_6/phase0_logs/p0i_id_injection.log`.

v2 fixes both, WITHOUT changing the causal question:
  At each device eval we inject the EXACT OSDI operating point evaluated at the
  live bias — id from OSDI col 0, plus OSDI's own analytic gm/gds/gmb mapped into
  the NN sign convention (probed against finite differences of the OSDI `id`):
      OSDI gm  = -∂id/∂Vg  == NN gm   (use directly)
      OSDI gmb = -∂id/∂Vb  == NN gmb  (use directly; Vbs≡0 in the RO ⇒ inert)
      OSDI gds = -∂id/∂Vd  → NN stamps floor(+∂id/∂Vd)=floor(-OSDI_gds)=|id|·0.5
  This is a C∞-smooth (id, gm, gds, gmb) — exactly what the normal LEVEL=72 path
  consumes — so circuit-level Newton converges quadratically.
  Two table-based predecessors (per-cell tangent plane, then bilinear) both
  CONVERGED but were ~35× too slow: a piecewise-constant table Jacobian makes
  Newton converge only linearly and the LTE controller sub-steps catastrophically
  (~4.2 ks for ~0.2 ns). The exact-bias analytic op-point removes the staircase.
  In the RO, Vs and Vb are rail-pinned (NMOS s=b=gnd, PMOS s=b=vdd); charges
  (qg/qd/qs/qb) and caps stay NN — P0-H proved the charge VALUES are exact.

WHY THIS STILL ISOLATES THE id VALUE (the P0-C cancellation argument):
  gm/gds/gmb are Jacobian-only: they enter the NR matrix AND a matching RHS
  offset (`_stamp_mosfet_dc:304`, `i_eq = i_leaving − g_ds·v_ds − g_m·v_gs −
  g_mb·v_bs`) and CANCEL at the converged fixed point. P0-C proved (causally)
  that swapping the exact OSDI gds/caps moves the period ≤0.01 ps. Injecting a
  *consistent* OSDI gm/gds/gmb here therefore does NOT change the converged
  trajectory's currents — it only steers Newton to the id-determined fixed
  point. So ANY period change measured here is owned by the injected `id` VALUE.
  Charges (qg/qd/qs/qb) and caps stay NN — P0-H already proved the charge VALUES
  are exact (≤2 aC), so they need no injection.

Variants (period re-measured each, %err vs NG 46.64 ps; baseline 50.82 ps):
  1. baseline        — hook OFF (must reproduce 50.82 ps; faithfulness check)
  2. id-inject NMOS  — consistent OSDI op-point on the NMOS pull-down only (primary)
  3. id-inject N+P   — consistent OSDI op-point on both NMOS and PMOS

Ground truth is ALWAYS the OSDI binary via PyCMG / NGSPICE (CLAUDE.md Validation).

Env knobs (for the cheap convergence smoke test before the full ~1.2 ns run):
  P0I_TSTOP_NS   override TRAN_TSTOP in ns (e.g. 0.15 for a smoke test)
  P0I_SETTLE_NS  override SETTLE in ns (default 0.3; lower it for short runs)
  P0I_VARIANTS   comma list of {baseline,nmos,np} (default all three)

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      conda run -n pycircuitsim python scripts/v6_4_6_p0i_id_injection_v2.py
"""
from __future__ import annotations

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
# solver import so the module-level flag read picks it up.
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


def _floor_gds(id_phys: float, gds_phys: float) -> float:
    """Mirror `_MOSFETNNBase._floor_gds` (Rule 5): max(|id|·0.5, 1e-12)."""
    return max(gds_phys, max(abs(id_phys) * 0.5, 1e-12))


class OsdiCache:
    """One PyCMG instance per (device_type, L), with a bias-rounded `id`
    cache. We only ever read the OSDI `id` field and finite-difference it on
    the grid — the FD neighbours are themselves grid-aligned, so they share the
    cache. OSDI `eval_single_point` is ~1.6 ms/call; the live RO trip revisits
    near-identical biases, so memoising on the grid keeps the run fast.
    """

    def __init__(self, bt) -> None:
        self.bt = bt
        self.tech = TECH_CONFIGS[bt.nn_tech]
        self._inst: Dict[Tuple[str, float], object] = {}
        self._cache: Dict[Tuple, Optional[Tuple[float, float, float, float]]] = {}
        self.n_calls = 0
        self.n_hits = 0

    def _inst_for(self, device_type: str, L: float):
        key = (device_type, round(L, 12))
        if key not in self._inst:
            _, inst, _ = _create_model_and_instance(
                self.tech, device_type, self.bt.vt, L, float(self.bt.nfin),
                300.15)
            self._inst[key] = inst
        return self._inst[key]

    def consistent_op(self, is_pmos: bool, L: float, vd: float, vg: float,
                      vs: float, vb: float
                      ) -> Optional[Tuple[float, float, float, float]]:
        """Return the EXACT OSDI operating point (id, gm, gds, gmb) at the live
        bias, mapped into the NN's sign convention. None if the OSDI eval fails.

        Why exact-bias analytic, not a grid table: a finite-difference / bilinear
        table device has a piecewise-constant Jacobian, so circuit-level Newton
        converges only LINEARLY and the LTE controller sub-steps catastrophically
        (the per-cell and bilinear 0.5 mV variants both took ~4.2 ks for ~0.2 ns,
        ~35x the baseline per-ns cost). Evaluating the OSDI binary directly at the
        live bias gives a C∞-smooth (id, gm, gds, gmb) — exactly what the normal
        LEVEL=72 path uses — so Newton converges quadratically and the run is fast.

        Sign mapping (probed against finite differences of the OSDI `id`):
          OSDI gm  = -∂id/∂Vg  == NN-convention gm   (use directly)
          OSDI gmb = -∂id/∂Vb  == NN-convention gmb  (use directly; Vbs≡0 ⇒ inert)
          OSDI gds = -∂id/∂Vd  (positive output conductance). The NN frame stamps
            gds = floor(+∂id/∂Vd) = floor(-OSDI_gds), i.e. the |id|·0.5 floor
            (P0-C already proved the gds swap is period-inert), so we reproduce
            the NN's own floored gds for an apples-to-apples Jacobian.
        """
        dt = "pmos" if is_pmos else "nmos"
        # Memo only on a TRULY-identical re-query (bias rounded to 1e-10 V — six
        # orders below the NR tolerance, so it cannot staircase the surface).
        # A coarser grid would hold id piecewise-constant while gm stays nonzero,
        # re-creating the v1 sub-cell limit cycle near convergence.
        rk = (dt, round(L, 12), round(vd, 10), round(vg, 10),
              round(vs, 10), round(vb, 10))
        cached = self._cache.get(rk)
        if cached is not None:
            self.n_hits += 1
            return cached
        self.n_calls += 1
        inst = self._inst_for(dt, L)
        o = eval_single_point(inst, vd, vg, vs, vb)
        if o is None:
            self._cache[rk] = None
            return None
        id_phys = float(o["id"])
        gm = float(o["gm"])                  # OSDI gm == NN gm  (-∂id/∂Vg)
        gmb = float(o["gmb"])                # OSDI gmb == NN gmb (-∂id/∂Vb); inert
        gds = _floor_gds(id_phys, -float(o["gds"]))  # NN frame: floor(+∂id/∂Vd)
        op = (id_phys, gm, gds, gmb)
        self._cache[rk] = op
        return op


def make_patch(cache: OsdiCache, inject_pmos: bool):
    """`_eval` replacement: inject a consistent OSDI (id, gm, gds, gmb) on the
    NMOS devices (and PMOS iff `inject_pmos`), keeping NN charges/caps."""

    def patched(self: _MOSFETNNBase, voltages: Dict[str, float]):
        r = _ORIG_EVAL(self, voltages)            # NN result (for charges/caps)
        if self._is_pmos and not inject_pmos:
            return r                              # NMOS-only: leave PMOS on NN
        vd = voltages.get(self.nodes[0], 0.0)
        vg = voltages.get(self.nodes[1], 0.0)
        vs = voltages.get(self.nodes[2], 0.0)
        vb = voltages.get(self.nodes[3], 0.0)
        op = cache.consistent_op(self._is_pmos, self.L, vd, vg, vs, vb)
        if op is None:
            return r                              # OSDI failed → keep NN (rare)
        id_lin, gm, gds, gmb = op
        r = dict(r)                               # don't mutate the cached dict
        r["id"] = id_lin
        r["gm"] = gm
        r["gds"] = gds
        r["gmb"] = gmb
        self._eval_cache = r
        self._cache_voltages = self._v_tuple(voltages)
        return r

    return patched


def run_variant(bt, label: str, mode: Optional[str], cache: OsdiCache):
    """Run one RO transient under the given injection mode; return waveform."""
    work_dir = Path(tempfile.mkdtemp(prefix=f"p0iv2_{label}_"))
    if mode is None:
        _MOSFETNNBase._eval = _ORIG_EVAL
    else:
        _MOSFETNNBase._eval = make_patch(cache, inject_pmos=(mode == "np"))
    try:
        dn, partial, err = run_directnet_ro(bt, work_dir)
    finally:
        _MOSFETNNBase._eval = _ORIG_EVAL          # ALWAYS restore (revert discipline)
    return dn, partial, err


def main() -> int:
    # ── env knobs ────────────────────────────────────────────────────────
    if os.environ.get("P0I_TSTOP_NS"):
        ro_mod.TRAN_TSTOP = float(os.environ["P0I_TSTOP_NS"]) * 1e-9
    if os.environ.get("P0I_SETTLE_NS"):
        ro_mod.SETTLE = float(os.environ["P0I_SETTLE_NS"]) * 1e-9
    want = os.environ.get("P0I_VARIANTS", "baseline,nmos,np").split(",")
    want = [w.strip() for w in want if w.strip()]

    tstop = ro_mod.TRAN_TSTOP
    settle = ro_mod.SETTLE
    tstep = ro_mod.TRAN_TSTEP
    print(f"[P0-I v2] TRAN_TSTOP={tstop*1e9:.3f} ns  SETTLE={settle*1e9:.3f} ns  "
          f"variants={want}")

    bt = BENCH["TSMC7"]
    mid = bt.vdd / 2.0
    cache = OsdiCache(bt)

    # NGSPICE ground truth (period + waveform for NRMSE/R2)
    ng_dir = Path(tempfile.mkdtemp(prefix="p0iv2_ng_"))
    ng = run_ngspice_ro(bt, ng_dir)
    ng_per = _period_from_wave(ng["time"], ng["v(n5)"], mid, settle)
    print(f"[P0-I v2] NGSPICE ground truth period = {ng_per*1e12:.2f} ps  "
          f"(gate <=5% -> DirectNet must reach <={ng_per*1e12*1.05:.2f} ps)")

    all_variants = (("baseline", None),
                    ("id-inject-NMOS", "nmos"),
                    ("id-inject-N+P", "np"))
    sel = {"baseline": "baseline", "nmos": "id-inject-NMOS",
           "np": "id-inject-N+P"}
    run_labels = {sel[w] for w in want if w in sel}

    rows: List[Tuple] = []
    for label, mode in all_variants:
        if label not in run_labels:
            continue
        t0 = time.time()
        dn, partial, err = run_variant(bt, label, mode, cache)
        wall = time.time() - t0
        per = _period_from_wave(dn["time"], dn["v(n5)"], mid, settle)
        per_err = (abs(per - ng_per) / ng_per * 100.0
                   if ng_per > 0 and np.isfinite(per) else float("nan"))
        t_lo, t_hi = settle, min(ng["time"][-1], dn["time"][-1])
        m = {"nrmse_pct": float("nan"), "r2": float("nan")}
        if t_hi > t_lo and len(dn["time"]) > 3:
            grid = np.arange(t_lo, t_hi, tstep)
            ng_i = np.interp(grid, ng["time"], ng["v(n5)"])
            dn_i = np.interp(grid, dn["time"], dn["v(n5)"])
            m = full_metrics(dn_i, ng_i)
        delta = per * 1e12 - 50.82          # ps moved vs the baseline RO period
        reached = dn["time"][-1] * 1e9 if len(dn["time"]) else float("nan")
        print(f"[P0-I v2] {label:16s} period = {per*1e12:7.2f} ps  "
              f"err = {per_err:6.2f}%  Δvs50.82 = {delta:+6.2f} ps  "
              f"NRMSE = {m['nrmse_pct']:6.2f}%  R2 = {m['r2']:8.4f}  "
              f"partial={partial} reached={reached:.3f}ns err={err!r}  "
              f"wall={wall:.0f}s osdi_calls={cache.n_calls} hits={cache.n_hits}")
        # Waveform diagnostics: distinguish a genuine ~92 ps oscillation from a
        # period-doubled / sub-harmonic artifact (rising-crossings 2x apart while
        # the fast cycle is ~46 ps). Print ALL midpoint crossings + swing.
        t_a, v_a = dn["time"], dn["v(n5)"]
        keep = t_a >= settle
        ta, va = t_a[keep], v_a[keep]
        if len(va) > 3:
            sgn = np.sign(va - mid)
            rises = np.where((sgn[:-1] < 0) & (sgn[1:] >= 0))[0]
            falls = np.where((sgn[:-1] > 0) & (sgn[1:] <= 0))[0]
            xt = np.sort(np.concatenate([ta[rises], ta[falls]]))
            half = np.diff(xt) * 1e12 if len(xt) > 1 else np.array([])
            print(f"           wave: vmin={va.min():.3f} vmax={va.max():.3f} "
                  f"swing={va.max()-va.min():.3f}V  n_rise={len(rises)} "
                  f"n_fall={len(falls)}  half-periods(ps)="
                  f"{np.array2string(half[:12], precision=1, separator=',')}")
        save_dir = os.environ.get("P0I_SAVE_DIR")
        if save_dir:
            sd = Path(save_dir); sd.mkdir(parents=True, exist_ok=True)
            np.savez(sd / f"wave_{label}.npz", time=t_a, vn5=v_a,
                     ng_time=ng["time"], ng_vn5=ng["v(n5)"])
            print(f"           saved waveform -> {sd / f'wave_{label}.npz'}")
        rows.append((label, per * 1e12, per_err, delta,
                     m["nrmse_pct"], m["r2"], partial, reached))

    print("\n" + "=" * 78)
    print("P0-I v2 SUMMARY — TSMC7 RO period under consistent OSDI-`id`-VALUE injection")
    print("=" * 78)
    print(f"  NGSPICE ground truth period = {ng_per*1e12:.2f} ps  "
          f"(gate <=5% -> <={ng_per*1e12*1.05:.2f} ps);  baseline 50.82 ps")
    print(f"  {'variant':16s} | {'period(ps)':>11s} | {'perErr%':>8s} | "
          f"{'Δvs50.82':>9s} | {'NRMSE%':>7s} | {'R2':>8s} | {'partial':>7s} | "
          f"{'reached':>8s}")
    for label, per, perr, delta, nr, r2, partial, reached in rows:
        print(f"  {label:16s} | {per:11.2f} | {perr:8.2f} | {delta:+9.2f} | "
              f"{nr:7.2f} | {r2:8.4f} | {str(partial):>7s} | {reached:7.3f}ns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
