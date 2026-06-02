#!/usr/bin/env python3
"""DirectNet V6.4.6 — Diagnostic P0-I: causal OSDI-`id`-VALUE injection (TSMC7 RO).

INSTRUMENTATION-ONLY. The decisive 0-GPU CAUSAL test for the open TSMC7
ring-oscillator gate — the analogue of P0-C, but for the `id` VALUE instead of
the gds/cap Jacobian surfaces.

Background (read first):
  - P0-C (`phase0_C_cap_swap_ablation.md`): causally swapped the NN autograd
    gds and caps into the live TSMC7 RO -> period moved <=0.01 ps. Those are
    Jacobian-only surfaces; they enter the NR matrix AND a matching RHS offset
    and CANCEL at the converged fixed point -> causally INERT on the period.
  - P0-G (`phase0G_ro_integration_study.md`): drove BE/Trap/BDF-2 truncation to
    zero (tstep->0). Both consistent integrators converge to a common ~50.4 ps
    continuum limit, ~3.7 ps above NG 46.64 ps. Integration owns only ~0.4 ps;
    the remaining ~3.7 ps is MODEL-owned.
  - P0-H (`phase0H_ro_value_overlay.md`): correlational VALUE overlay. The
    charge VALUES (qg/qd/qs) are EXACT (<=1.2% NRMSE); the `id` VALUE carries the
    residual (NMOS on-state NRMSE 9.6%, ~20% under-prediction of the dynamic
    peak pull-down current, direction-consistent with the longer DN period).

WHY THIS DIFFERS FROM P0-C (crucial mechanism note):
  The gds/cap swaps were inert because the Jacobian + its RHS offset cancel at
  the converged solution. The `id` VALUE does NOT cancel: the transient stamps
  the resistive companion current directly from the `id` VALUE
  (`_stamp_mosfet_dc:304`, ``i_eq = i_leaving - g_ds*v_ds - g_m*v_gs - ...``).
  Changing the `id` VALUE changes the converged solution and the WHOLE
  trajectory. So we cannot pre-tabulate OSDI `id` against the baseline
  trajectory — we must evaluate OSDI `id` AT THE EVOLVING BIAS, live, inside the
  transient/NR loop (the trajectory moves as we inject).

MECHANISM (same in-process monkeypatch as P0-C — no shipped file modified, no
retrain, no checkpoint mutation; ``git diff`` over ``pycircuitsim/`` stays
empty):
  Monkeypatch ``_MOSFETNNBase._eval`` so for the selected RO devices it returns
  the analytic OSDI ``id`` (from PyCMG ``eval_single_point`` at the CURRENT
  absolute Vd/Vg/Vs/Vb + geometry — the P0-H eval path) instead of the NN ``id``,
  while keeping the NN charges (qg/qd/qs/qb) and caps. The NN autograd gds would
  then be inconsistent with the injected OSDI id; P0-C proved a gds-swap is inert
  on the period, so to keep NR healthy AND isolate the id-VALUE effect cleanly we
  ALSO inject the OSDI gds alongside the OSDI id (a consistent (id, gds) pair,
  with the same |id|*0.5 floor the NN uses). Since gds-swap alone moved the period
  <=0.01 ps (P0-C), ANY period change here is attributable to the id VALUE.

Variants (period re-measured each, %err vs NG 46.64 ps):
  1. baseline        — hook OFF (must reproduce 50.82 ps)
  2. id-inject NMOS  — inject OSDI (id, gds) on the NMOS pull-down only (primary)
  3. id-inject N+P   — inject OSDI (id, gds) on both NMOS and PMOS

Ground truth is ALWAYS the OSDI binary via PyCMG (CLAUDE.md Validation rule).

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      conda run -n pycircuitsim python scripts/v6_4_6_p0i_id_injection.py
"""
from __future__ import annotations

import functools
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

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

from tests.common.complex import BENCH  # noqa: E402
from tests.verify_complex_ring_osc import (  # noqa: E402
    run_directnet_ro, run_ngspice_ro, _period_from_wave, SETTLE, TRAN_TSTEP,
)
from tests.common.complex import full_metrics  # noqa: E402
from pycircuitsim.models.mosfet_nn import _MOSFETNNBase  # noqa: E402
from pycmg.nn_generate import _create_model_and_instance, eval_single_point  # noqa: E402
from pycmg.nn_config import TECH_CONFIGS  # noqa: E402

_ORIG_EVAL = _MOSFETNNBase._eval


class OsdiCache:
    """One PyCMG instance per (device_type, L), with a bias-rounded result
    cache (identical strategy to P0-C).

    OSDI ``eval_single_point`` is ~1.6 ms/call. The live RO trip revisits
    near-identical biases, so results are memoised on the bias rounded to
    ``BIAS_Q`` volts; the rounding is identical across variants so it cannot
    bias the *comparison*. Note that with LIVE injection the trajectory moves,
    so the set of biases visited differs per variant — that is intended (the
    injected id changes the solution); the cache only deduplicates repeated
    biases WITHIN a run.
    """

    BIAS_Q = 1e-3  # 1 mV bias-cache granularity

    def __init__(self, bt) -> None:
        self.bt = bt
        self.tech = TECH_CONFIGS[bt.nn_tech]
        self._inst: Dict[Tuple[str, float], object] = {}
        self._cache: Dict[Tuple, Optional[Dict[str, float]]] = {}
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

    def eval(self, is_pmos: bool, L: float, vd: float, vg: float,
             vs: float, vb: float) -> Optional[Dict[str, float]]:
        dt = "pmos" if is_pmos else "nmos"
        q = self.BIAS_Q
        rk = (dt, round(L, 12), round(vd / q), round(vg / q),
              round(vs / q), round(vb / q))
        if rk in self._cache:
            self.n_hits += 1
            return self._cache[rk]
        self.n_calls += 1
        inst = self._inst_for(dt, L)
        out = eval_single_point(inst, rk[2] * q, rk[3] * q, rk[4] * q,
                                rk[5] * q)
        self._cache[rk] = out
        return out


def make_patch(cache: OsdiCache, inject_pmos: bool):
    """Return an ``_eval`` replacement that injects OSDI (id, gds) on the
    NMOS devices, and on the PMOS devices too iff ``inject_pmos``.

    Charges (qg/qd/qs/qb), caps, gm, gmb are kept from the NN. The OSDI gds is
    injected alongside the OSDI id so the NR Jacobian stays CONSISTENT with the
    injected current (P0-C proved the gds swap alone is period-inert, so any
    period change here is owned by the id VALUE, not the gds).
    """

    def patched(self: _MOSFETNNBase, voltages: Dict[str, float]):
        r = _ORIG_EVAL(self, voltages)
        if self._is_pmos and not inject_pmos:
            return r  # NMOS-only variant: leave PMOS on the NN id
        # absolute terminal biases (solver frame), exactly as P0-C/P0-H.
        vd = voltages.get(self.nodes[0], 0.0)
        vg = voltages.get(self.nodes[1], 0.0)
        vs = voltages.get(self.nodes[2], 0.0)
        vb = voltages.get(self.nodes[3], 0.0)
        o = cache.eval(self._is_pmos, self.L, vd, vg, vs, vb)
        if o is None:
            return r  # OSDI failed at this bias -> keep NN value (rare)
        r = dict(r)  # don't mutate the cached dict in place
        # id VALUE: trained directly on OSDI col 0 (NMOS<0, PMOS>0 conducting),
        # so OSDI id is already in the same convention the stamp consumes via
        # calculate_current() (-id for NMOS / +id for PMOS).
        r["id"] = o["id"]
        # gds: OSDI gds is the true physical slope (positive); apply the SAME
        # |id|*0.5 floor the NN uses so the Jacobian is consistent with the
        # injected id without a floor-policy change.
        r["gds"] = max(o["gds"], max(abs(o["id"]) * 0.5, 1e-12))
        self._eval_cache = r
        self._cache_voltages = self._v_tuple(voltages)
        return r

    return patched


def run_variant(bt, label: str, mode: Optional[str], cache: OsdiCache):
    """Run one RO transient under the given injection mode; return waveform."""
    work_dir = Path(tempfile.mkdtemp(prefix=f"p0i_{label}_"))
    if mode is None:
        _MOSFETNNBase._eval = _ORIG_EVAL
    else:
        _MOSFETNNBase._eval = make_patch(cache, inject_pmos=(mode == "np"))
    try:
        dn, partial, err = run_directnet_ro(bt, work_dir)
    finally:
        _MOSFETNNBase._eval = _ORIG_EVAL  # ALWAYS restore (revert discipline)
    return dn, partial, err


def main() -> int:
    bt = BENCH["TSMC7"]
    mid = bt.vdd / 2.0
    cache = OsdiCache(bt)

    # NGSPICE ground truth (period + waveform for NRMSE/R2)
    ng_dir = Path(tempfile.mkdtemp(prefix="p0i_ng_"))
    ng = run_ngspice_ro(bt, ng_dir)
    ng_per = _period_from_wave(ng["time"], ng["v(n5)"], mid, SETTLE)
    print(f"[P0-I] NGSPICE ground truth period = {ng_per*1e12:.2f} ps  "
          f"(gate <=5% -> DirectNet must reach <={ng_per*1e12*1.05:.2f} ps)")

    rows = []
    # (label, mode):  mode None=baseline, 'nmos'=NMOS-only, 'np'=NMOS+PMOS
    variants = (("baseline", None),
                ("id-inject-NMOS", "nmos"),
                ("id-inject-N+P", "np"))
    for label, mode in variants:
        t0 = time.time()
        dn, partial, err = run_variant(bt, label, mode, cache)
        wall = time.time() - t0
        per = _period_from_wave(dn["time"], dn["v(n5)"], mid, SETTLE)
        per_err = abs(per - ng_per) / ng_per * 100.0 if ng_per > 0 else float("nan")
        # waveform metrics vs NG on the common post-settle grid
        t_lo, t_hi = SETTLE, min(ng["time"][-1], dn["time"][-1])
        m = {"nrmse_pct": float("nan"), "r2": float("nan")}
        if t_hi > t_lo and len(dn["time"]) > 3:
            grid = np.arange(t_lo, t_hi, TRAN_TSTEP)
            ng_i = np.interp(grid, ng["time"], ng["v(n5)"])
            dn_i = np.interp(grid, dn["time"], dn["v(n5)"])
            m = full_metrics(dn_i, ng_i)
        delta = per * 1e12 - 50.82  # ps moved vs the baseline RO period
        print(f"[P0-I] {label:16s} period = {per*1e12:7.2f} ps  "
              f"err = {per_err:6.2f}%  Δvs50.82 = {delta:+6.2f} ps  "
              f"NRMSE = {m['nrmse_pct']:6.2f}%  R2 = {m['r2']:8.4f}  "
              f"partial={partial} err={err!r}  wall={wall:.0f}s "
              f"osdi_calls={cache.n_calls} hits={cache.n_hits}")
        rows.append((label, per * 1e12, per_err, delta,
                     m["nrmse_pct"], m["r2"]))

    print("\n" + "=" * 78)
    print("P0-I SUMMARY — TSMC7 RO period under causal OSDI-`id`-VALUE injection")
    print("=" * 78)
    print(f"  NGSPICE ground truth period = {ng_per*1e12:.2f} ps  "
          f"(gate <=5% -> <={ng_per*1e12*1.05:.2f} ps)")
    print(f"  {'variant':16s} | {'period(ps)':>11s} | {'perErr%':>8s} | "
          f"{'Δvs50.82':>9s} | {'NRMSE%':>7s} | {'R2':>8s}")
    for label, per, perr, delta, nr, r2 in rows:
        print(f"  {label:16s} | {per:11.2f} | {perr:8.2f} | {delta:+9.2f} | "
              f"{nr:7.2f} | {r2:8.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
