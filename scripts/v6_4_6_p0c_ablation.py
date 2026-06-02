#!/usr/bin/env python3
"""DirectNet V6.4.6 — Diagnostic P0-C: cap-swap / gds-swap RO ablation (TSMC7).

INSTRUMENTATION-ONLY. Confirms WHICH model surface the TSMC7 ring-oscillator
period actually depends on, by re-running the DN RO transient THREE ways and
measuring the period each time:

  (baseline)  unmodified NN                              (expect ~50.82 ps)
  (cap-swap)  replace ONLY the NN autograd caps (cgg,cgd,cgs,cdg,cdd) with
              analytic OSDI caps at the live bias; keep NN id/gds
  (gds-swap)  replace ONLY the NN gds with analytic OSDI gds; keep NN caps/id

Mechanism: monkeypatch ``_MOSFETNNBase._eval`` to call the original, then
substitute the chosen OSDI quantity(ies) at the device's CURRENT absolute
bias (reconstructed from ``self.nodes`` + ``self._is_pmos``). One PyCMG
instance is cached per (device_type, L). The original ``_eval`` is restored
between variants.

We force ``NN_BATCHED_EVAL=0`` for ALL three variants (incl. baseline) so the
per-device ``_eval`` path is the single source of truth the monkeypatch sees;
the baseline-under-this-flag period is validated to match the gate's 50.82 ps.

Convention (verified by P0-B / finite-difference vs the OSDI instance):
  gds   : NN result["gds"] = +d(id)/dVd, floored; OSDI gds = -d(id)/dVd, both
          positive -> inject OSDI gds, then re-apply the |id|*0.5 floor so the
          swap is gds-only (no floor-policy change).
  caps  : diagonals cgg/cdd same sign; off-diagonals cgd/cdg satisfy NN = -OSDI.
          Inject OSDI caps in NN convention (negate the off-diagonals). cgs is
          mapped d(qg)/dVs sign-consistently (NN = -OSDI for the s-derivative).

Ground truth is ALWAYS the OSDI binary via PyCMG (CLAUDE.md Validation rule).

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      conda run -n pycircuitsim python scripts/v6_4_6_p0c_ablation.py
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

# Stream progress even under `conda run` file redirection (which otherwise
# buffers stdout until process exit).
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

CAP_KEYS = ("cgg", "cgd", "cgs", "cdg", "cdd")
_ORIG_EVAL = _MOSFETNNBase._eval


class OsdiCache:
    """One PyCMG instance per (device_type, L), with a bias-rounded result
    cache.

    OSDI ``eval_single_point`` is ~1.6 ms/call (a full OSDI Newton solve), so
    calling it per device per NR iteration per timestep is the bottleneck. The
    RO trip revisits near-identical biases, so results are memoised on the bias
    rounded to ``BIAS_Q`` volts. The rounding error this injects is far below
    the bias scatter NR already produces and is identical across the cap-swap
    and gds-swap variants, so the period *comparison* is unaffected.
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
        # Evaluate at the rounded bias so the cached value is self-consistent
        # with its key (avoids cache aliasing of nearby raw biases).
        out = eval_single_point(inst, rk[2] * q, rk[3] * q, rk[4] * q,
                                rk[5] * q)
        self._cache[rk] = out
        return out


def _osdi_caps_nn_conv(o: Dict[str, float]) -> Dict[str, float]:
    """Map OSDI caps into the NN autograd convention.

    Diagonals (cgg, cdd) share sign; off-diagonals (cgd, cdg) and the
    source-derivative cgs satisfy NN = -OSDI (autograd +d(q)/dV vs the OSDI
    capacitance-matrix sign). Verified by finite-difference in P0-B.
    """
    return {
        "cgg": o["cgg"],
        "cdd": o["cdd"],
        "cgd": -o["cgd"],
        "cdg": -o["cdg"],
        "cgs": -o["cgs"],
    }


def make_patch(cache: OsdiCache, mode: str):
    """Return an _eval replacement that swaps `mode` in ('caps','gds')."""

    def patched(self: _MOSFETNNBase, voltages: Dict[str, float]):
        r = _ORIG_EVAL(self, voltages)
        # absolute terminal biases (solver frame)
        vd = voltages.get(self.nodes[0], 0.0)
        vg = voltages.get(self.nodes[1], 0.0)
        vs = voltages.get(self.nodes[2], 0.0)
        vb = voltages.get(self.nodes[3], 0.0)
        o = cache.eval(self._is_pmos, self.L, vd, vg, vs, vb)
        if o is None:
            return r  # OSDI failed at this bias -> keep NN value (rare)
        r = dict(r)  # don't mutate the cached dict in place
        if mode == "caps":
            r.update(_osdi_caps_nn_conv(o))
        elif mode == "gds":
            # OSDI gds is the true physical d-current slope (positive). Apply
            # the SAME |id|*0.5 floor the NN uses, so this is a gds-only swap
            # and not a floor-policy change.
            gds = max(o["gds"], max(abs(r["id"]) * 0.5, 1e-12))
            r["gds"] = gds
        elif mode == "gds_nofloor":
            # Disambiguator: inject raw OSDI gds WITHOUT the |id|*0.5 floor
            # (only the 1e-12 numerical floor). Tells us whether the gds floor
            # itself, rather than the trained slope, is what the period sees.
            r["gds"] = max(o["gds"], 1e-12)
        self._eval_cache = r
        self._cache_voltages = self._v_tuple(voltages)
        return r

    return patched


def run_variant(bt, mode: Optional[str], cache: OsdiCache):
    """Run one RO transient under the given swap mode; return waveform dict."""
    work_dir = Path(tempfile.mkdtemp(prefix=f"p0c_{mode or 'base'}_"))
    if mode is None:
        _MOSFETNNBase._eval = _ORIG_EVAL
    else:
        _MOSFETNNBase._eval = make_patch(cache, mode)
    try:
        dn, partial, err = run_directnet_ro(bt, work_dir)
    finally:
        _MOSFETNNBase._eval = _ORIG_EVAL
    return dn, partial, err


def main() -> int:
    bt = BENCH["TSMC7"]
    mid = bt.vdd / 2.0
    cache = OsdiCache(bt)

    # NGSPICE ground truth (period + waveform for NRMSE/R2)
    ng_dir = Path(tempfile.mkdtemp(prefix="p0c_ng_"))
    ng = run_ngspice_ro(bt, ng_dir)
    ng_per = _period_from_wave(ng["time"], ng["v(n5)"], mid, SETTLE)
    print(f"[P0-C] NGSPICE period = {ng_per*1e12:.2f} ps")

    rows = []
    variants = (("baseline", None), ("cap-swap", "caps"),
                ("gds-swap", "gds"), ("gds-swap-nofloor", "gds_nofloor"))
    for label, mode in variants:
        t0 = time.time()
        dn, partial, err = run_variant(bt, mode, cache)
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
        print(f"[P0-C] {label:16s} period = {per*1e12:7.2f} ps  "
              f"err = {per_err:6.2f}%  NRMSE = {m['nrmse_pct']:6.2f}%  "
              f"R2 = {m['r2']:8.4f}  partial={partial}  "
              f"wall={wall:.0f}s osdi_calls={cache.n_calls} hits={cache.n_hits}")
        rows.append((label, per * 1e12, per_err, m["nrmse_pct"], m["r2"]))

    print("\n" + "=" * 70)
    print("P0-C SUMMARY — TSMC7 RO period under cap-swap / gds-swap")
    print("=" * 70)
    print(f"  NGSPICE ground truth period = {ng_per*1e12:.2f} ps  (gate <=5%)")
    print(f"  {'variant':16s} | {'period(ps)':>11s} | {'perErr%':>8s} | "
          f"{'NRMSE%':>7s} | {'R2':>8s}")
    for label, per, perr, nr, r2 in rows:
        print(f"  {label:16s} | {per:11.2f} | {perr:8.2f} | {nr:7.2f} | {r2:8.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
