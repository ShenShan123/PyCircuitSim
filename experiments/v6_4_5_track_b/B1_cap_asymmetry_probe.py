#!/usr/bin/env python3
"""B1 — Adaptive cap-symmetry probe on the TSMC7 ring oscillator (Track B, Tier 1).

Falsifier for the hypothesis "TSMC7 RO period drift is a Cgd/Cdg asymmetry
problem". Instruments every DirectNet NN evaluation during the TSMC7 RO
transient and records

    delta = |cgd - cdg| / max(|cgd|, |cdg|, 1e-15)

per eval. Promotion (plan B1): delta > 5% on > 10% of NR evals *at the trip
region* AND NN_SYMMETRIC_CAPS=1 closes TSMC7 RO <= 5%. Hard kill: delta
uniformly < 1% -> RO drift is not cap-asymmetry; the flag stays dormant and
the gate is model-fidelity (-> Tier-2 B5/B6 / Tier-3).

Pure diagnostic: monkeypatches ``_MOSFETNNBase._unpack_eval`` so NO source
file is touched (trivially reversible). Writes a JSON + markdown summary.

Run:
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        conda run -n pycircuitsim python experiments/v6_4_5_track_b/B1_cap_asymmetry_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT,
           PROJECT_ROOT / "external_compact_models",
           PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Import the RO harness and the base class, THEN patch (patching the class
# method affects all later-instantiated devices regardless of import order).
import tests.verify_complex_ring_osc as ro  # noqa: E402
from pycircuitsim.models.mosfet_nn import _MOSFETNNBase  # noqa: E402

# (vds, cgd, cdg, delta, cmax) per eval
CAP_LOG: List[Tuple[float, float, float, float, float]] = []
_orig_unpack = _MOSFETNNBase._unpack_eval


def _patched_unpack(self, out_row, gi, gqg, gqd, v_d_nn, v_s_nn):
    res = _orig_unpack(self, out_row, gi, gqg, gqd, v_d_nn, v_s_nn)
    cgd = float(res["cgd"])
    cdg = float(res["cdg"])
    cmax = max(abs(cgd), abs(cdg), 1e-15)
    delta = abs(cgd - cdg) / cmax
    CAP_LOG.append((float(v_d_nn - v_s_nn), cgd, cdg, delta, cmax))
    return res


_MOSFETNNBase._unpack_eval = _patched_unpack  # type: ignore[assignment]


def _summarize(arr: np.ndarray, name: str) -> Dict[str, float]:
    if arr.size == 0:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": int(arr.size),
        "frac_gt_5pct": float(np.mean(arr > 0.05)),
        "frac_gt_1pct": float(np.mean(arr > 0.01)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def main() -> int:
    bt = ro.BENCH["TSMC7"]
    work_dir = ro.RESULTS_BASE / "ring_osc" / bt.name
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"B1 cap-asymmetry probe — {bt.name} RO transient ...")
    dn, partial, err = ro.run_directnet_ro(bt, work_dir)
    print(f"  RO transient done (partial={partial}); {len(CAP_LOG)} NN evals logged")

    log = np.array(CAP_LOG, dtype=np.float64)
    vds = log[:, 0]
    delta = log[:, 3]
    cmax = log[:, 4]

    # "Trip region" proxy: the inverter Miller cap peaks where the stage is
    # mid-switch (both devices conducting). Take evals in the top quartile of
    # max(|cgd|,|cdg|) as the trip-region subset, plus a |Vds| in [0.1,0.6]
    # band cut as a cross-check (mid-rail drain bias).
    if cmax.size:
        cmax_q75 = np.percentile(cmax, 75)
        trip_mask = cmax >= cmax_q75
    else:
        trip_mask = np.zeros_like(delta, dtype=bool)
    vds_band = (np.abs(vds) >= 0.10) & (np.abs(vds) <= 0.60)

    summaries = [
        _summarize(delta, "all_evals"),
        _summarize(delta[trip_mask], "trip_region_top25pct_cmax"),
        _summarize(delta[vds_band], "vds_band_0.1_0.6"),
    ]

    # Promotion test (plan B1): delta>5% on >10% of trip-region evals.
    trip = summaries[1]
    cap_asym_signal = bool(trip.get("n", 0) > 0 and trip.get("frac_gt_5pct", 0.0) > 0.10)
    uniformly_tiny = bool(summaries[0].get("frac_gt_1pct", 1.0) < 0.01)

    verdict = (
        "CAP-ASYMMETRY SIGNAL (promote: test NN_SYMMETRIC_CAPS=1)"
        if cap_asym_signal else
        "KILL — delta uniformly tiny; RO drift is not cap-asymmetry"
        if uniformly_tiny else
        "KILL — cap-asymmetry below promotion threshold (<10% of trip evals >5%)"
    )

    out = {
        "tech": bt.name,
        "ro_period_err_pct": None,
        "n_evals": int(log.shape[0]),
        "summaries": summaries,
        "cap_asym_signal": cap_asym_signal,
        "uniformly_tiny": uniformly_tiny,
        "verdict": verdict,
    }
    res_dir = PROJECT_ROOT / "results" / "v6_4_5_track_b"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "B1_cap_asymmetry.json").write_text(json.dumps(out, indent=2))

    print("\n=== B1 cap-asymmetry summary (TSMC7 RO) ===")
    for s in summaries:
        if s.get("n", 0) == 0:
            print(f"  {s['name']:30s}: (no evals)")
            continue
        print(f"  {s['name']:30s}: n={s['n']:6d}  "
              f">5%={s['frac_gt_5pct']*100:5.1f}%  >1%={s['frac_gt_1pct']*100:5.1f}%  "
              f"med={s['median']*100:6.3f}%  p90={s['p90']*100:6.3f}%  "
              f"p99={s['p99']*100:6.3f}%  max={s['max']*100:6.2f}%")
    print(f"\n  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
