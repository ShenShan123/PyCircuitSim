#!/usr/bin/env python3
"""V7.2.0 Phase 0 diagnostic — §8 branch-discontinuity hit rate.

Two hard discontinuities sit exactly at the SRAM off-device regime
(plan §8): the ``_guard_gds`` sign boundary (1 ULP across zero moves gds
by ~5 orders of magnitude) and the wrong-sign clamp (exact 0.0). A
float32 reorder (batched GEMM row vs single-row eval, CPU vs GPU kernel)
can flip which side of the boundary an eval lands on — this script
counts how often evals land NEAR those boundaries during a write
transient, which is the number that sizes the perturbation risk for the
{2t, 3a, 3b} flag bundle.

Counters (per NN eval, i.e. per ``_unpack_eval`` result):
- total            all NN device evals that reach the Vds correction
- guardF_neg       raw autograd gds was negative -> guard F clamped it
- vds_small        |Vds| < 1 mV (the id~0, Vds~0 corner both
                   discontinuities live in)
- id_zeroed        Vds correction emitted an exactly-0.0 id from a
                   nonzero raw id (wrong-sign clamp or taper zero)
- id_sub_1ulp_gds  |id|*k within 2 ULP of the emitted gds (evals whose
                   guard-F output could change at the last bit)

Denominator note: ``total`` counts evals (one ``_apply_vds_correction``
per eval); ``_guard_gds`` fires once in ``_unpack_eval`` and possibly a
second time inside the correction, so ``guardF_neg`` / ``id_sub_1ulp_gds``
are per-CALL counts that upper-bound the per-eval rate.

First measurement (4x4 short write transient, 2 ns / 40 steps):
total 46,272; guardF_neg 37.4%; vds_small 5.5%; id_zeroed 13.3%;
id_sub_1ulp_gds 37.3%. Reading: the guard-F discontinuity is the BULK
regime of an SRAM array (most devices off, raw autograd gds noise
negative there) — not a rare corner. Consequence for §8.4: the T1
branch-disagreement count can never be required to be zero; the binding
tiers are T2 (solved nodes within NR tolerance) and T4 (basin
agreement), which measured <=60 uV / 0 flips for {2t} and {2t+gpu}.

Counts are EXACT regardless of box load. Run:
    conda run -n pycircuitsim python scripts/bench_gpu/diag_branch_hits.py \
        [netlist.sp]   (default: scripts/bench_gpu/sram_tran_4x4.sp)
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from pycircuitsim.models import mosfet_nn  # noqa: E402
from pycircuitsim.models.mosfet_nn import _MOSFETNNBase  # noqa: E402

COUNTS = {
    "total": 0,
    "guardF_neg": 0,
    "vds_small": 0,
    "id_zeroed": 0,
    "id_sub_1ulp_gds": 0,
}

_orig_guard = _MOSFETNNBase._guard_gds  # class access unwraps staticmethod
_orig_corr = _MOSFETNNBase._apply_vds_correction


def _counting_guard(id_phys: float, gds_phys: float) -> float:
    if gds_phys <= 0.0:
        COUNTS["guardF_neg"] += 1
    out = _orig_guard(id_phys, gds_phys)
    # emitted gds within 2 ULP of the |id|*k clamp value -> a last-bit
    # perturbation of id or gds can move the emitted value
    clamp = abs(id_phys) * mosfet_nn._GDS_GUARD_K
    if out > 0.0 and clamp > 0.0:
        if abs(out - clamp) <= 2.0 * math.ulp(max(out, clamp)):
            COUNTS["id_sub_1ulp_gds"] += 1
    return out


def _counting_corr(self, result, vds):
    COUNTS["total"] += 1
    if abs(vds) < 1e-3:
        COUNTS["vds_small"] += 1
    id_in = result["id"]
    out = _orig_corr(self, result, vds)
    if id_in != 0.0 and out["id"] == 0.0:
        COUNTS["id_zeroed"] += 1
    return out


_MOSFETNNBase._guard_gds = staticmethod(_counting_guard)
_MOSFETNNBase._apply_vds_correction = _counting_corr


def main() -> None:
    deck = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        PROJECT_ROOT / "scripts" / "bench_gpu" / "sram_tran_4x4.sp")
    from pycircuitsim.simulation import run_simulation
    with tempfile.TemporaryDirectory(prefix="branch_hits_") as td:
        run_simulation(str(deck), output_dir=td)
    t = max(COUNTS["total"], 1)
    print(f"\n=== §8 branch-discontinuity hit rate — {deck.name} ===")
    for k, v in COUNTS.items():
        print(f"  {k:16s} {v:10d}  ({100.0 * v / t:6.3f}% of evals)")
    print("Interpretation: guardF_neg + id_zeroed are evals sitting ON a")
    print("discontinuity; vds_small is the corner both live in. A float32")
    print("reorder can flip only those — this bounds the §8.4 T1")
    print("branch-disagreement count from above.")


if __name__ == "__main__":
    main()
