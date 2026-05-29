#!/usr/bin/env python3
"""B2 — Test-time seed ensemble + rail voting for SRAM force_ic (Track B, Tier 1).

Falsifier for "an alternative seed lands SRAM force_ic on the rails". V6.4.4's
canonical TSMC7 checkpoint (seed-42 V6.4.1) settles the cross-coupled 6T cell
at the interior attractor q ≈ 0.82, qb ≈ 0.23 (rail residual r ≈ 0.7). This
sweeps the 8 TSMC7 seed siblings on disk
(``v6_4_2_p7_tsmc7_{stock,mono}_s{7,17,42,123}``) plus the canonical slot,
running each through the force_ic state-1 probe in an isolated
``BSIMAR_CHECKPOINT_DIR`` (reuses ``scripts/eval_v6_4_5_candidate.py``).

Promotion (plan B2): ≥ 1 seed snaps SRAM to rails (r < 0.05). Hard kill: no
seed snaps any SRAM cell → the q ≈ 0.18/0.82 attractor is intrinsic to the NN
family, not a single bad seed. Pure inference, no retrain.

Run:
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        conda run -n pycircuitsim python experiments/v6_4_5_track_b/B2_seed_ensemble.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL = PROJECT_ROOT / "scripts" / "eval_v6_4_5_candidate.py"
RES_DIR = PROJECT_ROOT / "results" / "v6_4_5_track_b"

TECH = "TSMC7"
RAIL_SNAP_GATE = 0.05            # r < 0.05 == rail-snapped

# 8 seed siblings (matched N&P per config) + canonical baseline.
CANDIDATES: List[Tuple[str, str, str]] = [
    ("canonical_seed42", "tsmc7_dn_medium_nmos", "tsmc7_dn_medium_pmos"),
]
for _recipe in ("stock", "mono"):
    for _seed in (7, 17, 42, 123):
        CANDIDATES.append((
            f"{_recipe}_s{_seed}",
            f"v6_4_2_p7_tsmc7_{_recipe}_s{_seed}_nmos",
            f"v6_4_2_p7_tsmc7_{_recipe}_s{_seed}_pmos",
        ))


def _eval_one(label: str, n_stem: str, p_stem: str) -> Optional[Dict]:
    cmd = [
        "conda", "run", "--no-capture-output", "-n", "pycircuitsim",
        "python", str(EVAL), "--tech", TECH,
        "--nmos", n_stem, "--pmos", p_stem, "--json",
        "--skip", "inv,ro,opamp",            # SRAM rail residual only — fast
    ]
    env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "CUDA_VISIBLE_DEVICES": "", "PATH": os.environ["PATH"]}
    out = subprocess.run(cmd, capture_output=True, text=True,
                         cwd=str(PROJECT_ROOT), env=env)
    res = None
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[len("RESULT "):])
    if res is None:
        sys.stderr.write(f"[B2 eval FAILED] {label}\n{out.stdout[-1200:]}\n{out.stderr[-1200:]}\n")
        return None
    res["label"] = label
    return res


def main() -> int:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"B2 seed-ensemble — {TECH} SRAM force_ic state-1 rail residual "
          f"over {len(CANDIDATES)} configs ...")
    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_eval_one, lab, n, p): lab
                for (lab, n, p) in CANDIDATES}
        for fut in futs:
            r = fut.result()
            if r is not None:
                results.append(r)

    # Order by rail residual (best first).
    results.sort(key=lambda r: r.get("sram_rail_snap_resid", float("inf")))
    snapped = [r for r in results
               if r.get("sram_rail_snap_resid", 1.0) < RAIL_SNAP_GATE]
    best = results[0] if results else None
    verdict = (
        f"PROMOTE — {len(snapped)} seed(s) rail-snap (r<{RAIL_SNAP_GATE})"
        if snapped else
        "KILL — no seed snaps SRAM to rails; q≈0.18/0.82 attractor is intrinsic"
    )

    out = {"tech": TECH, "gate": RAIL_SNAP_GATE,
           "results": results, "n_snapped": len(snapped), "verdict": verdict}
    (RES_DIR / "B2_seed_ensemble.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== B2 — {TECH} SRAM force_ic state-1 rail residual ===")
    print(f"{'config':20s} | {'rail_resid':>10s} | {'q':>7s} | {'qb':>7s}")
    print("-" * 52)
    for r in results:
        print(f"{r['label']:20s} | "
              f"{r.get('sram_rail_snap_resid', float('nan')):10.4f} | "
              f"{r.get('sram_q', float('nan')):7.3f} | "
              f"{r.get('sram_qb', float('nan')):7.3f}")
    print(f"\n  best resid = {best.get('sram_rail_snap_resid', float('nan')):.4f}"
          f" ({best['label']})" if best else "  no results")
    print(f"  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
