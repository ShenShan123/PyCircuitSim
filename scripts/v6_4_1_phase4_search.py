"""V6.4.1 Phase-4 best-of-N greedy pair search for ONE tech.

Greedy protocol (minimises inverter sims vs brute-force 64):
  1. Fix pmos at seed 42, sweep all 8 nmos seeds  -> rank by VTC MaxErr.
  2. Fix nmos at its best, sweep all 8 pmos seeds.
  3. Joint-refine: top-3 nmos x top-3 pmos.
Select the pair minimising inverter VTC MaxErr subject to transient
post-startup MaxErr <= the V6.4.1 baseline for that tech.

All evals go through scripts/eval_v6_4_1_pair.py (swap-eval-restore).
Results append to logs/v6_4_1_phase4/search_<tech>.jsonl.

The V6.4.1 baseline numbers below are measured with the SAME harness
(eval_v6_3_1_inverter.evaluate_inverter) on the on-disk seed-42 v6.4.1
checkpoints — see /tmp/v6_4_1_baseline_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "logs" / "v6_4_1_phase4"
SEEDS = [42, 123, 7, 17, 99, 256, 2024, 31337]

# V6.4.1 baseline (eval_v6_3_1_inverter harness):
#   VTC MaxErr (mV) / transient post-startup MaxErr (mV)
V641 = {
    "TSMC5":  (134.58, 39.64),
    "TSMC7":  (210.47, 49.39),
    "TSMC12": (63.09, 58.56),
    "TSMC16": (50.83, 55.09),
}


def stem(tech: str, dev: str, seed: int) -> str:
    return f"v6_4_1_p4_{tech.lower()}_s{seed}_{dev}"


def evaluate(tech: str, n_seed: int, p_seed: int, cache: dict) -> dict:
    """Run one inverter eval for the (n_seed, p_seed) pair (cached)."""
    key = (n_seed, p_seed)
    if key in cache:
        return cache[key]
    n_stem = stem(tech, "nmos", n_seed)
    p_stem = stem(tech, "pmos", p_seed)
    cmd = [
        "conda", "run", "--no-capture-output", "-n", "pycircuitsim",
        "python", str(ROOT / "scripts" / "eval_v6_4_1_pair.py"),
        "--tech", tech, "--nmos", n_stem, "--pmos", p_stem, "--json",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                         env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                              "PATH": __import__("os").environ["PATH"]})
    res = None
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[len("RESULT "):])
    if res is None:
        sys.stderr.write(out.stdout + "\n" + out.stderr + "\n")
        raise RuntimeError(f"eval failed for n{n_seed}/p{p_seed}")
    res["n_seed"] = n_seed
    res["p_seed"] = p_seed
    cache[key] = res
    jl = LOGDIR / f"search_{tech}.jsonl"
    with jl.open("a") as f:
        f.write(json.dumps(res) + "\n")
    print(f"  n{n_seed:>5} p{p_seed:>5}  "
          f"VTC={res['vtc_maxerr_mv']:7.1f}mV  "
          f"Tran={res['tran_post_maxerr_mv']:6.1f}mV  "
          f"R2={res['vtc_r2']:.4f}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", required=True)
    args = ap.parse_args()
    tech = args.tech
    LOGDIR.mkdir(parents=True, exist_ok=True)
    base_vtc, base_tran = V641[tech]
    cache: dict = {}

    print(f"\n=== V6.4.1 Phase-4 greedy pair search: {tech} ===")
    print(f"V6.4.1 baseline: VTC={base_vtc:.1f}mV  Tran={base_tran:.1f}mV")

    print("\n[Step 1] fix pmos=s42, sweep 8 nmos seeds")
    s1 = [evaluate(tech, ns, 42, cache) for ns in SEEDS]
    s1_sorted = sorted(s1, key=lambda r: r["vtc_maxerr_mv"])
    best_n = s1_sorted[0]["n_seed"]
    top3_n = [r["n_seed"] for r in s1_sorted[:3]]
    print(f"  -> best nmos seed={best_n}  top3={top3_n}")

    print(f"\n[Step 2] fix nmos=s{best_n}, sweep 8 pmos seeds")
    s2 = [evaluate(tech, best_n, ps, cache) for ps in SEEDS]
    s2_sorted = sorted(s2, key=lambda r: r["vtc_maxerr_mv"])
    best_p = s2_sorted[0]["p_seed"]
    top3_p = [r["p_seed"] for r in s2_sorted[:3]]
    print(f"  -> best pmos seed={best_p}  top3={top3_p}")

    print(f"\n[Step 3] joint-refine top3 nmos {top3_n} x top3 pmos {top3_p}")
    for ns in top3_n:
        for ps in top3_p:
            evaluate(tech, ns, ps, cache)

    # Selection: min VTC MaxErr s.t. transient <= V6.4.1 baseline.
    all_res = list(cache.values())
    feasible = [r for r in all_res
                if r["tran_post_maxerr_mv"] <= base_tran + 1e-6]
    pool = feasible if feasible else all_res
    pool_sorted = sorted(pool, key=lambda r: r["vtc_maxerr_mv"])
    win = pool_sorted[0]

    print(f"\n=== {tech} SELECTION ===")
    print(f"evals run: {len(cache)}")
    if not feasible:
        print("WARNING: no pair meets the transient gate; "
              "picking min-VTC overall.")
    print(f"winner: nmos=s{win['n_seed']} pmos=s{win['p_seed']}")
    print(f"  VTC  MaxErr={win['vtc_maxerr_mv']:.1f}mV "
          f"(V6.4.1 {base_vtc:.1f})  "
          f"NRMSE={win['vtc_nrmse_pct']:.3f}%  "
          f"MRE={win['vtc_mre_pct']:.2f}%  R2={win['vtc_r2']:.4f}")
    print(f"  Tran MaxErr={win['tran_post_maxerr_mv']:.1f}mV "
          f"(V6.4.1 {base_tran:.1f})  "
          f"NRMSE={win['tran_post_nrmse_pct']:.3f}%  "
          f"R2={win['tran_post_r2']:.4f}")
    beats = (win["vtc_maxerr_mv"] < base_vtc and
             win["tran_post_maxerr_mv"] <= base_tran + 1e-6)
    print(f"  verdict: {'PROMOTE' if beats else 'KEEP V6.4.1'}")

    summ = LOGDIR / f"search_{tech}_winner.json"
    win["_v641_vtc"] = base_vtc
    win["_v641_tran"] = base_tran
    win["_beats_v641"] = beats
    win["_n_evals"] = len(cache)
    summ.write_text(json.dumps(win, indent=2))
    print(f"  written: {summ}")


if __name__ == "__main__":
    main()
