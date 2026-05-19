"""V6.4.2 Phase-7 bake-off pair search.

For ONE (tech, recipe) pair, runs a greedy nmos/pmos seed search over the
4 bake-off seeds, exactly like v6_4_1_phase4_search.py but scoped to the
Phase-7 grid (recipe in {stock, mono}). Selects the (nmos-seed, pmos-seed)
pair minimising inverter VTC MaxErr subject to transient post-startup
MaxErr <= the V6.4.1 baseline for that tech.

All evals go through scripts/eval_v6_4_1_pair.py (swap-eval-restore against
the /tmp/v6_4_1_phase4_backup canonical V6.4.1 slots).

Recipe checkpoint stems: v6_4_2_p7_<tech>_<recipe>_s<S>_<dev>.

Usage:
    python scripts/v6_4_2_phase7_search.py --tech TSMC5 --recipe mono
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "logs" / "v6_4_2_phase7"
SEEDS = [42, 123, 7, 17]

# V6.4.1 baseline (eval_v6_3_1_inverter harness, on-disk seed-42 set):
#   VTC MaxErr (mV) / transient post-startup MaxErr (mV)
V641 = {
    "TSMC5":  (134.58, 39.64),
    "TSMC7":  (210.47, 49.39),
    "TSMC12": (63.09, 58.56),
    "TSMC16": (50.83, 55.09),
}


def stem(tech: str, recipe: str, dev: str, seed: int) -> str:
    return f"v6_4_2_p7_{tech.lower()}_{recipe}_s{seed}_{dev}"


def evaluate(tech: str, recipe: str, n_seed: int, p_seed: int,
             cache: dict) -> dict:
    key = (n_seed, p_seed)
    if key in cache:
        return cache[key]
    n_stem = stem(tech, recipe, "nmos", n_seed)
    p_stem = stem(tech, recipe, "pmos", p_seed)
    cmd = [
        "conda", "run", "--no-capture-output", "-n", "pycircuitsim",
        "python", str(ROOT / "scripts" / "eval_v6_4_1_pair.py"),
        "--tech", tech, "--nmos", n_stem, "--pmos", p_stem, "--json",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                         env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                              "PATH": os.environ["PATH"]})
    res = None
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[len("RESULT "):])
    if res is None:
        sys.stderr.write(out.stdout + "\n" + out.stderr + "\n")
        raise RuntimeError(f"eval failed for {recipe} n{n_seed}/p{p_seed}")
    res["n_seed"] = n_seed
    res["p_seed"] = p_seed
    res["recipe"] = recipe
    cache[key] = res
    jl = LOGDIR / f"search_{tech}_{recipe}.jsonl"
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
    ap.add_argument("--recipe", required=True, choices=["stock", "mono"])
    args = ap.parse_args()
    tech, recipe = args.tech, args.recipe
    LOGDIR.mkdir(parents=True, exist_ok=True)
    base_vtc, base_tran = V641[tech]
    cache: dict = {}

    print(f"\n=== V6.4.2 Phase-7 search: {tech} recipe={recipe} ===")
    print(f"V6.4.1 baseline: VTC={base_vtc:.1f}mV  Tran={base_tran:.1f}mV")

    # Step 1: fix pmos=s42, sweep nmos seeds.
    print("\n[Step 1] fix pmos=s42, sweep nmos seeds")
    s1 = [evaluate(tech, recipe, ns, 42, cache) for ns in SEEDS]
    s1_sorted = sorted(s1, key=lambda r: r["vtc_maxerr_mv"])
    best_n = s1_sorted[0]["n_seed"]
    top_n = [r["n_seed"] for r in s1_sorted[:2]]
    print(f"  -> best nmos seed={best_n}  top2={top_n}")

    # Step 2: fix nmos=best, sweep pmos seeds.
    print(f"\n[Step 2] fix nmos=s{best_n}, sweep pmos seeds")
    s2 = [evaluate(tech, recipe, best_n, ps, cache) for ps in SEEDS]
    s2_sorted = sorted(s2, key=lambda r: r["vtc_maxerr_mv"])
    best_p = s2_sorted[0]["p_seed"]
    top_p = [r["p_seed"] for r in s2_sorted[:2]]
    print(f"  -> best pmos seed={best_p}  top2={top_p}")

    # Step 3: joint-refine top2 x top2.
    print(f"\n[Step 3] joint-refine top2 nmos {top_n} x top2 pmos {top_p}")
    for ns in top_n:
        for ps in top_p:
            evaluate(tech, recipe, ns, ps, cache)

    # Selection: min VTC MaxErr s.t. transient <= V6.4.1 baseline.
    all_res = list(cache.values())
    feasible = [r for r in all_res
                if r["tran_post_maxerr_mv"] <= base_tran + 1e-6]
    pool = feasible if feasible else all_res
    win = sorted(pool, key=lambda r: r["vtc_maxerr_mv"])[0]

    print(f"\n=== {tech} {recipe} SELECTION ===")
    print(f"evals run: {len(cache)}")
    if not feasible:
        print("WARNING: no pair meets the transient gate; "
              "picking min-VTC overall.")
    print(f"winner: nmos=s{win['n_seed']} pmos=s{win['p_seed']}")
    print(f"  VTC  MaxErr={win['vtc_maxerr_mv']:.1f}mV "
          f"(V6.4.1 {base_vtc:.1f})  "
          f"NRMSE={win['vtc_nrmse_pct']:.3f}%  R2={win['vtc_r2']:.4f}")
    print(f"  Tran MaxErr={win['tran_post_maxerr_mv']:.1f}mV "
          f"(V6.4.1 {base_tran:.1f})")
    beats = (win["vtc_maxerr_mv"] < base_vtc and
             win["tran_post_maxerr_mv"] <= base_tran + 1e-6)
    print(f"  verdict: {'BEATS V6.4.1' if beats else 'does NOT beat V6.4.1'}")

    summ = LOGDIR / f"search_{tech}_{recipe}_winner.json"
    win["_v641_vtc"] = base_vtc
    win["_v641_tran"] = base_tran
    win["_beats_v641"] = beats
    win["_n_evals"] = len(cache)
    summ.write_text(json.dumps(win, indent=2))
    print(f"  written: {summ}")


if __name__ == "__main__":
    main()
