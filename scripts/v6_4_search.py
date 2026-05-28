"""V6.4 best-of-N greedy pair search for ONE tech.

Greedy protocol (minimizes inverter sims vs brute-force 64):
  1. Fix pmos at seed 42, sweep all 8 nmos seeds  -> rank by VTC MaxErr.
  2. Fix nmos at its best, sweep all 8 pmos seeds.
  3. Joint-refine: top-3 nmos x top-3 pmos.
Pick the pair minimizing inverter VTC MaxErr subject to transient
post-startup MaxErr <= the V6.3.1 harness baseline for that tech.

All evals go through scripts/eval_v6_4_pair.py (swap-eval-restore).
Each eval is sequential within a tech (they share canonical slots).
Results are appended to logs/v6_4_bestof/search_<tech>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "logs" / "v6_4_bestof"
SEEDS = [42, 123, 7, 17, 99, 256, 2024, 31337]

# V6.3.1 canonical harness baseline (measured by eval_v6_3_1_inverter):
#   VTC MaxErr (mV) / transient post-startup MaxErr (mV)
V631 = {
    "TSMC5":  (56.96, 39.48),
    "TSMC7":  (215.18, 50.25),
    "TSMC12": (260.65, 58.22),
    "TSMC16": (44.73, 55.32),
}

# Pre-existing repro stems (seed-42 / seed-123 stock trainings).
REPRO = {
    ("TSMC5", "nmos", 42): "v6_4_repro_tsmc5_dn_medium_nmos",
    ("TSMC5", "pmos", 42): "v6_4_repro_tsmc5_dn_medium_pmos",
    ("TSMC7", "nmos", 42): "v6_4_repro_tsmc7_dn_medium_nmos",
    ("TSMC7", "pmos", 42): "v6_4_repro_tsmc7_dn_medium_pmos",
    ("TSMC5", "nmos", 123): "v6_4_repro_seed123_tsmc5_dn_medium_nmos",
}


def stem(tech: str, dev: str, seed: int) -> str:
    if (tech, dev, seed) in REPRO:
        return REPRO[(tech, dev, seed)]
    return f"v6_4_bof_{tech.lower()}_s{seed}_{dev}"


def evaluate(tech: str, n_seed: int, p_seed: int,
             cache: dict) -> dict:
    """Run one inverter eval for the (n_seed, p_seed) pair (cached)."""
    key = (n_seed, p_seed)
    if key in cache:
        return cache[key]
    n_stem = stem(tech, "nmos", n_seed)
    p_stem = stem(tech, "pmos", p_seed)
    cmd = [
        "conda", "run", "--no-capture-output", "-n", "pycircuitsim",
        "python", str(ROOT / "scripts" / "eval_v6_4_pair.py"),
        "--tech", tech, "--nmos", n_stem, "--pmos", p_stem, "--json",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
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
    base_vtc, base_tran = V631[tech]
    cache: dict = {}

    print(f"\n=== V6.4 greedy pair search: {tech} ===")
    print(f"V6.3.1 baseline: VTC={base_vtc:.1f}mV  Tran={base_tran:.1f}mV")

    # Step 1: fix pmos=42, sweep nmos.
    print("\n[Step 1] fix pmos=s42, sweep 8 nmos seeds")
    s1 = [evaluate(tech, ns, 42, cache) for ns in SEEDS]
    s1_sorted = sorted(s1, key=lambda r: r["vtc_maxerr_mv"])
    best_n = s1_sorted[0]["n_seed"]
    top3_n = [r["n_seed"] for r in s1_sorted[:3]]
    print(f"  -> best nmos seed={best_n}  top3={top3_n}")

    # Step 2: fix nmos=best_n, sweep pmos.
    print(f"\n[Step 2] fix nmos=s{best_n}, sweep 8 pmos seeds")
    s2 = [evaluate(tech, best_n, ps, cache) for ps in SEEDS]
    s2_sorted = sorted(s2, key=lambda r: r["vtc_maxerr_mv"])
    best_p = s2_sorted[0]["p_seed"]
    top3_p = [r["p_seed"] for r in s2_sorted[:3]]
    print(f"  -> best pmos seed={best_p}  top3={top3_p}")

    # Step 3: joint refine top3 x top3.
    print(f"\n[Step 3] joint-refine top3 nmos {top3_n} x top3 pmos {top3_p}")
    for ns in top3_n:
        for ps in top3_p:
            evaluate(tech, ns, ps, cache)

    # Selection: min VTC MaxErr s.t. transient <= V6.3.1 baseline.
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
          f"(V6.3.1 {base_vtc:.1f})  "
          f"NRMSE={win['vtc_nrmse_pct']:.3f}%  "
          f"MRE={win['vtc_mre_pct']:.2f}%  R2={win['vtc_r2']:.4f}")
    print(f"  Tran MaxErr={win['tran_post_maxerr_mv']:.1f}mV "
          f"(V6.3.1 {base_tran:.1f})  "
          f"NRMSE={win['tran_post_nrmse_pct']:.3f}%  "
          f"R2={win['tran_post_r2']:.4f}")
    beats = (win["vtc_maxerr_mv"] < base_vtc and
             win["tran_post_maxerr_mv"] <= base_tran + 1e-6)
    print(f"  verdict: {'PROMOTE' if beats else 'KEEP V6.3.1'}")

    summ = LOGDIR / f"search_{tech}_winner.json"
    win["_v631_vtc"] = base_vtc
    win["_v631_tran"] = base_tran
    win["_beats_v631"] = beats
    win["_n_evals"] = len(cache)
    summ.write_text(json.dumps(win, indent=2))
    print(f"  written: {summ}")


if __name__ == "__main__":
    main()
