"""V6.4.5 Phase-3/5 multi-circuit candidate search (parallel).

Scores DirectNet (nmos-stem, pmos-stem) candidates for ONE tech against the
V6.4.5 multi-circuit selection vector via scripts/eval_v6_4_5_candidate.py
(one conda subprocess per candidate, BSIMAR_CHECKPOINT_DIR-isolated → safe to
run many concurrently; the real checkpoints/ dir is never mutated).

Each candidate eval is ~13 min single-threaded; with --jobs N they run N at a
time (the box has 152 cores). Default 8.

Two modes:

  --mode diagonal   Score same-seed N&P pairs across recipes (the cheap,
                    informative first pass) plus the baseline pair. Fully
                    parallel.
  --mode greedy     Greedy seed search within one recipe: fix pmos=s<ref>,
                    sweep nmos (parallel); fix best nmos, sweep pmos
                    (parallel); joint-refine top2 x top2 (parallel). Ranks
                    by ring_osc_period_err subject to the hard gates.

Candidate stems follow ``v6_4_2_p7_<tech>_<recipe>_s<seed>_<dev>``; the
baseline pair defaults to the canonical ``tsmc{X}_dn_medium_<dev>`` slot
(the V6.4.4 shipping checkpoint).

Selection objective (plan §Phase 3 lex/Pareto, re-calibrated per Phase-3
finding — see results/v6_4_5/phase3_multi_circuit_scorer.md):
  hard gates: inv_vtc_nrmse <= 5, inv_tran_post_nrmse <= 5,
              opamp_flat_flag <= baseline_flag  (relative regression guard:
              a candidate may not turn a non-flat baseline flat; the absolute
              ==0 form would reject the V6.4.4 TSMC7 baseline whose opamp is
              already a failing cell)
  primary   : minimise ring_osc_period_err  (the open TSMC7 gate)
  tiebreak  : minimise sram_rail_snap_resid, then inv_vtc_nrmse

Usage:
    python scripts/v6_4_5_search.py --tech TSMC7 --mode diagonal --jobs 8
    python scripts/v6_4_5_search.py --tech TSMC7 --mode greedy --recipe stock --jobs 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "results" / "v6_4_5" / "phase3_logs"
EVAL = ROOT / "scripts" / "eval_v6_4_5_candidate.py"

INV_VTC_GATE = 5.0
INV_TRAN_GATE = 5.0
RO_GATE = 5.0

# Baseline opamp_flat_flag per tech (canonical V6.4.4 slot), measured in the
# Phase-3 re-baseline. TSMC7's opamp is already a failing/flat cell, so the
# relative guard lets candidates through there.
BASELINE_OPAMP_FLAT = {"TSMC5": 0, "TSMC7": 1, "TSMC12": 1, "TSMC16": 1}


def p7_stem(tech: str, recipe: str, dev: str, seed: int) -> str:
    return f"v6_4_2_p7_{tech.lower()}_{recipe}_s{seed}_{dev}"


def canonical_stem(tech: str, dev: str) -> str:
    return f"{tech.lower()}_dn_medium_{dev}"


def _eval_one(tech: str, n_stem: str, p_stem: str, label: str) -> Optional[dict]:
    cmd = [
        "conda", "run", "--no-capture-output", "-n", "pycircuitsim",
        "python", str(EVAL), "--tech", tech,
        "--nmos", n_stem, "--pmos", p_stem, "--json",
    ]
    out = subprocess.run(
        cmd, capture_output=True, text=True, cwd=ROOT,
        env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
             "PATH": os.environ["PATH"]})
    res = None
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[len("RESULT "):])
    if res is None:
        sys.stderr.write(f"[eval FAILED] {label} {n_stem}/{p_stem}\n")
        sys.stderr.write(out.stdout[-1500:] + "\n" + out.stderr[-1500:] + "\n")
        return None
    res["label"] = label
    return res


def evaluate_many(tech: str, pairs: list[tuple[str, str, str]],
                  cache: dict, jobs: int) -> list[dict]:
    """pairs: list of (n_stem, p_stem, label). Skips already-cached keys."""
    todo = [(n, p, lbl) for (n, p, lbl) in pairs if (n, p) not in cache]
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_eval_one, tech, n, p, lbl): (n, p, lbl)
                for (n, p, lbl) in todo}
        for fut in futs:
            pass  # submitted
        for fut, (n, p, lbl) in list(futs.items()):
            r = fut.result()
            if r is None:
                continue
            cache[(n, p)] = r
            LOGDIR.mkdir(parents=True, exist_ok=True)
            with (LOGDIR / f"search_{tech}.jsonl").open("a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"  [{lbl:16s}] "
                  f"VTC={r.get('inv_vtc_nrmse', float('nan')):6.3f}%  "
                  f"tran={r.get('inv_tran_post_nrmse', float('nan')):6.3f}%  "
                  f"RO={r.get('ring_osc_period_err', float('nan')):6.2f}%  "
                  f"SRAM={r.get('sram_rail_snap_resid', float('nan')):.3f}  "
                  f"flat={r.get('opamp_flat_flag', '?')}", flush=True)
    for (n, p, lbl) in pairs:
        if (n, p) in cache:
            results.append(cache[(n, p)])
    return results


def feasible(r: dict, tech: str) -> bool:
    base_flat = BASELINE_OPAMP_FLAT.get(tech, 0)
    return (r.get("inv_vtc_nrmse", 1e9) <= INV_VTC_GATE
            and r.get("inv_tran_post_nrmse", 1e9) <= INV_TRAN_GATE
            and r.get("opamp_flat_flag", 1) <= base_flat)


def rank_key(r: dict) -> tuple:
    return (r.get("ring_osc_period_err", 1e9),
            r.get("sram_rail_snap_resid", 1e9),
            r.get("inv_vtc_nrmse", 1e9))


def _seed_of(stem: str) -> int:
    return int(stem.split("_s")[1].split("_")[0])


def report_winner(tech: str, cache: dict) -> dict:
    all_res = list(cache.values())
    feas = [r for r in all_res if feasible(r, tech)]
    pool = feas if feas else all_res
    win = sorted(pool, key=rank_key)[0]
    print(f"\n=== {tech} SELECTION (evals={len(cache)}, feasible={len(feas)}) ===")
    if not feas:
        print("WARNING: no candidate clears the hard gates; ranking all.")
    print(f"winner: nmos={win['nmos']} pmos={win['pmos']}  ({win.get('label','')})")
    print(f"  inv VTC NRMSE={win.get('inv_vtc_nrmse', float('nan')):.3f}%  "
          f"tran={win.get('inv_tran_post_nrmse', float('nan')):.3f}%")
    ro = win.get("ring_osc_period_err", float("nan"))
    print(f"  RO period_err={ro:.2f}%  (gate <= {RO_GATE}%  -> "
          f"{'PASS' if ro <= RO_GATE else 'FAIL'})")
    print(f"  SRAM rail_resid={win.get('sram_rail_snap_resid', float('nan')):.3f}  "
          f"opamp flat={win.get('opamp_flat_flag', '?')}")
    summ = LOGDIR / f"search_{tech}_winner.json"
    summ.write_text(json.dumps(win, indent=2))
    print(f"  written: {summ}")
    return win


def run_diagonal(tech, recipes, seeds, base_n, base_p, cache, jobs):
    print(f"\n=== {tech} DIAGONAL scan (recipes={recipes} seeds={seeds}) ===")
    pairs = [(base_n, base_p, "baseline")]
    for recipe in recipes:
        for s in seeds:
            pairs.append((p7_stem(tech, recipe, "nmos", s),
                          p7_stem(tech, recipe, "pmos", s), f"{recipe}_s{s}"))
    evaluate_many(tech, pairs, cache, jobs)
    report_winner(tech, cache)


def run_greedy(tech, recipe, seeds, ref_seed, cache, jobs):
    print(f"\n=== {tech} GREEDY recipe={recipe} seeds={seeds} ===")
    print(f"[Step 1] fix pmos=s{ref_seed}, sweep nmos")
    s1 = evaluate_many(tech, [
        (p7_stem(tech, recipe, "nmos", ns), p7_stem(tech, recipe, "pmos", ref_seed),
         f"{recipe}_n{ns}p{ref_seed}") for ns in seeds], cache, jobs)
    top_n = [_seed_of(r["nmos"]) for r in sorted(s1, key=rank_key)[:2]]
    print(f"  -> top2 nmos seeds={top_n}")

    print(f"[Step 2] fix nmos=s{top_n[0]}, sweep pmos")
    s2 = evaluate_many(tech, [
        (p7_stem(tech, recipe, "nmos", top_n[0]), p7_stem(tech, recipe, "pmos", ps),
         f"{recipe}_n{top_n[0]}p{ps}") for ps in seeds], cache, jobs)
    top_p = [_seed_of(r["pmos"]) for r in sorted(s2, key=rank_key)[:2]]
    print(f"  -> top2 pmos seeds={top_p}")

    print(f"[Step 3] joint-refine top2 nmos {top_n} x top2 pmos {top_p}")
    evaluate_many(tech, [
        (p7_stem(tech, recipe, "nmos", ns), p7_stem(tech, recipe, "pmos", ps),
         f"{recipe}_n{ns}p{ps}") for ns in top_n for ps in top_p], cache, jobs)
    report_winner(tech, cache)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", required=True)
    ap.add_argument("--mode", choices=["diagonal", "greedy"], default="diagonal")
    ap.add_argument("--recipe", default="stock", choices=["stock", "mono"])
    ap.add_argument("--recipes", default="stock,mono")
    ap.add_argument("--seeds", default="42,123,7,17")
    ap.add_argument("--ref-seed", type=int, default=42)
    ap.add_argument("--base-nmos", default=None)
    ap.add_argument("--base-pmos", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    tech = args.tech
    seeds = [int(s) for s in args.seeds.split(",")]
    base_n = args.base_nmos or canonical_stem(tech, "nmos")
    base_p = args.base_pmos or canonical_stem(tech, "pmos")
    cache: dict = {}

    if args.mode == "diagonal":
        run_diagonal(tech, [r.strip() for r in args.recipes.split(",")],
                     seeds, base_n, base_p, cache, args.jobs)
    else:
        run_greedy(tech, args.recipe, seeds, args.ref_seed, cache, args.jobs)


if __name__ == "__main__":
    main()
