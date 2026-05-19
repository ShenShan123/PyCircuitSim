"""V6.4.2 Phase-7 bake-off table collector.

Reads the per-(tech,recipe) winner JSON files written by
v6_4_2_phase7_search.py and prints the full scoped bake-off table:
stock vs +7a (monotonic) vs the V6.4.1 baseline, per tech.

Also dumps every individual (recipe, tech, n_seed, p_seed) eval from the
search jsonl logs so the full grid is on the record.

Usage:
    python scripts/v6_4_2_phase7_collect.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "logs" / "v6_4_2_phase7"
TECHS = ["TSMC5", "TSMC7"]
RECIPES = ["stock", "mono"]

V641 = {
    "TSMC5":  (134.58, 39.64),
    "TSMC7":  (210.47, 49.39),
}


def main() -> None:
    print("\n" + "=" * 78)
    print("V6.4.2 PHASE-7 SCOPED BAKE-OFF — full grid (every pair eval)")
    print("=" * 78)
    for tech in TECHS:
        for recipe in RECIPES:
            jl = LOGDIR / f"search_{tech}_{recipe}.jsonl"
            if not jl.exists():
                print(f"  [{tech} {recipe}] no jsonl yet")
                continue
            rows = [json.loads(l) for l in jl.read_text().splitlines() if l]
            print(f"\n  [{tech} {recipe}]  ({len(rows)} evals)")
            print(f"  {'nmos':>6} {'pmos':>6} | {'VTC mV':>9} {'Tran mV':>9} "
                  f"{'R2':>8}")
            for r in sorted(rows, key=lambda x: x["vtc_maxerr_mv"]):
                print(f"  s{r['n_seed']:>5} s{r['p_seed']:>5} | "
                      f"{r['vtc_maxerr_mv']:9.1f} "
                      f"{r['tran_post_maxerr_mv']:9.1f} "
                      f"{r['vtc_r2']:8.4f}")

    print("\n" + "=" * 78)
    print("WINNER SUMMARY — best pair per (tech, recipe) vs V6.4.1")
    print("=" * 78)
    print(f"  {'Tech':>7} {'Recipe':>7} | {'VTC mV':>9} {'Tran mV':>9} "
          f"| {'V6.4.1 VTC/Tran':>16} | verdict")
    for tech in TECHS:
        bv, bt = V641[tech]
        for recipe in RECIPES:
            w = LOGDIR / f"search_{tech}_{recipe}_winner.json"
            if not w.exists():
                print(f"  {tech:>7} {recipe:>7} | (no winner yet)")
                continue
            d = json.loads(w.read_text())
            verdict = "BEATS" if d.get("_beats_v641") else "loses"
            print(f"  {tech:>7} {recipe:>7} | "
                  f"{d['vtc_maxerr_mv']:9.1f} {d['tran_post_maxerr_mv']:9.1f} "
                  f"| {bv:7.1f}/{bt:<8.1f} | {verdict}  "
                  f"(nmos=s{d['n_seed']} pmos=s{d['p_seed']})")


if __name__ == "__main__":
    main()
