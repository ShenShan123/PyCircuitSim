#!/usr/bin/env python3
"""Render a compact PASS/FAIL grid + per-cell headline metrics across recipes
from results/recipe_bench/gate_iso/<recipe>/SUMMARY.txt (written by
scripts/gate_matrix_iso.sh). Usage: python scripts/gate_grid.py r1 r2 r3 ...
(default: clean + all crit* on disk)."""
from __future__ import annotations
import re, sys, os, glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recipe_retest_collect import PARSERS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "recipe_bench" / "gate_iso"
TECHS = ["TSMC5", "TSMC7", "TSMC12", "TSMC16"]
CIRCS = ["ring_osc", "opamp", "sram_snm", "switchcap"]
NCELL = len(TECHS) * len(CIRCS)   # 16 = 4 techs x 4 circuits


def load(recipe: str):
    f = BASE / recipe / "SUMMARY.txt"
    d = {}
    if f.exists():
        for line in f.read_text().splitlines():
            m = re.match(r"(\S+)\s+(\S+)\s+\|\s+rc=(\d+)\s+(PASS|FAIL)", line)
            if m:
                d[(m.group(1), m.group(2))] = (m.group(4), int(m.group(3)))
    # fallback: derive the verdict from the per-cell log's own status when
    # SUMMARY.txt lacks the rc line (subset re-runs used to clobber it);
    # rc=-1 marks a log-derived verdict
    for t in TECHS:
        for c in CIRCS:
            if (t, c) in d:
                continue
            p = BASE / recipe / f"{t.lower()}_{c}.log"
            if p.exists():
                st = PARSERS[c](p.read_text(errors="replace")).get("status")
                if st in ("PASS", "FAIL"):
                    d[(t, c)] = (st, -1)
    return d


def main():
    recipes = sys.argv[1:]
    if not recipes:
        found = sorted(p.name for p in BASE.iterdir()
                       if p.is_dir() and (p / "SUMMARY.txt").exists())
        recipes = (["clean"] if "clean" in found else []) + \
                  [r for r in found if r.startswith("crit")] + \
                  [r for r in found if r not in ("clean",) and not r.startswith("crit")]
    data = {r: load(r) for r in recipes}
    w = max(9, max((len(r) for r in recipes), default=9) + 1)
    hdr = "cell".ljust(20) + "".join(r[:w-1].rjust(w) for r in recipes)
    print(hdr); print("-" * len(hdr))
    for t in TECHS:
        for c in CIRCS:
            row = f"{t}-{c}".ljust(20)
            for r in recipes:
                v = data[r].get((t, c))
                cell = "  ." if v and v[0] == "PASS" else " XX" if v else "  ?"
                row += cell.rjust(w)
            print(row)
    print("-" * len(hdr))
    tot = f"TOTAL /{NCELL}".ljust(20)
    for r in recipes:
        n = sum(1 for t in TECHS for c in CIRCS
                if data[r].get((t, c), ("", 1))[0] == "PASS")
        tot += f"{n}/{NCELL}".rjust(w)
    print(tot)
    print("\nLegend: '.'=PASS  'XX'=FAIL  '?'=missing")


if __name__ == "__main__":
    main()
