#!/usr/bin/env python3
"""V6.6.1 recipe campaign — aggregate per-(recipe,size,tech) NN test logs into a
recipe-comparison matrix.

Parses  results/recipe_bench/<recipe>/<size>/<tech>/<suite>.log  (captured stdout
of the 8 NN suites) and emits:
  - results/recipe_bench/RECIPE_REPORT.md      (human-facing comparison tables)
  - results/recipe_bench/recipe_data.json      (machine-readable)

Reuses the tolerant gate/device/AC regexes from scripts/benchmark_collect.py so
the parsing is identical to the V6.6.0 capacity benchmark — only the directory
layout (recipe-major) differs.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "recipe_bench"
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_collect as bc  # reuse parse_device_log / parse_complex_log / AC parsers

TECHS = ["tsmc5", "tsmc7", "tsmc12", "tsmc16"]
CPX_SUITES = bc.CPX_SUITES
DEV_SUITES = bc.DEV_SUITES
AC_DEV, AC_CPX = bc.AC_DEV_SUITE, bc.AC_CPX_SUITE
CPX_SHORT = bc.CPX_SHORT


def discover():
    """recipes & sizes present on disk (recipe dirs each containing size dirs)."""
    recipes = sorted([p.name for p in BASE.iterdir()
                      if p.is_dir() and p.name not in ("train_logs",)
                      and any((p / s).is_dir() for s in ("small", "medium", "large", "xl"))])
    # canonical order: clean first (control), then the rest
    order = ["clean", "cor", "corft", "invtripft", "invtrip", "csob", "sob",
             "sobf", "ekv", "ekvhi", "e3", "csobekv"]
    recipes = sorted(recipes, key=lambda r: (order.index(r) if r in order else 99, r))
    sizes = set()
    for r in recipes:
        for s in ("small", "medium", "large", "xl"):
            if (BASE / r / s).is_dir():
                sizes.add(s)
    sizes = [s for s in ("small", "medium", "large", "xl") if s in sizes]
    return recipes, sizes


def load(recipes, sizes):
    data = {}
    for recipe in recipes:
        data[recipe] = {}
        for size in sizes:
            data[recipe][size] = {}
            for tech in TECHS:
                d = {"dev": {}, "cpx": {}, "ac": {}, "ckpt": None}
                cell = BASE / recipe / size / tech
                for suite in DEV_SUITES:
                    p = cell / f"{suite}.log"
                    if p.exists():
                        t = p.read_text(errors="replace")
                        d["dev"][suite] = bc.parse_device_log(t)
                        d["ckpt"] = d["ckpt"] or bc.resolver_line(t)
                for suite in CPX_SUITES:
                    p = cell / f"{suite}.log"
                    if p.exists():
                        t = p.read_text(errors="replace")
                        d["cpx"][suite] = bc.parse_complex_log(suite, t)
                        d["ckpt"] = d["ckpt"] or bc.resolver_line(t)
                p = cell / f"{AC_DEV}.log"
                if p.exists():
                    d["ac"][AC_DEV] = bc.parse_ac_device_log(p.read_text(errors="replace"))
                p = cell / f"{AC_CPX}.log"
                if p.exists():
                    d["ac"][AC_CPX] = bc.parse_ac_complex_log(p.read_text(errors="replace"))
                data[recipe][size][tech] = d
    return data


def cpx_pass(d):
    g = p = 0
    for suite in CPX_SUITES:
        c = d["cpx"].get(suite)
        if c:
            g += 1
            p += 1 if c["gate"] == "PASS" else 0
    return p, g


def ac_dev_pass(d):
    dv = d["ac"].get(AC_DEV)
    if not dv:
        return 0, 0
    return dv["n_pass"], dv["n"]


def dev_nrmse_mean(d):
    vals = []
    for suite in DEV_SUITES:
        dv = d["dev"].get(suite)
        if dv:
            for s in dv["by_dev"].values():
                vals.append(s["nrmse_mean"])
    return round(mean(vals), 2) if vals else None


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def report(data, recipes, sizes):
    L = ["# DirectNet (LEVEL=73) recipe-comparison accuracy benchmark", "",
         "Each recipe is a UNIFORM training addendum applied identically to all "
         "(tech × device × size) checkpoints on top of the V6.6.0 clean recipe "
         "(`--apply-filter off --swa-mode ema --seed 42`). `clean` is the control. "
         "Ground truth = NGSPICE BSIM-CMG (LEVEL=72), CPU-pinned. Gate matrix = "
         "4 circuits × 4 techs = 16 complex gates per (recipe,size).", ""]

    # ── headline: complex-gate pass-rate, recipe × size ──
    L += ["## Headline — complex-circuit gate pass-rate (X / 16)", "",
          "Higher is better. Clean control: large 13/16, xl 10/16 (V6.6.0).", ""]
    rows = []
    for recipe in recipes:
        cells = []
        for size in sizes:
            tp = tg = 0
            for tech in TECHS:
                p, g = cpx_pass(data[recipe][size][tech])
                tp += p; tg += g
            cells.append(f"{tp}/{tg}" if tg else "—")
        rows.append([recipe] + cells)
    L += [md_table(["Recipe"] + sizes, rows), ""]

    # ── AC device pass-rate, recipe × size ──
    L += ["## AC device CS-amp pass-rate (X / 8)", "",
          "Clean control: large 4/8-equiv (4/12 over all sizes in V6.6.0). "
          "Per (recipe,size) there are 8 device cells (4 techs × n/p).", ""]
    rows = []
    for recipe in recipes:
        cells = []
        for size in sizes:
            tp = tg = 0
            for tech in TECHS:
                p, g = ac_dev_pass(data[recipe][size][tech])
                tp += p; tg += g
            cells.append(f"{tp}/{tg}" if tg else "—")
        rows.append([recipe] + cells)
    L += [md_table(["Recipe"] + sizes, rows), ""]

    # ── device mean NRMSE%, recipe × size ──
    L += ["## Device-level mean NRMSE% (lower = better fit)", ""]
    rows = []
    for recipe in recipes:
        cells = []
        for size in sizes:
            vals = [dev_nrmse_mean(data[recipe][size][tech]) for tech in TECHS]
            vals = [v for v in vals if v is not None]
            cells.append(round(mean(vals), 2) if vals else "—")
        rows.append([recipe] + cells)
    L += [md_table(["Recipe"] + sizes, rows), ""]

    # ── per-(recipe,size) full 16-gate matrix vs clean ──
    for size in sizes:
        L += [f"## Complex-gate matrix — size = {size}", ""]
        hdr = ["Tech", "Circuit"] + recipes
        rows = []
        for tech in TECHS:
            for suite in CPX_SUITES:
                cells = []
                for recipe in recipes:
                    c = data[recipe][size][tech]["cpx"].get(suite)
                    if not c:
                        cells.append("—")
                    else:
                        hl = c.get("headline", "")
                        cells.append(f"{c['gate']}")
                rows.append([tech, CPX_SHORT[suite]] + cells)
        L += [md_table(hdr, rows), ""]
        # headline detail (errors) for the open gates
        L += [f"### Gate headline detail — size = {size} "
              "(period_err% / gain_err% / charge_err% / SNM)", ""]
        hdr = ["Tech", "Circuit"] + recipes
        rows = []
        for tech in TECHS:
            for suite in CPX_SUITES:
                cells = []
                for recipe in recipes:
                    c = data[recipe][size][tech]["cpx"].get(suite)
                    cells.append((c.get("headline") or c.get("gate") or "—") if c else "—")
                rows.append([tech, CPX_SHORT[suite]] + cells)
        L += [md_table(hdr, rows), ""]

    return "\n".join(L)


def main():
    recipes, sizes = discover()
    if not recipes:
        print("No recipe result dirs found under", BASE); return
    data = load(recipes, sizes)
    (BASE / "recipe_data.json").write_text(json.dumps(data, indent=1))
    (BASE / "RECIPE_REPORT.md").write_text(report(data, recipes, sizes))
    print(f"Recipes: {recipes}  Sizes: {sizes}")
    print(f"Wrote {BASE/'RECIPE_REPORT.md'} and recipe_data.json")
    # console tally
    print("\nComplex-gate pass-rate (X/16):")
    for recipe in recipes:
        line = f"  {recipe:9s}: "
        for size in sizes:
            tp = tg = 0
            for tech in TECHS:
                p, g = cpx_pass(data[recipe][size][tech]); tp += p; tg += g
            line += f"{size}={tp}/{tg}  "
        print(line)


if __name__ == "__main__":
    main()
