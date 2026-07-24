#!/usr/bin/env python3
"""Collect the V6.13.0 post-A3 re-gate into comparison tables.

Reads every ``results/a3_regate/<family>_<recipe>_<size>/<recipe>/SUMMARY.txt``
produced by ``scripts/gate_matrix_iso.sh`` and emits a markdown report: one
4-tech x 4-circuit matrix per checkpoint group, the pass count, and — where a
pre-fix number is known — the delta against it.

The pre-fix baselines are single-run counts transcribed from the accuracy
reports as they stood at commit ``a96112a`` (every one of them measured with
the gds sign bug present). They are here so the comparison is like-for-like:
single-run vs single-run, same harness, same techs. Strict OMP-swept counts
are NOT comparable to these and are not used.

Usage:
    python scripts/a3_regate_collect.py [--root results/a3_regate]
                                        [--out results/a3_regate/REPORT.md]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TECHS: List[str] = ["TSMC5", "TSMC7", "TSMC12", "TSMC16"]
CIRCS: List[str] = ["ring_osc", "opamp", "sram_snm", "switchcap"]

# Pre-fix single-run complex pass counts (out of 16), from the accuracy
# reports at commit a96112a. None = no directly comparable published number.
PRE_FIX: Dict[str, Optional[int]] = {
    # DirectNet — §3 capacity table (clean s/m/l/xl) and the Appendix A/B
    # "single-run OMP=1" columns.
    "dn/clean/small": 7,
    "dn/clean/medium": 10,
    "dn/clean/xl": 10,
    # NB "dn/clean/large" is the bare `tsmc{X}_dn_large_*` slot, which has
    # carried the crit30 curriculum weights since V6.6.4 — it is the PRODUCTION
    # checkpoint, not a clean one. Its pre-fix baseline is therefore crit30f's
    # 14/16, and `dn/crit30f/large` (the provenance copy of the same weights)
    # is an independent re-measurement of it. The genuine clean@large lives at
    # `dn/v660clean/large`.
    "dn/clean/large": 14,         # production slot (crit30 weights)
    "dn/crit30f/large": 14,       # same weights, provenance copy
    "dn/v660clean/large": 13,     # the archived genuine clean@large
    "dn/csob/large": 12,
    "dn/corroft/xl": 14,
    "dn/crit10/xl": 14,
    "dn/crit15m/xl": 14,
    # BSIM-AR — §3 scale table, §4 recipe table ("complex (single)"), §6 xl fill.
    "tf/clean/small": 12,
    "tf/clean/medium": 14,
    "tf/clean/large": 13,
    "tf/clean/xl": 13,
    "tf/corroft/medium": 15,
    "tf/corroft/large": 15,
    "tf/corroft/xl": 15,
    "tf/crit15m/large": 15,
    "tf/crit15m/xl": 15,
    "tf/crit30/large": 15,
    "tf/crit30/xl": 14,
    "tf/corro15/xl": 15,
    "tf/corro15/medium": None,    # never published at medium
    "tf/csob/xl": 13,
    "tf/invtrip/large": 13,
    # PFN — §5 gate table.
    "pfn/clean/small": 11,
    "pfn/clean/medium": 10,
    "pfn/clean/large": 8,
}

_LINE = re.compile(r"^(\S+)\s+(\S+)\s+\|\s+rc=(-?\d+)\s+(\S+)\s+\|\s*(.*)$")

# Headline metric extraction, per circuit — the number the gate actually scores.
_METRIC = {
    "ring_osc": re.compile(r"period error\s*=\s*(-?[\d.]+)\s*%"),
    "opamp": re.compile(r"gain error\s*=\s*(-?[\d.]+)\s*%"),
    "switchcap": re.compile(r"charge[_ ]err(?:or)?\s*=?\s*(-?[\d.]+)\s*%"),
    "sram_snm": re.compile(r"NRMSE\s*=?\s*(-?[\d.]+)\s*%"),
}


class Cell:
    __slots__ = ("tech", "circuit", "rc", "verdict", "raw")

    def __init__(self, tech: str, circuit: str, rc: int, verdict: str, raw: str):
        self.tech, self.circuit = tech, circuit
        self.rc, self.verdict, self.raw = rc, verdict, raw

    @property
    def passed(self) -> bool:
        return self.rc == 0 and self.verdict.upper() == "PASS"

    def metric(self) -> str:
        pat = _METRIC.get(self.circuit)
        if pat is not None:
            m = pat.search(self.raw)
            if m:
                return f"{float(m.group(1)):.2f}%"
        return "—"


def parse_summary(path: Path) -> List[Cell]:
    cells: List[Cell] = []
    for line in path.read_text(errors="replace").splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        tech, circ, rc, verdict, rest = m.groups()
        cells.append(Cell(tech.upper(), circ, int(rc), verdict, rest))
    return cells


def discover(root: Path) -> Dict[str, List[Cell]]:
    """Return {"family/recipe/size": cells} for every gated group.

    Reads the driver's per-cell verdict files rather than its SUMMARY.txt:
    SUMMARY.txt is only rebuilt when the whole group finishes, so a run that
    is still in flight (or that died partway) would otherwise report nothing.
    Both files carry the identical one-line-per-cell format.
    """
    groups: Dict[str, List[Cell]] = {}
    for recipe_dir in sorted(root.glob("*/*")):
        if not recipe_dir.is_dir():
            continue
        group_dir = recipe_dir.parent.name            # <family>_<recipe>_<size>
        parts = group_dir.split("_")
        if len(parts) < 3:
            continue
        family, size = parts[0], parts[-1]
        recipe = "_".join(parts[1:-1])
        cells: List[Cell] = []
        for cell_file in sorted(recipe_dir.glob(".cell_*")):
            cells.extend(parse_summary(cell_file))
        if not cells:
            summary = recipe_dir / "SUMMARY.txt"
            if summary.exists():
                cells = parse_summary(summary)
        if cells:
            groups[f"{family}/{recipe}/{size}"] = cells
    return groups


def matrix(cells: List[Cell]) -> Tuple[str, int, int]:
    """Render the 4x4 matrix, and return (markdown, passed, total)."""
    by_key = {(c.tech, c.circuit): c for c in cells}
    rows = ["| tech | " + " | ".join(CIRCS) + " |",
            "|---|" + "---|" * len(CIRCS)]
    passed = total = 0
    for tech in TECHS:
        cols = []
        for circ in CIRCS:
            c = by_key.get((tech, circ))
            if c is None:
                cols.append("·")
                continue
            total += 1
            passed += c.passed
            mark = "PASS" if c.passed else c.verdict.upper()
            cols.append(f"{mark} {c.metric()}".strip())
        rows.append(f"| {tech} | " + " | ".join(cols) + " |")
    return "\n".join(rows), passed, total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/a3_regate", type=Path)
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()

    groups = discover(args.root)
    if not groups:
        print(f"ERROR: no SUMMARY.txt found under {args.root}")
        return 1

    out: List[str] = [
        "# Post-A3 re-gate — every checkpoint on disk",
        "",
        "Gate verdict = the `verify_complex_*` script's exit code, CPU-pinned",
        "(`CUDA_VISIBLE_DEVICES=\"\" OMP=MKL=1`), repo ngspice, per-cell isolated",
        "results dir. Single-run, OMP=1 — directly comparable to the pre-fix",
        "single-run counts, NOT to strict OMP-swept counts.",
        "",
        "## Summary",
        "",
        "| group | post-fix | pre-fix | delta |",
        "|---|---|---|---|",
    ]
    detail: List[str] = ["", "## Per-group matrices", ""]
    incomplete: List[str] = []

    total_post = total_cells = 0
    for name in sorted(groups):
        md, passed, total = matrix(groups[name])
        total_post += passed
        total_cells += total
        pre = PRE_FIX.get(name)
        pre_s = "—" if pre is None else f"{pre}/16"
        if pre is None:
            delta = "—"
        elif total != 16:
            # Comparing a partial run against a full pre-fix count would
            # manufacture a regression out of cells that simply have not run.
            delta = "(incomplete)"
        else:
            d = passed - pre
            delta = f"**+{d}**" if d > 0 else (f"**{d}**" if d < 0 else "0")
        out.append(f"| `{name}` | {passed}/{total} | {pre_s} | {delta} |")
        if total != 16:
            incomplete.append(f"{name} ({total}/16 cells)")
        detail += [f"### `{name}` — {passed}/{total}", "", md, ""]

    out += ["", f"**Total: {total_post}/{total_cells} cells PASS across "
                f"{len(groups)} checkpoint groups.**", ""]
    if incomplete:
        out += ["> **Incomplete groups (fewer than 16 cells reported):** "
                + "; ".join(incomplete), ""]
    out += detail

    text = "\n".join(out)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
