#!/usr/bin/env python3
"""Collect the post-A3 universal-DirectNet re-gate into a comparison table.

Reads the per-cell logs under ``results/a3_regate_uni/<stem>/logs/`` rather than
SUMMARY.tsv, so the result survives a dispatch that rewrites the summary, and
reports both the single-run (OMP=1) count and the strict count (a cell counts
strict only if it passes at every OMP it was swept at).

Pre-fix baselines are the *strict* counts from DirectNet-L73-accuracy.md §9 —
the universal campaign published strict, not single-run, so this is the
like-for-like comparison.
"""
from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path

PRE_FIX_STRICT = {          # (strict, FLIPs) from §9 at commit a96112a
    "u716_dn_corroft_large": (10, 0),
    "u716_dn_clean_large": (9, 0),
    "u716_dn_crit30u_large": (9, 1),
    "u716_dn_csob_large": (8, 1),
    "u716_dn_clean_xl": (8, 1),
    "u716_dn_corroft_xl": (8, 0),
    "u716f5_plain_n1000000_large": (4, 0),
    "u716f5_plain_nfull_large": (3, 0),
}

_LOG = re.compile(r"(gates|omp)_verify_complex_(.+)_(TSMC\d+)_omp(\d+)\.log$")


def verdict(path: Path) -> str:
    text = path.read_text(errors="replace")
    results = re.findall(r"RESULT:\s*(\S+)", text)
    if results:
        return "PASS" if results[-1].upper().startswith(("ALL", "PASS")) else "FAIL"
    if re.search(r"->\s*PASS", text) and not re.search(r"->\s*FAIL", text):
        return "PASS"
    return "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/a3_regate_uni", type=Path)
    args = ap.parse_args()

    print("| stem | post-fix single | post-fix strict | FLIPs | pre-fix strict | delta |")
    print("|---|---|---|---|---|---|")
    for stem_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        cells: dict = collections.defaultdict(dict)
        for log in (stem_dir / "logs").glob("*.log"):
            m = _LOG.match(log.name)
            if m:
                _, circ, tech, omp = m.groups()
                cells[(circ, tech)][int(omp)] = verdict(log)
        if not cells:
            continue
        n = len(cells)
        single = sum(1 for v in cells.values() if v.get(1) == "PASS")
        strict = sum(1 for v in cells.values() if all(x == "PASS" for x in v.values()))
        flips = sum(1 for v in cells.values()
                    if len(v) > 1 and "PASS" in v.values()
                    and any(x != "PASS" for x in v.values()))
        pre = PRE_FIX_STRICT.get(stem_dir.name)
        if pre is None:
            pre_s, delta = "—", "—"
        else:
            pre_s = f"{pre[0]}/{n} ({pre[1]} FLIP)"
            d = strict - pre[0]
            delta = f"**+{d}**" if d > 0 else (f"**{d}**" if d < 0 else "0")
        print(f"| `{stem_dir.name}` | {single}/{n} | **{strict}/{n}** | {flips} | {pre_s} | {delta} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
