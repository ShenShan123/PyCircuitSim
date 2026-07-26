#!/usr/bin/env python3
"""Control: does HEAD reproduce the V6.13.0 (`d2ea720`) complex-gate verdicts?

The V7.1.0 re-gate runs on HEAD, which carries the V7.0.x performance work and
audit fix wave 1 on top of the frozen snapshot the V6.13.0 numbers were measured
at. Both were declared behaviour-preserving; this checks it against the only
thing that matters — the gate verdicts.

Every complex cell measured in BOTH campaigns is compared verdict-to-verdict.
Any disagreement means the two campaigns' numbers are NOT interchangeable and
the V7.1.0 tables must not be mixed with `results/a3_regate/REPORT.md`.

Usage: python scripts/v710_regate_control.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

CIRCS = ["ring_osc", "opamp", "sram_snm", "switchcap"]


def load_a3(path: pathlib.Path) -> dict:
    out, cur = {}, None
    for line in path.read_text().splitlines():
        m = re.match(r"^### `([^`]+)` — (\d+)/16", line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"^\| (TSMC\d+) \| (.+) \|$", line)
        if m and cur:
            for circ, val in zip(CIRCS, [p.strip() for p in m.group(2).split("|")]):
                out[(cur, m.group(1), circ)] = val
    return out


def a3_name(tag: str, variant: str) -> str:
    if variant in ("small", "medium", "large", "xl"):
        return f"{tag}/clean/{variant}"
    recipe, _, size = variant.rpartition("_")
    return f"{tag}/{recipe}/{size}"


def main() -> int:
    a3 = load_a3(pathlib.Path("results/a3_regate/REPORT.md"))
    v710 = json.loads(pathlib.Path("results/v710_regate/data.json").read_text())
    agree, rows = 0, []
    for tag, variants in v710.items():
        for variant, suites in variants.items():
            group = a3_name(tag, variant)
            for suite, techs in suites.items():
                if not suite.startswith("verify_complex_") or suite.endswith("_ac"):
                    continue
                circ = suite.replace("verify_complex_", "")
                for tech, omps in techs.items():
                    e = omps.get("omp1")
                    ref = a3.get((group, tech, circ))
                    if not e or ref is None:
                        continue
                    mine = "PASS" if e["rc"] == "0" else "FAIL"
                    if mine == ref.split()[0]:
                        agree += 1
                    else:
                        rows.append((group, tech, circ, ref, mine, e.get("metric")))
    print(f"HEAD vs d2ea720 complex-cell control: {agree} agree, {len(rows)} disagree")
    for r in rows:
        print(f"  MISMATCH {r[0]} {r[1]} {r[2]}: V6.13.0={r[3]} -> V7.1.0={r[4]} ({r[5]})")
    return 1 if rows else 0


if __name__ == "__main__":
    sys.exit(main())
