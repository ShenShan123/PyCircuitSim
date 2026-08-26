#!/usr/bin/env python3
"""Control: do the V7.3.0 re-measurements agree with the earlier passes?

Every complex cell measured in BOTH V7.3.0 and an earlier pass is compared
verdict-to-verdict. The V7.3.0 campaign re-gates BSIM-AR on HEAD, which
carries the V7.0.x performance work, audit fix wave 1 and the passive
thread-wait policy on top of the code the earlier numbers were measured at. All
were declared behaviour-preserving; this checks it against the only thing that
matters — the gate verdict.

**A disagreement is not automatically a bug.** The strict rule means a cell can
legitimately differ when the earlier pass measured only OMP=1 and this one
measured all three: single-run PASS + strict FAIL is a *flip*, which is real
information, not a contradiction. Those are reported separately from genuine
same-basis disagreements, which are the ones that would mean the two campaigns'
numbers cannot be mixed.

Usage: python scripts/v730_control.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_v730b", pathlib.Path(__file__).resolve().parent / "v730_docs_build.py")
_b = importlib.util.module_from_spec(_spec)
sys.modules["_v730b"] = _b
_spec.loader.exec_module(_b)

CIRCS = _b.CIRCS
TECHS = _b.TECHS


def verdict_from(data: dict, tag: str, variant: str, circ: str, tech: str,
                 omps) -> str | None:
    e = data.get(tag, {}).get(variant, {}).get(f"verify_complex_{circ}", {}).get(tech)
    if not e:
        return None
    got = [e[o] for o in omps if o in e]
    if not got:
        return None
    rcs = {g["rc"] for g in got}
    if rcs == {"0"}:
        return "PASS"
    return "FAIL" if "0" not in rcs else "FLIP"


def main() -> int:
    v710 = _b.load_json("v710_regate")
    v730 = _b.load_json("v730_regate")
    a3 = _b.A3

    agree = same_basis_disagree = flips_revealed = 0
    rows: list[str] = []

    for tag in ("dn", "tf"):
        for variant in sorted(set(v730.get(tag, {}))):
            for circ in CIRCS:
                for tech in TECHS:
                    new_all = verdict_from(v730, tag, variant, circ, tech,
                                           ("omp1", "omp2", "omp4"))
                    new_1 = verdict_from(v730, tag, variant, circ, tech, ("omp1",))
                    if new_1 is None:
                        continue
                    # Prefer a V7.1.0 counterpart; fall back to V6.13.0's
                    # single-run report.
                    old_1 = verdict_from(v710, tag, variant, circ, tech, ("omp1",))
                    src = "v710"
                    if old_1 is None:
                        a = a3.get((tag, variant, tech, circ))
                        old_1, src = (a[0] if a else None), "a3"
                    if old_1 is None:
                        continue

                    if new_1 == old_1:
                        agree += 1
                        # Same OMP=1 verdict, but the strict sweep disagrees:
                        # the cell flips. Informative, not a contradiction.
                        if new_all == "FLIP":
                            flips_revealed += 1
                            rows.append(f"  FLIP-REVEALED {tag}/{variant}/{tech}/{circ}"
                                        f": omp1={new_1} but strict={new_all}")
                    else:
                        same_basis_disagree += 1
                        rows.append(f"  DISAGREE      {tag}/{variant}/{tech}/{circ}"
                                    f": {src}={old_1} v730={new_1}")

    print(f"cells compared on the same basis (OMP=1): {agree + same_basis_disagree}")
    print(f"  agree                : {agree}")
    print(f"  disagree             : {same_basis_disagree}")
    print(f"  flips revealed by the strict sweep: {flips_revealed}")
    if rows:
        print("\n".join(rows))
    if same_basis_disagree:
        print("\nFAIL — the V7.3.0 numbers are NOT interchangeable with the "
              "earlier passes. Do not mix them in one table.")
        return 1
    print("\nOK — every jointly-measured cell reproduces at OMP=1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
