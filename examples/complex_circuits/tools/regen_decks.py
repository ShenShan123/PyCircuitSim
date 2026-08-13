"""Regenerate every built design's decks from its OWN design.json.

The deck emitters (build_amp.write_design, size_ldo.decks, sfe/size_vref/
size_cp emit, sfe_amp.emit) are the single source of deck text; after a
tools update their comment/text fixes only materialize on the next emit.
This script re-emits every design in the tree from the tree's own
design.json so no stale deck survives, without changing any device size:
it reuses port_tech's per-category translators with the reference tree
repointed at this tree, where the voltage scale factor is exactly 1 and
the L/NFIN/vt snaps are idempotent.

    python3 tools/regen_decks.py [category ...]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import port_tech
from port_tech import PORTERS, ROOT
from pycmg_lib import TECH


def main() -> None:
    # Re-emit from this tree's own design.json, not the TSMC16 reference.
    port_tech.REF = ROOT
    wanted: List[str] = sys.argv[1:] or list(PORTERS)
    n_ok = n_fail = 0
    for cat in wanted:
        cdir = ROOT / cat
        if not cdir.is_dir():
            continue
        for d in sorted(cdir.iterdir()):
            if not (d / "design.json").exists():
                continue
            try:
                PORTERS[cat](d.name)
                n_ok += 1
                print(f"  regen {cat}/{d.name}", flush=True)
            except Exception as exc:
                n_fail += 1
                print(f"  FAIL  {cat}/{d.name}: {type(exc).__name__}: {exc}",
                      flush=True)
    print(f"\n{TECH}: {n_ok} regenerated, {n_fail} failed")


if __name__ == "__main__":
    main()
