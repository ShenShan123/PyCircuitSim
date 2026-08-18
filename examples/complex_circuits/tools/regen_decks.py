"""Regenerate every built design's decks from its OWN design.json.

The deck emitters (build_amp.write_design, size_ldo.decks, sfe/size_vref/
size_cp emit, sfe_amp.emit) are the single source of deck text; after a
tools update their comment/text fixes only materialize on the next emit.
This script re-emits every design in the tree from the tree's own
design.json so no stale deck survives, without changing any device size:
it reuses port_tech's per-category translators with the reference tree
repointed at this tree, where the voltage scale factor is exactly 1 and
the L/NFIN/vt snaps are idempotent.

The untracked upstream AnalogGym topology corpus at ``../designs`` is a
required input. Generated netlists cannot replace it: their role multipliers
and source device identities have already been lowered. Missing source data is
therefore a loud nonzero failure, not a partial regeneration.

    python3 tools/regen_decks.py [category ...]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import port_tech
from port_tech import PORTERS, ROOT
from pycmg_lib import TECH


SOURCELESS_DESIGNS = {
    ("sensing_front_end", "SMCNR_SE_2st_AMP"),
}


def _selected_designs(wanted: List[str]) -> List[Tuple[str, Path]]:
    """Return the generated designs selected for one regeneration pass."""
    unknown = sorted(set(wanted) - set(PORTERS))
    if unknown:
        raise SystemExit(
            "Unknown regeneration categories: " + ", ".join(unknown)
        )

    selected: List[Tuple[str, Path]] = []
    for category in wanted:
        category_dir = ROOT / category
        if not category_dir.is_dir():
            continue
        selected.extend(
            (category, design_dir)
            for design_dir in sorted(category_dir.iterdir())
            if (design_dir / "design.json").exists()
        )
    return selected


def _preflight_sources(
    source_root: Path, selected: List[Tuple[str, Path]]
) -> None:
    """Reject an incomplete source corpus before any generated file changes."""
    missing: List[Path] = []
    for category, design_dir in selected:
        if (category, design_dir.name) in SOURCELESS_DESIGNS:
            continue
        source_dir = source_root / category / design_dir.name
        required = [source_dir / "netlist.spice"]
        if category == "charge_pump":
            required.append(source_dir / "param.spice")
        missing.extend(path for path in required if not path.is_file())

    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "AnalogGym source corpus is incomplete; missing source inputs:\n"
            f"{rendered}\nNo decks were regenerated."
        )


def main() -> None:
    source_root = ROOT.parent / "designs"
    if not source_root.is_dir():
        raise SystemExit(
            "AnalogGym source corpus is missing: "
            f"{source_root}. Deck regeneration needs the untracked source "
            "topologies; the generated designs alone are not a faithful "
            "substitute."
        )

    wanted: List[str] = sys.argv[1:] or list(PORTERS)
    selected = _selected_designs(wanted)
    _preflight_sources(source_root, selected)

    # Re-emit from this tree's own design.json, not the TSMC16 reference.
    # Source preflight must remain above this mutation boundary.
    port_tech.REF = ROOT
    n_ok = n_fail = 0
    for category, design_dir in selected:
        try:
            PORTERS[category](design_dir.name)
            n_ok += 1
            print(f"  regen {category}/{design_dir.name}", flush=True)
        except Exception as exc:
            n_fail += 1
            print(
                f"  FAIL  {category}/{design_dir.name}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
    print(f"\n{TECH}: {n_ok} regenerated, {n_fail} failed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
