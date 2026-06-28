#!/usr/bin/env python3
"""Equivalence canaries for the complex-circuit sweep builders (plan C2/C4).

The parametric builders in ``tests/common/complex.py`` must, at their default
(baseline) stimulus, reproduce the single-point ship-gate decks line-for-line —
otherwise a sweep "baseline" silently diverges from the authoritative
``verify_complex_*.py`` gate. These canaries assert that, normalized for
whitespace/comments, the sweep builder line-set equals the REAL ship-gate deck
for BOTH the DirectNet and the NGSPICE-ground-truth side of every circuit:

  opamp / ring / switchcap / sram, ×
  DirectNet deck        : sweep builder == the single-point ``*_deck(bt)``
  NGSPICE ground truth  : sweep builder body == the single-point ``*_body`` body

Crucially this diffs against the **actual** single-point deck-producing
functions imported from the verify scripts (``directnet_opamp_deck``,
``ngspice_opamp_body``, …) — NOT a hand-copied replica that can silently drift
from the ship-gate rewrite logic (bug report B8). The single-point and sweep
NGSPICE bodies share the SAME baked modelcard here, so the ``.include`` line is
identical and is compared like every other line.

No simulation is run; this is a pure string-equivalence guard. Run cheap, often.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

print = functools.partial(print, flush=True)  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))

from tests.common.complex import (  # noqa: E402
    BENCH, BENCH_TECHS,
    OpAmpParams, RingOscParams, SwitchCapParams, SramParams,
    directnet_opamp, directnet_ringosc, directnet_switchcap, directnet_sram_lobe,
    ngspice_opamp, ngspice_ringosc, ngspice_switchcap, ngspice_sram_lobe,
    get_baked_modelcard,
)
# REAL single-point ship-gate deck producers (B8): diff the sweep builders
# against these, not against hand-copied replicas.
from tests.verify_complex_opamp import (  # noqa: E402
    directnet_opamp_deck, ngspice_opamp_body)
from tests.verify_complex_ring_osc import (  # noqa: E402
    directnet_ring_deck, ngspice_ring_body)
from tests.verify_complex_switchcap import (  # noqa: E402
    directnet_sc_deck, ngspice_sc_body)
from tests.verify_complex_sram_snm import (  # noqa: E402
    directnet_sram_lobe_deck, ngspice_sram_lobe_body)

# The sweep ring/switchcap probe window the orchestrator uses at baseline.
RING_TSTOP = 1.2e-9


def norm_set(text: str) -> set:
    return set(" ".join(l.split()) for l in text.splitlines()
              if l.strip() and not l.strip().startswith("*"))


def _diff(tag: str, side: str, sweep_text: str, ship_text: str) -> int:
    """Compare a sweep builder's line-set against the REAL ship-gate deck.

    Returns 1 on mismatch (and prints the asymmetric diff), 0 on identity.
    """
    s, r = norm_set(sweep_text), norm_set(ship_text)
    if s != r:
        print(f"  [{side}] {tag:9s} FAIL")
        print("     sweep-only:", s - r)
        print("     ship-only :", r - s)
        return 1
    print(f"  [{side}] {tag:9s} line-set identical (vs REAL ship-gate deck)")
    return 0


def main() -> int:
    fails = 0
    print("=" * 70)
    print("Complex-circuit sweep builder equivalence canaries")
    print("  (sweep baseline builders  ==  REAL single-point ship-gate decks)")
    print("=" * 70)

    for name in BENCH_TECHS:
        bt = BENCH[name]
        print(f"\n--- {name} ---")
        wd = PROJECT_ROOT / "tests" / "verify_complex_results" / "_canary" / name
        try:
            baked = get_baked_modelcard(bt, bt.nfin, wd,
                                        nfin_p=bt.effective_nfin_p)
        except Exception as exc:  # noqa: BLE001
            print(f"  bake ERROR ({exc}) — cannot run NGSPICE-side checks")
            fails += 1
            continue

        # --- DirectNet decks: sweep builder == single-point ship-gate deck ---
        fails += _diff("opamp", "DN",
                       directnet_opamp(bt, OpAmpParams()),
                       directnet_opamp_deck(bt))
        fails += _diff("ring", "DN",
                       directnet_ringosc(bt, RingOscParams(), RING_TSTOP),
                       directnet_ring_deck(bt))
        fails += _diff("switchcap", "DN",
                       directnet_switchcap(bt, SwitchCapParams()),
                       directnet_sc_deck(bt))
        fails += _diff("sram", "DN",
                       directnet_sram_lobe(bt, SramParams(), bt.nfin),
                       directnet_sram_lobe_deck(bt, bt.nfin))

        # --- NGSPICE ground-truth bodies: sweep builder == single-point body --
        # Same baked modelcard is passed to both sides, so the `.include` line
        # is identical and is part of the compared line-set (B8: the old canary
        # only asserted the sweep body was non-degenerate and never compared).
        fails += _diff("opamp", "ng",
                       ngspice_opamp(bt, OpAmpParams(), baked)["body"],
                       ngspice_opamp_body(bt, baked)["body"])
        fails += _diff("ring", "ng",
                       ngspice_ringosc(bt, RingOscParams(), baked, RING_TSTOP)["body"],
                       ngspice_ring_body(bt, baked)["body"])
        fails += _diff("switchcap", "ng",
                       ngspice_switchcap(bt, SwitchCapParams(), baked)["body"],
                       ngspice_sc_body(bt, baked)["body"])
        fails += _diff("sram", "ng",
                       ngspice_sram_lobe(bt, SramParams(), bt.nfin, baked)["body"],
                       ngspice_sram_lobe_body(bt, bt.nfin, baked)["body"])

    print("\n" + "-" * 70)
    print(f"RESULT: {'ALL CANARIES PASS' if fails == 0 else f'{fails} CANARY FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
