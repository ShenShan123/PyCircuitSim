"""What the PyCircuitSim netlist parser actually implements.

The harness renders one deck for NGSPICE and one for PyCircuitSim from the
same template, then compares them for physical parity. That comparison assumes
both engines read every card the same way. Where they do not, the assumption is
silent — so this module states the parser's real support surface once, and the
compatibility guard in ``tests/test_deck_engine_compatibility.py`` holds every
rendered deck against it.

Facts here are read from ``pycircuitsim.parser`` where the parser exposes them
and asserted against it where it does not, so this file cannot drift into
describing a parser that no longer exists.
"""
from __future__ import annotations

import re
from typing import Dict, Set, Tuple

from pycircuitsim.parser import Parser

#: Directives ``Parser.parse_line``/``parse_file`` act on. ``.end`` is listed
#: because both engines treat it as a terminator, so it is safe in a deck.
IMPLEMENTED_DIRECTIVES: Set[str] = {
    ".ac", ".dc", ".end", ".ic", ".include", ".model", ".temp", ".tran",
}

#: ``.op`` has no branch in ``parse_line``, so it leaves ``analysis_type`` at
#: ``None`` — which ``simulation.run_simulation`` then reads as "run a single
#: DC operating point". The card is therefore honoured by both engines through
#: different mechanisms, not dropped. It is separated from the implemented set
#: so the equivalence is asserted rather than assumed.
IMPLIED_DIRECTIVES: Set[str] = {".op"}

#: Directives NGSPICE honours and ``parse_line`` drops without a diagnostic.
#: A deck carrying one of these makes the two engines solve different problems
#: while every text-level parity check still reports a match.
SILENTLY_IGNORED_DIRECTIVES: Set[str] = {
    ".disto", ".four", ".global", ".measure", ".meas", ".nodeset", ".noise",
    ".option", ".options", ".param", ".print", ".pz", ".save", ".sens",
}

#: SPICE scale factors and the magnitude NGSPICE gives them, measured on
#: NGSPICE 45.2 by reading back ``V1 out 0 DC <token>`` for each spelling.
#: This is the independent reference the parser is held against — the whole
#: point is that PyCircuitSim must not be merely self-consistent here.
#:
#: Until V7.6.9 ``m`` was 1e6 in ``Parser.UNIT_SUFFIXES`` and ``meg``/``mil``
#: were unparseable, so ``Rload out 0 1m`` was 1 MOhm to the candidate and
#: 1 mOhm to the reference on a byte-identical deck. Deck-to-deck parity
#: cannot see that, which is why the guard below scans values as well as
#: cards.
SPICE_SCALE_FACTORS: Dict[str, float] = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "mil": 25.4e-6,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}

#: A number with an optional scale factor and trailing unit text. Assignments
#: (``L=16n``), node names and keywords are excluded by anchoring the match to
#: the whole token.
_VALUE = re.compile(
    r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Z]*)$")


def spice_value(token: str) -> float:
    """Return the magnitude NGSPICE gives ``token``.

    Independent of ``Parser``: longest scale factor first, case-insensitive,
    trailing unit text ignored. ``value_disagrees_with_spice`` compares the
    two implementations rather than trusting either alone.
    """
    match = _VALUE.match(token.strip())
    if match is None:
        raise ValueError(f"not a SPICE value: {token!r}")
    number, suffix = match.group(1), match.group(2).lower()
    for spelling in sorted(SPICE_SCALE_FACTORS, key=len, reverse=True):
        if suffix.startswith(spelling):
            return float(number) * SPICE_SCALE_FACTORS[spelling]
    return float(number)


def value_disagrees_with_spice(token: str) -> bool:
    """Whether the two engines would read ``token`` as different magnitudes.

    A token that is not a numeric literal at all (a node name, a keyword, an
    assignment) is not a value and is reported as agreeing.
    """
    try:
        expected = spice_value(token)
    except ValueError:
        return False
    try:
        actual = Parser._parse_value(token)
    except ValueError:
        # NGSPICE reads it and this parser refuses: the deck cannot run on
        # both engines, which is the property being asked about.
        return True
    return abs(actual - expected) > 1e-12 * max(abs(expected), 1e-300)


def ambiguous_value_tokens(deck: str) -> Tuple[str, ...]:
    """Return every deck token whose magnitude depends on which engine reads it."""
    found: list[str] = []
    for line in deck.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("*"):
            continue
        for token in stripped.split():
            if value_disagrees_with_spice(token):
                found.append(token)
    return tuple(sorted(set(found)))
