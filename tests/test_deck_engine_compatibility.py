"""Every card in a rendered deck must mean the same thing to both engines.

``physical_deck_mismatch`` compares the candidate deck against the reference
deck. That answers "are these two decks the same text?" — it cannot answer "do
NGSPICE and PyCircuitSim read this text the same way?". Two identical decks
still solve two different problems whenever a card is honoured by one engine
and dropped or reinterpreted by the other, and the parity check reports a
clean match while it happens.

Two such seams exist today and neither had a test:

1. ``Parser.parse_line`` deliberately ignores every ``.``-directive it does not
   implement (``.end``, ``.option``, ``.measure``, ... fall through in
   silence). A template that gained ``.options rshunt=1e12`` or ``.nodeset``
   would change the NGSPICE problem and not the PyCircuitSim one.
2. ``Parser.UNIT_SUFFIXES`` read ``m`` as mega where SPICE — and therefore the
   NGSPICE reference — reads it as milli, and refused ``meg``, ``mil`` and any
   trailing unit text. ``Rload out 0 1m`` was 1 MOhm to the candidate and
   1 mOhm to the reference on a byte-identical deck. V7.6.9 corrected the
   parser to SPICE semantics; the tests below now hold it there, against a
   scale-factor table measured on NGSPICE 45.2 rather than against the parser
   itself.

This module keeps both seams closed for everything the harness can render.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Set, Tuple

import pytest

from tests.common.base import TEMPLATES_DIR, deck_tokens
from tests.common.circuit_benchmarks import BENCH
from tests.common.parser_support import (
    IMPLEMENTED_DIRECTIVES, IMPLIED_DIRECTIVES, SPICE_SCALE_FACTORS,
    SILENTLY_IGNORED_DIRECTIVES, ambiguous_value_tokens, spice_value,
)
from tests.common.simple_circuit_catalog import CASES
from tests.common.simple_circuit_harness import CORNERS, render_case_decks

#: Directives a template may carry. ``.subckt``/``.ends`` are expansion cards
#: handled in ``parse_file`` rather than ``parse_line``; ``.op`` is honoured
#: through the no-analysis fallback (see ``IMPLIED_DIRECTIVES``); the rest come
#: straight from the ``parse_line`` dispatch.
ALLOWED_DIRECTIVES: Set[str] = (
    IMPLEMENTED_DIRECTIVES | IMPLIED_DIRECTIVES | {".subckt", ".ends"}
)

_DIRECTIVE = re.compile(r"^\s*(\.[a-zA-Z_]+)")


def _directives(deck: str) -> Set[str]:
    """Return the lowercased ``.``-directives a deck actually carries."""
    found: Set[str] = set()
    for line in deck.splitlines():
        if line.lstrip().startswith("*"):
            continue
        match = _DIRECTIVE.match(line)
        if match:
            found.add(match.group(1).lower())
    return found


def _rendered_catalog_decks() -> Iterator[Tuple[str, str]]:
    """Yield ``(label, deck)`` for both engines of every catalog analysis."""
    baked = Path("/frozen/compatibility.lib")
    for case in CASES:
        for analysis in case.analyses:
            candidate, reference = render_case_decks(
                case, analysis, BENCH["TSMC12"], CORNERS["nominal"],
                baked_lib=baked,
            )
            label = f"{case.case_id}/{analysis.name}"
            yield f"{label} candidate", candidate
            yield f"{label} reference", reference


# ---------------------------------------------------------------------------
# Directive support
# ---------------------------------------------------------------------------
def test_template_sources_carry_only_implemented_directives() -> None:
    """A fixture or control template is scanned as written, tokens and all."""
    offenders: dict[str, Set[str]] = {}
    for path in sorted(TEMPLATES_DIR.rglob("*.spice.tmpl")):
        unsupported = _directives(path.read_text()) - ALLOWED_DIRECTIVES
        if unsupported:
            offenders[str(path.relative_to(TEMPLATES_DIR))] = unsupported

    assert not offenders, (
        "templates carry directives PyCircuitSim ignores while NGSPICE honours "
        f"them: {offenders}"
    )


def test_rendered_catalog_decks_carry_only_implemented_directives() -> None:
    """The injected ``MODEL_SETUP`` and ``ANALYSIS`` cards are scanned too."""
    offenders: dict[str, Set[str]] = {}
    for label, deck in _rendered_catalog_decks():
        unsupported = _directives(deck) - ALLOWED_DIRECTIVES
        if unsupported:
            offenders[label] = unsupported

    assert not offenders, f"unsupported directives reach a rendered deck: {offenders}"


def test_op_card_is_honoured_through_the_no_analysis_fallback(
    tmp_path: Path,
) -> None:
    """``.op`` reaches 8 candidate decks and has no branch in the dispatch.

    It is nonetheless not a divergence: the card leaves ``analysis_type`` at
    ``None``, and ``run_simulation`` reads that as "single DC operating point".
    Pin both halves — if a later change gives ``analysis_type`` a non-``None``
    default, ``.op`` becomes a silently dropped analysis and this fails.
    """
    from pycircuitsim.parser import Parser

    deck = tmp_path / "op.sp"
    deck.write_text("* op only\nV1 in 0 1\nR1 in 0 1k\n.op\n.end\n")
    parser = Parser()
    parser.parse_file(str(deck))

    assert parser.analysis_type is None
    assert IMPLIED_DIRECTIVES == {".op"}
    assert ".op" not in IMPLEMENTED_DIRECTIVES


def test_dropping_a_physics_changing_directive_is_announced(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Silence is the defect: NGSPICE applies these and this parser does not.

    ``.options cshunt`` is the worked example — NGSPICE stamps a capacitor
    from every node to ground at parse time, which was measured at 14 % on an
    amplifier slew rate in V7.5.10. The card still parses (real decks carry
    it), but the run says so.
    """
    from pycircuitsim.parser import Parser

    deck = tmp_path / "ignored.sp"
    deck.write_text(
        "* ignored directives\n"
        "V1 in 0 1\n"
        "R1 in 0 1k\n"
        ".options cshunt=1e-14\n"
        ".options rshunt=1e12\n"
        ".nodeset V(in)=0.5\n"
        ".print v(in)\n"
        ".op\n"
        ".end\n"
    )
    parser = Parser()
    with caplog.at_level("WARNING", logger="pycircuitsim.parser"):
        parser.parse_file(str(deck))

    warned = {record.args[0] for record in caplog.records}
    assert warned == {".options", ".nodeset"}, "one warning per dropped card"
    assert [item.name for item in parser.circuit.components] == ["V1", "R1"]


@pytest.mark.parametrize("directive", (".print", ".save", ".measure", ".end"))
def test_presentation_only_directive_is_dropped_without_noise(
    directive: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """These change no circuit, so warning about them would train the eye off."""
    from pycircuitsim.parser import Parser

    deck = tmp_path / "quiet.sp"
    deck.write_text(
        f"* quiet\nV1 in 0 1\nR1 in 0 1k\n{directive} v(in)\n.op\n.end\n"
    )
    with caplog.at_level("WARNING", logger="pycircuitsim.parser"):
        Parser().parse_file(str(deck))

    assert caplog.records == []


def test_physical_directive_set_covers_the_guarded_directives() -> None:
    """The parser's warn list and this module's guard list must not drift."""
    from pycircuitsim.parser import Parser

    assert Parser.PHYSICAL_DIRECTIVES <= SILENTLY_IGNORED_DIRECTIVES


@pytest.mark.parametrize("directive", sorted(SILENTLY_IGNORED_DIRECTIVES))
def test_named_unsupported_directive_is_still_unsupported(directive: str) -> None:
    """Pin the list this guard defends against to the parser's real behaviour.

    If one of these gains an implementation, this test fails and the guard's
    allow-list must be widened deliberately rather than by a template that
    happened to start using it.
    """
    assert directive not in IMPLEMENTED_DIRECTIVES


# ---------------------------------------------------------------------------
# Value syntax
# ---------------------------------------------------------------------------
def test_template_sources_avoid_engine_divergent_value_suffixes() -> None:
    offenders: dict[str, Tuple[str, ...]] = {}
    for path in sorted(TEMPLATES_DIR.rglob("*.spice.tmpl")):
        text = path.read_text()
        # Placeholders are substituted before a simulator sees them; the
        # rendered-deck scan below covers whatever they resolve to.
        for token in deck_tokens(text):
            text = text.replace(f"<{token}>", " ")
        ambiguous = ambiguous_value_tokens(text)
        if ambiguous:
            offenders[str(path.relative_to(TEMPLATES_DIR))] = ambiguous

    assert not offenders, (
        "templates use a value suffix NGSPICE and PyCircuitSim read "
        f"differently: {offenders}"
    )


def test_rendered_catalog_decks_avoid_engine_divergent_value_suffixes() -> None:
    offenders: dict[str, Tuple[str, ...]] = {}
    for label, deck in _rendered_catalog_decks():
        ambiguous = ambiguous_value_tokens(deck)
        if ambiguous:
            offenders[label] = ambiguous

    assert not offenders, (
        f"a rendered deck uses an engine-divergent value suffix: {offenders}"
    )


@pytest.mark.parametrize(
    ("token", "expected"),
    (
        ("1m", 1e-3),        # milli, not mega
        ("2.5M", 2.5e-3),    # scale factors are case-insensitive
        ("1meg", 1e6),
        ("1MEG", 1e6),
        ("1mil", 25.4e-6),
        ("2.2kohm", 2.2e3),  # trailing unit text is ignored
        ("10uF", 10e-6),
        ("1megohm", 1e6),
        ("1k", 1e3),
        ("10n", 10e-9),
        ("10f", 10e-15),
        ("1e-3", 1e-3),
        ("1e3", 1e3),
        ("0.7", 0.7),
        (".5n", 0.5e-9),
        ("-2.5u", -2.5e-6),
    ),
)
def test_parser_reads_every_scale_factor_the_way_ngspice_does(
    token: str,
    expected: float,
) -> None:
    """The expected column was read back from NGSPICE 45.2, not from us.

    Each value was measured by sourcing ``V1 out 0 DC <token>`` into NGSPICE
    and reading ``v(out)``. A parser that is merely self-consistent here still
    simulates a different circuit than the reference does.
    """
    from pycircuitsim.parser import Parser

    assert Parser._parse_value(token) == pytest.approx(expected, rel=1e-12)
    assert spice_value(token) == pytest.approx(expected, rel=1e-12)
    assert ambiguous_value_tokens(f"R1 a b {token}\n") == ()


@pytest.mark.parametrize("suffix", sorted(SPICE_SCALE_FACTORS))
def test_every_declared_scale_factor_is_implemented(suffix: str) -> None:
    """A scale factor NGSPICE honours must not be a parse error here."""
    from pycircuitsim.parser import Parser

    assert Parser._parse_value(f"1{suffix}") == pytest.approx(
        SPICE_SCALE_FACTORS[suffix], rel=1e-12,
    )


@pytest.mark.parametrize(
    ("expression", "params", "expected"),
    (
        ("1k", {}, 1e3),
        ("1m", {}, 1e-3),
        ("2meg", {}, 2e6),
        ("2.2kohm", {}, 2.2e3),
        ("10n", {}, 10e-9),          # repr() is "1e-08"; the exponent is not
        ("1.5u*2", {}, 3e-6),        # a parameter name
        ("1e-3*2", {}, 2e-3),        # nor is a scientific-notation exponent
        ("2*1e3", {}, 2e3),
        ("3*1k+2", {}, 3002.0),
        ("(1k+1k)/2", {}, 1e3),
        ("2*RBASE", {"RBASE": 1000.0}, 2000.0),
        ("GAIN*RV", {"GAIN": 3.0, "RV": 1e3}, 3000.0),
    ),
)
def test_subcircuit_expressions_read_values_like_a_card_does(
    expression: str,
    params: dict[str, float],
    expected: float,
) -> None:
    """A ``{...}`` parameter must scale exactly as the same literal on a card.

    The expression evaluator carried its own suffix regex, so it inherited the
    pre-V7.6.9 ``m``-is-mega reading and could not evaluate ``10n`` or
    ``1e-3`` at all — the substituted float's own exponent was looked up as a
    parameter name. A parameterized ``.subckt`` is the only place a value
    reaches the circuit through this path.
    """
    from pycircuitsim.parser import Parser

    assert Parser()._eval_expr(expression, params) == pytest.approx(
        expected, rel=1e-12,
    )


@pytest.mark.parametrize(
    "expression", ("__import__('os')", "NOSUCH", "1/0", "R1.attr"),
)
def test_subcircuit_expression_rejects_anything_it_cannot_evaluate(
    expression: str,
) -> None:
    """The evaluator reaches ``eval``; a silent wrong number is the hazard."""
    from pycircuitsim.parser import Parser

    with pytest.raises(ValueError):
        Parser()._eval_expr(expression, {"R1": 1.0})


@pytest.mark.parametrize("token", ("PULSE", "vdd", "L=16n", "DC=0.8"))
def test_value_scanner_ignores_tokens_that_are_not_values(token: str) -> None:
    """A false positive would block a valid deck, so the scanner must be exact."""
    assert ambiguous_value_tokens(f"R1 a b {token}\n") == ()


def test_value_scanner_flags_a_parser_that_drifts_from_spice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must be able to fail, or it is documentation.

    Restoring the pre-V7.6.9 reading of ``m`` has to be caught.
    """
    from pycircuitsim.parser import Parser

    monkeypatch.setattr(
        Parser, "UNIT_SUFFIXES",
        tuple(("m", 1e6) if s == "m" else (s, v)
              for s, v in Parser.UNIT_SUFFIXES),
    )

    assert ambiguous_value_tokens("R1 a b 1m\n") == ("1m",)
