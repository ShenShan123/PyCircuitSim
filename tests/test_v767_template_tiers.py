"""Contracts for the difficulty-tiered template tree and its token defaults.

These tests exist because two specific mistakes are easy to make here and
neither shows up as an exception:

1. Parameterizing a frozen deck. The simple-v1 opamp cell is a published
   score. Turning its hard-coded sources into tokens must leave the rendered
   deck byte-identical, or the score silently measures a different circuit.

2. Freezing a derived default too early. ``VINP_SPEC`` defaults to the value
   of ``VCM``. If that default is resolved while the base substitution table
   is built, a caller that overrides ``VCM`` — the parametric opamp sweep does
   — has its override dropped, and the sweep quietly re-measures the nominal
   bias at every point.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.common.base import (
    CIRCUIT_TIERS, CONTROLS_DIR, TEMPLATES_DIR, deck_tokens, template_deck,
)
from tests.common.circuit_benchmarks import BENCH, OpAmpParams, directnet_opamp
from tests.common.simple_circuit_catalog import CASES, SIMPLE_V2, cases
from tests.common.simple_circuit_harness import CORNERS, render_case_decks


def _source_lines(deck: str, prefixes: tuple[str, ...]) -> list[str]:
    return [line for line in deck.splitlines() if line.startswith(prefixes)]


# ---------------------------------------------------------------------------
# Tier resolution
# ---------------------------------------------------------------------------
def test_every_case_template_lives_in_its_declared_tier() -> None:
    for case in CASES:
        assert case.tier in CIRCUIT_TIERS, case.case_id
        resolved = template_deck(case.template, tier=case.tier)
        assert resolved.is_file()
        assert resolved.parent.name == case.tier


def test_template_deck_rejects_unknown_names_and_tiers() -> None:
    with pytest.raises(FileNotFoundError):
        template_deck("no_such_topology.spice.tmpl")
    with pytest.raises(ValueError):
        template_deck("mosfet.spice.tmpl", tier="L9_imaginary")
    with pytest.raises(FileNotFoundError):
        template_deck("mosfet.spice.tmpl", tier="L4_systems")


def test_template_deck_rejects_a_topology_duplicated_across_tiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two copies of one topology is the drift this tree exists to prevent."""
    from tests.common import base

    tiers = {tier: tmp_path / tier for tier in CIRCUIT_TIERS}
    for directory in tiers.values():
        directory.mkdir()
    for tier in ("L2_stages", "L3_blocks"):
        (tiers[tier] / "dup.spice.tmpl").write_text("* dup\n.end\n")
    monkeypatch.setattr(base, "TIER_DIRS", tiers)
    with pytest.raises(ValueError, match="duplicated across tiers"):
        base.template_deck("dup.spice.tmpl")


def test_tier_directories_are_the_only_template_homes() -> None:
    owned = {path for tier in CIRCUIT_TIERS
             for path in (TEMPLATES_DIR / tier).glob("*.spice.tmpl")}
    owned |= set((TEMPLATES_DIR / "subcircuits").glob("*.spice.tmpl"))
    owned |= set(CONTROLS_DIR.glob("*.spice.tmpl"))
    assert set(TEMPLATES_DIR.rglob("*.spice.tmpl")) == owned


# ---------------------------------------------------------------------------
# Frozen simple-v1 rendering
# ---------------------------------------------------------------------------
def test_frozen_opamp_deck_still_renders_plain_dc_sources() -> None:
    """The scored opamp cell must not become an AC deck by accident."""
    case = next(c for c in CASES if c.case_id == "opamp")
    _, reference = render_case_decks(
        case, case.analyses[0], BENCH["TSMC12"], CORNERS["nominal"],
        baked_lib=Path("/nonexistent/baked.lib"),
    )
    assert _source_lines(reference, ("Vdd ", "Vinn ", "Vinp ")) == [
        "Vdd vdd 0 0.8", "Vinn inn 0 0.44", "Vinp inp 0 0.44",
    ]


def test_frozen_current_mirror_output_biases_are_unchanged() -> None:
    case = next(c for c in CASES if c.case_id == "current_mirror")
    _, reference = render_case_decks(
        case, case.analyses[0], BENCH["TSMC12"], CORNERS["nominal"],
        baked_lib=Path("/nonexistent/baked.lib"),
    )
    assert _source_lines(reference, ("Voutn ", "Voutp ")) == [
        "Voutn on 0 0", "Voutp op 0 0.8",
    ]


# ---------------------------------------------------------------------------
# Derived source-spec defaults
# ---------------------------------------------------------------------------
def test_opamp_common_mode_override_reaches_the_rendered_deck() -> None:
    """A VCM override must survive the VINP_SPEC/VINN_SPEC indirection."""
    nominal = directnet_opamp(BENCH["TSMC12"], OpAmpParams())
    shifted = directnet_opamp(BENCH["TSMC12"], OpAmpParams(vcm_frac=0.70))
    assert _source_lines(nominal, ("Vinp ", "Vinn ")) == [
        "Vinn inn 0 0.44", "Vinp inp 0 0.44",
    ]
    assert _source_lines(shifted, ("Vinp ", "Vinn ")) == [
        "Vinn inn 0 0.56", "Vinp inp 0 0.56",
    ]


def test_rejection_analyses_drive_the_declared_stimulus() -> None:
    """Each rejection experiment must excite exactly one input."""
    case = next(c for c in CASES if c.case_id == "opamp_rejection")
    expected = {
        "differential_ac": ("AC=1", "AC=0", "AC=0"),
        "common_mode_ac": ("AC=1", "AC=1", "AC=0"),
        "supply_ac": ("AC=0", "AC=0", "AC=1"),
    }
    for analysis in case.analyses:
        _, reference = render_case_decks(
            case, analysis, BENCH["TSMC12"], CORNERS["nominal"],
            baked_lib=Path("/nonexistent/baked.lib"),
        )
        inp, inn, vdd = expected[analysis.name]
        lines = {line.split()[0]: line
                 for line in _source_lines(reference, ("Vdd ", "Vinp ", "Vinn "))}
        assert inp in lines["Vinp"], analysis.name
        assert inn in lines["Vinn"], analysis.name
        assert vdd in lines["Vdd"], analysis.name


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------
def test_closed_loop_systems_declare_a_cold_start_transient() -> None:
    """L4 is the only tier that exercises solving an OP before integrating."""
    systems = [case for case in cases(score_version=SIMPLE_V2)
               if case.tier == "L4_systems"]
    assert systems, "the L4 tier must not be empty"
    for case in systems:
        transients = [spec for spec in case.analyses if spec.kind == "tran"]
        assert transients, case.case_id
        assert any("uic" not in spec.card.lower() for spec in transients), \
            case.case_id
        # Only real cards count; the templates discuss `.ic` in comments.
        cards = [line.strip().lower()
                 for line in template_deck(
                     case.template, tier=case.tier).read_text().splitlines()
                 if not line.lstrip().startswith("*")]
        assert not [line for line in cards if line.startswith(".ic")], \
            f"{case.case_id}: a cold-start deck must not carry .ic"


def test_l4_templates_expose_no_body_token_they_do_not_use() -> None:
    for case in cases(score_version=SIMPLE_V2):
        if case.device_kinds:
            continue
        tokens = deck_tokens(
            template_deck(case.template, tier=case.tier).read_text())
        assert not [t for t in tokens if t.startswith("BODY_")], case.case_id
