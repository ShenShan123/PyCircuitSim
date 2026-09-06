"""One technology fact, three hand-maintained registries.

``tests/common/base.py:ALL_TECHS`` is the nominal master, ``BENCH`` in
``circuit_benchmarks.py`` feeds the catalog and five of the seven campaign
device suites, and ``ALL_TEST_TECHS`` in ``nn_gate.py`` feeds the parametric
device and inverter suites.  They are separately typed tables of the same
facts, so a later cleanup of any one of them would move a bias point in one
part of the scoreboard and not the others with nothing firing.  The V7.7.2
audit (A2) found they agree today and diverge on exactly one thing, the
inverter geometry; this module pins both the agreement and the divergence.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.v710_regate_jobs import DEVICE_SUITES
from scripts.v730_coverage import NON_SIMPLE_SUITES
from tests.common.base import ALL_TECHS
from tests.common.circuit_benchmarks import BENCH, BENCH_TECHS
from tests.common.nn_gate import ALL_TEST_TECHS, INV_CLOAD
from tests.common.simple_circuit_catalog import get_case
from tests.common.simple_circuit_harness import CORNERS, render_case_decks

CAMPAIGN_TECHS = ("TSMC5", "TSMC6", "TSMC7", "TSMC12", "TSMC16")


def test_campaign_technology_set_is_the_benchmark_set() -> None:
    assert tuple(BENCH_TECHS) == CAMPAIGN_TECHS
    assert set(CAMPAIGN_TECHS) <= set(ALL_TECHS)
    assert set(CAMPAIGN_TECHS) <= set(ALL_TEST_TECHS)


@pytest.mark.parametrize("tech", CAMPAIGN_TECHS)
def test_registries_agree_on_supply_vt_models_and_device_geometry(
    tech: str,
) -> None:
    master = ALL_TECHS[tech]
    bench = BENCH[tech]
    legacy = ALL_TEST_TECHS[tech]
    pair = master.default_vt_pair

    assert bench.vdd == legacy.vdd == master.vdd
    assert bench.tfin == legacy.tfin == master.tfin
    assert (bench.vt == bench.effective_nmos_vt == bench.effective_pmos_vt
            == legacy.nn_vt == legacy.effective_pmos_vt == master.default_vt)
    assert bench.nmos_model == legacy.nmos_model == pair.nmos_model
    assert bench.pmos_model == legacy.pmos_model == pair.pmos_model
    assert bench.nn_tech == legacy.nn_tech_key == tech.lower()
    assert bench.nfin == bench.effective_nfin_p == legacy.nfin == master.default_nfin
    assert bench.l_nmos == legacy.l_nmos == master.default_l_nmos == 16e-9
    assert bench.l_pmos == legacy.effective_l_pmos == master.default_l_pmos == 20e-9


@pytest.mark.parametrize("tech", CAMPAIGN_TECHS)
def test_default_length_membership_in_the_sweep_set_is_declared(tech: str) -> None:
    """A default outside its own sweep list is allowed only where recorded.

    TSMC7 dropped L=16 nm from ``l_values`` because the symmetric 16/16 ULVT
    inverter diverged in the reference; the campaign's asymmetric 16/20 nm
    default still resolves to a modelcard bin, so the default stayed.  That
    exception is asserted here rather than left as a comment, so a cleanup of
    either list is a visible decision.
    """
    master = ALL_TECHS[tech]
    assert master.default_l_pmos in master.l_values
    assert (master.default_l_nmos in master.l_values) == (tech != "TSMC7")


def test_inverter_geometry_divergence_is_exactly_the_declared_table() -> None:
    """The catalog and the parametric suite report two different inverters.

    Both results are labelled "CMOS inverter" in one campaign.  The
    parametric suite aligns its lengths to the per-technology training bins
    and drives 1 fF; the catalog keeps the 16/20 nm benchmark device and
    drives the shared 5 fF logic load.  The divergence is deliberate and is
    pinned so neither side can drift toward the other silently.
    """
    parametric = {
        tech: (ALL_TEST_TECHS[tech].effective_inv_l_nmos,
               ALL_TEST_TECHS[tech].effective_inv_l_pmos)
        for tech in CAMPAIGN_TECHS
    }
    assert parametric == {
        "TSMC5": (20e-9, 20e-9), "TSMC6": (20e-9, 20e-9),
        "TSMC7": (20e-9, 20e-9), "TSMC12": (16e-9, 20e-9),
        "TSMC16": (16e-9, 20e-9),
    }
    assert INV_CLOAD == 1e-15

    case = get_case("inverter_energy")
    switching = next(item for item in case.analyses if item.kind == "tran")
    for tech in CAMPAIGN_TECHS:
        candidate, reference = render_case_decks(
            case, switching, BENCH[tech], CORNERS["nominal"],
            baked_lib=Path("/frozen/lib"), model_level=75,
        )
        for deck in (candidate, reference):
            assert "Cload out 0 5f" in deck
        # The reference bakes geometry into its modelcard; the candidate
        # carries it on the device line.
        assert "L=16n NFIN=2" in candidate and "L=20n NFIN=2" in candidate


def test_coverage_denominator_is_the_dispatched_device_suite_list() -> None:
    """The coverage tool and the job generator must not name two suite sets."""
    assert list(NON_SIMPLE_SUITES) == list(DEVICE_SUITES)
    assert all(omps == ("1",) for omps in NON_SIMPLE_SUITES.values())
