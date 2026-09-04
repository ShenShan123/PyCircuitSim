"""Fail-closed contracts at the standalone subcircuit harness seam."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest

from tests.common.base import PROJECT_ROOT, SUBCIRCUIT_DECKS, render_template
from tests.common.simple_circuit_catalog import AnalysisSpec
from tests.common.simple_circuit_harness import (
    analysis_axis_limits,
    analysis_minimum_points,
)
from tests.simple_circuits.verify_subckt import (
    parse_deck,
    require_converged_solution,
    run_ac,
    validate_transient_results,
)


def test_unconverged_prerequisite_cannot_be_reported_as_a_result() -> None:
    solver = SimpleNamespace(_last_solve_converged=False)

    with pytest.raises(RuntimeError, match="did not converge"):
        require_converged_solution(solver, {"out": 0.5}, "AC operating point")


def test_truncated_hierarchical_transient_is_rejected() -> None:
    results = {
        "time": np.array([0.0, 1.0e-9, 2.0e-9]),
        "out": np.array([0.0, 0.5, 0.8]),
    }

    with pytest.raises(ValueError, match="requested stop"):
        validate_transient_results(results, tstop=3.0e-9, tstep=1.0e-9)


def test_hierarchical_ac_uses_the_complete_declared_decade_grid() -> None:
    deck = render_template(
        SUBCIRCUIT_DECKS / "rc_lowpass_hierarchical.spice.tmpl",
        {
            "TEMP": "27",
            "INPUT_DC": "0",
            "INPUT_AC": "1",
            "INPUT_PHASE": "0",
            "RESISTANCE": "1k",
            "CAPACITANCE": "1n",
            "ANALYSIS": ".ac dec 2 10 1k",
        },
    )

    artifact_root = PROJECT_ROOT / "results" / "tests" / "subckt_contracts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=artifact_root) as temporary:
        frequencies, results = run_ac(
            parse_deck(deck, Path(temporary) / "ac.sp"),
        )

    assert frequencies.size == 5
    assert frequencies[0] == pytest.approx(10.0)
    assert frequencies[-1] == pytest.approx(1.0e3)
    assert results["out"].size == frequencies.size


def test_production_ac_grid_matches_spice_cardinality() -> None:
    from pycircuitsim.simulation import build_ac_frequencies

    frequencies = build_ac_frequencies({
        "sweep_type": "dec",
        "num_points": 2,
        "fstart": 10.0,
        "fstop": 1.0e3,
    })

    assert frequencies.size == 5
    assert frequencies[0] == pytest.approx(10.0)
    assert frequencies[-1] == pytest.approx(1.0e3)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (
            {
                "sweep_type": "dec",
                "num_points": 2,
                "fstart": 10.0,
                "fstop": 31.55,
            },
            np.array([10.0, 31.6227766016838]),
        ),
        (
            {
                "sweep_type": "oct",
                "num_points": 2,
                "fstart": 10.0,
                "fstop": 14.13,
            },
            np.array([10.0, 14.142135623731]),
        ),
        (
            {
                "sweep_type": "dec",
                "num_points": 2,
                "fstart": 10.0,
                "fstop": 50.0,
            },
            np.array([10.0, 31.6227766016838]),
        ),
        (
            {
                "sweep_type": "dec",
                "num_points": 3,
                "fstart": 10.0,
                "fstop": 50.0,
            },
            np.array([10.0, 21.5443469003188, 46.4158883361278]),
        ),
        (
            {
                "sweep_type": "dec",
                "num_points": 2,
                "fstart": 10.0,
                "fstop": 200.0,
            },
            np.array([10.0, 44.7213595499958, 200.0]),
        ),
        (
            {
                "sweep_type": "dec",
                "num_points": 2,
                "fstart": 10.0,
                "fstop": 999.0,
            },
            np.array([10.0, 46.4004112, 215.299816, 999.0]),
        ),
        (
            {
                "sweep_type": "dec",
                "num_points": 3,
                "fstart": 10.0,
                "fstop": 999.0,
            },
            np.array([
                10.0,
                25.1138385,
                63.0704886,
                158.394207,
                397.788653,
                999.0,
            ]),
        ),
        (
            {
                "sweep_type": "oct",
                "num_points": 2,
                "fstart": 10.0,
                "fstop": 70.0,
            },
            np.array([
                10.0,
                14.142135623731,
                20.0,
                28.284271247462,
                40.0,
                56.568542494924,
            ]),
        ),
    ],
)
def test_production_ac_grid_matches_ngspice_fractional_bands(
    params: dict[str, float | int | str],
    expected: np.ndarray,
) -> None:
    from pycircuitsim.simulation import build_ac_frequencies

    assert build_ac_frequencies(params) == pytest.approx(expected)


def test_catalog_ac_contract_uses_the_last_native_fractional_point() -> None:
    analysis = AnalysisSpec(
        "fractional_oct",
        "ac",
        "ac oct 2 10 70",
        ("v(out)",),
    )

    assert analysis_axis_limits(analysis) == pytest.approx(
        (10.0, 56.568542494924),
    )
    assert analysis_minimum_points(analysis) == 6


@pytest.mark.parametrize(
    ("fstop", "expected_count"),
    [(315.3, 3), (316.2, 3), (316.3, 4)],
)
def test_long_decade_cardinality_uses_the_untolerated_bound(
    fstop: float,
    expected_count: int,
) -> None:
    from pycircuitsim.simulation import build_ac_frequencies

    frequencies = build_ac_frequencies({
        "sweep_type": "dec",
        "num_points": 2,
        "fstart": 10.0,
        "fstop": fstop,
    })

    assert frequencies.size == expected_count
