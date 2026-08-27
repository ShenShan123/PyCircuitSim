"""Focused V7.6.0 checks for the exact reduced-OSDI diagnostic boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple
from unittest.mock import patch

import numpy as np
import pytest

from examples.complex_circuits.pycircuitsim_bench import DeckOptions, run_compare
from pycircuitsim import solver


class _StampProbe:
    """Minimal four-terminal device that records which solver seam was used."""

    def __init__(self, evaluator_boundary: str = "native") -> None:
        self.nodes = ["d", "g", "s", "b"]
        self.evaluator_boundary = evaluator_boundary
        self.full_calls = 0
        self.classic_calls = 0
        self.current_calls = 0

    def get_terminal_stamp(
        self, voltages: Dict[str, float]
    ) -> Tuple[list[float], np.ndarray]:
        self.full_calls += 1
        return [0.0] * 4, np.eye(4)

    def get_charge_stamp(
        self, voltages: Dict[str, float]
    ) -> Tuple[list[float], np.ndarray]:
        return [0.0] * 4, np.eye(4)

    def get_conductance(
        self, voltages: Dict[str, float]
    ) -> Tuple[float, float, float]:
        self.classic_calls += 1
        return 2.0, 3.0, 4.0

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        self.current_calls += 1
        return 5.0


def _stamp(device: _StampProbe) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((4, 4))
    rhs = np.zeros(4)
    solver._stamp_mosfet_dc(
        device,
        matrix,
        rhs,
        {"d": 0, "g": 1, "s": 2, "b": 3},
        {"d": 0.4, "g": 0.7, "s": 0.1, "b": 0.0},
        1e-12,
    )
    return matrix, rhs


def test_native_boundary_keeps_full_level72_stamp() -> None:
    device = _StampProbe()

    matrix, rhs = _stamp(device)

    assert device.full_calls == 1
    assert device.classic_calls == 0
    assert device.current_calls == 0
    expected = np.eye(4)
    for first, second in ((0, 2), (0, 3), (2, 3)):
        expected[first, first] += 1e-12
        expected[second, second] += 1e-12
        expected[first, second] -= 1e-12
        expected[second, first] -= 1e-12
    assert np.array_equal(matrix, expected)
    assert np.array_equal(rhs, expected @ np.array([0.4, 0.7, 0.1, 0.0]))
    assert solver._has_full_stamp_device(SimpleNamespace(components=[device]))


def test_reduced_osdi_boundary_uses_classic_drain_source_stamp() -> None:
    device = _StampProbe("reduced-osdi")

    _stamp(device)

    assert device.full_calls == 0
    assert device.classic_calls == 1
    assert device.current_calls == 1
    assert not solver._has_full_stamp_device(SimpleNamespace(components=[device]))


def test_parsed_level72_devices_are_configured_before_solving() -> None:
    device = SimpleNamespace(evaluator_boundary="native")
    circuit = SimpleNamespace(components=[device])
    opts = run_compare.SimOptions(evaluator_boundary="reduced-osdi")

    with patch.object(run_compare, "_mosfets", return_value=[device]):
        run_compare._configure_evaluator_boundary(circuit, 72, opts)

    assert device.evaluator_boundary == "reduced-osdi"


def test_boundary_validation_rejects_invalid_values_and_level_pair() -> None:
    with pytest.raises(ValueError, match="evaluator boundary"):
        run_compare.SimOptions(evaluator_boundary="unknown")

    opts = run_compare.SimOptions(evaluator_boundary="reduced-osdi")
    with pytest.raises(ValueError, match="LEVEL=72"):
        run_compare._validate_evaluator_boundary(73, opts)

    with pytest.raises(SystemExit) as exc:
        run_compare.main([
            "--category", "amplifier",
            "--model-level", "73",
            "--evaluator-boundary", "reduced-osdi",
        ])
    assert exc.value.code == 2


def test_boundary_is_emitted_in_options_and_model_provenance(
    tmp_path: Path,
) -> None:
    td = run_compare.TranslatedDeck(
        tech="tsmc5",
        category="amplifier",
        design="probe",
        deck="tb_gain.cir",
        design_dir=run_compare.BENCH_ROOT,
        netlist_text="",
        modelcard_path=run_compare.BENCH_ROOT / "unused.spice",
        plans=[],
        meas=[],
        nodesets={},
        ic={},
        params={},
        options=DeckOptions(),
        devices=0,
        multipliers={},
        stability=None,
        temp_c=None,
        warnings=[],
        model_level=72,
    )
    opts = run_compare.SimOptions(evaluator_boundary="reduced-osdi")

    assert run_compare._model_provenance(td) == {
        "family": "bsim_cmg", "level": 72, "tech": "tsmc5",
    }
    assert opts.evaluator_boundary == "reduced-osdi"
    assert run_compare._model_provenance(td, opts)["evaluator_boundary"] == (
        "reduced-osdi"
    )
    with patch.object(
        run_compare,
        "_reference_provenance",
        return_value={"family": "bsim_cmg", "level": 72},
    ):
        row = run_compare.compare_translated(td, tmp_path, opts)
    assert row["options"]["evaluator_boundary"] == "reduced-osdi"
    assert row["py_model"]["evaluator_boundary"] == "reduced-osdi"
