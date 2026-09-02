"""Focused checks for the reduced-OSDI solver boundary."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import pytest

from pycircuitsim import solver
from pycircuitsim.models.mosfet_bsimar_full import NMOS_TFF
from pycircuitsim.models.mosfet_directnet_full import NMOS_DNF


class _StampProbe:
    """Minimal four-terminal device that records which solver seam was used."""

    def __init__(self, evaluator_boundary: str = "native") -> None:
        self.nodes = ["d", "g", "s", "b"]
        self.evaluator_boundary = evaluator_boundary
        self.full_calls = 0
        self.classic_calls = 0
        self.current_calls = 0

    def get_terminal_stamp(
        self, voltages: Dict[str, float],
    ) -> Tuple[list[float], np.ndarray]:
        del voltages
        self.full_calls += 1
        return [0.0] * 4, np.eye(4)

    def get_charge_stamp(
        self, voltages: Dict[str, float],
    ) -> Tuple[list[float], np.ndarray]:
        del voltages
        return [0.0] * 4, np.eye(4)

    def get_conductance(
        self, voltages: Dict[str, float],
    ) -> Tuple[float, float, float]:
        del voltages
        self.classic_calls += 1
        return 2.0, 3.0, 4.0

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        del voltages
        self.current_calls += 1
        return 5.0


def _stamp(device: _StampProbe) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((4, 4))
    rhs = np.zeros(4)
    solver._stamp_mosfet_dc(
        device, matrix, rhs,
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


@pytest.mark.parametrize(
    ("device_type", "override", "expected"),
    (
        (NMOS_DNF, None, 0.1),
        (NMOS_DNF, 0.25, 0.25),
        (NMOS_TFF, None, 0.1),
        (NMOS_TFF, 0.25, 0.25),
    ),
)
def test_full_terminal_nn_gets_support_safe_newton_step(
    device_type: type,
    override: float | None,
    expected: float,
) -> None:
    """LEVEL=75/76 stay in support unless the caller overrides the cap."""
    device = object.__new__(device_type)
    circuit = SimpleNamespace(components=[device], _topo_version=0)

    assert solver._has_directnet_full_device(circuit)
    assert solver.DCSolver(circuit, dv_limit=override).dv_limit == expected
