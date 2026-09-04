"""Focused checks for the full-terminal solver boundary."""

from __future__ import annotations

import numpy as np
import pytest

from pycircuitsim import solver
from pycircuitsim.circuit import Circuit
from pycircuitsim.models.mosfet_bsimar_full import NMOS_TFF
from pycircuitsim.models.mosfet_directnet_full import NMOS_DNF


class _StampProbe:
    """Minimal device exposing the mandatory four-terminal stamp."""

    def __init__(self) -> None:
        self.name = "Mprobe"
        self.nodes = ["d", "g", "s", "b"]
        self.full_calls = 0

    def get_terminal_stamp(
        self, voltages: dict[str, float],
    ) -> tuple[list[float], np.ndarray]:
        del voltages
        self.full_calls += 1
        return [0.0] * 4, np.eye(4)


def test_mosfet_dc_uses_the_complete_terminal_stamp() -> None:
    device = _StampProbe()
    matrix = np.zeros((4, 4))
    rhs = np.zeros(4)

    solver._stamp_mosfet_dc(
        device, matrix, rhs,
        {"d": 0, "g": 1, "s": 2, "b": 3},
        {"d": 0.4, "g": 0.7, "s": 0.1, "b": 0.0},
        1e-12,
    )

    assert device.full_calls == 1
    expected = np.eye(4)
    for first, second in ((0, 2), (0, 3), (2, 3)):
        expected[first, first] += 1e-12
        expected[second, second] += 1e-12
        expected[first, second] -= 1e-12
        expected[second, first] -= 1e-12
    assert np.array_equal(matrix, expected)
    assert np.array_equal(rhs, expected @ np.array([0.4, 0.7, 0.1, 0.0]))


def test_mosfet_without_a_full_stamp_fails_loud() -> None:
    class MissingStamp:
        def __init__(self) -> None:
            self.name = "Mbad"
            self.nodes = ["d", "g", "s", "b"]

    device = MissingStamp()
    with pytest.raises(TypeError, match="get_terminal_stamp"):
        solver._stamp_mosfet_dc(
            device, np.zeros((4, 4)), np.zeros(4),
            {"d": 0, "g": 1, "s": 2, "b": 3},
            {"d": 0.4, "g": 0.7, "s": 0.1, "b": 0.0},
            1e-12,
        )


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
    device = object.__new__(device_type)
    circuit = Circuit()
    circuit.components.append(device)
    circuit.invalidate_topology()

    assert solver._has_full_terminal_nn_device(circuit)
    assert solver.DCSolver(circuit, dv_limit=override).dv_limit == expected
