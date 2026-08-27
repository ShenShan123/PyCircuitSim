"""Focused checks for the V7.6.0 same-state evaluator probe."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import numpy as np
import pytest

from examples.complex_circuits.pycircuitsim_bench import evaluator_probe
from pycircuitsim.circuit import Circuit
from pycircuitsim.models.passive import Resistor, VoltageSource


class _OSDIProbe:
    """Deterministic full/reduced evaluator with already-scaled public APIs."""

    def __init__(self, multiplier: float = 3.0) -> None:
        self.name = "XAMP.M1"
        self.nodes = ["out", "in", "0", "0"]
        self.m = multiplier
        self.L = 16e-9
        self.NFIN = 2.0
        self.temperature = 300.15
        self.evaluator_boundary = "native"

    def get_terminal_stamp(
        self, voltages: Dict[str, float],
    ) -> Tuple[List[float], np.ndarray]:
        del voltages
        currents = np.asarray([2.0, 1.0, -2.5, -0.5]) * self.m
        jacobian = np.asarray([
            [4.0, 5.0, -8.0, -1.0],
            [1.0, 2.0, -4.0, 1.0],
            [-4.0, -6.0, 11.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0],
        ]) * self.m
        return currents.tolist(), jacobian

    def get_charge_stamp(
        self, voltages: Dict[str, float],
    ) -> Tuple[List[float], np.ndarray]:
        del voltages
        charges = np.asarray([2.0, 1.0, -2.5, -0.5]) * 1e-15 * self.m
        jacobian = np.asarray([
            [4.0, 5.0, -8.0, -1.0],
            [1.0, 2.0, -4.0, 1.0],
            [-4.0, -6.0, 11.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0],
        ]) * 1e-15 * self.m
        return charges.tolist(), jacobian

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        del voltages
        return 2.0 * self.m

    def get_conductance(
        self, voltages: Dict[str, float],
    ) -> Tuple[float, float, float]:
        del voltages
        return 4.0 * self.m, 5.0 * self.m, -1.0 * self.m


class _DirectNetProbe:
    """Deterministic NN evaluator whose raw and production values differ."""

    def __init__(self, multiplier: float = 3.0) -> None:
        self.name = "xamp.m1"
        self.nodes = ["OUT", "IN", "0", "0"]
        self.m = multiplier
        self.L = 16e-9
        self.NFIN = 2.0
        self.temperature = 300.15
        self.evaluator_boundary = "native"

    def configure_evaluator(
        self, boundary: str, correction_trace: bool = False,
    ) -> None:
        del correction_trace
        self.evaluator_boundary = boundary

    def _result(self) -> Dict[str, float]:
        factor = 1.0 if self.evaluator_boundary == "raw-directnet" else 2.0
        return {
            "id": -2.0 * factor,
            "gm": 5.0 * factor,
            "gds": 4.0 * factor,
            "gmb": -1.0 * factor,
            "qg": 1e-15 * factor,
            "qd": 2e-15 * factor,
            "qs": -2.5e-15 * factor,
            "qb": -0.5e-15 * factor,
            "cgg": 2e-15 * factor,
            "cgd": 1e-15 * factor,
            "cgs": -3e-15 * factor,
            "cdg": 4e-15 * factor,
            "cdd": 5e-15 * factor,
        }

    def _eval(self, voltages: Dict[str, float]) -> Dict[str, float]:
        del voltages
        return self._result()

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        del voltages
        return -self.m * self._result()["id"]

    def get_conductance(
        self, voltages: Dict[str, float],
    ) -> Tuple[float, float, float]:
        del voltages
        result = self._result()
        return tuple(
            self.m * result[key] for key in ("gds", "gm", "gmb")
        )

    def get_charges(self, voltages: Dict[str, float]) -> Dict[str, float]:
        del voltages
        result = self._result()
        return {
            key: self.m * result[key] for key in ("qg", "qd", "qs", "qb")
        }

    def get_capacitances(self, voltages: Dict[str, float]) -> Dict[str, float]:
        del voltages
        result = self._result()
        return {
            key: self.m * result[key]
            for key in ("cgg", "cgd", "cgs", "cdg", "cdd")
        }


def test_device_probe_preserves_signs_shapes_closure_and_multiplier() -> None:
    osdi = _OSDIProbe(multiplier=3.0)
    directnet = _DirectNetProbe(multiplier=3.0)

    row = evaluator_probe.compare_device_at_state(
        osdi, directnet,
        {"out": 0.4, "in": 0.6, "0": 0.0},
        {"OUT": 0.4, "IN": 0.6, "0": 0.0},
    )

    assert row["instance"] == "xamp.m1"
    assert row["terminals"] == ["out", "in", "0", "0"]
    assert row["multiplier"] == 3.0
    full = row["full_osdi"]
    assert np.asarray(full["terminal_currents"]).shape == (4,)
    assert np.asarray(full["current_jacobian"]).shape == (4, 4)
    assert np.asarray(full["charges"]).shape == (4,)
    assert np.asarray(full["charge_jacobian"]).shape == (4, 4)
    assert sum(full["terminal_currents"]) == pytest.approx(0.0)
    assert sum(full["charges"]) == pytest.approx(0.0, abs=1e-28)
    assert full["terminal_currents"][0] == pytest.approx(6.0)
    assert row["reduced_osdi"]["osdi_terminal_id"] == pytest.approx(-6.0)
    assert row["reduced_osdi"]["drain_current"] == pytest.approx(6.0)

    raw = row["raw_directnet"]
    production = row["production_directnet"]
    assert raw["model_id"] == pytest.approx(-6.0)
    assert raw["drain_current"] == pytest.approx(6.0)
    assert production["model_id"] == pytest.approx(-12.0)
    assert raw["charges"]["qg"] == pytest.approx(3e-15)
    assert sum(raw["charges"].values()) == pytest.approx(0.0, abs=1e-28)


def test_device_alignment_rejects_topology_or_multiplier_mismatch() -> None:
    osdi = _OSDIProbe()
    directnet = _DirectNetProbe()
    pairs = evaluator_probe.align_devices([osdi], [directnet])
    assert pairs == [(osdi, directnet)]

    directnet.nodes[0] = "different"
    with pytest.raises(ValueError, match="terminal topology"):
        evaluator_probe.align_devices([osdi], [directnet])

    directnet.nodes[0] = "OUT"
    directnet.m = 7.0
    with pytest.raises(ValueError, match="multiplier"):
        evaluator_probe.align_devices([osdi], [directnet])


def test_circuit_residual_fits_voltage_source_tail_and_uses_current_scale() -> None:
    circuit = Circuit()
    circuit.add_component(VoltageSource("VDD", ["vdd", "0"], 1.0))
    circuit.add_component(Resistor("R1", ["vdd", "0"], 1e3))

    exact = evaluator_probe.circuit_residual(
        circuit, {"vdd": 1.0, "0": 0.0}, reltol=1e-3,
    )
    assert exact["residual_inf"] < 1e-12
    assert exact["current_rhs_scale"] == 0.0
    # The 1 V ideal-source constraint row must not inflate an ampere tolerance.
    assert exact["tolerance"] == pytest.approx(1e-6)
    assert exact["normalized"] < 1e-6

    wrong = evaluator_probe.circuit_residual(
        circuit, {"vdd": 0.5, "0": 0.0}, reltol=1e-3,
    )
    assert wrong["residual_inf"] == pytest.approx(0.5)
    assert wrong["normalized"] == pytest.approx(5e5)


def test_state_loader_accepts_explicit_and_unambiguous_run_result(
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "states.json"
    explicit.write_text(json.dumps({
        "states": [
            {"label": "ng-op", "nodes": {"VDD": 0.8, "out": 0.4}},
            {"label": "hot", "temperature_c": 125.0,
             "nodes": {"VDD": 0.8, "out": 0.3}},
        ],
    }))
    states = evaluator_probe.load_states(explicit)
    assert [state.label for state in states] == ["ng-op", "hot"]
    assert states[1].temperature_c == 125.0

    run_result = tmp_path / "result.json"
    run_result.write_text(json.dumps({
        "schema": "1",
        "pycircuitsim": {
            "operating_point": {"VDD": 0.8, "out": 0.4},
        },
    }))
    loaded = evaluator_probe.load_states(run_result)
    assert loaded[0].label == "pycircuitsim-operating-point"

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"states": [{"nodes": {"out": "nan"}}]}))
    with pytest.raises(ValueError, match="finite"):
        evaluator_probe.load_states(bad)

    ambiguous = tmp_path / "ambiguous.json"
    ambiguous.write_text(json.dumps({
        "schema": "1", "pycircuitsim": {"sweeps": [{}, {}]},
    }))
    with pytest.raises(ValueError, match="unambiguous"):
        evaluator_probe.load_states(ambiguous)


def test_cli_writes_probe_schema_and_provenance(tmp_path: Path) -> None:
    state_path = tmp_path / "states.json"
    state_path.write_text(json.dumps({
        "states": [{"label": "op", "nodes": {"0": 0.0}}],
    }))
    out = tmp_path / "probe.json"
    expected: Dict[str, Any] = {
        "schema": evaluator_probe.SCHEMA_VERSION,
        "provenance": {"state_source": {"sha256": "abc"}},
        "states": [],
    }
    with patch.object(evaluator_probe, "probe_deck", return_value=expected) as run:
        status = evaluator_probe.main([
            "--root", str(tmp_path), "--tech", "tsmc5",
            "--category", "ldo", "--design", "ldo_1",
            "--deck", "tb_line_max.cir", "--states", str(state_path),
            "--work", str(tmp_path / "work"), "--out", str(out),
        ])

    assert status == 0
    assert json.loads(out.read_text()) == expected
    assert run.call_args.kwargs["state_path"] == state_path

