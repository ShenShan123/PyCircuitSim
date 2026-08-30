"""Proof contracts for the V7.6.8 NGSPICE-grounded LDO harvester."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pycircuitsim.models.passive import CurrentSource, Resistor, VoltageSource
from scripts.harvest_dnf_unrolled_ldo import (
    EXPECTED_STATES,
    GMIN_TERMINAL_PAIRS,
    aligned_sweep_state,
    build_parser,
    compute_arm_scale,
    harvest,
    partition_nodes,
    terminal_incidence,
    validate_device_support,
    validate_sweep_plan,
)


class _Circuit:
    def __init__(self, nodes: list[str], components: list[object]) -> None:
        self._nodes = nodes
        self.components = components

    def get_nodes(self) -> list[str]:
        return list(self._nodes)


class _FakeDevice:
    def __init__(self) -> None:
        self.name = "Mpass"
        self.nodes = ["out", "gate", "vdd", "0"]
        self.L = 8e-9
        self.NFIN = 10.0
        self.temperature = 300.15
        self.m = 100.0
        self._norm_stats = SimpleNamespace(
            input_min=np.asarray([-1.0, -1.0, 0.0, -1.0, 0.0, 1e-9, 250.0]),
            input_max=np.asarray([1.0, 1.0, 0.0, 1.0, 8.0, 1e-6, 400.0]),
        )

    def _raw_inputs(self, state: dict[str, float]) -> np.ndarray:
        source = state["vdd"]
        return np.asarray([
            state["out"] - source,
            state["gate"] - source,
            0.0,
            -source,
            np.log2(self.NFIN),
            self.L,
            self.temperature,
        ])

    def get_terminal_stamp(
        self, state: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        del state
        return np.asarray([2e-3, 1e-4, -2.1e-3, 0.0]), np.zeros((4, 4))


def test_voltage_sources_become_fixed_nodes_without_an_mna_tail() -> None:
    circuit = _Circuit(
        ["vdd", "out", "gate", "vref"],
        [VoltageSource("VDD", ["vdd", "0"], 0.65),
         VoltageSource("VREF", ["0", "vref"], -0.2)],
    )
    free, fixed = partition_nodes(circuit)
    assert free == ["out", "gate"]
    assert fixed == ["0", "vdd", "vref"]

    unsupported = _Circuit(
        ["a", "b"], [VoltageSource("VFLOAT", ["a", "b"], 0.1)])
    with pytest.raises(ValueError, match="unsupported voltage-source topology"):
        partition_nodes(unsupported)


def test_raw_truth_requires_every_translated_physical_node() -> None:
    circuit = _Circuit(["VDD", "out"], [])
    sweep = SimpleNamespace(v={
        "vdd": np.asarray([0.65]),
        "out": np.asarray([0.48]),
        "n.x1.mm1#di": np.asarray([0.2]),
    })
    assert aligned_sweep_state(circuit, sweep, 0) == {
        "0": 0.0, "VDD": 0.65, "out": 0.48,
    }
    del sweep.v["out"]
    with pytest.raises(ValueError, match="missing=.*out"):
        aligned_sweep_state(circuit, sweep, 0)


def test_terminal_incidence_has_exactly_one_partition_per_terminal() -> None:
    free, fixed = terminal_incidence(
        [["out", "gate", "vdd", "0"]],
        ["out", "gate"], ["0", "vdd"],
    )
    assert free.tolist() == [[0, 1, -1, -1]]
    assert fixed.tolist() == [[-1, -1, 1, 0]]
    assert np.all((free >= 0) ^ (fixed >= 0))

    with pytest.raises(ValueError, match="maps to 0 partitions"):
        terminal_incidence([["out", "missing"]], ["out"], ["0"])


def test_parent_support_is_a_hard_boundary_not_a_clamp() -> None:
    device = _FakeDevice()
    inside = {"out": 0.48, "gate": 0.3, "vdd": 0.65, "0": 0.0}
    lower, upper, margin = validate_device_support([inside], [device])
    assert lower.shape == upper.shape == (1, 7)
    assert margin.shape == (1,)
    assert margin[0] > 0.0

    outside = dict(inside, out=2.0)
    with pytest.raises(ValueError, match="support excursion"):
        validate_device_support([outside], [device])


def test_line_sweep_identity_and_denominator_are_fixed() -> None:
    plans = (
        SimpleNamespace(
            label="up", kind="dc_source", source="V1",
            start=0.65, stop=0.715, step=0.005,
        ),
        SimpleNamespace(
            label="dn", kind="dc_source", source="V1",
            start=0.65, stop=0.585, step=-0.005,
        ),
    )
    grids = [validate_sweep_plan(plan) for plan in plans]
    assert [len(grid) for grid in grids] == [14, 14]
    assert sum(len(grid) for grid in grids) == EXPECTED_STATES

    changed = SimpleNamespace(**vars(plans[0]))
    changed.source = "V2"
    with pytest.raises(ValueError, match="plan changed"):
        validate_sweep_plan(changed)


def test_arm_scale_tracks_largest_physical_node_contribution() -> None:
    state = {"out": 0.48, "gate": 0.3, "vdd": 0.65, "0": 0.0}
    arms = compute_arm_scale(
        [state], ["out", "gate"], [_FakeDevice()],
        [Resistor("Rout", ["out", "0"], 1e3)],
        [CurrentSource("Iload", ["vdd", "out"], 3e-3)],
    )
    assert arms.shape == (1, 2)
    assert arms[0, 0] == pytest.approx(3e-3)
    assert arms[0, 1] == pytest.approx(1e-4)
    assert GMIN_TERMINAL_PAIRS.tolist() == [[0, 2], [0, 3], [2, 3]]


@pytest.mark.skipif(
    os.environ.get("PYCIRCUITSIM_RUN_PRIVATE_INTEGRATION") != "1",
    reason="requires the private TSMC5 card, OSDI binary, and NGSPICE 45.2",
)
def test_real_harvest_uses_only_fresh_ldo_1_line_max_truth(
    tmp_path: Path,
) -> None:
    """Opt-in: exercise both real translations and both fresh raw sweeps."""
    root = Path(__file__).resolve().parents[1]
    checkpoint_dir = root / "results" / "v764_terminal_l_matched_checkpoints"
    args = build_parser().parse_args([
        "--parent-nmos", str(checkpoint_dir / "tsmc5_dnf_medium_nmos_best.pt"),
        "--parent-pmos", str(checkpoint_dir / "tsmc5_dnf_medium_pmos_best.pt"),
        "--work-dir", str(tmp_path / "work"),
        "--output", str(tmp_path / "ldo.npz"),
    ])
    environment_names = (
        "AG_TREE", "PYCIRCUITSIM_NN_CHECKPOINT_DNF_NMOS",
        "PYCIRCUITSIM_NN_CHECKPOINT_DNF_PMOS",
        "PYCIRCUITSIM_NN_STRICT_TECH_CODE",
    )
    environment_before = {name: os.environ.get(name) for name in environment_names}
    artifact, arrays = harvest(args)
    assert {
        name: os.environ.get(name) for name in environment_names
    } == environment_before
    assert artifact.is_file()
    assert arrays["free_voltage_l72"].shape == (28, 8)
    assert arrays["fixed_voltage_l72"].shape == (28, 4)
    assert arrays["mos_term_free"].shape == (9, 4)
    assert arrays["state_group"].tolist() == list(range(28))
    assert set(arrays["group_direction"].tolist()) == {"up", "dn"}
    assert np.array_equal(arrays["vout_target"],
                          arrays["free_voltage_l72"][:, int(arrays["vout_index"])])
    provenance = json.loads(str(arrays["provenance_json"]))
    assert provenance["scope"]["design"] == "ldo_1"
    assert provenance["scope"]["deck"] == "tb_line_max.cir"
    assert provenance["scope"]["excluded_designs"] == ["ldo_2"]
    assert len(provenance["raw_sweeps"]) == 2
    marker = json.loads(artifact.with_suffix(".npz.complete").read_text())
    assert marker["artifact_sha256"]
    assert set(marker["parent_checkpoints"]) == {"nmos", "pmos"}
