#!/usr/bin/env python3
"""Harvest one NGSPICE-grounded LDO topology for unrolled DirectNet training.

Only ``TSMC5/ldo_1/tb_line_max.cir`` is admitted.  Node targets come from
fresh NGSPICE LEVEL=72 rawfiles; the native LEVEL=72 circuit is built only to
verify that its flattened topology matches the LEVEL=75 training circuit.
No PyCircuitSim solve or device evaluation supplies a target.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "external_compact_models"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from examples.complex_circuits.pycircuitsim_bench import (  # noqa: E402
    run_compare,
    translate,
)
from examples.complex_circuits.pycircuitsim_bench.evaluator_probe import (  # noqa: E402
    align_devices,
)
from examples.complex_circuits.pycircuitsim_bench.provenance import (  # noqa: E402
    artifact_record,
    file_sha256,
)
from pycircuitsim.models.passive import (  # noqa: E402
    Capacitor,
    CurrentSource,
    Resistor,
    VoltageSource,
)
from pycircuitsim.solver import _is_pmos  # noqa: E402


SCHEMA_VERSION = 1
SCHEMA_NAME = "v768-dnf-unrolled-ldo"
TECH = "tsmc5"
CATEGORY = "ldo"
DESIGN = "ldo_1"
DECK = "tb_line_max.cir"
OUTPUT_NODE = "vo"
GMIN = 1e-12
GMIN_TERMINAL_PAIRS = np.asarray(((0, 2), (0, 3), (2, 3)), dtype=np.int64)
TERMINAL_ORDER = ("d", "g", "s", "b")
EXPECTED_STATES = 28
EXPECTED_SWEEPS = {
    "up": ("v1", 0.65, 0.715, 0.005, 14),
    "dn": ("v1", 0.65, 0.585, -0.005, 14),
}
_SCOPED_ENVIRONMENT = (
    "AG_TREE",
    "PYCIRCUITSIM_NN_CHECKPOINT_DNF_NMOS",
    "PYCIRCUITSIM_NN_CHECKPOINT_DNF_PMOS",
    "PYCIRCUITSIM_NN_STRICT_TECH_CODE",
)


def checkpoint_override_prefix(path: Path, device: str) -> str:
    """Return the explicit parser pin for one full-terminal checkpoint."""
    suffix = f"_{device}_best.pt"
    value = str(path.resolve())
    if not value.endswith(suffix):
        raise ValueError(f"checkpoint must end in {suffix!r}: {path}")
    return value.removesuffix("_best.pt")


def validate_sweep_plan(plan: Any) -> np.ndarray:
    """Return the one admitted 14-point line-regulation sweep grid."""
    expected = EXPECTED_SWEEPS.get(str(plan.label))
    if expected is None:
        raise ValueError(f"unexpected tb_line_max plan label {plan.label!r}")
    source, start, stop, step, points = expected
    actual = (str(plan.source).lower(), plan.start, plan.stop, plan.step)
    wanted = (source, start, stop, step)
    if actual != wanted or plan.kind != "dc_source":
        raise ValueError(
            f"tb_line_max {plan.label} plan changed: {actual!r}, "
            f"expected {wanted!r}")
    grid = np.asarray(run_compare.plan_points(plan), dtype=np.float64)
    if grid.shape != (points,):
        raise ValueError(
            f"tb_line_max {plan.label} must contain {points} points, "
            f"got {len(grid)}")
    return grid


def aligned_sweep_state(circuit: Any, sweep: Any, index: int) -> dict[str, float]:
    """Return one raw NGSPICE point after an exact physical-node-set check."""
    native_by_folded: dict[str, str] = {}
    for raw_node in circuit.get_nodes():
        node = str(raw_node)
        folded = node.lower()
        if folded in native_by_folded and native_by_folded[folded] != node:
            raise ValueError(
                f"circuit nodes differ only by case: "
                f"{native_by_folded[folded]!r}/{node!r}")
        native_by_folded[folded] = node

    supplied: dict[str, np.ndarray] = {}
    for raw_name, raw_values in sweep.v.items():
        name = str(raw_name).lower()
        if "#" in name:
            continue
        if name in supplied:
            raise ValueError(f"NGSPICE rawfile repeats node {name!r}")
        supplied[name] = np.asarray(raw_values)
    missing = sorted(set(native_by_folded) - set(supplied))
    unknown = sorted(set(supplied) - set(native_by_folded))
    if missing or unknown:
        raise ValueError(
            "NGSPICE state does not exactly match translated nodes: "
            f"missing={missing}, unknown={unknown}")

    state: dict[str, float] = {"0": 0.0}
    for folded, native in native_by_folded.items():
        values = supplied[folded]
        if index < 0 or index >= len(values):
            raise IndexError(f"raw point {index} is outside node {folded!r}")
        value = float(np.real(values[index]))
        if not math.isfinite(value):
            raise ValueError(f"raw point {index} node {folded!r} is not finite")
        state[native] = value
    return state


def partition_nodes(circuit: Any) -> tuple[list[str], list[str]]:
    """Partition voltage-source-fixed nodes from nonlinear free nodes.

    The compact residual has no MNA branch-current tail, so it supports only
    ideal voltage sources tied directly to ground.  Ground is fixed node zero.
    """
    nodes = [str(node) for node in circuit.get_nodes()]
    if len({node.lower() for node in nodes}) != len(nodes):
        raise ValueError("circuit node names must be case-insensitively unique")
    canonical = {node.lower(): node for node in nodes}
    fixed = ["0"]
    claimed: dict[str, str] = {}
    for component in circuit.components:
        if not isinstance(component, VoltageSource):
            continue
        if type(component) is not VoltageSource:
            raise ValueError(
                f"unsupported voltage-source component {component.name}: "
                f"{type(component).__name__}")
        positive, negative = (str(node) for node in component.nodes)
        positive_ground = positive.lower() in ("0", "gnd")
        negative_ground = negative.lower() in ("0", "gnd")
        if positive_ground == negative_ground:
            raise ValueError(
                f"unsupported voltage-source topology {component.name}: "
                f"{positive!r}, {negative!r}; exactly one terminal must be ground")
        non_ground = negative if positive_ground else positive
        folded = non_ground.lower()
        if folded not in canonical:
            raise ValueError(
                f"voltage source {component.name} names unknown node {non_ground!r}")
        if folded in claimed:
            raise ValueError(
                f"fixed node {non_ground!r} is driven by both "
                f"{claimed[folded]!r} and {component.name!r}")
        claimed[folded] = str(component.name)
        fixed.append(canonical[folded])
    fixed_folded = {node.lower() for node in fixed}
    free = [node for node in nodes if node.lower() not in fixed_folded]
    if not free:
        raise ValueError("topology has no free nonlinear nodes")
    return free, fixed


def terminal_incidence(
    terminals: Sequence[Sequence[str]],
    free_nodes: Sequence[str],
    fixed_nodes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Encode terminal nodes into mutually exclusive free/fixed indices."""
    free_index = {str(node).lower(): index for index, node in enumerate(free_nodes)}
    fixed_index = {
        str(node).lower(): index for index, node in enumerate(fixed_nodes)
    }
    if set(free_index) & set(fixed_index):
        raise ValueError("free and fixed node sets overlap")
    free = np.full((len(terminals), 4 if terminals and len(terminals[0]) == 4 else 2),
                   -1, dtype=np.int64)
    fixed = np.full_like(free, -1)
    for row, nodes in enumerate(terminals):
        if len(nodes) != free.shape[1]:
            raise ValueError("terminal rows must have one consistent width")
        for column, raw_node in enumerate(nodes):
            node = str(raw_node).lower()
            in_free = node in free_index
            in_fixed = node in fixed_index
            if in_free == in_fixed:
                raise ValueError(
                    f"terminal node {raw_node!r} maps to "
                    f"{int(in_free) + int(in_fixed)} partitions")
            if in_free:
                free[row, column] = free_index[node]
            else:
                fixed[row, column] = fixed_index[node]
    return free, fixed


def validate_voltage_source_state(
    circuit: Any,
    state: Mapping[str, float],
    *,
    swept_source: str,
    swept_value: float,
    atol: float = 2e-9,
) -> None:
    """Require every raw fixed-node voltage to satisfy its source constraint."""
    matches = 0
    for component in circuit.components:
        if not isinstance(component, VoltageSource):
            continue
        if type(component) is not VoltageSource:
            raise ValueError(
                f"unsupported voltage-source component {component.name}: "
                f"{type(component).__name__}")
        value = float(component.voltage)
        if str(component.name).lower() == swept_source.lower():
            value = float(swept_value)
            matches += 1
        positive, negative = (str(node) for node in component.nodes)
        actual = float(state.get(positive, 0.0) - state.get(negative, 0.0))
        if not math.isclose(actual, value, rel_tol=0.0, abs_tol=atol):
            raise ValueError(
                f"raw state violates {component.name}: "
                f"V+ - V-={actual:.17g}, expected {value:.17g}")
    if matches != 1:
        raise ValueError(
            f"DC plan source {swept_source!r} matched {matches} voltage sources")


def validate_component_scope(circuit: Any, mos_devices: Sequence[Any]) -> None:
    """Reject a component whose DC contribution is absent from the schema."""
    supported_ids = {id(device) for device in mos_devices}
    for component in circuit.components:
        if id(component) in supported_ids:
            continue
        if type(component) in (Resistor, CurrentSource, Capacitor, VoltageSource):
            continue
        raise ValueError(
            f"unsupported DC component {component.name}: "
            f"{type(component).__name__}")


def validate_device_support(
    states: Sequence[Mapping[str, float]],
    devices: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reject exact states outside a parent device's certified input support."""
    support_min: list[np.ndarray] = []
    support_max: list[np.ndarray] = []
    minimum_margin = np.full(len(devices), np.inf, dtype=np.float64)
    for device_index, device in enumerate(devices):
        lower = np.asarray(device._norm_stats.input_min, dtype=np.float64)
        upper = np.asarray(device._norm_stats.input_max, dtype=np.float64)
        if lower.shape != (7,) or upper.shape != (7,):
            raise ValueError(f"{device.name} parent support must have 7 inputs")
        support_min.append(lower)
        support_max.append(upper)
        for state_index, state in enumerate(states):
            raw = np.asarray(device._raw_inputs(dict(state)), dtype=np.float64)
            if raw.shape != (7,) or not np.all(np.isfinite(raw)):
                raise ValueError(
                    f"{device.name} state {state_index} has invalid raw inputs")
            outside = np.flatnonzero((raw < lower) | (raw > upper))
            if outside.size:
                column = int(outside[0])
                raise ValueError(
                    f"support excursion at state {state_index}, {device.name}, "
                    f"input {column}: {raw[column]} outside "
                    f"[{lower[column]}, {upper[column]}]")
            span = upper - lower
            variable = span > 0.0
            if not np.any(variable):
                raise ValueError(f"{device.name} parent support has no span")
            margin = (
                np.minimum(raw[variable] - lower[variable],
                           upper[variable] - raw[variable])
                / span[variable]
            )
            minimum_margin[device_index] = min(
                minimum_margin[device_index], float(np.min(margin)))
    return (
        np.stack(support_min), np.stack(support_max), minimum_margin,
    )


def compute_arm_scale(
    states: Sequence[Mapping[str, float]],
    free_nodes: Sequence[str],
    mos_devices: Sequence[Any],
    resistors: Sequence[Resistor],
    current_sources: Sequence[CurrentSource],
    *,
    gmin: float = GMIN,
    floor: float = 1e-7,
) -> np.ndarray:
    """Return per-state node current arms from the frozen LEVEL=75 parent."""
    free_index = {str(node).lower(): index for index, node in enumerate(free_nodes)}
    out = np.full((len(states), len(free_nodes)), float(floor), dtype=np.float64)

    def include(row: np.ndarray, node: str, contribution: float) -> None:
        index = free_index.get(str(node).lower())
        if index is not None:
            row[index] = max(row[index], abs(float(contribution)))

    for state_index, state_mapping in enumerate(states):
        state = dict(state_mapping)
        row = out[state_index]
        for device in mos_devices:
            currents, _ = device.get_terminal_stamp(state)
            currents_array = np.asarray(currents, dtype=np.float64)
            if currents_array.shape != (4,) or not np.all(np.isfinite(currents_array)):
                raise ValueError(f"{device.name} parent terminal currents are invalid")
            for node, current in zip(device.nodes, currents_array):
                include(row, str(node), float(current))
            for first, second in GMIN_TERMINAL_PAIRS:
                node_a = str(device.nodes[int(first)])
                node_b = str(device.nodes[int(second)])
                current = gmin * (
                    float(state.get(node_a, 0.0)) - float(state.get(node_b, 0.0)))
                include(row, node_a, current)
                include(row, node_b, -current)
        for resistor in resistors:
            first, second = (str(node) for node in resistor.nodes)
            current = float(resistor.conductance) * (
                float(state.get(first, 0.0)) - float(state.get(second, 0.0)))
            include(row, first, current)
            include(row, second, -current)
        for source in current_sources:
            first, second = (str(node) for node in source.nodes)
            current = float(source.current)
            include(row, first, current)
            include(row, second, -current)
    if not np.all(np.isfinite(out)) or np.any(out < floor):
        raise ValueError("node arm scale is non-finite or below its floor")
    return out


def _component_map(circuit: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for component in circuit.components:
        key = str(component.name).lower()
        if key in result:
            raise ValueError(f"duplicate flattened component {component.name!r}")
        result[key] = component
    return result


def validate_translated_alignment(circuit72: Any, circuit75: Any) -> list[tuple[Any, Any]]:
    """Validate complete LEVEL=72/75 topology, not only MOS instance names."""
    map72 = _component_map(circuit72)
    map75 = _component_map(circuit75)
    if set(map72) != set(map75):
        raise ValueError("LEVEL=72/75 flattened component names differ")
    mos_pairs = align_devices(
        run_compare._mosfets(circuit72), run_compare._mosfets(circuit75))
    for device72, device75 in mos_pairs:
        if _is_pmos(device72) != _is_pmos(device75):
            raise ValueError(
                f"LEVEL=72/75 polarity differs for {device72.name}")
        if float(device72.temperature) != float(device75.temperature):
            raise ValueError(
                f"LEVEL=72/75 temperature differs for {device72.name}")
    mos_names = {str(pair[0].name).lower() for pair in mos_pairs}
    for name in sorted(set(map72) - mos_names):
        left = map72[name]
        right = map75[name]
        if type(left) is not type(right):
            raise ValueError(
                f"LEVEL=72/75 component type differs for {name}: "
                f"{type(left).__name__}/{type(right).__name__}")
        if tuple(str(node).lower() for node in left.nodes) != tuple(
                str(node).lower() for node in right.nodes):
            raise ValueError(f"LEVEL=72/75 terminals differ for {name}")
        if getattr(left, "value", None) != getattr(right, "value", None):
            raise ValueError(f"LEVEL=72/75 value differs for {name}")
    return mos_pairs


def _checkpoint_bundle(path: Path) -> dict[str, dict[str, str]]:
    checkpoint = path.resolve()
    norm = checkpoint.parent / (checkpoint.stem.replace("_best", "_norm") + ".npz")
    complete = checkpoint.with_suffix(checkpoint.suffix + ".complete")
    return {
        "checkpoint": artifact_record(checkpoint),
        "normalization": artifact_record(norm),
        "completion": artifact_record(complete),
    }


def _git_identity() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True, check=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip())
    return {"commit": commit, "dirty": dirty}


def _artifact_arrays(
    circuit75: Any,
    states: Sequence[Mapping[str, float]],
    state_group: np.ndarray,
    group_direction: Sequence[str],
    group_sweep: np.ndarray,
    group_raw_point: np.ndarray,
    provenance: Mapping[str, object],
) -> dict[str, np.ndarray]:
    free_nodes, fixed_nodes = partition_nodes(circuit75)
    if OUTPUT_NODE not in {node.lower() for node in free_nodes}:
        raise ValueError(f"output node {OUTPUT_NODE!r} is not a free node")
    mos_devices = sorted(run_compare._mosfets(circuit75), key=lambda item: item.name.lower())
    validate_component_scope(circuit75, mos_devices)
    resistors = sorted(
        (item for item in circuit75.components if isinstance(item, Resistor)),
        key=lambda item: item.name.lower(),
    )
    current_sources = sorted(
        (item for item in circuit75.components if isinstance(item, CurrentSource)),
        key=lambda item: item.name.lower(),
    )
    support_min, support_max, support_margin = validate_device_support(
        states, mos_devices)
    arm_scale = compute_arm_scale(
        states, free_nodes, mos_devices, resistors, current_sources)

    mos_free, mos_fixed = terminal_incidence(
        [device.nodes for device in mos_devices], free_nodes, fixed_nodes)
    resistor_free, resistor_fixed = terminal_incidence(
        [item.nodes for item in resistors], free_nodes, fixed_nodes)
    current_free, current_fixed = terminal_incidence(
        [item.nodes for item in current_sources], free_nodes, fixed_nodes)
    free_voltage = np.asarray([
        [float(state[node]) for node in free_nodes] for state in states
    ], dtype=np.float64)
    fixed_voltage = np.asarray([
        [float(state.get(node, 0.0)) for node in fixed_nodes] for state in states
    ], dtype=np.float64)
    output_index = next(
        index for index, node in enumerate(free_nodes)
        if node.lower() == OUTPUT_NODE)

    return {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "schema_name": np.asarray(SCHEMA_NAME),
        "node_names": np.asarray(["0", *[str(node) for node in circuit75.get_nodes()]]),
        "free_nodes": np.asarray(free_nodes),
        "fixed_nodes": np.asarray(fixed_nodes),
        "state_group": np.asarray(state_group, dtype=np.int64),
        "group_direction": np.asarray(group_direction),
        "group_segment_offsets": np.arange(len(states) + 1, dtype=np.int64),
        "group_sweep": np.asarray(group_sweep, dtype=np.float64),
        "group_raw_point": np.asarray(group_raw_point, dtype=np.int64),
        "free_voltage_l72": free_voltage,
        "fixed_voltage_l72": fixed_voltage,
        "arm_scale": arm_scale,
        "vout_index": np.asarray(output_index, dtype=np.int64),
        "vout_target": free_voltage[:, output_index],
        "output_node": np.asarray(OUTPUT_NODE),
        "mos_name": np.asarray([str(device.name) for device in mos_devices]),
        "mos_is_pmos": np.asarray([_is_pmos(device) for device in mos_devices]),
        "mos_term_free": mos_free,
        "mos_term_fixed": mos_fixed,
        "mos_length": np.asarray([device.L for device in mos_devices], dtype=np.float64),
        "mos_nfin": np.asarray([device.NFIN for device in mos_devices], dtype=np.float64),
        "mos_temperature": np.asarray(
            [device.temperature for device in mos_devices], dtype=np.float64),
        "mos_multiplier": np.asarray([device.m for device in mos_devices], dtype=np.float64),
        "mos_code": np.asarray([device._tech_code for device in mos_devices], dtype=np.int64),
        "mos_support_min": support_min,
        "mos_support_max": support_max,
        "mos_support_min_fractional_margin": support_margin,
        "resistor_name": np.asarray([str(item.name) for item in resistors]),
        "resistor_term_free": resistor_free,
        "resistor_term_fixed": resistor_fixed,
        "resistor_conductance": np.asarray(
            [item.conductance for item in resistors], dtype=np.float64),
        "current_name": np.asarray([str(item.name) for item in current_sources]),
        "current_term_free": current_free,
        "current_term_fixed": current_fixed,
        "current_value": np.asarray(
            [item.current for item in current_sources], dtype=np.float64),
        "gmin": np.asarray(GMIN, dtype=np.float64),
        "gmin_terminal_pairs": GMIN_TERMINAL_PAIRS.copy(),
        "gmin_scales_with_multiplier": np.asarray(False),
        "terminal_order": np.asarray(TERMINAL_ORDER),
        "provenance_json": np.asarray(json.dumps(provenance, sort_keys=True)),
    }


def _harvest(args: argparse.Namespace) -> tuple[Path, dict[str, np.ndarray]]:
    """Harvest, validate, and atomically publish the compact training artifact."""
    source_identity = _git_identity()
    if bool(source_identity["dirty"]):
        raise RuntimeError("LDO harvester source has tracked changes")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise RuntimeError(f"{variable}=1 is required for the fixed CPU arm")
    torch.set_num_threads(1)
    parent_nmos = Path(args.parent_nmos).resolve()
    parent_pmos = Path(args.parent_pmos).resolve()
    for path in (parent_nmos, parent_pmos):
        if not path.is_file():
            raise FileNotFoundError(path)
    os.environ["PYCIRCUITSIM_NN_CHECKPOINT_DNF_NMOS"] = (
        checkpoint_override_prefix(parent_nmos, "nmos"))
    os.environ["PYCIRCUITSIM_NN_CHECKPOINT_DNF_PMOS"] = (
        checkpoint_override_prefix(parent_pmos, "pmos"))
    os.environ["PYCIRCUITSIM_NN_STRICT_TECH_CODE"] = "1"

    bench_root = Path(args.bench_root).resolve()
    design_dir = bench_root / "designs_tsmc5" / CATEGORY / DESIGN
    source_deck = design_dir / DECK
    if not source_deck.is_file():
        raise FileNotFoundError(source_deck)
    run_compare._pin_design_tree(bench_root, TECH)
    td72 = translate.translate_deck(
        design_dir, DECK, tech=TECH, category=CATEGORY, model_level=72)
    td75 = translate.translate_deck(
        design_dir, DECK, tech=TECH, category=CATEGORY, model_level=75)
    if len(td72.plans) != 2 or {plan.label for plan in td72.plans} != {"up", "dn"}:
        raise ValueError("tb_line_max must translate to exactly up/dn DC plans")
    if td72.plans != td75.plans:
        raise ValueError("LEVEL=72/75 analysis plans differ")
    expected_grids = {
        str(plan.label): validate_sweep_plan(plan) for plan in td72.plans
    }

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    circuit72, netlist72 = run_compare.build_circuit(td72, work_dir / "l72")
    circuit75, netlist75 = run_compare.build_circuit(td75, work_dir / "l75")
    validate_translated_alignment(circuit72, circuit75)
    if partition_nodes(circuit72) != partition_nodes(circuit75):
        raise ValueError("LEVEL=72/75 free/fixed node partitions differ")

    states: list[dict[str, float]] = []
    state_group: list[int] = []
    directions: list[str] = []
    sweeps: list[float] = []
    raw_points: list[int] = []
    raw_records: list[dict[str, object]] = []
    for plan_index, plan in enumerate(td72.plans):
        plan_work = work_dir / f"ngspice_{plan_index}_{plan.label}"
        before = time.time_ns()
        sweep, _, seconds = run_compare.ngspice_sweep(
            td72, plan, plan_work, float(args.ng_timeout))
        raw_path = Path(str(sweep.meta["rawfile"])).resolve()
        if not raw_path.is_file() or raw_path.stat().st_mtime_ns < before:
            raise RuntimeError(f"NGSPICE did not create a fresh rawfile: {raw_path}")
        if sweep.kind != "dc":
            raise ValueError("NGSPICE raw sweep is not a DC transfer curve")
        raw_grid = np.asarray(sweep.x, dtype=np.float64)
        if not np.array_equal(raw_grid, expected_grids[str(plan.label)]):
            raise ValueError(
                f"NGSPICE {plan.label} abscissa differs from the fixed plan")
        for point_index, raw_sweep in enumerate(raw_grid):
            state = aligned_sweep_state(circuit72, sweep, point_index)
            validate_voltage_source_state(
                circuit72, state, swept_source=str(plan.source),
                swept_value=float(raw_sweep))
            state_group.append(len(states))
            directions.append(str(plan.label))
            sweeps.append(float(raw_sweep))
            raw_points.append(point_index)
            states.append(state)
        raw_records.append({
            "plan_index": plan_index,
            "label": plan.label,
            "control": plan.control,
            "points": int(len(sweep.x)),
            "seconds": float(seconds),
            "rawfile": artifact_record(raw_path),
        })
    if len(states) != EXPECTED_STATES:
        raise ValueError(
            f"expected {EXPECTED_STATES} accepted states, got {len(states)}")

    source_files = [source_deck, design_dir / "netlist.spice",
                    design_dir / "tsmc5_models.spice", design_dir / "design.json"]
    parent_bundles = {"nmos": _checkpoint_bundle(parent_nmos),
                      "pmos": _checkpoint_bundle(parent_pmos)}
    provenance: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "ground_truth": "NGSPICE LEVEL=72 identical BSIM-CMG OSDI",
        "scope": {"tech": TECH, "category": CATEGORY, "design": DESIGN,
                  "deck": DECK, "excluded_designs": ["ldo_2"],
                  "excluded_decks": ["tb_load.cir", "tb_line_min.cir"]},
        "source": source_identity,
        "harvester": artifact_record(Path(__file__)),
        "source_files": [artifact_record(path) for path in source_files if path.is_file()],
        "translated_level72": artifact_record(netlist72),
        "translated_level75": artifact_record(netlist75),
        "reference": run_compare._reference_provenance(td72),
        "parent": parent_bundles,
        "raw_sweeps": raw_records,
        "target_origin": "NGSPICE raw node voltages only",
        "arm_origin": "frozen LEVEL=75 parent terminal/passive/source currents",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "omp_num_threads": os.environ["OMP_NUM_THREADS"],
            "mkl_num_threads": os.environ["MKL_NUM_THREADS"],
        },
        "gmin_contract": {
            "siemens": GMIN,
            "terminal_pairs": GMIN_TERMINAL_PAIRS.tolist(),
            "scales_with_instance_multiplier": False,
        },
    }
    arrays = _artifact_arrays(
        circuit75, states, np.asarray(state_group, dtype=np.int64), directions,
        np.asarray(sweeps, dtype=np.float64),
        np.asarray(raw_points, dtype=np.int64), provenance)

    output = Path(args.output).resolve()
    marker = output.with_suffix(output.suffix + ".complete")
    if (output.exists() or marker.exists()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output} or {marker}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output)
    completion = {
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "artifact": output.name,
        "artifact_sha256": file_sha256(output),
        "states": len(states),
        "groups": len(states),
        "devices": int(len(arrays["mos_name"])),
        "parent_checkpoints": {
            polarity: {
                "checkpoint_sha256": bundle["checkpoint"]["sha256"],
                "normalization_sha256": bundle["normalization"]["sha256"],
                "completion_sha256": bundle["completion"]["sha256"],
            }
            for polarity, bundle in parent_bundles.items()
        },
        "harvester_sha256": file_sha256(Path(__file__)),
    }
    marker.write_text(json.dumps(completion, sort_keys=True, indent=2) + "\n")
    return output, arrays


def harvest(args: argparse.Namespace) -> tuple[Path, dict[str, np.ndarray]]:
    """Run one harvest without leaking its parser/checkpoint environment."""
    previous = {name: os.environ.get(name) for name in _SCOPED_ENVIRONMENT}
    try:
        return _harvest(args)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def build_parser() -> argparse.ArgumentParser:
    """Build the deliberately fixed-scope command-line interface."""
    checkpoint_dir = ROOT / "results" / "v764_terminal_l_matched_checkpoints"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-nmos", type=Path,
        default=checkpoint_dir / "tsmc5_dnf_medium_nmos_best.pt")
    parser.add_argument(
        "--parent-pmos", type=Path,
        default=checkpoint_dir / "tsmc5_dnf_medium_pmos_best.pt")
    parser.add_argument(
        "--bench-root", type=Path,
        default=ROOT / "examples" / "complex_circuits")
    parser.add_argument(
        "--work-dir", type=Path,
        default=ROOT / "results" / "v768_dnf_unrolled_ldo" / "work")
    parser.add_argument(
        "--output", type=Path,
        default=(ROOT / "results" / "v768_dnf_unrolled_ldo"
                 / "tsmc5_ldo_1_tb_line_max_unrolled.npz"))
    parser.add_argument("--ng-timeout", type=float, default=120.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed-scope harvest and print its content identity."""
    args = build_parser().parse_args(argv)
    output, arrays = harvest(args)
    print(json.dumps({
        "artifact": str(output),
        "sha256": file_sha256(output),
        "states": int(len(arrays["state_group"])),
        "devices": int(len(arrays["mos_name"])),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
