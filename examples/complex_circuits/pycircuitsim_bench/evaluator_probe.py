"""Truth-relative, same-state compact-model evaluator probe.

This module is deliberately measurement-only.  It rebuilds one translated
AnalogGym deck at LEVEL=72 and LEVEL=73, evaluates both circuits at caller-
supplied physical node states, and records the four Phase-1 evaluator seams:
the full OSDI terminal interface, exact reduced OSDI interface, raw DirectNet,
and production DirectNet.  It never runs Newton or changes the simulator's
default evaluator selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import CONTROLS, SimFailure
from . import run_compare
from . import translate
from .provenance import artifact_record
from pycircuitsim.config import DEFAULT_TEMPERATURE


SCHEMA_VERSION: str = "pycircuitsim-evaluator-probe-v1"
_TERMINAL_ORDER: Tuple[str, ...] = ("d", "g", "s", "b")
_CAP_KEYS: Tuple[str, ...] = ("cgg", "cgd", "cgs", "cdg", "cdd")
_CHARGE_KEYS: Tuple[str, ...] = ("qg", "qd", "qs", "qb")


@dataclass(frozen=True)
class ProbeState:
    """One explicit physical node-voltage state and optional temperature."""

    label: str
    nodes: Dict[str, float]
    temperature_c: Optional[float] = None


def _finite_float(value: object, context: str) -> float:
    """Return one finite scalar, rejecting strings, booleans, and NaN/Inf."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _state_from_mapping(value: object, index: int) -> ProbeState:
    """Parse one explicit state record without accepting ambiguous aliases."""
    if not isinstance(value, Mapping):
        raise ValueError(f"state {index} must be a JSON object")
    nodes_value = value.get("nodes")
    if not isinstance(nodes_value, Mapping) or not nodes_value:
        raise ValueError(f"state {index} must contain a non-empty 'nodes' object")
    nodes: Dict[str, float] = {}
    for raw_name, raw_voltage in nodes_value.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"state {index} contains an invalid node name")
        name = raw_name.strip()
        folded = name.lower()
        if folded in nodes:
            raise ValueError(
                f"state {index} repeats node {name!r} case-insensitively")
        nodes[folded] = _finite_float(
            raw_voltage, f"state {index} node {name!r}")
    for ground in ("0", "gnd"):
        if ground in nodes and nodes[ground] != 0.0:
            raise ValueError(
                f"state {index} ground node {ground!r} must be exactly 0 V")

    label_value = value.get("label", f"state-{index}")
    if not isinstance(label_value, str) or not label_value.strip():
        raise ValueError(f"state {index} label must be a non-empty string")
    temp_value = value.get("temperature_c")
    temperature = (
        None if temp_value is None
        else _finite_float(temp_value, f"state {index} temperature_c")
    )
    return ProbeState(label_value.strip(), nodes, temperature)


def _run_result_state(data: Mapping[str, object]) -> Optional[ProbeState]:
    """Extract a state only when a run-result JSON contains exactly one OP."""
    candidates: List[Tuple[str, Mapping[str, object]]] = []
    for side in ("pycircuitsim", "ngspice"):
        side_value = data.get(side)
        if not isinstance(side_value, Mapping):
            continue
        direct = side_value.get("operating_point")
        if isinstance(direct, Mapping):
            candidates.append((f"{side}-operating-point", direct))
        sweeps = side_value.get("sweeps")
        if isinstance(sweeps, list):
            for sweep_index, sweep in enumerate(sweeps):
                if not isinstance(sweep, Mapping):
                    continue
                operating_point = sweep.get("operating_point")
                if isinstance(operating_point, Mapping):
                    candidates.append((
                        f"{side}-sweep-{sweep_index}-operating-point",
                        operating_point,
                    ))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(
            "run_compare result contains more than one operating point; "
            "write the intended state explicitly under a top-level 'states' list"
        )
    label, nodes = candidates[0]
    return _state_from_mapping({"label": label, "nodes": nodes}, 0)


def load_states(path: Path) -> List[ProbeState]:
    """Load explicit states or one unambiguous saved run_compare OP.

    Current scored run_compare rows contain sweep summaries, not complete node
    states.  Such a row is rejected rather than manufacturing a state from its
    scalar ``op_delta`` summary.  A producer may persist exactly one
    ``operating_point`` under either engine (or one sweep) and this loader will
    consume it.
    """
    resolved = path.resolve()
    try:
        data = json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load state JSON {resolved}: {exc}") from exc

    if isinstance(data, Mapping) and "states" in data:
        raw_states = data["states"]
        if not isinstance(raw_states, list) or not raw_states:
            raise ValueError("top-level 'states' must be a non-empty list")
        states = [
            _state_from_mapping(value, index)
            for index, value in enumerate(raw_states)
        ]
    elif isinstance(data, list):
        if not data:
            raise ValueError("state list must not be empty")
        states = [
            _state_from_mapping(value, index)
            for index, value in enumerate(data)
        ]
    elif isinstance(data, Mapping):
        state = _run_result_state(data)
        if state is None:
            raise ValueError(
                "run_compare JSON has no unambiguous saved operating point; "
                "provide explicit states with complete node voltages"
            )
        states = [state]
    else:
        raise ValueError("state JSON must be an object or list")

    labels = [state.label for state in states]
    if len(set(labels)) != len(labels):
        raise ValueError("state labels must be unique")
    return states


def _device_map(devices: Sequence[Any], family: str) -> Dict[str, Any]:
    """Index flattened devices by case-insensitive instance name."""
    out: Dict[str, Any] = {}
    for device in devices:
        name_value = getattr(device, "name", None)
        if not isinstance(name_value, str) or not name_value:
            raise ValueError(f"{family} device has no flattened instance name")
        key = name_value.lower()
        if key in out:
            raise ValueError(
                f"{family} contains duplicate instance name {name_value!r}")
        out[key] = device
    return out


def _folded_nodes(device: Any) -> Tuple[str, ...]:
    """Return one device's four-terminal topology in canonical casing."""
    nodes = getattr(device, "nodes", None)
    if not isinstance(nodes, (list, tuple)) or len(nodes) != 4:
        raise ValueError(
            f"device {getattr(device, 'name', '<unknown>')!r} must have "
            "four ordered terminals"
        )
    if not all(isinstance(node, str) and node for node in nodes):
        raise ValueError("device terminal names must be non-empty strings")
    return tuple(node.lower() for node in nodes)


def align_devices(
    osdi_devices: Sequence[Any], directnet_devices: Sequence[Any],
) -> List[Tuple[Any, Any]]:
    """Align LEVEL=72/73 devices by flattened name and ordered topology."""
    osdi = _device_map(osdi_devices, "LEVEL=72")
    directnet = _device_map(directnet_devices, "LEVEL=73")
    if set(osdi) != set(directnet):
        missing_nn = sorted(set(osdi) - set(directnet))
        missing_osdi = sorted(set(directnet) - set(osdi))
        raise ValueError(
            "device-name mismatch between translated levels: "
            f"missing LEVEL=73={missing_nn}, missing LEVEL=72={missing_osdi}"
        )

    pairs: List[Tuple[Any, Any]] = []
    for key in sorted(osdi):
        full_device = osdi[key]
        nn_device = directnet[key]
        if _folded_nodes(full_device) != _folded_nodes(nn_device):
            raise ValueError(
                f"terminal topology mismatch for {key}: "
                f"LEVEL=72={_folded_nodes(full_device)}, "
                f"LEVEL=73={_folded_nodes(nn_device)}"
            )
        full_m = _finite_float(getattr(full_device, "m", 1.0), f"{key} LEVEL=72 m")
        nn_m = _finite_float(getattr(nn_device, "m", 1.0), f"{key} LEVEL=73 m")
        if full_m != nn_m:
            raise ValueError(
                f"instance multiplier mismatch for {key}: "
                f"LEVEL=72={full_m:g}, LEVEL=73={nn_m:g}"
            )
        for attribute in ("L", "NFIN"):
            if hasattr(full_device, attribute) and hasattr(nn_device, attribute):
                full_value = _finite_float(
                    getattr(full_device, attribute), f"{key} LEVEL=72 {attribute}")
                nn_value = _finite_float(
                    getattr(nn_device, attribute), f"{key} LEVEL=73 {attribute}")
                if full_value != nn_value:
                    raise ValueError(
                        f"geometry mismatch for {key} {attribute}: "
                        f"LEVEL=72={full_value:g}, LEVEL=73={nn_value:g}"
                    )
        pairs.append((full_device, nn_device))
    return pairs


def _array(value: object, shape: Tuple[int, ...], context: str) -> np.ndarray:
    """Return one finite float64 array of the exact evaluator-contract shape."""
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{context} has shape {result.shape}, expected {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{context} contains NaN/Inf")
    return result


def _finite_dict(
    value: Mapping[str, object], keys: Sequence[str], context: str,
) -> Dict[str, float]:
    """Copy a named scalar block while rejecting missing/non-finite values."""
    out: Dict[str, float] = {}
    for key in keys:
        if key not in value:
            raise ValueError(f"{context} is missing {key!r}")
        out[key] = _finite_float(value[key], f"{context}.{key}")
    return out


def _configure(device: Any, boundary: str) -> None:
    """Set one explicit measurement seam and clear any incompatible cache."""
    configure = getattr(device, "configure_evaluator", None)
    if callable(configure):
        configure(boundary, correction_trace=False)
    else:
        device.evaluator_boundary = boundary
        clear = getattr(device, "clear_cache", None)
        if callable(clear):
            clear()


def _nn_payload(
    device: Any, voltages: Dict[str, float], boundary: str,
) -> Dict[str, Any]:
    """Evaluate one DirectNet seam through the public solver-facing APIs."""
    _configure(device, boundary)
    capacitances = _finite_dict(
        device.get_capacitances(voltages), _CAP_KEYS,
        f"{device.name} {boundary} capacitances",
    )
    # get_capacitances guarantees the cache contains the qg/qd autograd block.
    raw_result = device._eval(voltages)
    if not isinstance(raw_result, Mapping):
        raise ValueError(f"{device.name} {boundary} returned no evaluator mapping")
    multiplier = _finite_float(getattr(device, "m", 1.0), f"{device.name} m")
    model_id = multiplier * _finite_float(
        raw_result.get("id"), f"{device.name} {boundary} id")
    drain_current = _finite_float(
        device.calculate_current(voltages),
        f"{device.name} {boundary} drain current",
    )
    gds, gm, gmb = device.get_conductance(voltages)
    conductance = {
        "gds": _finite_float(gds, f"{device.name} {boundary} gds"),
        "gm": _finite_float(gm, f"{device.name} {boundary} gm"),
        "gmb": _finite_float(gmb, f"{device.name} {boundary} gmb"),
    }
    charges = _finite_dict(
        device.get_charges(voltages), _CHARGE_KEYS,
        f"{device.name} {boundary} charges",
    )
    return {
        "model_id": model_id,
        "drain_current": drain_current,
        "conductance": conductance,
        "charges": charges,
        "charge_closure": float(sum(charges.values())),
        "capacitances": capacitances,
    }


def compare_device_at_state(
    osdi_device: Any,
    directnet_device: Any,
    osdi_voltages: Dict[str, float],
    directnet_voltages: Dict[str, float],
) -> Dict[str, Any]:
    """Compare all four evaluator boundaries for one aligned device/state."""
    if _folded_nodes(osdi_device) != _folded_nodes(directnet_device):
        raise ValueError(
            f"terminal topology mismatch for {osdi_device.name!r}")
    multiplier = _finite_float(
        getattr(osdi_device, "m", 1.0), f"{osdi_device.name} multiplier")
    nn_multiplier = _finite_float(
        getattr(directnet_device, "m", 1.0),
        f"{directnet_device.name} multiplier",
    )
    if multiplier != nn_multiplier:
        raise ValueError(f"instance multiplier mismatch for {osdi_device.name}")

    _configure(osdi_device, "native")
    currents_value, current_jacobian_value = osdi_device.get_terminal_stamp(
        osdi_voltages)
    charges_value, charge_jacobian_value = osdi_device.get_charge_stamp(
        osdi_voltages)
    currents = _array(
        currents_value, (4,), f"{osdi_device.name} terminal currents")
    current_jacobian = _array(
        current_jacobian_value, (4, 4),
        f"{osdi_device.name} current Jacobian",
    )
    charges = _array(charges_value, (4,), f"{osdi_device.name} charges")
    charge_jacobian = _array(
        charge_jacobian_value, (4, 4),
        f"{osdi_device.name} charge Jacobian",
    )

    # Exact reduced OSDI uses calculate_current(), which is explicitly backed
    # by OSDI terminal `id`.  Do not derive or expose the channel `ids` opvar.
    _configure(osdi_device, "reduced-osdi")
    drain_current = _finite_float(
        osdi_device.calculate_current(osdi_voltages),
        f"{osdi_device.name} reduced drain current",
    )
    gds, gm, gmb = osdi_device.get_conductance(osdi_voltages)
    reduced = {
        "osdi_terminal_id": float(-currents[0]),
        "drain_current": drain_current,
        "conductance": {
            "gds": _finite_float(gds, f"{osdi_device.name} reduced gds"),
            "gm": _finite_float(gm, f"{osdi_device.name} reduced gm"),
            "gmb": _finite_float(gmb, f"{osdi_device.name} reduced gmb"),
        },
    }

    raw_directnet = _nn_payload(
        directnet_device, directnet_voltages, "raw-directnet")
    production_directnet = _nn_payload(
        directnet_device, directnet_voltages, "native")

    def _reduced_delta(candidate: Mapping[str, Any]) -> Dict[str, Any]:
        """Candidate minus exact reduced OSDI in solver-consumed quantities."""
        candidate_g = candidate["conductance"]
        reference_g = reduced["conductance"]
        return {
            "model_id": float(
                candidate["model_id"] - reduced["osdi_terminal_id"]),
            "drain_current": float(
                candidate["drain_current"] - reduced["drain_current"]),
            "conductance": {
                key: float(candidate_g[key] - reference_g[key])
                for key in ("gds", "gm", "gmb")
            },
        }

    full_charge = {
        key: float(charges[index])
        for index, key in enumerate(("qd", "qg", "qs", "qb"))
    }
    comparisons = {
        "delta_direction": "candidate_minus_reference",
        "full_to_reduced_osdi": {
            # Exact reduced OSDI keeps the drain terminal `id` value but routes
            # its opposite solely through source. Gate/bulk rows disappear.
            "drain_terminal_current": 0.0,
            "omitted_gate_terminal_current": float(currents[1]),
            "omitted_bulk_terminal_current": float(currents[3]),
            "reduced_source_minus_full_source": float(
                -currents[0] - currents[2]),
        },
        "reduced_osdi_to_raw_directnet": _reduced_delta(raw_directnet),
        "reduced_osdi_to_production_directnet": _reduced_delta(
            production_directnet),
        "raw_to_production_directnet": {
            "model_id": float(
                production_directnet["model_id"] - raw_directnet["model_id"]),
            "drain_current": float(
                production_directnet["drain_current"]
                - raw_directnet["drain_current"]),
            "conductance": {
                key: float(
                    production_directnet["conductance"][key]
                    - raw_directnet["conductance"][key])
                for key in ("gds", "gm", "gmb")
            },
        },
        "full_osdi_to_raw_directnet_charge": {
            key: float(raw_directnet["charges"][key] - full_charge[key])
            for key in _CHARGE_KEYS
        },
        "full_osdi_to_production_directnet_charge": {
            key: float(
                production_directnet["charges"][key] - full_charge[key])
            for key in _CHARGE_KEYS
        },
    }

    return {
        "instance": str(osdi_device.name).lower(),
        "terminals": list(_folded_nodes(osdi_device)),
        "terminal_order": list(_TERMINAL_ORDER),
        "multiplier": multiplier,
        "geometry": {
            "L": _finite_float(getattr(osdi_device, "L"), "L"),
            "NFIN": _finite_float(getattr(osdi_device, "NFIN"), "NFIN"),
        },
        "full_osdi": {
            "terminal_currents": currents.tolist(),
            "current_jacobian": current_jacobian.tolist(),
            "current_closure": float(np.sum(currents)),
            "charges": charges.tolist(),
            "charge_jacobian": charge_jacobian.tolist(),
            "charge_closure": float(np.sum(charges)),
        },
        "reduced_osdi": reduced,
        "raw_directnet": raw_directnet,
        "production_directnet": production_directnet,
        "comparisons": comparisons,
    }


def _state_for_circuit(circuit: Any, state: ProbeState) -> Dict[str, float]:
    """Map a case-insensitive complete physical state onto native node names."""
    table: Dict[str, str] = {}
    for node in circuit.get_nodes():
        key = node.lower()
        if key in table and table[key] != node:
            raise ValueError(
                f"circuit nodes {table[key]!r}/{node!r} differ only by case")
        table[key] = node
    supplied = set(state.nodes) - {"0", "gnd"}
    missing = sorted(set(table) - supplied)
    extra = sorted(supplied - set(table))
    if missing or extra:
        raise ValueError(
            f"state {state.label!r} does not match circuit nodes: "
            f"missing={missing}, unknown={extra}"
        )
    out = {native: state.nodes[key] for key, native in table.items()}
    out["0"] = 0.0
    out["GND"] = 0.0
    return out


def circuit_residual(
    circuit: Any,
    voltages: Dict[str, float],
    *,
    reltol: float = run_compare.NGSPICE_RELTOL,
) -> Dict[str, float]:
    """Measure the physical-GMIN DC residual with the branch tail fitted.

    The implementation deliberately reuses :meth:`DCSolver._dc_residual_at`.
    That helper solves the augmented ideal-voltage-source branch-current tail
    before measuring the residual and derives its tolerance scale only from
    current-valued node rows.
    """
    from pycircuitsim.solver import DCSolver, _RESID_ABS_FLOOR

    reltol_value = _finite_float(reltol, "reltol")
    if reltol_value <= 0.0:
        raise ValueError("reltol must be positive")
    nodes = list(circuit.get_nodes())
    required = set(nodes)
    missing = sorted(required - set(voltages))
    if missing:
        raise ValueError(f"residual state is missing circuit nodes {missing}")
    solver = DCSolver(
        circuit, use_source_stepping=False, reltol=reltol_value,
    )
    node_map = circuit.get_node_map()
    num_nodes = len(nodes)
    matrix_size = num_nodes + circuit.count_voltage_sources()
    residual_inf, current_scale = solver._dc_residual_at(
        voltages, node_map, nodes, num_nodes, matrix_size, solver.gmin)
    residual_inf = _finite_float(residual_inf, "circuit residual")
    current_scale = _finite_float(current_scale, "current RHS scale")
    tolerance = max(_RESID_ABS_FLOOR, 100.0 * reltol_value * current_scale)
    return {
        "residual_inf": residual_inf,
        "current_rhs_scale": current_scale,
        "tolerance": float(tolerance),
        "normalized": float(residual_inf / tolerance),
    }


def _configure_circuit(circuit: Any, boundary: str) -> None:
    """Select one measurement seam on every MOSFET in a circuit."""
    for device in run_compare._mosfets(circuit):
        _configure(device, boundary)


def _temperature_for_state(td: Any, state: ProbeState) -> Optional[float]:
    """Resolve a state temperature, rejecting ambiguous temperature sweeps."""
    if state.temperature_c is not None:
        return state.temperature_c
    if any(plan.kind == "dc_temp" for plan in td.plans):
        raise ValueError(
            f"state {state.label!r} needs temperature_c for a temperature-sweep deck"
        )
    if td.temp_c is not None:
        return float(td.temp_c)
    return float(DEFAULT_TEMPERATURE - 273.15)


def _set_temperature(circuit: Any, temperature_c: Optional[float]) -> None:
    """Apply one saved-state temperature through the shared bench helper."""
    if temperature_c is not None:
        run_compare._apply_temperature(circuit, temperature_c)


def _loaded_directnet_provenance(circuit: Any) -> List[Dict[str, Any]]:
    """Fingerprint the exact checkpoint/norm files the parser actually loaded."""
    records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for device in run_compare._mosfets(circuit):
        model_key = getattr(device, "_model_key", None)
        norm_key = getattr(device, "_norm_key", None)
        if not isinstance(model_key, tuple) or not isinstance(norm_key, tuple):
            raise SimFailure(
                f"DirectNet device {device.name!r} exposes no loaded artifact keys")
        checkpoint = Path(str(model_key[0]))
        norm = Path(str(norm_key[0]))
        key = (str(checkpoint.resolve()), str(norm.resolve()))
        if key not in records:
            records[key] = {
                "checkpoint": artifact_record(checkpoint),
                "normalization": artifact_record(norm),
                "device_classes": [],
                "tech_codes": [],
            }
        device_class = type(device).__name__
        tech_code = int(getattr(device, "_tech_code"))
        if device_class not in records[key]["device_classes"]:
            records[key]["device_classes"].append(device_class)
        if tech_code not in records[key]["tech_codes"]:
            records[key]["tech_codes"].append(tech_code)
    return [records[key] for key in sorted(records)]


def _code_commit() -> Optional[str]:
    """Return the pinned campaign commit or the local repository HEAD."""
    pinned = os.environ.get("PYCIRCUITSIM_BENCH_CODE_COMMIT")
    if pinned:
        return pinned
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=run_compare.REPO_ROOT,
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def probe_deck(
    *,
    root: Path,
    tech: str,
    category: str,
    design: str,
    deck: str,
    state_path: Path,
    work: Path,
    reltol: float = run_compare.NGSPICE_RELTOL,
) -> Dict[str, Any]:
    """Rebuild one deck at LEVEL=72/73 and probe all supplied states."""
    tech_key = tech.lower()
    if category not in CONTROLS:
        raise ValueError(f"unsupported AnalogGym category {category!r}")
    design_dir = root / f"designs_{tech_key}" / category / design
    if not design_dir.is_dir():
        raise FileNotFoundError(f"AnalogGym design directory not found: {design_dir}")
    if not (design_dir / deck).is_file():
        raise FileNotFoundError(f"AnalogGym deck not found: {design_dir / deck}")

    run_compare._pin_design_tree(root, tech_key)
    states = load_states(state_path)
    td72 = translate.translate_deck(
        design_dir, deck, tech=tech_key, category=category, model_level=72)
    td73 = translate.translate_deck(
        design_dir, deck, tech=tech_key, category=category, model_level=73)
    work.mkdir(parents=True, exist_ok=True)
    circuit72, netlist72 = run_compare.build_circuit(td72, work / "level72")
    circuit73, netlist73 = run_compare.build_circuit(td73, work / "level73")
    pairs = align_devices(
        run_compare._mosfets(circuit72), run_compare._mosfets(circuit73))
    effective_reltol = (
        td72.options.reltol if td72.options.reltol is not None else reltol)
    state_rows: List[Dict[str, Any]] = []
    for state in states:
        temperature_c = _temperature_for_state(td72, state)
        _set_temperature(circuit72, temperature_c)
        _set_temperature(circuit73, temperature_c)
        voltages72 = _state_for_circuit(circuit72, state)
        voltages73 = _state_for_circuit(circuit73, state)
        devices = [
            compare_device_at_state(full, nn, voltages72, voltages73)
            for full, nn in pairs
        ]

        residuals: Dict[str, Dict[str, float]] = {}
        for boundary, circuit, voltages in (
            ("full_osdi", circuit72, voltages72),
            ("reduced_osdi", circuit72, voltages72),
            ("raw_directnet", circuit73, voltages73),
            ("production_directnet", circuit73, voltages73),
        ):
            evaluator = {
                "full_osdi": "native",
                "reduced_osdi": "reduced-osdi",
                "raw_directnet": "raw-directnet",
                "production_directnet": "native",
            }[boundary]
            _configure_circuit(circuit, evaluator)
            residuals[boundary] = circuit_residual(
                circuit, voltages, reltol=effective_reltol)
        state_rows.append({
            "label": state.label,
            "temperature_c": temperature_c,
            "nodes": {key: state.nodes[key] for key in sorted(state.nodes)},
            "residuals": residuals,
            "devices": devices,
        })

    return {
        "schema": SCHEMA_VERSION,
        "kind": "same-state-evaluator-boundary-probe",
        "tech": tech_key,
        "category": category,
        "design": design,
        "deck": deck,
        "device_count": len(pairs),
        # This is the unflattened source-card count. It can legitimately be
        # smaller than device_count when one subcircuit is instantiated more
        # than once, so it is evidence rather than an equality assertion.
        "source_device_cards": td72.devices,
        "state_count": len(state_rows),
        "provenance": {
            "code_commit": _code_commit(),
            "probe_module": artifact_record(Path(__file__)),
            "state_source": artifact_record(state_path),
            "source_deck": artifact_record(design_dir / deck),
            "modelcard": artifact_record(td72.modelcard_path),
            "translated_level72": artifact_record(netlist72),
            "translated_level73": artifact_record(netlist73),
            "ground_truth": run_compare._reference_provenance(td72),
            "level72": run_compare._model_provenance(td72),
            "level73": run_compare._model_provenance(td73),
            "loaded_directnet": _loaded_directnet_provenance(circuit73),
            "checkpoint_pin_environment": {
                name: os.environ[name]
                for name in (
                    "PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS",
                    "PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS",
                    "PYCIRCUITSIM_NN_CHECKPOINT_NMOS",
                    "PYCIRCUITSIM_NN_CHECKPOINT_PMOS",
                    "PYCIRCUITSIM_NN_CHECKPOINT_OVERRIDE",
                )
                if os.environ.get(name)
            },
        },
        "residual_contract": {
            "physical_gmin_s": 1e-12,
            "reltol": effective_reltol,
            "tolerance": "max(1e-6 A, 100*reltol*current_node_rhs_inf)",
            "ideal_source_tail": "least-squares fitted",
        },
        "states": state_rows,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI for one AnalogGym deck and an explicit JSON state artifact."""
    parser = argparse.ArgumentParser(
        description=(
            "Probe full/reduced OSDI and raw/production DirectNet at the "
            "same saved physical node states. DirectNet checkpoints follow "
            "the existing PYCIRCUITSIM_NN_CHECKPOINT_* environment pins."
        ))
    parser.add_argument("--root", type=Path, default=run_compare.BENCH_ROOT)
    parser.add_argument("--tech", required=True)
    parser.add_argument("--category", required=True, choices=sorted(CONTROLS))
    parser.add_argument("--design", required=True)
    parser.add_argument("--deck", required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--reltol", type=float, default=run_compare.NGSPICE_RELTOL,
        help="residual tolerance scale when the deck has no .options reltol",
    )
    args = parser.parse_args(argv)
    report = probe_deck(
        root=args.root, tech=args.tech, category=args.category,
        design=args.design, deck=args.deck, state_path=args.states,
        work=args.work, reltol=args.reltol,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
